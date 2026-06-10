from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .text import token_counts


@dataclass(frozen=True)
class Chunk:
    id: int
    doc_id: int
    filename: str
    page_start: int
    page_end: int
    title: str
    category: str
    text: str
    tokens: dict[str, int]
    embedding: list[float] | None


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    bm25_rank: int | None
    vector_rank: int | None


@dataclass(frozen=True)
class ContextBlock:
    label: str
    filename: str
    page_start: int
    page_end: int
    category: str
    text: str

    @property
    def citation(self) -> str:
        if self.page_start == self.page_end:
            return f"{self.filename}, p.{self.page_start}"
        return f"{self.filename}, pp.{self.page_start}-{self.page_end}"


def load_chunks(conn: sqlite3.Connection) -> list[Chunk]:
    rows = conn.execute(
        """
        SELECT id, doc_id, filename, page_start, page_end, title, category, text,
               tokens_json, embedding_json
        FROM chunks
        """
    ).fetchall()
    chunks: list[Chunk] = []
    for row in rows:
        embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
        chunks.append(
            Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                filename=row["filename"],
                page_start=row["page_start"],
                page_end=row["page_end"],
                title=row["title"],
                category=row["category"],
                text=row["text"],
                tokens=json.loads(row["tokens_json"]),
                embedding=embedding,
            )
        )
    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def bm25_rank(query: str, chunks: list[Chunk], limit: int) -> list[tuple[int, float]]:
    query_terms = token_counts(query)
    if not query_terms:
        return []

    n_docs = len(chunks)
    doc_lengths = [sum(chunk.tokens.values()) for chunk in chunks]
    avgdl = sum(doc_lengths) / max(n_docs, 1)
    dfs: Counter[str] = Counter()
    for term in query_terms:
        dfs[term] = sum(1 for chunk in chunks if term in chunk.tokens)

    k1 = 1.5
    b = 0.75
    scored: list[tuple[int, float]] = []
    for chunk, doc_len in zip(chunks, doc_lengths):
        score = 0.0
        for term, query_tf in query_terms.items():
            tf = chunk.tokens.get(term, 0)
            if tf == 0:
                continue
            df = dfs[term]
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-9))
            score += idf * (tf * (k1 + 1) / denom) * min(query_tf, 3)
        if score > 0:
            scored.append((chunk.id, score))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def vector_rank(
    query: str,
    chunks: list[Chunk],
    embedder: Callable[[str], list[float]] | None,
    limit: int,
) -> list[tuple[int, float]]:
    if embedder is None:
        return []
    query_embedding = embedder(query)
    scored: list[tuple[int, float]] = []
    for chunk in chunks:
        if chunk.embedding is None:
            continue
        scored.append((chunk.id, cosine_similarity(query_embedding, chunk.embedding)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]


def rerank_search(
    conn: sqlite3.Connection,
    query: str,
    embedder: Callable[[str], list[float]],
    reranker: Callable[[str, list[str]], list[float]],
    top_k: int = 3,
    candidate_k: int = 30,
) -> list[SearchResult]:
    chunks = load_chunks(conn)
    by_id = {chunk.id: chunk for chunk in chunks}
    bm25 = bm25_rank(query, chunks, candidate_k)
    vector = vector_rank(query, chunks, embedder, candidate_k)

    bm25_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(bm25, start=1)}
    vector_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(vector, start=1)}

    candidate_ids: list[int] = []
    seen: set[int] = set()
    for ranked in (bm25, vector):
        for chunk_id, _ in ranked:
            if chunk_id not in seen:
                seen.add(chunk_id)
                candidate_ids.append(chunk_id)

    passages = [
        f"{by_id[chunk_id].filename} | {by_id[chunk_id].title}\n{by_id[chunk_id].text}"
        for chunk_id in candidate_ids
    ]
    rerank_scores = reranker(query, passages)
    if len(rerank_scores) != len(candidate_ids):
        raise ValueError("Reranker returned an unexpected number of scores.")

    ranked = sorted(zip(candidate_ids, rerank_scores), key=lambda item: item[1], reverse=True)
    return [
        SearchResult(
            chunk=by_id[chunk_id],
            score=score,
            bm25_rank=bm25_ranks.get(chunk_id),
            vector_rank=vector_ranks.get(chunk_id),
        )
        for chunk_id, score in ranked[:top_k]
    ]


def build_context(
    conn: sqlite3.Connection,
    results: list[SearchResult],
    window: int = 1,
    max_chars: int = 12000,
    max_blocks: int = 12,
) -> list[ContextBlock]:
    blocks: list[ContextBlock] = []
    seen: set[tuple[int, int]] = set()
    total_chars = 0

    for result in results:
        chunk = result.chunk
        start_page = max(1, chunk.page_start - window)
        end_page = chunk.page_end + window
        rows = conn.execute(
            """
            SELECT doc_id, filename, page_start, page_end, category, text
            FROM chunks
            WHERE doc_id = ? AND page_start BETWEEN ? AND ?
            ORDER BY page_start
            """,
            (chunk.doc_id, start_page, end_page),
        ).fetchall()

        for row in rows:
            if len(blocks) >= max_blocks:
                return blocks
            key = (row["doc_id"], row["page_start"])
            if key in seen:
                continue
            text = row["text"]
            if total_chars + len(text) > max_chars and blocks:
                return blocks
            seen.add(key)
            label = f"S{len(blocks) + 1}"
            blocks.append(
                ContextBlock(
                    label=label,
                    filename=row["filename"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                    category=row["category"],
                    text=text,
                )
            )
            total_chars += len(text)
    return blocks
