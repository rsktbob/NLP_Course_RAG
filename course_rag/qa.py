from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .ollama_client import OllamaClient
from .reranker import get_reranker
from .retriever import ContextBlock, SearchResult, build_context, rerank_search
from .text import normalize_traditional


SYSTEM_PROMPT = """你是自然語言處理課程的 QA 助教。
請只根據提供的來源回答，不要補充來源以外的資訊。
如果來源不足，請明確說「目前資料中找不到足夠資訊」。
回答要精簡、準確；行政資訊要保留日期、時間、比例、地點、email 等細節。
如果問題限定特定面向，例如「評分、時間、地點、範圍、減少模型大小的方法」，只回答該面向；不要把同頁中其他特色或背景也列入答案。
預設用繁體中文回答，嚴禁混用簡體字；英文專有名詞可保留英文，或使用「英文專有名詞 + 繁體中文解釋」的形式，例如 RAG、Retriever、Generator、Text-to-SQL、ALBERT。
只有使用者明確要求英文回答時，才全英文回答。
如果來源已經足以回答問題，不要在答案最後再補「目前資料中找不到足夠資訊」。
答案控制在 1 到 5 句，或 3 到 5 個重點條列；不要加入問題沒有問的研究背景。
注意：SOP / Sentence Order Prediction 是 ALBERT 的訓練任務特色，不是減少模型大小的主要方法；只有問題詢問 ALBERT 全部特色時才提到 SOP。
每個重要句子或條列後方要標示來源，例如 [S1]。
"""


@dataclass(frozen=True)
class Answer:
    question: str
    text: str
    sources: list[ContextBlock]
    results: list[SearchResult]


def format_context(blocks: list[ContextBlock]) -> str:
    parts: list[str] = []
    for block in blocks:
        parts.append(
            f"[{block.label}] {block.citation} | category={block.category}\n{block.text}"
        )
    return "\n\n".join(parts)


def answer_question(
    conn: sqlite3.Connection,
    question: str,
    client: OllamaClient,
    embed_model: str,
    chat_model: str,
    rerank_model: str,
    top_k: int = 3,
    candidate_k: int = 30,
    context_window: int = 0,
) -> Answer:
    embedder = lambda text: client.embed(embed_model, text)
    reranker = get_reranker(rerank_model)
    results = rerank_search(
        conn,
        question,
        embedder=embedder,
        reranker=reranker.score,
        top_k=top_k,
        candidate_k=candidate_k,
    )
    sources = build_context(conn, results, window=context_window)

    context = format_context(sources)
    user_prompt = f"""問題：
{question}

可用來源：
{context}

請根據來源回答。"""
    text = client.chat(
        chat_model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return Answer(question=question, text=normalize_traditional(text), sources=sources, results=results)
