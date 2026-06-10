from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

from .ollama_client import OllamaClient, OllamaError
from .pdf_loader import PageChunk, load_pdf
from .store import connect, database_stats, init_db, reset_db, store_document


DEFAULT_DB = "storage/course_rag.sqlite"
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def embed_chunks(client: OllamaClient, model: str, chunks: list[PageChunk], batch_size: int = 16) -> list[PageChunk]:
    embedded: list[PageChunk] = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [f"{chunk.filename} p.{chunk.page_start}\n{chunk.text}" for chunk in batch]
        vectors = client.embed_many(model, texts)
        embedded.extend(replace(chunk, embedding=vector) for chunk, vector in zip(batch, vectors))
    return embedded


def find_pdfs(data_dir: Path) -> tuple[Path, list[Path]]:
    pdfs = sorted(data_dir.glob("*.pdf"))
    if pdfs:
        return data_dir, pdfs

    child_dirs = [path for path in sorted(data_dir.iterdir()) if path.is_dir()]
    candidates: list[tuple[Path, list[Path]]] = []
    for child in child_dirs:
        child_pdfs = sorted(child.glob("*.pdf"))
        if child_pdfs:
            candidates.append((child, child_pdfs))

    if len(candidates) == 1:
        return candidates[0]
    return data_dir, []


def main() -> None:
    configure_output()
    parser = argparse.ArgumentParser(description="Build the course RAG index from PDF files.")
    parser.add_argument("--data-dir", default=".", help="Directory containing course PDFs.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama host URL.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    parser.add_argument("--reset", action="store_true", help="Reset the database before ingesting.")
    parser.add_argument("--no-embeddings", action="store_true", help="Build a BM25-only index.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir, pdfs = find_pdfs(data_dir)
    if not pdfs:
        raise SystemExit(f"No PDF files found in {data_dir.resolve()}")
    if Path(args.data_dir) != data_dir:
        print(f"No PDFs found in {Path(args.data_dir).resolve()}; using {data_dir.resolve()}")

    conn = connect(args.db)
    if args.reset:
        reset_db(conn)
    else:
        init_db(conn)

    client = OllamaClient(args.ollama_host)
    total_chunks = 0
    try:
        for pdf in pdfs:
            if not pdf.exists():
                print(f"Skipping missing PDF: {pdf}")
                continue
            doc, chunks = load_pdf(pdf)
            if not args.no_embeddings:
                chunks = embed_chunks(client, args.embed_model, chunks)
            count = store_document(conn, doc, chunks)
            total_chunks += count
            print(f"Indexed {pdf.name}: {count} chunks")
    except OllamaError as exc:
        raise SystemExit(
            f"\nOllama embedding failed: {exc}\n"
            f"Make sure Ollama is running and the model is available:\n"
            f"  ollama pull {args.embed_model}\n"
            f"Or build a BM25-only index with --no-embeddings.\n"
        ) from exc

    stats = database_stats(conn)
    print(
        f"Done. documents={stats['documents']}, chunks={stats['chunks']}, "
        f"embedded_chunks={stats['embedded_chunks']}, db={args.db}"
    )
    if total_chunks == 0:
        raise SystemExit("No text chunks were extracted.")


if __name__ == "__main__":
    main()
