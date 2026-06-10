from __future__ import annotations

import argparse
import os
import sys

from .ollama_client import OllamaClient, OllamaError
from .qa import answer_question
from .reranker import RerankerError
from .store import connect


DEFAULT_DB = "storage/course_rag.sqlite"
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
DEFAULT_RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def print_answer(answer, show_sources: bool) -> None:
    print(answer.text)
    if show_sources:
        print()
        for source in answer.sources:
            print(f"- [{source.label}] {source.citation}")


def ask_once(args, question: str) -> None:
    conn = connect(args.db)
    client = OllamaClient(args.ollama_host)
    try:
        answer = answer_question(
            conn,
            question,
            client=client,
            embed_model=args.embed_model,
            chat_model=args.chat_model,
            rerank_model=args.rerank_model,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            context_window=args.context_window,
        )
    except (OllamaError, RerankerError) as exc:
        raise SystemExit(
            f"\nRAG request failed: {exc}\n"
            f"Check dependencies and Ollama models:\n"
            f"  python -m pip install -r requirements.txt\n"
            f"  ollama pull {args.embed_model}\n"
            f"  ollama pull {args.chat_model}\n"
        ) from exc
    print_answer(answer, args.show_sources)


def main() -> None:
    configure_output()
    parser = argparse.ArgumentParser(description="Ask questions against the course RAG index.")
    parser.add_argument("question", nargs="*", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama host URL.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model.")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL, help="Hugging Face reranker model.")
    parser.add_argument("--candidate-k", type=int, default=30, help="Candidates from each retriever.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of reranked chunks.")
    parser.add_argument("--context-window", type=int, default=0, help="Neighbor pages to include around hits.")
    parser.add_argument("--show-sources", action="store_true", help="Print source citations after the answer.")
    args = parser.parse_args()

    if args.question:
        ask_once(args, " ".join(args.question))
        return

    print("Course RAG interactive mode. Press Enter on an empty line to exit.")
    while True:
        question = input("\nQuestion> ").strip()
        if not question:
            break
        ask_once(args, question)


if __name__ == "__main__":
    main()
