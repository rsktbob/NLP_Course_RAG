from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

from .ollama_client import OllamaClient, OllamaError
from .qa import answer_question
from .store import connect


DEFAULT_DB = "storage/course_rag.sqlite"
DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_CITATION_RANGE_RE = re.compile(r"\s*\[S\d+\]\s*(?:至|到|~|-|–|—)\s*\[S\d+\]")
_CITATION_WITH_LABEL_RE = re.compile(r"\s*\[(?:來源|Source)\s*:\s*S\d+\]", re.IGNORECASE)
_CITATION_RE = re.compile(r"\s*\[S\d+(?:\s*(?:[-,]|至|到|~|–|—)\s*S?\d+)*\]")


def configure_output() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def read_questions(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        questions = [row[0].strip() for row in reader if row and row[0].strip()]
    if questions and questions[0] == "題目":
        raise ValueError("Input CSV must not contain a 題目 header. Write questions directly.")
    return questions


def strip_citations(text: str) -> str:
    text = _CITATION_RANGE_RE.sub("", text)
    text = _CITATION_WITH_LABEL_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    configure_output()
    parser = argparse.ArgumentParser(description="Answer a CSV of questions with the course RAG system.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite database path.")
    parser.add_argument("--ollama-host", default=DEFAULT_OLLAMA_HOST, help="Ollama host URL.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model.")
    parser.add_argument("--chat-model", default=DEFAULT_CHAT_MODEL, help="Ollama chat model.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of retrieved chunks.")
    parser.add_argument("--context-window", type=int, default=1, help="Neighbor pages to include around hits.")
    parser.add_argument("--limit", type=int, help="Only process the first N questions.")
    parser.add_argument("--include-sources", action="store_true", help="Add a source citation column to the CSV.")
    parser.add_argument("--keep-citations", action="store_true", help="Keep [S1] markers inside answers.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        questions = read_questions(input_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.limit:
        questions = questions[: args.limit]

    conn = connect(args.db)
    client = OllamaClient(args.ollama_host)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            for index, question in enumerate(questions, start=1):
                print(f"[{index}/{len(questions)}] {question}")
                answer = answer_question(
                    conn,
                    question,
                    client=client,
                    embed_model=args.embed_model,
                    chat_model=args.chat_model,
                    top_k=args.top_k,
                    context_window=args.context_window,
                )
                answer_text = answer.text if args.keep_citations else strip_citations(answer.text)
                if args.include_sources:
                    writer.writerow(
                        [answer_text, "; ".join(source.citation for source in answer.sources)]
                    )
                else:
                    writer.writerow([answer_text])
    except OllamaError as exc:
        raise SystemExit(
            f"\nOllama request failed: {exc}\n"
            f"Check models:\n"
            f"  ollama pull {args.embed_model}\n"
            f"  ollama pull {args.chat_model}\n"
        ) from exc

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
