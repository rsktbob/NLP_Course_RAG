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


def read_questions(path: Path, question_column: str | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(2048)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if not rows:
                return []
            column = question_column or ("題目" if "題目" in rows[0] else reader.fieldnames[0])
            for row in rows:
                row["_question"] = row.get(column, "")
            return rows

        reader = csv.reader(handle)
        rows = []
        for row in reader:
            if row:
                rows.append({"題目": row[0], "_question": row[0]})
        return rows


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
    parser.add_argument("--question-column", help="Question column name. Defaults to 題目 or first column.")
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
    rows = read_questions(input_path, args.question_column)
    if args.limit:
        rows = rows[: args.limit]

    conn = connect(args.db)
    client = OllamaClient(args.ollama_host)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [name for name in rows[0].keys() if name != "_question"] if rows else ["題目"]
    if "答案" not in fieldnames:
        fieldnames.append("答案")
    if args.include_sources and "來源" not in fieldnames:
        fieldnames.append("來源")

    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, row in enumerate(rows, start=1):
                question = row.get("_question", "").strip()
                print(f"[{index}/{len(rows)}] {question}")
                answer = answer_question(
                    conn,
                    question,
                    client=client,
                    embed_model=args.embed_model,
                    chat_model=args.chat_model,
                    top_k=args.top_k,
                    context_window=args.context_window,
                )
                row["答案"] = answer.text if args.keep_citations else strip_citations(answer.text)
                if args.include_sources:
                    row["來源"] = "; ".join(source.citation for source in answer.sources)
                else:
                    row.pop("來源", None)
                row.pop("_question", None)
                writer.writerow({name: row.get(name, "") for name in fieldnames})
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
