from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .text import normalize_text, short_title, token_counts


@dataclass(frozen=True)
class PdfDocument:
    path: Path
    filename: str
    sha256: str
    mtime: float
    pages: int
    category: str
    lecture_no: int | None
    title: str


@dataclass(frozen=True)
class PageChunk:
    filename: str
    page_start: int
    page_end: int
    title: str
    category: str
    text: str
    tokens: dict[str, int]
    embedding: list[float] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_document_metadata(path: Path, pages: int) -> PdfDocument:
    filename = path.name
    lower = filename.lower()
    lecture_no: int | None = None
    category = "lecture"

    lecture_match = re.match(r"c(\d+)[_\-]", lower)
    if lecture_match:
        lecture_no = int(lecture_match.group(1))
    elif "course_introduction" in lower:
        lecture_no = 0
        category = "course_info"

    if "course_introduction" in lower or filename.startswith("c0_"):
        category = "course_info"
    elif "期末專題" in filename:
        category = "project"
    elif "學習活動" in filename:
        category = "activity"

    stem = path.stem
    title = re.sub(r"^c\d+[_\-]?", "", stem, flags=re.IGNORECASE)
    title = title.replace("_", " ").replace("-", " ").strip() or stem
    return PdfDocument(
        path=path,
        filename=filename,
        sha256=sha256_file(path),
        mtime=path.stat().st_mtime,
        pages=pages,
        category=category,
        lecture_no=lecture_no,
        title=title,
    )


def infer_chunk_category(doc_category: str, filename: str, text: str) -> str:
    if doc_category in {"project", "activity", "course_info"}:
        return doc_category

    normalized = normalize_text(text).lower()
    admin_markers = (
        "announcement",
        "期中考",
        "期末考",
        "office hour",
        "範圍",
        "地點",
        "評分",
        "繳交",
        "出席",
        "加選",
    )
    if any(marker in normalized for marker in admin_markers):
        return "announcement"
    return doc_category


def load_pdf(path: Path) -> tuple[PdfDocument, list[PageChunk]]:
    reader = PdfReader(str(path))
    doc = infer_document_metadata(path, len(reader.pages))
    chunks: list[PageChunk] = []

    for index, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if not text:
            continue

        category = infer_chunk_category(doc.category, doc.filename, text)
        title = short_title(text, fallback=f"{doc.title} p.{index}")
        chunk_text = f"{doc.title}\n{text}"
        chunks.append(
            PageChunk(
                filename=doc.filename,
                page_start=index,
                page_end=index,
                title=title,
                category=category,
                text=chunk_text,
                tokens=token_counts(f"{doc.filename} {doc.title} {category} {chunk_text}"),
            )
        )

    return doc, chunks


def chunk_to_record(chunk: PageChunk) -> dict[str, Any]:
    return {
        "filename": chunk.filename,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "title": chunk.title,
        "category": chunk.category,
        "text": chunk.text,
        "tokens": chunk.tokens,
        "embedding": chunk.embedding,
    }
