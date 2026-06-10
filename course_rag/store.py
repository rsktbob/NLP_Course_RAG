from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from .pdf_loader import PageChunk, PdfDocument


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL UNIQUE,
    path TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    mtime REAL NOT NULL,
    pages INTEGER NOT NULL,
    category TEXT NOT NULL,
    lecture_no INTEGER,
    title TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL,
    filename TEXT NOT NULL,
    page_start INTEGER NOT NULL,
    page_end INTEGER NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    text TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    embedding_json TEXT,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_page ON chunks(doc_id, page_start, page_end);
CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def reset_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS chunks;
        DROP TABLE IF EXISTS documents;
        """
    )
    init_db(conn)


def store_document(conn: sqlite3.Connection, doc: PdfDocument, chunks: Iterable[PageChunk]) -> int:
    conn.execute("DELETE FROM documents WHERE filename = ?", (doc.filename,))
    cursor = conn.execute(
        """
        INSERT INTO documents (filename, path, sha256, mtime, pages, category, lecture_no, title)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc.filename,
            str(doc.path),
            doc.sha256,
            doc.mtime,
            doc.pages,
            doc.category,
            doc.lecture_no,
            doc.title,
        ),
    )
    doc_id = int(cursor.lastrowid)
    rows = []
    for chunk in chunks:
        rows.append(
            (
                doc_id,
                chunk.filename,
                chunk.page_start,
                chunk.page_end,
                chunk.title,
                chunk.category,
                chunk.text,
                json.dumps(chunk.tokens, ensure_ascii=False),
                json.dumps(chunk.embedding) if chunk.embedding is not None else None,
            )
        )
    conn.executemany(
        """
        INSERT INTO chunks (
            doc_id, filename, page_start, page_end, title, category, text,
            tokens_json, embedding_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def database_stats(conn: sqlite3.Connection) -> dict[str, int]:
    docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding_json IS NOT NULL").fetchone()[0]
    return {"documents": docs, "chunks": chunks, "embedded_chunks": embedded}
