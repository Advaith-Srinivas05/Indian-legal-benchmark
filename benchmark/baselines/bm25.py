"""BM25 baselines over the canonical corpus, on SQLite's own FTS5 index.

Two systems, one index:

* :class:`Bm25Windows` — rank fixed-size windows of canonical text directly.
  The sparse floor. It knows nothing about sections, which is the point: it is
  what a system that ignores legal structure can manage.
* :class:`Bm25TwoStage` — resolve the document first, then rank windows inside
  the documents that survived. The design the archived ``corpusdb/query.py``
  implemented, whose notes warn it "may be surprisingly hard to beat on
  citation lookups".

FTS5 ships with Python's own sqlite3, so this adds no dependency and needs no
network. The tables are **contentless** (``content=''``): the index stores no
copy of the text, only the postings, which keeps a 712 MB corpus to an index
that fits beside it. Spans are resolved from the canonical text at query time,
so what a baseline returns is the same coordinate system every other system
must use — no chunk ids anywhere.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .. import config, terms
from ..score import Span

#: Window geometry, named in the results report because a baseline that does not
#: name its chunk size is not reproducible. The overlap keeps a provision that
#: straddles a boundary reachable from one window rather than two.
WINDOW_CHARS = 1200
STRIDE_CHARS = 900
#: Terms beyond this add little and cost a lot on a corpus this size.
MAX_QUERY_TERMS = 24
INDEX_SCHEMA_VERSION = 1


def _query_string(question: str, max_terms: int = MAX_QUERY_TERMS) -> str:
    """The question as an FTS5 OR-query of its content words, in order, deduplicated."""
    seen: list[str] = []
    for word in terms.content_words(question):
        if word not in seen:
            seen.append(word)
    return " OR ".join(f'"{w}"' for w in seen[:max_terms])


def windows_of(text: str, *, window: int = WINDOW_CHARS,
               stride: int = STRIDE_CHARS) -> Iterable[tuple[int, int]]:
    if stride <= 0 or window <= 0:
        raise ValueError("window and stride must be positive")
    start = 0
    while start < len(text):
        yield start, min(start + window, len(text))
        if start + window >= len(text):
            return
        start += stride


def build_index(corpus_dir: Path, index_path: Path, *, window: int = WINDOW_CHARS,
                stride: int = STRIDE_CHARS, progress: bool = False) -> dict:
    """Build the FTS5 index. Rebuilds from scratch: an index is derived, never patched."""
    corpus_dir, index_path = Path(corpus_dir), Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    db = sqlite3.connect(index_path)
    db.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE window (rowid INTEGER PRIMARY KEY, document_id TEXT NOT NULL,
                             start INTEGER NOT NULL, end INTEGER NOT NULL);
        CREATE TABLE document (rowid INTEGER PRIMARY KEY, document_id TEXT NOT NULL,
                               chars INTEGER NOT NULL);
        CREATE VIRTUAL TABLE window_fts USING fts5(body, content='');
        CREATE VIRTUAL TABLE document_fts USING fts5(body, content='');
    """)

    documents = [json.loads(line) for line in
                 open(corpus_dir / config.DOCUMENTS_FILENAME, encoding="utf-8")]
    window_rowid = 0
    for n, record in enumerate(documents, start=1):
        did = record["document_id"]
        meta = json.loads((corpus_dir / config.META_DIRNAME / f"{did}.json").read_text(encoding="utf-8"))
        with open(corpus_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        db.execute("INSERT INTO document (rowid, document_id, chars) VALUES (?, ?, ?)",
                   (n, did, len(text)))
        db.execute("INSERT INTO document_fts (rowid, body) VALUES (?, ?)",
                   (n, f"{meta.get('title', '')}\n{text}"))
        rows, bodies = [], []
        for start, end in windows_of(text, window=window, stride=stride):
            window_rowid += 1
            rows.append((window_rowid, did, start, end))
            bodies.append((window_rowid, text[start:end]))
        db.executemany("INSERT INTO window (rowid, document_id, start, end) VALUES (?, ?, ?, ?)", rows)
        db.executemany("INSERT INTO window_fts (rowid, body) VALUES (?, ?)", bodies)
        if n % 200 == 0:
            db.commit()
            if progress:
                print(f"  {n}/{len(documents)} documents, {window_rowid:,} windows", flush=True)

    db.execute("CREATE INDEX window_document ON window (document_id)")
    report = {"index_schema_version": INDEX_SCHEMA_VERSION, "documents": len(documents),
              "windows": window_rowid, "window_chars": window, "stride_chars": stride}
    db.executemany("INSERT INTO meta (key, value) VALUES (?, ?)",
                   [(k, json.dumps(v)) for k, v in report.items()])
    db.commit()
    db.execute("VACUUM")
    db.close()
    report["index_bytes"] = index_path.stat().st_size
    return report


class Bm25Index:
    """Read-only access to a built index. Shared by both BM25 baselines."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no BM25 index at {self.path}; run 'python -m benchmark index'")
        self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.meta = {k: json.loads(v) for k, v in self.db.execute("SELECT key, value FROM meta")}

    def windows(self, query: str, limit: int, documents: Optional[Sequence[str]] = None
                ) -> list[tuple[str, int, int]]:
        if not query:
            return []
        if documents is None:
            sql = ("SELECT w.document_id, w.start, w.end FROM window_fts f "
                   "JOIN window w ON w.rowid = f.rowid "
                   "WHERE window_fts MATCH ? ORDER BY bm25(window_fts) LIMIT ?")
            return list(self.db.execute(sql, (query, limit)))
        if not documents:
            return []
        marks = ",".join("?" * len(documents))
        sql = ("SELECT w.document_id, w.start, w.end FROM window_fts f "
               f"JOIN window w ON w.rowid = f.rowid WHERE window_fts MATCH ? "
               f"AND w.document_id IN ({marks}) ORDER BY bm25(window_fts) LIMIT ?")
        return list(self.db.execute(sql, (query, *documents, limit)))

    def documents(self, query: str, limit: int) -> list[str]:
        if not query:
            return []
        sql = ("SELECT d.document_id FROM document_fts f JOIN document d ON d.rowid = f.rowid "
               "WHERE document_fts MATCH ? ORDER BY bm25(document_fts) LIMIT ?")
        return [row[0] for row in self.db.execute(sql, (query, limit))]

    def close(self) -> None:
        self.db.close()


def _fill(candidates: Sequence[tuple[str, int, int]], budget_chars: int) -> list[Span]:
    """Take ranked windows until the budget is covered. The scorer trims the last one."""
    spans: list[Span] = []
    used = 0
    for did, start, end in candidates:
        spans.append((did, start, end))
        used += end - start
        if used >= budget_chars:
            break
    return spans


class Bm25Windows:
    """BM25 over fixed-size windows: the sparse floor."""

    name = "bm25-windows"

    def __init__(self, index: Bm25Index, *, max_windows: int = 64) -> None:
        self.index = index
        self.max_windows = max_windows

    def retrieve(self, question: str, budget_chars: int) -> list[Span]:
        query = _query_string(question)
        limit = max(1, min(self.max_windows, budget_chars // 200 + 2))
        return _fill(self.index.windows(query, limit), budget_chars)


class Bm25TwoStage:
    """Resolve the document, then search within it."""

    name = "bm25-two-stage"

    def __init__(self, index: Bm25Index, *, documents: int = 5, max_windows: int = 64) -> None:
        self.index = index
        self.documents = documents
        self.max_windows = max_windows

    def retrieve(self, question: str, budget_chars: int) -> list[Span]:
        query = _query_string(question)
        chosen = self.index.documents(query, self.documents)
        if not chosen:
            return []
        limit = max(1, min(self.max_windows, budget_chars // 200 + 2))
        return _fill(self.index.windows(query, limit, documents=chosen), budget_chars)
