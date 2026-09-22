"""The one sanctioned reader of what processing wrote.

``pages.json`` stores **two** readings of every page and a field saying which one
applies:

===============  ==================================================
``text``         the extraction backend's reading — empty on a scanned page
``ocr.text``     the OCR engine's reading, present whether accepted or not
``text_source``  ``"backend"`` or ``"ocr"`` — which of the two is authoritative
===============  ==================================================

44,827 pages in the corpus were read by OCR. A consumer that reads
``page["text"]`` gets an empty string for every one of them and reports no error.
This module rebuilds real :class:`processing.models.PageText` objects so that
``page.selected_text`` — the property processing itself used — resolves the
reading exactly as it did then.

It also closes two quieter traps:

**The line-index contract.** Every unit in ``document.json`` addresses its text
by index into the stream built by :func:`processing.structure.build_line_stream`
over the pages processing *parsed* — the indexable pages, or all pages when none
were indexable. Rebuilding over all pages shifts every index in a bilingual
document and attaches real text to the wrong section, silently.
:func:`line_stream` checks the rebuilt length against ``counts.line_count`` and
raises rather than guessing.

**Field drift.** Files written on 2026-08-24 declare schema version 2 but lack
``text_source``, ``ocr`` and ``quality``. They pre-date OCR, so their ``text``
genuinely is their reading; the page-quality verdict is unknown and is reported
as such. Every absence is returned to the caller, never hidden.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from processing.models import FootnoteBlock, FurnitureLine, PageText
from processing.structure import Line, build_line_stream

from . import config
from .errors import LineStreamMismatchError, NotEligibleError, SourceError


def document_dir(data_dir: Path, document_id: str) -> Path:
    return Path(data_dir) / config.PROCESSED_SUBDIR / document_id


def page_text(page: dict) -> str:
    """The authoritative reading of one raw ``pages.json`` page dictionary.

    For callers holding dictionaries rather than :class:`PageText` objects.
    Identical in behaviour to :attr:`processing.models.PageText.selected_text`.
    """
    if page.get("text_source") == "ocr":
        text = (page.get("ocr") or {}).get("text")
        if text:
            return text
    return page.get("text") or ""


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise SourceError(f"{path} does not exist; has this document been processed?")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise SourceError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SourceError(f"{path} does not contain a JSON object")
    return payload


def load_document(data_dir: Path, document_id: str) -> dict:
    """Read one ``document.json``, checking it is a shape we understand."""
    payload = _read_json(document_dir(data_dir, document_id) / config.DOCUMENT_FILENAME)
    version = payload.get("schema_version")
    if version not in config.SUPPORTED_PROCESSING_SCHEMA_VERSIONS:
        raise SourceError(
            f"{document_id}: document.json has schema_version {version!r}; this "
            f"package understands {config.SUPPORTED_PROCESSING_SCHEMA_VERSIONS}. "
            "Refusing to read it rather than half-understanding it."
        )
    if "structure" not in payload:
        raise SourceError(f"{document_id}: document.json carries no 'structure' block")
    return payload


def load_pages(data_dir: Path, document_id: str) -> tuple[list[PageText], list[str]]:
    """Rebuild ``pages.json`` as :class:`PageText` objects.

    Returns the pages and the sorted names of any optional fields the file did
    not carry.
    """
    payload = _read_json(document_dir(data_dir, document_id) / config.PAGES_FILENAME)
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise SourceError(f"{document_id}: pages.json carries no 'pages' list")

    missing: set[str] = set()
    pages: list[PageText] = []
    for raw in raw_pages:
        for name in config.OPTIONAL_PAGE_FIELDS:
            if name not in raw:
                missing.add(name)
        page = PageText(
            page_number=raw["page_number"],
            text=raw.get("text") or "",
            char_count=raw.get("char_count", 0),
        )
        # The fields that decide which reading applies. The default is the
        # pre-OCR one, which is correct for the drifted files: OCR did not exist
        # when they were written, so their backend text is their only reading.
        page.text_source = raw.get("text_source") or "backend"
        page.ocr = raw.get("ocr")
        page.quality = raw.get("quality")
        page.language = raw.get("language")
        page.indexable = raw.get("indexable", True)
        page.non_english_lines = raw.get("non_english_lines") or []
        page.tables = raw.get("tables") or []
        page.furniture = [
            FurnitureLine(
                line_index=item["line_index"],
                text=item.get("text", ""),
                kind=item.get("kind", ""),
                reason=item.get("reason", ""),
            )
            for item in raw.get("furniture") or []
        ]
        page.footnotes = [
            FootnoteBlock(
                page_number=item.get("page_number", page.page_number),
                line_start=item["line_start"],
                line_end=item["line_end"],
                text=item.get("text", ""),
                detected_by=item.get("detected_by", ""),
            )
            for item in raw.get("footnotes") or []
        ]
        pages.append(page)
    return pages, sorted(missing)


def assessed_pages(pages: Iterable[PageText]) -> list[PageText]:
    """The pages processing parsed structure over.

    Mirrors ``processing/process.py`` exactly: the indexable pages, or every page
    when the language gate marked none.
    """
    pages = list(pages)
    return [page for page in pages if page.indexable] or pages


def line_stream(document: dict, pages: Iterable[PageText]) -> list[Line]:
    """Rebuild the exact line stream the structure was parsed on.

    Verified against ``structure.counts.line_count``; a mismatch is refused,
    never approximated.
    """
    stream = build_line_stream(assessed_pages(pages))
    recorded = (document.get("structure") or {}).get("counts", {}).get("line_count")
    if recorded is None:
        raise LineStreamMismatchError(
            f"{_document_id(document)}: document.json records no line_count, so the "
            "rebuilt line stream cannot be checked. Refusing to build offsets from it."
        )
    if recorded != len(stream):
        raise LineStreamMismatchError(
            f"{_document_id(document)}: rebuilt line stream has {len(stream)} lines "
            f"but document.json recorded {recorded}. Every unit addresses its text by "
            "index into that stream, so offsets built now would point at the wrong text."
        )
    return stream


def require_eligible(document: dict) -> None:
    """Refuse a document processing quarantined.

    Reads the document's own record, never the processing journal: the journal
    is stale for the 351 documents restored in place on 2026-08-29.
    """
    if not document.get("eligible_for_indexing"):
        raise NotEligibleError(
            f"{_document_id(document)} is not eligible_for_indexing; "
            "quarantined documents are not part of the corpus."
        )


def is_eligible(data_dir: Path, document_id: str) -> Optional[bool]:
    """Whether a processed document is eligible, or ``None`` if unreadable."""
    try:
        return bool(load_document(data_dir, document_id).get("eligible_for_indexing"))
    except SourceError:
        return None


def _document_id(document: dict) -> str:
    return (document.get("source") or {}).get("document_id") or "?"
