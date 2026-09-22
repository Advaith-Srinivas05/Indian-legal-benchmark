"""Canonical text for one document: the coordinate system every gold span lives in.

The canonical text is the line stream processing parsed the legal structure over,
joined with ``"\\n"``. Because unit line indices address that stream, a unit's
character span is a prefix sum over line lengths, and for every provision

    text[char_start:char_end] == the provision's text

holds byte for byte. :func:`build_canonical` asserts that for every provision and
raises :class:`~benchmark.errors.SpanMismatchError` rather than emit a document
where it fails.

Offsets are Python string indices — Unicode code points into the UTF-8-decoded
file — and spans are half-open, ``[char_start, char_end)``.

One normalisation is applied, and it is length-preserving: characters that
``str.splitlines`` and text-mode readers treat as line boundaries (``\\r``, form
feed, ``U+2028`` and friends) are replaced inside lines by a single space. Left in
place, they would make an ordinary ``open(...).read()`` shorter than the file and
shift every offset after them. The count is recorded per document.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

from processing.models import PageText

from . import config
from .errors import SpanMismatchError
from .source import assessed_pages, line_stream

#: Every character ``str.splitlines`` breaks on, other than ``"\n"`` itself.
_LINE_BOUNDARIES = "".join(map(chr, (0x0D, 0x0B, 0x0C, 0x1C, 0x1D, 0x1E, 0x85, 0x2028, 0x2029)))
_TO_SPACE = str.maketrans({c: " " for c in _LINE_BOUNDARIES})


def _clean(text: str) -> str:
    return text.translate(_TO_SPACE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CanonicalDocument:
    """Everything derived from one processed document, ready to be written."""

    document_id: str
    text: str
    line_count: int
    page_map: list[dict]
    structure: list[dict]
    preamble: Optional[dict]
    provisions: list[dict]
    ocr_pages: list[int]
    replaced_line_boundaries: int
    non_bmp_chars: int
    duplicate_provision_keys: int
    ambiguous_provisions: int
    stats: dict = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        return _sha256(self.text)


def build_canonical(document: dict, pages: Iterable[PageText]) -> CanonicalDocument:
    """Build the canonical text, page map and unit spans for one document.

    *document* is the parsed ``document.json``; *pages* come from
    :func:`benchmark.source.load_pages`. Raises
    :class:`~benchmark.errors.LineStreamMismatchError` or
    :class:`~benchmark.errors.SpanMismatchError`; never returns a partial result.
    """
    pages = list(pages)
    document_id = (document.get("source") or {}).get("document_id") or "?"
    stream = line_stream(document, pages)

    raw_lines = [line.text for line in stream]
    lines = [_clean(line) for line in raw_lines]
    replaced = sum(a != b for raw, clean in zip(raw_lines, lines) for a, b in zip(raw, clean))
    text = "\n".join(lines)

    offsets: list[int] = []
    cursor = 0
    for line in lines:
        offsets.append(cursor)
        cursor += len(line) + 1

    def span(line_start: int, line_end: int) -> Optional[tuple[int, int]]:
        if line_start is None or line_end is None:
            return None
        if not (0 <= line_start <= line_end < len(lines)):
            return None
        return offsets[line_start], offsets[line_end] + len(lines[line_end])

    page_map = _page_map(stream, offsets, len(text))
    ocr_pages = sorted(p.page_number for p in assessed_pages(pages)
                       if getattr(p, "text_source", "backend") == "ocr")
    ocr_in_document = bool(ocr_pages)
    ocr_page_set = set(ocr_pages)

    provisions: list[dict] = []
    key_counts: dict[str, int] = {}
    duplicates = 0

    def convert(unit: dict, ancestors: list[dict]) -> dict:
        nonlocal duplicates
        path = ancestors + [{"unit_type": unit.get("unit_type"), "number": unit.get("number")}]
        s = span(unit.get("line_start"), unit.get("line_end"))
        node = {
            "unit_type": unit.get("unit_type"),
            "number": unit.get("number"),
            "heading": unit.get("heading"),
            "page_start": unit.get("page_start"),
            "page_end": unit.get("page_end"),
            "char_start": s[0] if s else None,
            "char_end": s[1] if s else None,
            "detected_by": unit.get("detected_by"),
            "text_matches_span": bool(s) and text[s[0]:s[1]] == _clean(unit.get("text") or ""),
        }
        if unit.get("unit_type") in config.PROVISION_UNIT_TYPES:
            if not node["text_matches_span"]:
                raise SpanMismatchError(
                    f"{document_id}: {unit.get('unit_type')} {unit.get('number')!r} at lines "
                    f"{unit.get('line_start')}-{unit.get('line_end')} does not reproduce its "
                    "text from the canonical line stream."
                )
            key = "/".join(f"{p['unit_type']}:{p['number'] if p['number'] is not None else '_'}"
                           for p in path)
            key_counts[key] = key_counts.get(key, 0) + 1
            if key_counts[key] > 1:
                duplicates += 1
                key = f"{key}#{key_counts[key]}"
            node["key"] = key
            provisions.append(_provision(node, path, text, ocr_in_document, ocr_page_set))
        children = [convert(child, path) for child in unit.get("children") or []]
        if children:
            node["children"] = children
        return node

    structure_block = document.get("structure") or {}
    structure = [convert(unit, []) for unit in structure_block.get("units") or []]

    preamble = None
    raw_preamble = structure_block.get("preamble")
    if raw_preamble:
        s = span(raw_preamble.get("line_start"), raw_preamble.get("line_end"))
        preamble = {
            "page_start": raw_preamble.get("page_start"),
            "page_end": raw_preamble.get("page_end"),
            "char_start": s[0] if s else None,
            "char_end": s[1] if s else None,
            "detected_by": raw_preamble.get("detected_by"),
            "text_matches_span": bool(s) and text[s[0]:s[1]] == _clean(raw_preamble.get("text") or ""),
        }

    # A citation path shared by several provisions is ambiguous for all of them,
    # including the first, which carries no "#n" suffix. Mark the whole group so
    # nobody can mistake the first occurrence for a unique citation.
    for p in provisions:
        p["key_ambiguous"] = key_counts[p["key"].split("#", 1)[0]] > 1
    ambiguous = sum(p["key_ambiguous"] for p in provisions)

    tiers = {tier: 0 for tier in config.CONFIDENCE_TIERS}
    for p in provisions:
        tiers[p["evidence_confidence"]] += 1

    return CanonicalDocument(
        document_id=document_id,
        text=text,
        line_count=len(lines),
        page_map=page_map,
        structure=structure,
        preamble=preamble,
        provisions=provisions,
        ocr_pages=ocr_pages,
        replaced_line_boundaries=replaced,
        non_bmp_chars=sum(1 for c in text if ord(c) > 0xFFFF),
        duplicate_provision_keys=duplicates,
        ambiguous_provisions=ambiguous,
        stats={"provisions": len(provisions), "confidence": tiers},
    )


def _page_map(stream, offsets: list[int], text_length: int) -> list[dict]:
    """Contiguous ``[char_start, char_end)`` ranges, one per page present in the text.

    Each page owns its lines and the newline that follows its last line, so the
    ranges tile the whole text with no gaps and no overlaps. Pages that contribute
    no lines (blank, furniture-only, not indexable) are absent.
    """
    starts: list[tuple[int, int]] = []
    for index, line in enumerate(stream):
        if not starts or starts[-1][0] != line.page_number:
            starts.append((line.page_number, offsets[index]))
    page_map = []
    for i, (page_number, start) in enumerate(starts):
        end = starts[i + 1][1] if i + 1 < len(starts) else text_length
        page_map.append({"page_number": page_number, "char_start": start, "char_end": end})
    return page_map


def _provision(node: dict, path: list[dict], text: str, ocr_in_document: bool,
               ocr_pages: set[int]) -> dict:
    """One provision record, with its evidence-confidence tier.

    The OCR test is on the provision's **own pages**, not the whole document. A
    provision's text comes only from its own lines, so an OCR-read schedule on
    page 24 cannot corrupt Section 8 on page 10 — and a document-level test
    demoted all 28 sections of the born-digital Right to Information Act, 2005
    for exactly that reason.
    """
    strong = node["detected_by"] in config.STRONG_DETECTORS
    pages = range(node["page_start"] or 0, (node["page_end"] or -1) + 1)
    ocr_in_span = any(p in ocr_pages for p in pages)
    if strong and not ocr_in_span:
        tier = "high"
    elif strong or not ocr_in_span:
        tier = "medium"
    else:
        tier = "low"
    return {
        "key": node["key"],
        "unit_type": node["unit_type"],
        "number": node["number"],
        "heading": node["heading"],
        "unit_path": path,
        "char_start": node["char_start"],
        "char_end": node["char_end"],
        "page_start": node["page_start"],
        "page_end": node["page_end"],
        "detected_by": node["detected_by"],
        "evidence_confidence": tier,
        "ocr_in_span": ocr_in_span,
        "ocr_in_document": ocr_in_document,
        "text_sha256": _sha256(text[node["char_start"]:node["char_end"]]),
    }
