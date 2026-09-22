"""The published metadata record for one document: identity, source and quality.

Three inputs, each authoritative for different fields:

* ``document.json`` ``source`` — identity as processing recorded it;
* ``data/raw/.../metadata.json`` — India Code's own metadata as ingestion saw it
  (long title, Act number, ministry, enforcement date, download record);
* the discovery inventory — the only place ``department`` is kept.

Fields are copied, never re-derived. One distinction is enforced rather than
copied: rules and regulations are not DSpace items, and the ``handle`` ingestion
stored for them is their **parent Act's**. Publishing it as the document's own
persistent identifier would be false, so only Acts get ``handle`` /
``handle_uri``; subordinate legislation records its parent under ``parent``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import config
from .canonical import CanonicalDocument

#: Document types that are DSpace items with a handle of their own.
_HANDLE_TYPES = frozenset({"central_act", "state_act"})
_HANDLE_RESOLVER = "http://hdl.handle.net/"


def raw_metadata_path(data_dir: Path, document: dict) -> Optional[Path]:
    """Where ingestion left this document's ``metadata.json``, beside its PDF."""
    relpath = (document.get("source") or {}).get("pdf_relpath")
    if not relpath:
        return None
    return Path(data_dir) / Path(relpath).parent / "metadata.json"


def load_raw_metadata(data_dir: Path, document: dict) -> Optional[dict]:
    path = raw_metadata_path(data_dir, document)
    if path is None or not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inventory(data_dir: Path) -> dict[str, dict]:
    """The discovery inventory, keyed by ``document_id``. Empty if absent."""
    path = Path(data_dir) / config.INVENTORY_RELPATH
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return {entry["document_id"]: entry for entry in payload.get("documents") or []}


def build_meta(document: dict, canonical: CanonicalDocument, *,
               raw: Optional[dict], inventory_entry: Optional[dict],
               missing_page_fields: list[str]) -> dict:
    """Assemble ``meta/<document_id>.json``."""
    source = document.get("source") or {}
    raw = raw or {}
    inv = inventory_entry or {}
    download = raw.get("download") or {}
    document_type = source.get("document_type")
    own_handle = source.get("handle") if document_type in _HANDLE_TYPES else None

    parent = None
    if raw.get("parent_title") or raw.get("parent_handle") or inv.get("parent_title"):
        parent = {
            "title": raw.get("parent_title") or inv.get("parent_title"),
            "handle": raw.get("parent_handle") or inv.get("parent_handle"),
            "url": raw.get("parent_url"),
        }

    extraction_quality = document.get("extraction_quality") or {}
    structure = document.get("structure") or {}
    detectors: dict[str, int] = {}
    for p in canonical.provisions:
        detectors[p["detected_by"]] = detectors.get(p["detected_by"], 0) + 1

    return {
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "document_id": canonical.document_id,
        "title": source.get("title"),
        "short_title": raw.get("short_title") or inv.get("short_title"),
        "long_title": raw.get("long_title"),
        "hindi_title": raw.get("hindi_title"),
        "category": source.get("category"),
        "document_type": document_type,
        "jurisdiction": source.get("jurisdiction"),
        "year": source.get("year"),
        "act_number": raw.get("act_number") or inv.get("act_number"),
        "act_year": raw.get("act_year"),
        "india_code_act_id": raw.get("india_code_act_id") or inv.get("india_code_act_id"),
        "enactment_date": source.get("enactment_date") or raw.get("enactment_date"),
        "enforcement_date": raw.get("enforcement_date"),
        "ministry": raw.get("ministry") or inv.get("ministry"),
        "department": inv.get("department"),
        "parent": parent,
        "source": {
            "provider": source.get("provider") or "India Code",
            "handle": own_handle,
            "handle_uri": f"{_HANDLE_RESOLVER}{own_handle}" if own_handle else None,
            "india_code_url": raw.get("india_code_handle_url") or inv.get("india_code_url"),
            "source_url": source.get("source_url"),
            "final_pdf_url": download.get("final_url"),
            "pdf_sha256": source.get("sha256"),
            "pdf_bytes": source.get("bytes") or download.get("bytes"),
            "downloaded_at": download.get("downloaded_at"),
        },
        "text": {
            "path": f"{config.TEXT_DIRNAME}/{canonical.document_id}.txt",
            "sha256": canonical.text_sha256,
            "encoding": "utf-8",
            "offsets": "unicode code points, half-open [char_start, char_end)",
            "char_count": len(canonical.text),
            "line_count": canonical.line_count,
            "page_count": len(canonical.page_map),
        },
        "page_map": canonical.page_map,
        "provisions": {
            "count": len(canonical.provisions),
            "confidence": canonical.stats["confidence"],
            "detected_by": dict(sorted(detectors.items())),
            "duplicate_keys": canonical.duplicate_provision_keys,
            "ambiguous_keys": canonical.ambiguous_provisions,
        },
        "quality": {
            "structure_confidence": structure.get("confidence"),
            "unit_vocabulary": structure.get("unit_vocabulary"),
            "structure_warnings": len(structure.get("warnings") or []),
            "extraction_quality": extraction_quality.get("classification"),
            "extraction_quality_score": extraction_quality.get("score"),
            "content_language": (document.get("content_language") or {}).get("content_language"),
            "pdf_type": (document.get("extraction") or {}).get("pdf_type"),
            "ocr_text_source": (document.get("ocr_decision") or {}).get("text_source"),
            "ocr_pages": canonical.ocr_pages,
            "missing_page_fields": missing_page_fields,
            "raw_metadata_found": bool(raw),
            "replaced_line_boundaries": canonical.replaced_line_boundaries,
            "non_bmp_chars": canonical.non_bmp_chars,
        },
    }


def structure_record(canonical: CanonicalDocument) -> dict:
    """``structure/<document_id>.json``: the optional unit tree, with spans."""
    return {
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "document_id": canonical.document_id,
        "note": ("Optional aid. No metric depends on using it. char_start/char_end "
                 "index text/<document_id>.txt. text_matches_span is false for "
                 "containers by design: their stored text is the heading only."),
        "preamble": canonical.preamble,
        "units": canonical.structure,
        "provisions": canonical.provisions,
    }
