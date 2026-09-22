"""The gold evidence pool: every provision a question may be written against.

A provision enters the pool only if every one of these holds, and the reason for
each exclusion is counted, never silently dropped:

=====================  ===========================================================
``tier``               ``evidence_confidence == "high"``: a strong detector and no
                       OCR-read page in its span
``ambiguous``          its citation path is unique in its document
``title_conflict``     its document is not in ``duplicates.json`` title conflicts
                       (some are India Code attaching the wrong PDF to a listing)
``text_source``        its document is born digital. A scanned PDF's own text layer
                       is OCR too — done by someone else and never checked
``structure``          its document's structure is ``structured``, not ``partial``
``too_short``          at least 150 characters
``too_long``           at most 6,000 characters. The longest "provision" is a
                       428,087-character parse failure, and gold must fit the
                       8,000-character headline budget
``stub``               not an ``[Omitted]`` / ``[Repealed]`` placeholder (at most
                       12 words, one of them omitted / repealed / deleted)
=====================  ===========================================================

Each pooled provision carries difficulty signals and author tags. Signals say how
hard it is to find; tags say which question category it suits. Neither decides
anything — they are for stratifying the sample and for the author's choice.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from . import config
from .duplicates import _title_words

_STUB = re.compile(r"\b(omitted|repealed|deleted)\b", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z]{2,}")
#: "57. Act not to prevent use of Aadhaar number …—Omitted by the Aadhaar and
#: Other Laws (Amendment) Act 2019 (Act 14 of 2019), s. 25" — a placeholder whose
#: heading and citation defeat a word count. The editorial form is "<omitted|
#: repealed|rep.> <by|vide|w.e.f.>"; "is hereby repealed" (a real repeal section)
#: and "shall be omitted" (an operative amendment) do not have it.
_EDITORIAL_STUB = re.compile(r"\[?\s*\b(omitted|repealed|rep\.)\s+(by|vide|w\.\s*e\.\s*f)", re.IGNORECASE)
_EDITORIAL_STUB_WINDOW = 300
_EDITORIAL_STUB_MAX_CHARS = 600
_DEFINITION = re.compile(r"\b(definitions?|interpretation|meaning|defined)\b", re.IGNORECASE)
#: A number followed by a unit a legal threshold is stated in.
_NUMERIC = re.compile(
    r"\b(\d[\d,.]*|one|two|three|four|five|six|seven|eight|nine|ten|twelve|fifteen|"
    r"twenty|thirty|forty|fifty|sixty|ninety|hundred)\s+"
    r"(days?|weeks?|months?|years?|hours?|rupees|lakhs?|crores?|per\s*cent|percent)\b",
    re.IGNORECASE)
_SECTION_REF = re.compile(r"\bsections?\s+(\d{1,4}[A-Z]{0,3})\b")


def _load(corpus_dir: Path):
    metas, structures = {}, {}
    for path in sorted((corpus_dir / config.META_DIRNAME).glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        metas[meta["document_id"]] = meta
        structures[meta["document_id"]] = json.loads(
            (corpus_dir / config.STRUCTURE_DIRNAME / f"{meta['document_id']}.json").read_text(encoding="utf-8"))
    return metas, structures


def _read_text(corpus_dir: Path, meta: dict) -> str:
    with open(corpus_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
        return handle.read()


def exclusion(provision: dict, meta: dict, conflicted: set[str], body: str) -> str | None:
    """The first rule a provision fails, or ``None`` if it is eligible gold."""
    length = provision["char_end"] - provision["char_start"]
    if provision["evidence_confidence"] != "high":
        return "tier"
    if provision["key_ambiguous"]:
        return "ambiguous"
    if meta["document_id"] in conflicted:
        return "title_conflict"
    if meta["quality"]["ocr_text_source"] not in config.GOLD_TEXT_SOURCES:
        return "text_source"
    if meta["quality"]["structure_confidence"] != "structured":
        return "structure"
    if length < config.GOLD_MIN_CHARS:
        return "too_short"
    if length > config.GOLD_MAX_CHARS:
        return "too_long"
    if _STUB.search(body) and len(_WORD.findall(body)) <= config.GOLD_STUB_MAX_WORDS:
        # "12. [Omitted.]" — a placeholder, not law.
        return "stub"
    if length <= _EDITORIAL_STUB_MAX_CHARS and _EDITORIAL_STUB.search(body[:_EDITORIAL_STUB_WINDOW]):
        return "stub"
    return None


def build_pool(corpus_dir: Path) -> tuple[list[dict], dict]:
    """Return (pool rows, report). Rows are sorted by (document_id, key)."""
    corpus_dir = Path(corpus_dir)
    metas, structures = _load(corpus_dir)
    duplicates = json.loads((corpus_dir / config.DUPLICATES_FILENAME).read_text(encoding="utf-8"))
    conflicted = {d for e in duplicates["title_conflicts"] for d in (e["a"], e["b"])}
    cluster_of, cluster_size = {}, {}
    for c in duplicates["clusters"]:
        for d in c["documents"]:
            cluster_of[d["document_id"]] = c["cluster_id"]
            cluster_size[d["document_id"]] = len(c["documents"])
    parallel_docs = {d for e in duplicates["parallel"] for d in (e["a"], e["b"])}

    classes: dict[tuple[str, str], dict] = {}
    with open(corpus_dir / config.PROVISION_EQUIVALENTS_FILENAME, "r", encoding="utf-8") as handle:
        for line in handle:
            c = json.loads(line)
            for m in c["members"]:
                classes[(m["document_id"], m["key"])] = c

    # Corpus-wide difficulty: how many documents carry a provision with this
    # label ("section 3" is in most of them), and how many share this title.
    number_docs: dict[tuple, set] = defaultdict(set)
    for did, s in structures.items():
        for p in s["provisions"]:
            number_docs[(p["unit_type"], p["number"])].add(did)
    title_key = {did: " ".join(sorted(set(_title_words(m["title"])))) for did, m in metas.items()}
    title_count = Counter(title_key.values())

    rows: list[dict] = []
    excluded: Counter = Counter()
    for did in sorted(metas):
        meta, provisions = metas[did], structures[did]["provisions"]
        text = _read_text(corpus_dir, meta)
        by_number = defaultdict(list)
        for p in provisions:
            if p["unit_type"] == "section" and not p["key_ambiguous"]:
                by_number[p["number"]].append(p["key"])
        for p in provisions:
            body = text[p["char_start"]:p["char_end"]]
            reason = exclusion(p, meta, conflicted, body)
            if reason:
                excluded[reason] += 1
                continue
            refs = sorted({by_number[n][0] for n in _SECTION_REF.findall(body)
                           if len(by_number.get(n, [])) == 1 and by_number[n][0] != p["key"]})
            cls = classes.get((did, p["key"]))
            tags = []
            if _DEFINITION.search(p["heading"] or ""):
                tags.append("definitional")
            if _NUMERIC.search(body):
                tags.append("numeric")
            if refs:
                tags.append("cross_reference")
            if did in parallel_docs:
                tags.append("has_parallel_law")
            rows.append({
                "document_id": did,
                "key": p["key"],
                "category": meta["category"],
                "jurisdiction": meta["jurisdiction"],
                "title": meta["title"],
                "year": meta["year"],
                "unit_type": p["unit_type"],
                "number": p["number"],
                "heading": p["heading"],
                "char_start": p["char_start"],
                "char_end": p["char_end"],
                "page_start": p["page_start"],
                "page_end": p["page_end"],
                "text_sha256": p["text_sha256"],
                "document_text_sha256": meta["text"]["sha256"],
                "cluster_id": cluster_of.get(did),
                "equivalence_class": cls["class_id"] if cls else None,
                "tags": tags,
                "references": refs,
                "signals": {
                    "char_count": p["char_end"] - p["char_start"],
                    "number_ambiguity": len(number_docs[(p["unit_type"], p["number"])]),
                    "title_collisions": title_count[title_key[did]],
                    "cluster_size": cluster_size.get(did, 1),
                    "equivalents": (cls["documents"] - 1) if cls else 0,
                    "cross_instrument_equivalents": bool(cls) and not cls["same_instrument"],
                },
            })

    report = {
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "rules": {"tier": "high", "text_sources": sorted(config.GOLD_TEXT_SOURCES),
                  "min_chars": config.GOLD_MIN_CHARS, "max_chars": config.GOLD_MAX_CHARS,
                  "stub_max_words": config.GOLD_STUB_MAX_WORDS},
        "provisions": sum(len(s["provisions"]) for s in structures.values()),
        "pool": len(rows),
        "excluded": {r: excluded[r] for r in config.GOLD_EXCLUSION_ORDER},
        "pool_by_category": dict(sorted(Counter(r["category"] for r in rows).items())),
        "pool_documents": len({r["document_id"] for r in rows}),
        "tags": dict(sorted(Counter(t for r in rows for t in r["tags"]).items())),
    }
    return rows, report
