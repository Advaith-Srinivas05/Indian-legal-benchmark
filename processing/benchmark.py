"""The PDF extraction benchmark: select ~100 documents, process them, report.

This is the deliverable of Phase 2's first step. It answers one question — *how
reliably can 19,802 India Code PDFs be turned into structured, citable legal
text?* — on a deterministic sample, before anything is scaled to the full
corpus.

It writes four things under ``data/benchmark/pdf_extraction/``:

``sample.json``     the exact documents chosen, and why each was chosen
``report.json``     every metric, per document and aggregated
``report.md``       the same, readable, with worked examples
``examples/``       extracted text and parsed structure for one document of
                    each kind, so the numbers can be checked against reality

Per-document output goes to ``data/processed/indiacode/<document_id>/`` as
usual. Nothing under ``data/raw/`` is touched.

Usage::

    python -m processing.benchmark                  # the full ~100-document run
    python -m processing.benchmark --limit 5        # a quick smoke run
    python -m processing.benchmark --dry-run        # choose the sample only
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ingestion import config as ingestion_config
from ingestion.utils import atomic_write_text, utcnow_iso

from . import __version__, config, ocr, sample
from .backends import default_backend
from .corpus import Corpus
from .errors import BackendUnavailableError, ProcessingError
from .models import ProcessedDocument
from .process import process_document

log = logging.getLogger("processing.benchmark")

#: The example documents the project spec asks the benchmark to show.
EXAMPLE_SLOTS = (
    ("central_act", "one Central Act"),
    ("state_act", "one State Act"),
    ("rule", "one Rule"),
    ("regulation", "one Regulation"),
    ("scanned", "one scanned PDF"),
    ("difficult", "one difficult / unusual PDF"),
)


# --- Running --------------------------------------------------------------------


def run(
    data_dir: Path,
    *,
    sample_size: Optional[int] = None,
    limit: Optional[int] = None,
    workers: int = 4,
    detect_tables: bool = True,
    write_documents: bool = True,
) -> tuple[list[dict], list[ProcessedDocument], dict]:
    """Select, process and measure.

    Returns ``(sample records, per-document results, report)``. The results come
    back alongside the report because the worked examples are rendered from the
    live objects, not from the report's summary rows.
    """
    corpus = Corpus.load(data_dir, require_inventory=True)
    missing = corpus.missing()
    if missing:
        log.warning(
            "%d manifest entries have no PDF on disk; they are excluded from "
            "sampling (first: %s)", len(missing), missing[0].document_id,
        )
    records = sample.select(corpus.present(), target=sample_size)
    if limit:
        records = records[:limit]

    by_id = {d.document_id: d for d in corpus}
    documents = [by_id[r["document_id"]] for r in records]

    backend = default_backend()
    started = time.monotonic()
    results = _process_all(
        documents, data_dir, backend=backend, workers=workers,
        detect_tables=detect_tables, write=write_documents,
    )
    report = build_report(
        records, results, backend_name=backend.name,
        seconds=time.monotonic() - started,
    )
    return records, results, report


def _process_all(documents, data_dir, *, backend, workers, detect_tables, write):
    """Process every document, reporting progress as it goes.

    Each worker opens its own PDF; no document object is shared between threads.
    """
    results: list[ProcessedDocument] = []
    total = len(documents)
    done = 0

    def work(document) -> ProcessedDocument:
        return process_document(
            document, data_dir, backend=backend,
            detect_tables=detect_tables, write=write,
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        for result in executor.map(work, documents):
            results.append(result)
            done += 1
            if done % 10 == 0 or done == total:
                log.info("  … %d/%d documents processed", done, total)
    return results


# --- Measuring ------------------------------------------------------------------


def problem_flags(result: ProcessedDocument) -> list[str]:
    """Why a document counts as an "unusual extraction problem".

    Deliberately explicit rather than a score: each flag names something a human
    would have to look at before this document could be trusted in a retrieval
    corpus.
    """
    flags: list[str] = []
    if not result.ok:
        return [f"extraction_failed:{result.error_type}"]
    extraction, structure = result.extraction, result.structure
    assert extraction is not None and structure is not None

    if extraction.text_extraction_status != "ok":
        flags.append(f"text_extraction_{extraction.text_extraction_status}")
    if extraction.pdf_type == "vector_outlined":
        flags.append("vector_outlined_text")
    # The *document-level* verdict, not "has a sideways page anywhere". A
    # landscape fold-out schedule in a 347-page act is typesetting, not a
    # problem; the page is still counted in the orientation section of the
    # report either way.
    if extraction.orientation.get("orientation_suspect"):
        flags.append("orientation_suspect")
    if extraction.direction_inconsistent_page_count:
        flags.append("reversed_writing_direction")
    if (result.language is not None
            and result.language.signals.get("page_language", {}).get("mixed_language")):
        flags.append("mixed_language_pages")
    if result.quality is not None and result.quality.classification != "good":
        flags.append(f"extraction_quality_{result.quality.classification}")
    if result.language is not None and result.language.content_language != "en":
        flags.append(f"content_language_{result.language.content_language}")
    if extraction.empty_page_count:
        flags.append("empty_pages")
    if structure.confidence == "unstructured":
        flags.append("no_structure_identified")
    if any("could not be loaded" in w for w in extraction.warnings):
        flags.append("page_load_error")
    if any("not in ascending order" in w for w in structure.warnings):
        flags.append("section_numbering_disordered")
    return flags


def _document_row(record: dict, result: ProcessedDocument) -> dict:
    row = {
        "document_id": record["document_id"],
        "category": record["category"],
        "document_type": record["document_type"],
        "title": record["title"],
        "jurisdiction": record["jurisdiction"],
        "year": record["year"],
        "bytes": record["bytes"],
        "era_band": record["era_band"],
        "size_band": record["size_band"],
        "ok": result.ok,
        "error_type": result.error_type,
        "error_message": result.error_message,
        "seconds": round(result.seconds, 3),
        "problem_flags": problem_flags(result),
    }
    if result.ok and result.extraction and result.structure:
        extraction, structure = result.extraction, result.structure
        counts = structure.counts
        row.update({
            "pages": extraction.page_count,
            "chars": extraction.char_count,
            "empty_pages": extraction.empty_page_count,
            "pdf_type": extraction.pdf_type,
            "text_extraction_status": extraction.text_extraction_status,
            "text_quality_suspect": bool(
                extraction.classification_evidence.get("text_quality_suspect")
            ),
            "image_backed_pages": extraction.classification_evidence.get(
                "image_backed_pages", 0),
            "vector_outlined_pages": extraction.vector_outlined_page_count,
            "sideways_pages": extraction.sideways_page_count,
            "direction_inconsistent_pages": (
                extraction.direction_inconsistent_page_count),
            "declared_rotations": extraction.orientation.get(
                "declared_rotations", []),
            "mixed_language_pages": (
                result.language.signals.get("page_language", {}).get(
                    "non_english_pages", 0)
                if result.language else 0),
            "ocr_preprocessing": (
                result.ocr_decision.preprocessing if result.ocr_decision else []),
            "tables": extraction.table_count,
            "table_candidates": extraction.table_candidate_count,
            "pages_with_tables": sum(
                1 for p in extraction.pages if p.retained_tables),
            "furniture_lines": sum(len(p.furniture) for p in extraction.pages),
            "footnotes": extraction.footnote_count,
            "toc_pages": len(structure.toc_pages),
            "structure_confidence": structure.confidence,
            "unit_vocabulary": structure.unit_vocabulary,
            "content_language": (
                result.language.content_language if result.language else None),
            "language_function_word_rate": (
                result.language.signals.get("function_word_rate")
                if result.language else None),
            "extraction_quality": (
                result.quality.classification if result.quality else None),
            "quality_score": round(result.quality.score, 4) if result.quality else None,
            "quality_failed_checks": (
                result.quality.failed_checks if result.quality else []),
            "ocr_action": result.ocr_decision.action if result.ocr_decision else None,
            "ocr_text_source": (
                result.ocr_decision.text_source if result.ocr_decision else None),
            "ocr_pages": (
                result.ocr_decision.estimated_pages if result.ocr_decision else 0),
            "eligible_for_indexing": result.eligible_for_indexing,
            "parts": counts.get("part", 0),
            "chapters": counts.get("chapter", 0),
            "sections": counts.get("section", 0),
            "articles": counts.get("article", 0),
            "subsections": counts.get("subsection", 0),
            "clauses": counts.get("clause", 0),
            "subclauses": counts.get("subclause", 0),
            "provisos": counts.get("proviso", 0),
            "explanations": counts.get("explanation", 0),
            "schedules": counts.get("schedule", 0),
            "has_preamble": bool(structure.preamble),
            "producer": (extraction.pdf_metadata or {}).get("producer"),
            "extraction_warnings": extraction.warnings,
            "structure_warnings": structure.warnings,
        })
    return row


def build_report(
    records: list[dict], results: list[ProcessedDocument], *,
    backend_name: str, seconds: float,
) -> dict:
    """Aggregate per-document results into the benchmark report."""
    by_id = {r.document.document_id: r for r in results}
    rows = [
        _document_row(record, by_id[record["document_id"]])
        for record in records
        if record["document_id"] in by_id
    ]
    ok_rows = [r for r in rows if r["ok"]]

    pages = [r["pages"] for r in ok_rows]
    chars = [r["chars"] for r in ok_rows]

    totals = {
        "documents_selected": len(records),
        "documents_processed": len(rows),
        "successful_extraction": len(ok_rows),
        "extraction_failures": len(rows) - len(ok_rows),
        "pages_extracted": sum(pages),
        "characters_extracted": sum(chars),
        "average_pages_per_document": round(statistics.fmean(pages), 1) if pages else 0,
        "median_pages_per_document": statistics.median(pages) if pages else 0,
        "average_characters_per_document": round(statistics.fmean(chars)) if chars else 0,
        "median_characters_per_document": int(statistics.median(chars)) if chars else 0,
        "empty_pages": sum(r["empty_pages"] for r in ok_rows),
        "documents_with_empty_pages": sum(1 for r in ok_rows if r["empty_pages"]),
        "wall_clock_seconds": round(seconds, 1),
        "seconds_per_document": round(seconds / len(rows), 2) if rows else 0,
    }

    pdf_types = Counter(r["pdf_type"] for r in ok_rows)
    statuses = Counter(r["text_extraction_status"] for r in ok_rows)

    structure_metrics = {
        "documents_with_sections": sum(1 for r in ok_rows if r["sections"]),
        "documents_with_articles": sum(1 for r in ok_rows if r["articles"]),
        "documents_with_footnotes": sum(1 for r in ok_rows if r["footnotes"]),
        "total_articles": sum(r["articles"] for r in ok_rows),
        "total_footnotes": sum(r["footnotes"] for r in ok_rows),
        "unit_vocabulary": dict(Counter(
            r["unit_vocabulary"] or "none" for r in ok_rows)),
        "documents_with_chapters": sum(1 for r in ok_rows if r["chapters"]),
        "documents_with_parts": sum(1 for r in ok_rows if r["parts"]),
        "documents_with_subsections": sum(1 for r in ok_rows if r["subsections"]),
        "documents_with_clauses": sum(1 for r in ok_rows if r["clauses"]),
        "documents_with_schedules": sum(1 for r in ok_rows if r["schedules"]),
        "documents_with_preamble": sum(1 for r in ok_rows if r["has_preamble"]),
        "documents_with_toc_detected": sum(1 for r in ok_rows if r["toc_pages"]),
        "confidence": dict(Counter(r["structure_confidence"] for r in ok_rows)),
        "total_sections": sum(r["sections"] for r in ok_rows),
        "total_subsections": sum(r["subsections"] for r in ok_rows),
        "total_clauses": sum(r["clauses"] for r in ok_rows),
    }

    table_metrics = {
        "documents_with_tables": sum(1 for r in ok_rows if r["tables"]),
        "documents_with_table_candidates": sum(
            1 for r in ok_rows if r["table_candidates"]),
        "total_tables_retained": sum(r["tables"] for r in ok_rows),
        "total_table_candidates": sum(r["table_candidates"] for r in ok_rows),
        "candidates_rejected": sum(
            r["table_candidates"] - r["tables"] for r in ok_rows),
    }

    flag_counter: Counter = Counter()
    for row in rows:
        flag_counter.update(row["problem_flags"])
    problems = {
        "documents_with_problems": sum(1 for r in rows if r["problem_flags"]),
        "by_flag": dict(flag_counter.most_common()),
        "documents": [
            {"document_id": r["document_id"], "category": r["category"],
             "title": r["title"], "flags": r["problem_flags"]}
            for r in rows if r["problem_flags"]
        ],
    }

    per_category: dict[str, dict] = {}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in ok_rows:
        grouped[row["category"]].append(row)
    for category, group in sorted(grouped.items()):
        per_category[category] = {
            "documents": len(group),
            "pages": sum(r["pages"] for r in group),
            "average_pages": round(statistics.fmean([r["pages"] for r in group]), 1),
            "average_characters": round(
                statistics.fmean([r["chars"] for r in group])),
            "pdf_types": dict(Counter(r["pdf_type"] for r in group)),
            "text_extraction_status": dict(
                Counter(r["text_extraction_status"] for r in group)),
            "with_sections": sum(1 for r in group if r["sections"]),
            "with_chapters": sum(1 for r in group if r["chapters"]),
            "with_subsections": sum(1 for r in group if r["subsections"]),
            "with_tables": sum(1 for r in group if r["tables"]),
            "ocr_quality_suspect": sum(1 for r in group if r["text_quality_suspect"]),
        }

    return {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "backend": backend_name,
        "sample": {**sample.describe(records), "quotas": config.CATEGORY_QUOTAS},
        "totals": totals,
        "pdf_types": dict(pdf_types),
        "text_extraction_status": dict(statuses),
        "ocr_quality": {
            "documents_flagged_suspect": sum(
                1 for r in ok_rows if r["text_quality_suspect"]),
            "scanned_documents_with_text_layer": sum(
                1 for r in ok_rows
                if r["pdf_type"] in ("scanned", "mixed")
                and r["text_extraction_status"] != "requires_ocr"
            ),
        },
        "structure": structure_metrics,
        "tables": table_metrics,
        "orientation": _orientation_metrics(ok_rows),
        "content_language": _language_metrics(ok_rows),
        "extraction_quality": _quality_metrics(ok_rows),
        "ocr": _ocr_metrics(results),
        "eligibility": {
            "eligible_for_indexing": sum(
                1 for r in ok_rows if r["eligible_for_indexing"]),
            "quarantined": sum(
                1 for r in ok_rows if not r["eligible_for_indexing"]),
            "gate": (
                "content_language == 'en' AND extraction_quality == 'good' "
                "AND the pages are not substantially sideways or reversed "
                "AND the document is not awaiting OCR. "
                "Quarantine is a routing decision: nothing is deleted, and the "
                "extracted text stays on disk."
            ),
        },
        "problems": problems,
        "by_category": per_category,
        "thresholds": _threshold_snapshot(),
        "documents": rows,
        "examples": choose_examples(rows),
    }


def _orientation_metrics(rows: list[dict]) -> dict:
    """How many pages are not the right way round, and where.

    Two distinct faults, reported separately because they need different fixes.
    A *sideways* page needs rotating before OCR. A page whose writing direction
    is *reversed* relative to its own document has a broken character map: no
    rotation helps it, and its extracted characters are not the ones printed on
    the page.
    """
    sideways = [r for r in rows if r.get("sideways_pages")]
    reversed_rows = [r for r in rows if r.get("direction_inconsistent_pages")]
    declared: Counter = Counter()
    for row in rows:
        declared.update(str(value) for value in row.get("declared_rotations") or [])
    return {
        "documents_with_sideways_pages": len(sideways),
        "total_sideways_pages": sum(r["sideways_pages"] for r in sideways),
        "documents_with_reversed_direction": len(reversed_rows),
        "total_reversed_direction_pages": sum(
            r["direction_inconsistent_pages"] for r in reversed_rows),
        "documents_with_declared_rotation": sum(
            1 for r in rows if r.get("declared_rotations")),
        "declared_rotation_values": dict(sorted(declared.items())),
        "documents": [
            {"document_id": r["document_id"], "title": r["title"],
             "sideways_pages": r.get("sideways_pages", 0),
             "reversed_direction_pages": r.get("direction_inconsistent_pages", 0),
             "pages": r["pages"]}
            for r in sorted(
                {r["document_id"]: r for r in sideways + reversed_rows}.values(),
                key=lambda r: r["document_id"],
            )
        ],
        "note": (
            "A PDF may declare /Rotate and still be upright; a PDF may declare "
            "nothing and be sideways. These counts are measured from the writing "
            "direction of the extracted lines, not from what the file claims."
        ),
    }


def _language_metrics(rows: list[dict]) -> dict:
    mixed = [r for r in rows if r.get("mixed_language_pages")]
    return {
        "by_content_language": dict(Counter(r["content_language"] for r in rows)),
        "documents_not_english": [
            {"document_id": r["document_id"], "title": r["title"],
             "content_language": r["content_language"],
             "function_word_rate": r["language_function_word_rate"]}
            for r in rows if r["content_language"] != "en"
        ],
        "documents_with_non_english_pages": [
            {"document_id": r["document_id"], "title": r["title"],
             "non_english_pages": r["mixed_language_pages"], "pages": r["pages"],
             "content_language": r["content_language"]}
            for r in sorted(mixed, key=lambda r: -r["mixed_language_pages"])
        ],
        "note": (
            "India Code metadata called every one of these English. This check "
            "reads the extracted content instead — as a whole, and page by page, "
            "because a document can be part English and part not."
        ),
    }


def _quality_metrics(rows: list[dict]) -> dict:
    failed: Counter = Counter()
    for row in rows:
        failed.update(row["quality_failed_checks"])
    return {
        "by_classification": dict(Counter(r["extraction_quality"] for r in rows)),
        "failed_checks": dict(failed.most_common()),
        "worst_documents": [
            {"document_id": r["document_id"], "title": r["title"],
             "score": r["quality_score"], "pdf_type": r["pdf_type"],
             "failed": r["quality_failed_checks"]}
            for r in sorted(rows, key=lambda r: r["quality_score"] or 0)[:10]
        ],
    }


def _ocr_metrics(results: list[ProcessedDocument]) -> dict:
    decisions = [r.ocr_decision for r in results if r.ocr_decision]
    summary = ocr.summarise(decisions)
    summary["note"] = (
        "Decisions only. No OCR was run over the corpus; see the OCR engine "
        "evaluation for what running it would cost and produce."
    )
    return summary


def _threshold_snapshot() -> dict:
    """Every threshold the run used, so a number can be reproduced later."""
    return {
        "min_page_alpha_chars": config.MIN_PAGE_ALPHA_CHARS,
        "image_backed_area_ratio": config.IMAGE_BACKED_AREA_RATIO,
        "scanned_page_ratio": config.SCANNED_PAGE_RATIO,
        "text_based_page_ratio": config.TEXT_BASED_PAGE_RATIO,
        "text_ok_page_ratio": config.TEXT_OK_PAGE_RATIO,
        "requires_ocr_page_ratio": config.REQUIRES_OCR_PAGE_RATIO,
        "quality_suspect_page_ratio": config.QUALITY_SUSPECT_PAGE_RATIO,
        "furniture_min_page_ratio": config.FURNITURE_MIN_PAGE_RATIO,
        "table_min_rows": config.TABLE_MIN_ROWS,
        "table_min_cols": config.TABLE_MIN_COLS,
        "table_min_filled_ratio": config.TABLE_MIN_FILLED_RATIO,
        "table_min_column_fill": config.TABLE_MIN_COLUMN_FILL,
        "table_min_row_consistency": config.TABLE_MIN_ROW_CONSISTENCY,
        "toc_run_length": config.TOC_RUN_LENGTH,
        "max_heading_chars": config.MAX_HEADING_CHARS,
        "vector_outline_min_paths": config.VECTOR_OUTLINE_MIN_PATHS,
        "vector_outline_page_ratio": config.VECTOR_OUTLINE_PAGE_RATIO,
        "quality_good_score": config.QUALITY_GOOD_SCORE,
        "quality_questionable_score": config.QUALITY_QUESTIONABLE_SCORE,
        "quality_mean_word_length_min": config.QUALITY_MEAN_WORD_LENGTH_MIN,
        "quality_single_char_rate_max": config.QUALITY_SINGLE_CHAR_RATE_MAX,
        "quality_mixed_case_rate_max": config.QUALITY_MIXED_CASE_RATE_MAX,
        "language_english_rate": config.LANGUAGE_ENGLISH_RATE,
        "language_non_english_rate": config.LANGUAGE_NON_ENGLISH_RATE,
        "language_min_clean_words": config.LANGUAGE_MIN_CLEAN_WORDS,
        "language_page_min_letters": config.LANGUAGE_PAGE_MIN_LETTERS,
        "language_non_latin_letter_ratio": config.LANGUAGE_NON_LATIN_LETTER_RATIO,
        "language_mixed_non_en_page_ratio": config.LANGUAGE_MIXED_NON_EN_PAGE_RATIO,
        "orientation_min_lines": config.ORIENTATION_MIN_LINES,
        "orientation_sideways_line_ratio": config.ORIENTATION_SIDEWAYS_LINE_RATIO,
        "orientation_suspect_page_ratio": config.ORIENTATION_SUSPECT_PAGE_RATIO,
        "footnote_min_page_fraction": config.FOOTNOTE_MIN_PAGE_FRACTION,
    }


def choose_examples(rows: list[dict]) -> dict:
    """Pick one document per example slot, deterministically.

    For the four category slots the pick is the document with the most sections
    (the one that best shows what structure parsing does); for ``scanned`` it is
    the most page-covering scan; for ``difficult`` it is the document carrying
    the most problem flags. Ties break on ``document_id`` so the choice is
    reproducible.
    """
    ok_rows = [r for r in rows if r["ok"]]
    examples: dict[str, Optional[str]] = {}

    for slot, _ in EXAMPLE_SLOTS:
        if slot in ("scanned", "difficult"):
            continue
        candidates = [r for r in ok_rows if r["document_type"] == slot]
        examples[slot] = (
            max(candidates, key=lambda r: (r["sections"], r["document_id"]))["document_id"]
            if candidates else None
        )

    scanned = [r for r in ok_rows if r["pdf_type"] in ("scanned", "mixed")]
    examples["scanned"] = (
        max(scanned, key=lambda r: (r["image_backed_pages"], r["document_id"]))["document_id"]
        if scanned else None
    )

    difficult = [r for r in rows if r["problem_flags"]]
    examples["difficult"] = (
        max(difficult, key=lambda r: (len(r["problem_flags"]), r["document_id"]))["document_id"]
        if difficult else None
    )
    return examples


# --- Writing --------------------------------------------------------------------


def benchmark_dir(data_dir: Path) -> Path:
    return Path(data_dir) / config.BENCHMARK_SUBDIR


# --- Comparing two runs ---------------------------------------------------------


def compare(before: dict, after: dict) -> dict:
    """Diff two benchmark reports over the same sample.

    Only meaningful because the sampler is deterministic: the two runs are the
    same 100 documents, so every difference is a change in the processing rather
    than a change in what was processed. That is checked, not assumed.
    """
    before_ids = {r["document_id"] for r in before.get("documents", [])}
    after_ids = {r["document_id"] for r in after.get("documents", [])}
    same_sample = before_ids == after_ids

    def counts(report: dict, path: list[str]) -> dict:
        node = report
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        return node if isinstance(node, dict) else {}

    def totals(report: dict, key: str):
        return report.get("totals", {}).get(key)

    sections = {}
    for label, path in (
        ("pdf_types", ["pdf_types"]),
        ("text_extraction_status", ["text_extraction_status"]),
        ("structure_confidence", ["structure", "confidence"]),
        ("problem_flags", ["problems", "by_flag"]),
    ):
        sections[label] = _diff_counts(counts(before, path), counts(after, path))

    scalar = {}
    for key in ("documents_processed", "successful_extraction", "extraction_failures",
                "pages_extracted", "empty_pages", "average_characters_per_document"):
        scalar[key] = {"before": totals(before, key), "after": totals(after, key)}
    for label, path in (
        ("documents_with_sections", ["structure", "documents_with_sections"]),
        ("documents_with_chapters", ["structure", "documents_with_chapters"]),
        ("documents_with_subsections", ["structure", "documents_with_subsections"]),
        ("documents_with_tables", ["tables", "documents_with_tables"]),
        ("total_tables_retained", ["tables", "total_tables_retained"]),
        ("documents_with_problems", ["problems", "documents_with_problems"]),
        ("documents_with_sideways_pages",
         ["orientation", "documents_with_sideways_pages"]),
        ("documents_with_reversed_direction",
         ["orientation", "documents_with_reversed_direction"]),
        ("eligible_for_indexing", ["eligibility", "eligible_for_indexing"]),
        ("quarantined", ["eligibility", "quarantined"]),
    ):
        node_before, node_after = before, after
        for key in path:
            node_before = (node_before or {}).get(key)
            node_after = (node_after or {}).get(key)
        scalar[label] = {"before": node_before, "after": node_after}

    changed = []
    before_rows = {r["document_id"]: r for r in before.get("documents", [])}
    for row in after.get("documents", []):
        old = before_rows.get(row["document_id"])
        if not old:
            continue
        deltas = {}
        for key in ("pdf_type", "text_extraction_status", "structure_confidence",
                    "sections", "tables"):
            if old.get(key) != row.get(key):
                deltas[key] = {"before": old.get(key), "after": row.get(key)}
        if deltas:
            changed.append({
                "document_id": row["document_id"],
                "title": row.get("title"),
                "changes": deltas,
            })

    return {
        "same_sample": same_sample,
        "sample_size": {"before": len(before_ids), "after": len(after_ids)},
        "scalar": scalar,
        "counts": sections,
        "new_in_after": {
            "content_language": after.get("content_language", {}).get(
                "by_content_language"),
            "extraction_quality": after.get("extraction_quality", {}).get(
                "by_classification"),
            "ocr_actions": after.get("ocr", {}).get("by_action"),
            "documents_with_articles": after.get("structure", {}).get(
                "documents_with_articles"),
            "documents_with_footnotes": after.get("structure", {}).get(
                "documents_with_footnotes"),
        },
        "documents_changed": changed,
    }


def _diff_counts(before: dict, after: dict) -> dict:
    keys = sorted(set(before) | set(after))
    return {
        key: {"before": before.get(key, 0), "after": after.get(key, 0),
              "delta": after.get(key, 0) - before.get(key, 0)}
        for key in keys
    }


def render_comparison(diff: dict) -> str:
    lines = [
        "# Benchmark comparison — before vs after remediation",
        "",
        f"Same 100-document sample: **{diff['same_sample']}** "
        f"({diff['sample_size']['before']} → {diff['sample_size']['after']} documents). "
        "The sampler is deterministic, so every difference below is a change in "
        "the processing, not in what was processed.",
        "",
        "## Totals",
        "",
        "| metric | before | after |",
        "| --- | ---: | ---: |",
    ]
    for key, values in diff["scalar"].items():
        lines.append(f"| {key} | {values['before']} | {values['after']} |")

    for label, table in diff["counts"].items():
        lines += ["", f"## {label}", "", "| value | before | after | delta |",
                  "| --- | ---: | ---: | ---: |"]
        for key, values in table.items():
            delta = values["delta"]
            lines.append(
                f"| `{key}` | {values['before']} | {values['after']} | "
                f"{delta:+d} |"
            )

    lines += ["", "## New in the remediated run", ""]
    for key, value in diff["new_in_after"].items():
        lines.append(f"- **{key}**: `{value}`")

    lines += ["", f"## Documents whose classification changed "
                  f"({len(diff['documents_changed'])})", ""]
    for entry in diff["documents_changed"][:40]:
        changes = ", ".join(
            f"{key} {value['before']!r} → {value['after']!r}"
            for key, value in entry["changes"].items()
        )
        lines.append(f"- `{entry['document_id'][:60]}`: {changes}")
    return "\n".join(lines)


def write_sample(data_dir: Path, records: list[dict]) -> Path:
    path = benchmark_dir(data_dir) / config.SAMPLE_FILENAME
    payload = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "method": (
            "Stratified by category x era x stored size; ordered within each "
            "stratum by sha256(document_id); ties broken toward the least-used "
            "jurisdiction. No RNG and no clock, so the same corpus always yields "
            "the same sample."
        ),
        "quotas": config.CATEGORY_QUOTAS,
        "era_bands": [list(b) for b in config.ERA_BANDS],
        "size_bands": [list(b) for b in config.SIZE_BANDS],
        "coverage": sample.describe(records),
        "documents": records,
    }
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def write_report(data_dir: Path, report: dict) -> tuple[Path, Path]:
    directory = benchmark_dir(data_dir)
    json_path = directory / config.REPORT_JSON_FILENAME
    md_path = directory / config.REPORT_MARKDOWN_FILENAME
    atomic_write_text(json_path, json.dumps(report, indent=2, ensure_ascii=False))
    atomic_write_text(md_path, render_markdown(report))
    return json_path, md_path


def write_examples(data_dir: Path, report: dict, results: list[ProcessedDocument]) -> Path:
    """Write a readable excerpt for each example slot.

    The point is falsifiability: every headline number in the report can be
    checked against the actual extracted text of a real document.
    """
    directory = benchmark_dir(data_dir) / config.EXAMPLES_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    by_id = {r.document.document_id: r for r in results}
    for slot, description in EXAMPLE_SLOTS:
        document_id = report["examples"].get(slot)
        if not document_id or document_id not in by_id:
            continue
        atomic_write_text(
            directory / f"{slot}.md",
            render_example(description, by_id[document_id]),
        )
    return directory


def render_example(description: str, result: ProcessedDocument, *, page_chars: int = 1800) -> str:
    """A single document's extraction and structure, as text."""
    document = result.document
    lines = [
        f"# Example: {description}",
        "",
        f"- **document_id**: `{document.document_id}`",
        f"- **title**: {document.title}",
        f"- **type / jurisdiction / year**: {document.document_type} / "
        f"{document.jurisdiction} / {document.year}",
        f"- **source**: {document.source_url}",
        f"- **raw PDF**: `{document.pdf_relpath}` ({document.bytes:,} bytes)",
        f"- **sha256**: `{document.sha256}`",
        "",
    ]
    if not result.ok:
        lines += ["## Extraction failed", "", f"`{result.error_type}`: {result.error_message}", ""]
        return "\n".join(lines)

    extraction, structure = result.extraction, result.structure
    assert extraction is not None and structure is not None
    lines += [
        "## Extraction",
        "",
        f"- pages: **{extraction.page_count}**, characters: **{extraction.char_count:,}**",
        f"- pdf_type: **{extraction.pdf_type}**, "
        f"text_extraction_status: **{extraction.text_extraction_status}**",
        f"- image-backed pages: "
        f"{extraction.classification_evidence.get('image_backed_pages')}"
        f" / {extraction.page_count}; vector-outlined pages: "
        f"{extraction.vector_outlined_page_count}",
        f"- tables retained: {extraction.table_count} "
        f"(of {extraction.table_candidate_count} candidates)",
        f"- footnote blocks separated: {extraction.footnote_count}",
        f"- empty pages: {extraction.empty_page_count}",
    ]
    if result.quality is not None:
        lines += [
            f"- extraction quality: **{result.quality.classification}** "
            f"(score {result.quality.score:.2f})",
        ]
        lines += [f"  - {r}" for r in result.quality.reasons[:6]]
    if result.language is not None:
        lines += [
            f"- content language: **{result.language.content_language}**",
        ]
        lines += [f"  - {r}" for r in result.language.reasons[:3]]
        page_language = result.language.signals.get("page_language") or {}
        if page_language.get("non_english_pages"):
            lines.append(
                f"  - pages in a non-Latin script: "
                f"{page_language['non_english_pages']} of "
                f"{page_language['measurable_pages']} measurable "
                f"({page_language['non_english_page_numbers'][:10]})"
            )
    if extraction.sideways_page_count or extraction.direction_inconsistent_page_count:
        lines.append(
            f"- orientation: **{extraction.sideways_page_count}** sideways pages, "
            f"**{extraction.direction_inconsistent_page_count}** with a reversed "
            f"writing direction; declared /Rotate values "
            f"`{extraction.orientation.get('declared_rotations')}`"
        )
    if result.ocr_decision is not None:
        lines += [
            f"- OCR decision: **{result.ocr_decision.action}** "
            f"(`{result.ocr_decision.text_source}`) — {result.ocr_decision.reason}",
            f"- required preprocessing: "
            f"`{result.ocr_decision.preprocessing or 'none'}`",
            f"- eligible for indexing: **{result.eligible_for_indexing}**",
        ]
    if extraction.warnings:
        lines.append("- warnings:")
        lines += [f"  - {w}" for w in extraction.warnings]
    lines.append("")

    lines += [
        "## Structure",
        "",
        f"- confidence: **{structure.confidence}**",
        f"- title: {structure.title!r} (source: {structure.title_source})",
        f"- counts: `{structure.counts}`",
        f"- contents pages excluded from the section index: {structure.toc_pages}",
    ]
    if structure.warnings:
        lines.append("- warnings:")
        lines += [f"  - {w}" for w in structure.warnings]
    lines.append("")

    sections = [u for u in structure.all_units() if u.unit_type == "section"][:3]
    if sections:
        lines += ["### First identified sections", ""]
        for unit in sections:
            lines += [
                f"**Section {unit.number} — {unit.heading}**  "
                f"(pages {unit.page_start}–{unit.page_end}, via `{unit.detected_by}`)",
                "",
                "```text",
                unit.text[:900].rstrip(),
                "```",
                "",
            ]
            for child in unit.children[:3]:
                lines.append(
                    f"- `{child.unit_type}` ({child.number}) "
                    f"p{child.page_start}–{child.page_end} "
                    f"[{child.detected_by}]: {child.text[:140].strip()!r}"
                )
            lines.append("")

    first_table = next(
        (t for page in extraction.pages for t in page.retained_tables), None)
    if first_table:
        lines += [
            "### First retained table",
            "",
            f"page {first_table.page_number}, {first_table.row_count} rows x "
            f"{first_table.col_count} columns, "
            f"{first_table.filled_ratio:.0%} of cells filled",
            "",
            "```json",
            json.dumps(first_table.rows[:6], indent=2, ensure_ascii=False),
            "```",
            "",
        ]

    first_footnote = next(
        (f for page in extraction.pages for f in page.footnotes), None)
    if first_footnote:
        lines += [
            "### First separated footnote block",
            "",
            f"page {first_footnote.page_number}, lines "
            f"{first_footnote.line_start}–{first_footnote.line_end}, via "
            f"`{first_footnote.detected_by}`",
            "",
            "```text",
            first_footnote.text[:700].rstrip(),
            "```",
            "",
            "(Still present verbatim in `pages.json`; separated here so it is "
            "not read as part of the provision above it.)",
            "",
        ]

    lines += ["### Raw extracted text, first page with text", "", "```text"]
    page = next((p for p in extraction.pages if p.has_text), extraction.pages[0])
    lines += [f"[page {page.page_number}]", page.text[:page_chars].rstrip(), "```", ""]
    if page.furniture:
        lines += ["Detected page furniture on this page (labelled, not removed):", ""]
        lines += [f"- line {f.line_index} `{f.kind}`: {f.text.strip()!r} — {f.reason}"
                  for f in page.furniture]
        lines.append("")
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    """The human-readable benchmark report."""
    totals = report["totals"]
    coverage = report["sample"]
    out: list[str] = [
        "# PDF extraction benchmark — India Code corpus",
        "",
        f"Generated {report['generated_at']} by {report['processor']} "
        f"using `{report['backend']}`.",
        "",
        "> Phase 2 measurement run over a deterministic sample. No embeddings, "
        "no index, no OCR, no full-corpus processing.",
        "",
        "## Sample",
        "",
        f"- documents selected: **{totals['documents_selected']}**",
        f"- by category: `{coverage['by_category']}`",
        f"- by era: `{coverage['by_era']}`",
        f"- by size band: `{coverage['by_size_band']}`",
        f"- distinct jurisdictions: **{coverage['distinct_jurisdictions']}**",
        f"- year range: {coverage['year_range'][0]}–{coverage['year_range'][1]}",
        f"- size range: {coverage['bytes_range'][0]:,}–{coverage['bytes_range'][1]:,} bytes",
        "",
        "## Extraction totals",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| documents selected | {totals['documents_selected']} |",
        f"| documents processed | {totals['documents_processed']} |",
        f"| successful extraction | {totals['successful_extraction']} |",
        f"| extraction failures | {totals['extraction_failures']} |",
        f"| pages extracted | {totals['pages_extracted']:,} |",
        f"| average pages / document | {totals['average_pages_per_document']} |",
        f"| median pages / document | {totals['median_pages_per_document']} |",
        f"| average characters / document | {totals['average_characters_per_document']:,} |",
        f"| median characters / document | {totals['median_characters_per_document']:,} |",
        f"| empty pages | {totals['empty_pages']} |",
        f"| documents with empty pages | {totals['documents_with_empty_pages']} |",
        f"| wall clock | {totals['wall_clock_seconds']}s "
        f"({totals['seconds_per_document']}s/doc) |",
        "",
        "## PDF type and text availability",
        "",
        f"- `pdf_type`: `{report['pdf_types']}`",
        f"- `text_extraction_status`: `{report['text_extraction_status']}`",
        f"- scanned/mixed documents that nevertheless carry a text layer: "
        f"**{report['ocr_quality']['scanned_documents_with_text_layer']}**",
        "",
        "## Extraction quality",
        "",
        f"- classification: `{report['extraction_quality']['by_classification']}`",
        f"- checks failed across the sample: "
        f"`{report['extraction_quality']['failed_checks']}`",
        "",
        "## Content-level language",
        "",
        f"- `content_language`: `{report['content_language']['by_content_language']}`",
        f"- India Code called every one of these documents English; this is what "
        f"their extracted text says.",
        "",
    ]
    for entry in report["content_language"]["documents_not_english"]:
        out.append(
            f"  - `{entry['content_language']}` — {entry['title']!r} "
            f"(English function words: {entry['function_word_rate']:.1%})"
        )
    mixed = report["content_language"].get("documents_with_non_english_pages") or []
    if mixed:
        out += [
            "",
            "Documents whose pages are not all the same language. A pooled "
            "average hides these; the page-level check is what finds them.",
            "",
        ]
        for entry in mixed:
            out.append(
                f"  - {entry['title']!r} — {entry['non_english_pages']} of "
                f"{entry['pages']} pages are not English "
                f"(document reads as `{entry['content_language']}`)"
            )

    orientation = report.get("orientation") or {}
    out += [
        "",
        "## Page orientation",
        "",
        "> Measured from the writing direction of the extracted lines, not from "
        "the PDF's `/Rotate`. A file can declare a rotation and be upright, or "
        "declare nothing and be sideways.",
        "",
        f"- documents with sideways pages: "
        f"**{orientation.get('documents_with_sideways_pages', 0)}** "
        f"({orientation.get('total_sideways_pages', 0)} pages)",
        f"- documents with pages whose writing direction is reversed relative to "
        f"the rest of the document: "
        f"**{orientation.get('documents_with_reversed_direction', 0)}** "
        f"({orientation.get('total_reversed_direction_pages', 0)} pages) — a "
        f"broken character map, not a rotation",
        f"- documents declaring a page `/Rotate`: "
        f"**{orientation.get('documents_with_declared_rotation', 0)}**, values "
        f"`{orientation.get('declared_rotation_values', {})}`",
        "",
    ]
    for entry in orientation.get("documents", []):
        out.append(
            f"  - {entry['title']!r} — {entry['sideways_pages']} sideways, "
            f"{entry['reversed_direction_pages']} reversed, of {entry['pages']} pages"
        )

    out += [
        "",
        "## OCR decisions (no OCR was run)",
        "",
        f"- by action: `{report['ocr']['by_action']}`",
        f"- by text source: `{report['ocr']['by_text_source']}`",
        f"- required preprocessing: `{report['ocr'].get('by_preprocessing', {})}`",
        f"- documents needing OCR: **{report['ocr']['documents_needing_ocr']}**, "
        f"covering **{report['ocr']['pages_needing_ocr']:,}** pages",
        "",
        "## Eligibility for downstream indexing",
        "",
        f"- eligible: **{report['eligibility']['eligible_for_indexing']}**, "
        f"quarantined: **{report['eligibility']['quarantined']}**",
        f"- gate: {report['eligibility']['gate']}",
        "",
        "## Legal structure detection",
        "",
        "| metric | documents |",
        "| --- | ---: |",
    ]
    structure = report["structure"]
    for label, key in (
        ("with detectable sections", "documents_with_sections"),
        ("with detectable articles", "documents_with_articles"),
        ("with detected footnote blocks", "documents_with_footnotes"),
        ("with detectable chapters", "documents_with_chapters"),
        ("with detectable parts", "documents_with_parts"),
        ("with detectable subsections", "documents_with_subsections"),
        ("with detectable clauses", "documents_with_clauses"),
        ("with detectable schedules", "documents_with_schedules"),
        ("with an identified preamble", "documents_with_preamble"),
        ("with a contents listing detected", "documents_with_toc_detected"),
    ):
        out.append(f"| {label} | {structure[key]} |")
    out += [
        "",
        f"- structure confidence: `{structure['confidence']}`",
        f"- units identified in total: {structure['total_sections']} sections, "
        f"{structure['total_subsections']} subsections, "
        f"{structure['total_clauses']} clauses",
        "",
        "## Tables",
        "",
        "> Reported as a conservative **lower bound**. A candidate must look "
        "like a populated grid and must not look like a contents listing or a "
        "footnote block; a real table that fails those tests is recorded as a "
        "rejected candidate with its reason.",
        "",
        f"- documents containing at least one retained table: "
        f"**{report['tables']['documents_with_tables']}**",
        f"- tables retained: {report['tables']['total_tables_retained']} "
        f"of {report['tables']['total_table_candidates']} candidates "
        f"({report['tables']['candidates_rejected']} rejected)",
        "",
        "## Documents with unusual extraction problems",
        "",
        f"**{report['problems']['documents_with_problems']}** of "
        f"{totals['documents_processed']} documents carry at least one flag.",
        "",
        "| flag | documents |",
        "| --- | ---: |",
    ]
    for flag, count in report["problems"]["by_flag"].items():
        out.append(f"| `{flag}` | {count} |")

    out += ["", "## Per category", "", "| category | docs | avg pages | avg chars | "
            "text_based | scanned | mixed | with sections | with tables | OCR suspect |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category, stats in report["by_category"].items():
        types = stats["pdf_types"]
        out.append(
            f"| {category} | {stats['documents']} | {stats['average_pages']} | "
            f"{stats['average_characters']:,} | {types.get('text_based', 0)} | "
            f"{types.get('scanned', 0)} | {types.get('mixed', 0)} | "
            f"{stats['with_sections']} | {stats['with_tables']} | "
            f"{stats['ocr_quality_suspect']} |"
        )

    out += ["", "## Worked examples", "",
            "Full text and parsed structure for each are in "
            f"`{config.BENCHMARK_SUBDIR.as_posix()}/{config.EXAMPLES_DIRNAME}/`.", ""]
    for slot, description in EXAMPLE_SLOTS:
        out.append(f"- **{description}**: `{report['examples'].get(slot)}`")

    failures = [r for r in report["documents"] if not r["ok"]]
    if failures:
        out += ["", "## Extraction failures", "", "| document | error |", "| --- | --- |"]
        for row in failures:
            out.append(
                f"| `{row['document_id']}` | `{row['error_type']}`: "
                f"{(row['error_message'] or '')[:160]} |"
            )

    out += ["", "## Thresholds used", "", "```json",
            json.dumps(report["thresholds"], indent=2), "```", ""]
    return "\n".join(out)


# --- CLI ------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m processing.benchmark",
        description=(
            "Benchmark PDF text extraction and legal-structure parsing on a "
            "deterministic, representative sample of the India Code corpus. "
            "Processes only the sample; never the whole corpus."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=ingestion_config.DEFAULT_DATA_DIR,
                        help="Root data directory (default: ./data).")
    parser.add_argument("--sample-size", type=int, default=config.BENCHMARK_SAMPLE_SIZE,
                        metavar="N",
                        help=f"Documents to sample (default: {config.BENCHMARK_SAMPLE_SIZE}).")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Process only the first N of the sample (smoke runs).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent documents (default: 4).")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip table detection (much faster; tables unreported).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Select and record the sample; process nothing.")
    parser.add_argument("--compare-with", type=Path, metavar="REPORT.JSON",
                        help="An earlier report.json to diff this run against. "
                             "Writes comparison.{json,md} beside the report.")
    parser.add_argument("--no-write", action="store_true",
                        help="Measure without writing data/processed/ output.")
    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    logging_group.add_argument("-q", "--quiet", action="store_true",
                               help="Warnings and errors only.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def render_summary(report: dict) -> str:
    totals = report["totals"]
    structure = report["structure"]
    return "\n".join([
        "",
        "============= PDF EXTRACTION BENCHMARK =============",
        f"  documents selected      : {totals['documents_selected']:>6,}",
        f"  documents processed     : {totals['documents_processed']:>6,}",
        f"  successful extraction   : {totals['successful_extraction']:>6,}",
        f"  extraction failures     : {totals['extraction_failures']:>6,}",
        "  ------------------------------------------------",
        f"  pages extracted         : {totals['pages_extracted']:>6,}",
        f"  avg pages / document    : {totals['average_pages_per_document']:>6}",
        f"  avg chars / document    : {totals['average_characters_per_document']:>6,}",
        f"  empty pages             : {totals['empty_pages']:>6,}",
        "  ------------------------------------------------",
        f"  text_based / scanned / mixed / vector_outlined : "
        f"{report['pdf_types'].get('text_based', 0)} / "
        f"{report['pdf_types'].get('scanned', 0)} / "
        f"{report['pdf_types'].get('mixed', 0)} / "
        f"{report['pdf_types'].get('vector_outlined', 0)}",
        f"  text ok / partial / requires_ocr : "
        f"{report['text_extraction_status'].get('ok', 0)} / "
        f"{report['text_extraction_status'].get('partial', 0)} / "
        f"{report['text_extraction_status'].get('requires_ocr', 0)}",
        f"  quality good / questionable / bad : "
        f"{report['extraction_quality']['by_classification'].get('good', 0)} / "
        f"{report['extraction_quality']['by_classification'].get('questionable', 0)} / "
        f"{report['extraction_quality']['by_classification'].get('bad', 0)}",
        f"  language en / uncertain / non_en : "
        f"{report['content_language']['by_content_language'].get('en', 0)} / "
        f"{report['content_language']['by_content_language'].get('uncertain', 0)} / "
        f"{report['content_language']['by_content_language'].get('non_en', 0)}",
        "  ------------------------------------------------",
        f"  with sections           : {structure['documents_with_sections']:>6,}",
        f"  with articles           : {structure['documents_with_articles']:>6,}",
        f"  with chapters           : {structure['documents_with_chapters']:>6,}",
        f"  with subsections        : {structure['documents_with_subsections']:>6,}",
        f"  with footnote blocks    : {structure['documents_with_footnotes']:>6,}",
        f"  with tables             : {report['tables']['documents_with_tables']:>6,}",
        f"  with problems           : "
        f"{report['problems']['documents_with_problems']:>6,}",
        "  ------------------------------------------------",
        f"  eligible for indexing   : "
        f"{report['eligibility']['eligible_for_indexing']:>6,}",
        f"  quarantined             : {report['eligibility']['quarantined']:>6,}",
        f"  needing OCR             : "
        f"{report['ocr']['documents_needing_ocr']:>6,}"
        f"  ({report['ocr']['pages_needing_ocr']:,} pages)",
        "====================================================",
    ])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, args.quiet)

    try:
        if args.dry_run:
            corpus = Corpus.load(args.data_dir, require_inventory=True)
            records = sample.select(corpus.present(), target=args.sample_size)
            path = write_sample(args.data_dir, records)
            print(json.dumps(sample.describe(records), indent=2))
            print(f"\nSample: {path}")
            return 0

        records, results, report = run(
            args.data_dir,
            sample_size=args.sample_size,
            limit=args.limit,
            workers=args.workers,
            detect_tables=not args.no_tables,
            write_documents=not args.no_write,
        )
        write_sample(args.data_dir, records)
        json_path, md_path = write_report(args.data_dir, report)
        write_examples(args.data_dir, report, results)
        if args.compare_with:
            previous = json.loads(
                Path(args.compare_with).read_text(encoding="utf-8"))
            diff = compare(previous, report)
            directory = benchmark_dir(args.data_dir)
            atomic_write_text(directory / "comparison.json",
                              json.dumps(diff, indent=2, ensure_ascii=False))
            atomic_write_text(directory / "comparison.md", render_comparison(diff))
            log.info("Comparison written to %s", directory / "comparison.md")
    except BackendUnavailableError as exc:
        log.error("%s", exc)
        return 3
    except (ProcessingError, OSError) as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("Benchmark interrupted.")
        return 130

    print(render_summary(report))
    print(f"\nReport:  {md_path}")
    print(f"JSON:    {json_path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
