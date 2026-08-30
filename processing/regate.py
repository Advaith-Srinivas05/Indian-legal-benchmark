"""Re-decide eligibility from what Phase 2 already wrote to disk.

Why this exists
---------------
A language gate changed. Every stage after extraction now reaches a different
verdict on some documents, and the honest way to act on that is to run those
stages again. Running *extraction* again is not required to do it: ``pages.json``
already holds both readings of every page and ``document.json`` holds the
counters the later stages consult, so the only thing a re-extraction would buy is
95 hours of re-running Tesseract over pages whose OCR text is already stored.

This module therefore replays exactly the part of :mod:`processing.process` that
comes after extraction and OCR -- language, structure, quality, OCR routing,
eligibility -- over the stored artefacts, and rewrites the two files when the
verdict moves. It calls the same functions ``process.py`` calls, in the same
order, for the same reasons; where that order is subtle the comment lives there
and is not repeated here.

What it is not
--------------
It is **not** a second processing pipeline and must not grow into one. It cannot
change a document's text, because it never opens a PDF. It cannot rescue a
document whose failure was in extraction itself -- those need
``processing.run --force``, which reads the source.

It also never promotes a document on its own authority: it recomputes
:attr:`processing.models.ProcessedDocument.eligible_for_indexing` from the same
four gates and records what moved, in both directions. A document the new gate
makes *worse* is demoted with the same lack of ceremony.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config, language, ocr, quality
from .models import (CorpusDocument, ExtractedDocument, ExtractedTable,
                     FootnoteBlock, FurnitureLine, PageText, ProcessedDocument)
from .ocr_engine import OcrRun
from .orientation import PageOrientation
from .process import write_outputs
from .structure import parse_structure

log = logging.getLogger(__name__)

NEWLINE = "\n"


class RegateError(RuntimeError):
    """Raised when a document's stored artefacts cannot be replayed."""


# --- reading back what Phase 2 wrote ---------------------------------------------


def _page_from_dict(raw: dict) -> PageText:
    """One ``pages.json`` entry as a :class:`PageText`.

    Every field the later stages read is restored, including the three that
    decide which of a page's two readings applies (docs/KNOWN_ISSUES.md B9).
    ``chunking.source.load_pages`` is the downstream twin of this function and
    restores the subset chunking needs; ``tests/test_regate.py`` asserts the two
    agree on the fields they share, so this copy cannot drift silently.
    """
    page = PageText(
        page_number=raw["page_number"],
        text=raw.get("text") or "",
        char_count=raw.get("char_count", 0),
        alpha_char_count=raw.get("alpha_char_count", 0),
        line_count=raw.get("line_count", 0),
        width=raw.get("width", 0.0),
        height=raw.get("height", 0.0),
        image_count=raw.get("image_count", 0),
        image_area_ratio=raw.get("image_area_ratio", 0.0),
        is_image_backed=raw.get("is_image_backed", False),
        drawing_path_count=raw.get("drawing_path_count", 0),
        is_vector_outlined=raw.get("is_vector_outlined", False),
        has_text=raw.get("has_text", False),
        is_empty=raw.get("is_empty", False),
        text_quality_suspect=raw.get("text_quality_suspect", False),
        quality_signals=raw.get("quality_signals") or {},
    )
    # Defaults match chunking.source: a file written before the OCR stage
    # existed has no text_source, and its backend text is its only reading.
    page.text_source = raw.get("text_source") or "backend"
    page.ocr = raw.get("ocr")
    page.quality = raw.get("quality")
    page.language = raw.get("language")
    page.indexable = raw.get("indexable", True)
    page.non_english_lines = raw.get("non_english_lines") or []
    page.warnings = raw.get("warnings") or []
    orientation = raw.get("orientation")
    if isinstance(orientation, dict):
        fields = {key: value for key, value in orientation.items()
                  if key in PageOrientation.__dataclass_fields__}
        fields.setdefault("page_number", page.page_number)
        page.orientation = PageOrientation(**fields)
    page.tables = [
        ExtractedTable(**{key: value for key, value in item.items()
                          if key in ExtractedTable.__dataclass_fields__})
        for item in raw.get("tables") or []
    ]
    page.furniture = [
        FurnitureLine(line_index=item["line_index"], text=item.get("text", ""),
                      kind=item.get("kind", ""), reason=item.get("reason", ""))
        for item in raw.get("furniture") or []
    ]
    page.footnotes = [
        FootnoteBlock(page_number=item.get("page_number", page.page_number),
                      line_start=item["line_start"], line_end=item["line_end"],
                      text=item.get("text", ""),
                      detected_by=item.get("detected_by", ""))
        for item in raw.get("footnotes") or []
    ]
    return page


def load_artefacts(data_dir: Path, document_id: str
                   ) -> tuple[dict, ExtractedDocument]:
    """``document.json`` and an :class:`ExtractedDocument` rebuilt from disk.

    The counters extraction established -- ``pdf_type``,
    ``text_extraction_status``, ``classification_evidence``, ``orientation`` --
    are restored verbatim rather than recomputed. They are statements about the
    PDF, and this module has not read the PDF.
    """
    directory = data_dir / "processed" / "indiacode" / document_id
    doc_path = directory / config.DOCUMENT_FILENAME
    pages_path = directory / config.PAGES_FILENAME
    if not doc_path.exists() or not pages_path.exists():
        raise RegateError(f"{document_id}: no processed output on disk")
    document = json.loads(doc_path.read_text(encoding="utf-8"))
    payload = json.loads(pages_path.read_text(encoding="utf-8"))
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise RegateError(f"{document_id}: pages.json carries no 'pages' list")
    meta = document.get("extraction") or {}
    extraction = ExtractedDocument(
        document_id=document_id,
        page_count=meta.get("page_count", len(raw_pages)),
        pages=[_page_from_dict(raw) for raw in raw_pages],
        pdf_type=meta.get("pdf_type", "text_based"),
        text_extraction_status=meta.get("text_extraction_status", "ok"),
        backend=meta.get("backend", ""),
        toc_pages=meta.get("toc_pages") or [],
        pdf_metadata=meta.get("pdf_metadata") or {},
        is_encrypted=meta.get("is_encrypted", False),
        classification_evidence=meta.get("classification_evidence") or {},
        orientation=meta.get("orientation") or {},
        warnings=meta.get("warnings") or [],
    )
    if extraction.page_count != len(extraction.pages):
        raise RegateError(
            f"{document_id}: document.json says {extraction.page_count} pages, "
            f"pages.json holds {len(extraction.pages)}")
    return document, extraction


# --- replaying the decision ------------------------------------------------------


def ocr_run_from_journal(record: Optional[dict]) -> Optional[OcrRun]:
    """The recorded OCR run, rebuilt from a ``processing_status.jsonl`` record.

    The journal is the authority for this and ``document.json`` is not, because
    this module **rewrites document.json**. The first version of it did not carry
    ``ocr_run`` through, so the rewrite recorded ``ocr_run: null`` -- and on the
    next pass every one of those documents looked like one OCR had never touched,
    which routes them straight back to ``ocr_required``. The journal is never
    rewritten here, so reading it cannot lose the fact that OCR ran.
    """
    if not isinstance(record, dict) or record.get("ocr_executed") is None:
        return None
    run = OcrRun()
    run.executed = bool(record.get("ocr_executed"))
    run.attempted = record.get("ocr_pages_attempted") or 0
    run.accepted = record.get("ocr_pages_accepted") or 0
    run.seconds = record.get("ocr_seconds") or 0.0
    run.truncated = bool(record.get("ocr_truncated"))
    return run


def _ocr_run_from_dict(raw: Optional[dict]) -> Optional[OcrRun]:
    """The ``ocr_run`` block of a ``document.json``, when it still has one."""
    if not isinstance(raw, dict):
        return None
    run = OcrRun()
    run.executed = raw.get("executed", False)
    run.attempted = raw.get("pages_attempted", 0)
    run.accepted = raw.get("pages_accepted", 0)
    run.seconds = raw.get("seconds", 0.0)
    run.truncated = raw.get("truncated", False)
    run.unavailable_reason = raw.get("unavailable_reason", "")
    return run


def _corpus_document(data_dir: Path, source: dict) -> CorpusDocument:
    """Rebuild the ingestion record from the ``source`` block, field for field.

    Rebuilt rather than re-read from the manifest so that ``write_outputs``
    reproduces the same block it was given: this module re-decides a verdict and
    must not quietly restate a document's provenance.
    """
    fields = {name: source.get(name)
              for name in CorpusDocument.__dataclass_fields__
              if name in source}
    fields["document_id"] = source["document_id"]
    fields["category"] = source.get("category") or ""
    fields["document_type"] = source.get("document_type") or ""
    fields["pdf_relpath"] = source.get("pdf_relpath") or ""
    fields["sha256"] = source.get("sha256") or ""
    fields["bytes"] = source.get("bytes", 0)
    fields["pdf_path"] = data_dir / (source.get("pdf_relpath") or "")
    return CorpusDocument(**fields)


@dataclass
class RegateResult:
    """What one document's verdict was, and what it is now."""

    document_id: str
    ok: bool = False
    changed: bool = False
    written: bool = False
    was_eligible: Optional[bool] = None
    now_eligible: Optional[bool] = None
    was_quality: str = ""
    now_quality: str = ""
    was_language: str = ""
    now_language: str = ""
    was_action: str = ""
    now_action: str = ""
    pages_total: int = 0
    pages_indexable_before: int = 0
    pages_indexable_after: int = 0
    pages_mangled: int = 0
    #: Word-validity rate, recorded for promotions only (see `regate_document`).
    word_validity: float = 0.0
    word_validity_measurable: bool = False
    #: True when every gate passed but the admission bar did not.
    held_back: bool = False
    error: str = ""

    @property
    def promoted(self) -> bool:
        return bool(self.now_eligible) and not self.was_eligible

    @property
    def demoted(self) -> bool:
        return bool(self.was_eligible) and not self.now_eligible

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "ok": self.ok,
            "changed": self.changed,
            "written": self.written,
            "was_eligible": self.was_eligible,
            "now_eligible": self.now_eligible,
            "was_quality": self.was_quality,
            "now_quality": self.now_quality,
            "was_language": self.was_language,
            "now_language": self.now_language,
            "was_action": self.was_action,
            "now_action": self.now_action,
            "pages_total": self.pages_total,
            "pages_indexable_before": self.pages_indexable_before,
            "pages_indexable_after": self.pages_indexable_after,
            "pages_mangled": self.pages_mangled,
            "word_validity": self.word_validity,
            "word_validity_measurable": self.word_validity_measurable,
            "held_back": self.held_back,
            "error": self.error,
        }


def regate_document(data_dir: Path, document_id: str, *,
                    write: bool = False,
                    baseline_eligible: Optional[bool] = None,
                    journal_record: Optional[dict] = None) -> RegateResult:
    """Re-run every stage after extraction, and rewrite the output if it moved.

    *baseline_eligible* is what the original Phase 2 run decided, read from
    ``data/processing_status.jsonl``. It matters because this module **rewrites
    the file it would otherwise read that from**: after one pass, a promoted
    document's ``document.json`` says ``eligible_for_indexing: true``, so a
    second pass would see no promotion, would not apply the admission bar, and
    would quietly admit the documents the first pass refused. The journal is not
    rewritten here, so it is the stable answer to "was this eligible before any
    of this started". Falling back to ``document.json`` is correct for a
    first pass and for a single document inspected by hand.
    """
    result = RegateResult(document_id=document_id)
    try:
        document, extraction = load_artefacts(data_dir, document_id)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    source = document.get("source") or {}
    result.pages_total = len(extraction.pages)
    result.was_eligible = (bool(document.get("eligible_for_indexing"))
                           if baseline_eligible is None else baseline_eligible)
    result.was_quality = (
        document.get("extraction_quality") or {}).get("classification", "")
    result.was_language = (
        document.get("content_language") or {}).get("content_language", "")
    result.was_action = (document.get("ocr_decision") or {}).get("action", "")
    result.pages_indexable_before = sum(1 for p in extraction.pages if p.indexable)

    try:
        # The order below mirrors processing/process.py exactly. The OCR stage is
        # absent because its output is already on the pages; nothing here can add
        # or remove a reading, only re-judge the ones already taken.
        metadata_language = source.get("language")
        assessment = language.assess_pages(
            extraction.pages, metadata_language=metadata_language)
        for page in extraction.pages:
            page.language = language.classify_page(page)
            page.non_english_lines = language.non_english_lines(page)
        result.pages_mangled = sum(
            1 for page in extraction.pages
            if isinstance(page.language, dict)
            and page.language.get("mangled_script"))

        english_of = language.english_line_text
        indexable = language.indexable_pages(extraction.pages, assessment)
        assessed = indexable or extraction.pages
        structure = parse_structure(assessed, metadata_title=source.get("title"))
        quality_assessment = quality.assess(
            assessed, structure=structure,
            text=NEWLINE.join(english_of(page) for page in assessed),
        )
        for page in extraction.pages:
            page.quality = quality.page_verdict(english_of(page))
        indexable_numbers = {page.page_number for page in indexable}
        for page in extraction.pages:
            page.indexable = page.page_number in indexable_numbers
        # The recorded run is replayed rather than reinvented. This module did
        # not run OCR, but OCR *did* run over this document and what it did is
        # on disk; passing None here would tell `decide` that no OCR has ever
        # been attempted, which is a different and false statement.
        ocr_run = (ocr_run_from_journal(journal_record)
                   or _ocr_run_from_dict(document.get("ocr_run")))
        decision = ocr.decide(extraction, quality_assessment,
                              language=assessment,
                              ocr_run=ocr_run)
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    processed = ProcessedDocument(
        document=_corpus_document(data_dir, source),
        extraction=extraction,
        structure=structure,
        language=assessment,
        quality=quality_assessment,
        ocr_decision=decision,
        ocr_run=ocr_run,
        ok=True,
    )
    result.ok = True
    result.now_eligible = bool(processed.eligible_for_indexing)
    # --- the admission bar, and why it applies here and not in process.py ------
    #
    # A document being *promoted* out of quarantine is being admitted on a
    # judgement that was already made once and came out the other way. The
    # quality panel measures word shape, and damaged English keeps the shape of
    # English, so the panel alone is not enough evidence to overturn a
    # quarantine. Word validity is asked as well.
    #
    # It is asked **only of promotions**, which is a deliberate asymmetry rather
    # than an oversight. Applying it to the corpus at large would demote a
    # quarter of the documents already accepted (25th percentile 0.937, 5th
    # 0.770) on a signal that has never been validated against human labels --
    # the blunt-instrument mistake DECISIONS D21 recorded. Documents admitted
    # under the old rules keep their place; documents arriving now clear the bar.
    # That is a transitional policy, and it is written down as one.
    #
    # If a later phase validates this signal against ground truth, the right move
    # is to apply one rule to the whole corpus. Until then this is where the
    # evidence supports drawing it. See DECISIONS D28.
    if result.now_eligible and not result.was_eligible:
        validity = (quality_assessment.signals or {}).get("word_validity") or {}
        result.word_validity = validity.get("rate", 0.0)
        result.word_validity_measurable = bool(validity.get("measurable"))
        if (not result.word_validity_measurable
                or result.word_validity < config.QUALITY_WORD_VALIDITY_MIN):
            result.now_eligible = False
            result.held_back = True
    result.now_quality = quality_assessment.classification
    result.now_language = assessment.content_language
    result.now_action = decision.action
    result.pages_indexable_after = len(indexable)
    result.changed = (
        result.now_eligible != result.was_eligible
        or result.pages_indexable_after != result.pages_indexable_before
        or result.now_quality != result.was_quality
        or result.now_action != result.was_action
    )
    if write and result.changed:
        write_outputs(processed, data_dir)
        if result.held_back:
            # `write_outputs` derives the file's `eligible_for_indexing` from
            # ProcessedDocument, which does not know about the admission bar --
            # so left alone it would record this document as eligible and hand
            # the database exactly what the bar just refused. The verdict is
            # corrected here, with its reason, rather than by not writing at all:
            # the page labelling this pass produced is worth keeping, and a file
            # that silently disagrees with the runner would be worse than either.
            _record_held_back(data_dir, document_id, result)
        result.written = True
    return result


def _record_held_back(data_dir: Path, document_id: str,
                      result: RegateResult) -> None:
    """Mark a written document ineligible because it failed the admission bar."""
    path = (data_dir / "processed" / "indiacode" / document_id
            / config.DOCUMENT_FILENAME)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["eligible_for_indexing"] = False
    payload["held_back"] = {
        "by": "processing.regate admission bar",
        "word_validity": result.word_validity,
        "measurable": result.word_validity_measurable,
        "threshold": config.QUALITY_WORD_VALIDITY_MIN,
        "reason": (
            "every eligibility gate passed, but this document is being promoted "
            "out of quarantine and its word-validity rate does not clear the bar "
            "for that. See processing/regate.py and docs/DECISIONS.md D28."
        ),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


# --- the runner ------------------------------------------------------------------


def _work(args) -> dict:
    data_dir, document_id, write, baseline, record = args
    return regate_document(Path(data_dir), document_id, write=write,
                           baseline_eligible=baseline,
                           journal_record=record).to_dict()


def documents_from_journal(journal: Path) -> dict[str, dict]:
    """Every document Phase 2 recorded, mapped to its final journal record.

    Newest record per document wins. The record carries both the baseline the
    admission bar is measured against and what OCR did, neither of which can be
    read back from ``document.json`` once this module has rewritten it. See
    :func:`regate_document` and :func:`ocr_run_from_journal`.
    """
    last: dict[str, dict] = {}
    with journal.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                record = json.loads(line)
                last[record["document_id"]] = record
    return dict(sorted(last.items()))


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    from concurrent.futures import ProcessPoolExecutor

    parser = argparse.ArgumentParser(
        prog="python -m processing.regate",
        description="Re-decide eligibility from the stored Phase 2 output. "
                    "Opens no PDF and runs no OCR.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--write", action="store_true",
                        help="rewrite document.json/pages.json where the verdict "
                             "moved. Without this nothing is written.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None,
                        help="write the per-document results as JSON")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    journal = args.data_dir / "processing_status.jsonl"
    baseline = documents_from_journal(journal)
    ids = list(baseline)
    if args.limit:
        ids = ids[:args.limit]
    log.info("re-gating %s documents%s", f"{len(ids):,}",
             "" if args.write else " (dry run -- nothing will be written)")

    results: list[dict] = []
    payload = [(str(args.data_dir), d, args.write,
                bool(baseline[d].get("eligible_for_indexing")), baseline[d])
               for d in ids]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(_work, payload, chunksize=16), 1):
            results.append(result)
            if index % 2000 == 0:
                log.info("  %s/%s", f"{index:,}", f"{len(ids):,}")

    promoted = [r for r in results if r["now_eligible"] and not r["was_eligible"]]
    demoted = [r for r in results if r["was_eligible"] and not r["now_eligible"]]
    held = [r for r in results if r.get("held_back")]
    errors = [r for r in results if r["error"]]
    written = [r for r in results if r["written"]]
    eligible_now = sum(1 for r in results if r["now_eligible"])

    log.info("")
    log.info("  documents re-gated     : %s", f"{len(results):,}")
    log.info("  errors                 : %s", f"{len(errors):,}")
    log.info("  promoted               : %s", f"{len(promoted):,}")
    log.info("  demoted                : %s", f"{len(demoted):,}")
    log.info("  held back at the bar   : %s", f"{len(held):,}")
    log.info("  damaged pages excluded : %s over %s documents",
             f"{sum(r['pages_mangled'] for r in results):,}",
             f"{sum(1 for r in results if r['pages_mangled']):,}")
    log.info("  eligible after         : %s", f"{eligible_now:,}")
    log.info("  files rewritten        : %s", f"{len(written):,}")
    if args.report:
        args.report.write_text(json.dumps(results, indent=0), encoding="utf-8")
        log.info("  report                 : %s", args.report)
    return 1 if errors else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
