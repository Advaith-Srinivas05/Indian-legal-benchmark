"""One document, end to end: PDF -> ``data/processed/indiacode/<id>/``.

Two files per document, as the project spec specifies:

``pages.json``
    The page-level record. Page boundaries are preserved because a citation
    names a page, and the text of each page is stored **verbatim** — no
    normalisation, no furniture removal, no re-wrapping. Detected furniture and
    table candidates are recorded alongside it as annotations, so every
    transformation a later stage applies can be checked against what the PDF
    actually produced.

``document.json``
    Identity and provenance (carried forward from the ingestion manifest, never
    re-derived), the extraction result and its classification evidence, and the
    legal structure with page ranges.

The raw PDF is opened read-only and never written to. Output goes only under
``data/processed/``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

from ingestion.utils import atomic_write_text, utcnow_iso

from . import (__version__, config, furniture, language, ocr, ocr_engine,
               quality)
from .backends import PdfBackend
from .errors import ProcessingError
from .extract import extract_document
from .models import CorpusDocument, ProcessedDocument
from .structure import parse_structure

log = logging.getLogger(__name__)

NEWLINE = chr(10)


def output_dir(data_dir: Path, document_id: str) -> Path:
    return Path(data_dir) / config.PROCESSED_SUBDIR / document_id


def process_document(
    document: CorpusDocument,
    data_dir: Path,
    *,
    backend: Optional[PdfBackend] = None,
    detect_tables: bool = True,
    write: bool = True,
    run_ocr: bool = False,
    ocr_reader=None,
) -> ProcessedDocument:
    """Extract, parse and (optionally) write one document.

    Never raises for a per-document problem: a failure is returned as a
    :class:`~processing.models.ProcessedDocument` with ``ok=False`` and the error
    recorded, so a run over 100 documents reports 100 outcomes rather than
    stopping at the first unreadable PDF.

    *run_ocr* is **off by default**, unlike every other stage. OCR costs seconds
    per page rather than milliseconds and needs a Tesseract install, so a caller
    gets it by asking. ``processing.run`` asks (see ``--ocr``/``--no-ocr``, on by
    default), which puts the policy in the command that spends the time and
    keeps this function cheap for everything else that calls it.
    """
    started = time.monotonic()
    result = ProcessedDocument(document=document)
    try:
        if not document.pdf_path.exists():
            raise ProcessingError(
                f"{document.pdf_relpath} is in the manifest but not on disk."
            )
        extraction = extract_document(
            document.pdf_path,
            document.document_id,
            backend=backend,
            detect_tables=detect_tables,
        )
        # Order matters, and language now comes first.
        #
        # Language is judged over *every* page, because deciding a document is
        # bilingual means comparing the pages against each other. Everything
        # after it is judged over the pages that survive: structure, because a
        # Hindi translation of an act contains no English sections to find, and
        # quality, because pooling a Devanagari half into an English document's
        # word-shape signals reports the translation as extraction damage. A
        # bilingual document quarantined for `extraction_quality_bad` is the same
        # document lost for a different stated reason.
        #
        # For a document that is entirely English -- which is almost all of them
        # -- nothing is filtered and this is exactly what it was before.
        #
        # Quality still reads the parsed structure (section numbering is an
        # independent witness to text quality), and the OCR decision still reads
        # both quality and language.
        language_assessment = language.assess_pages(
            extraction.pages, metadata_language=document.language)
        for page in extraction.pages:
            page.language = language.classify_page(page)
            # Marked whatever the document turns out to be: a stray Devanagari
            # line in an otherwise English act is the same finding as a block of
            # them, and the old page-ratio rule let exactly that through.
            page.non_english_lines = language.non_english_lines(page)
        # OCR, page by page, over what extraction could not read. The only
        # stage that adds text rather than measuring it -- and it adds it beside
        # page.text, never over it. See processing.ocr_engine.
        if run_ocr:
            result.ocr_run = _run_ocr(
                extraction, document, language_assessment, reader=ocr_reader)
            if result.ocr_run.accepted:
                language_assessment = language.assess_pages(
                    extraction.pages, metadata_language=document.language)

        indexable = language.indexable_pages(extraction.pages, language_assessment)
        indexable_numbers = {p.page_number for p in indexable}
        for page in extraction.pages:
            page.indexable = page.page_number in indexable_numbers

        # Judged over the indexable pages -- but never over an empty list. A
        # document whose language was never established has none, and measuring
        # its structure and quality against nothing would report an empty
        # document rather than an unreadable one. It is quarantined either way;
        # the difference is whether the record says why.
        assessed = indexable or extraction.pages
        structure = parse_structure(assessed, metadata_title=document.title)
        # Quality is given the English lines explicitly rather than the pages,
        # because it pools page text and a translation left in that pool reads
        # as extraction damage: unfamiliar word shapes, no English function
        # words. A bilingual act would be quarantined as `extraction_quality_bad`
        # -- the same document lost, for a different stated reason.
        quality_assessment = quality.assess(
            assessed, structure=structure,
            text=NEWLINE.join(language.english_line_text(p) for p in assessed),
        )
        decision = ocr.decide(
            extraction, quality_assessment, language=language_assessment)

        result.extraction = extraction
        result.structure = structure
        result.language = language_assessment
        result.quality = quality_assessment
        result.ocr_decision = decision
        result.ok = True
        if write:
            result.output_dir = write_outputs(result, data_dir)
    except ProcessingError as exc:
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        log.warning("%s failed: %s: %s", document.document_id, result.error_type, exc)
    except Exception as exc:                       # a backend can raise anything
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        log.warning("%s failed: %s: %s", document.document_id, result.error_type, exc)
    result.seconds = time.monotonic() - started
    return result


def _run_ocr(extraction, document, language_assessment, *, reader=None):
    """Run the OCR stage and repair what replacing a page's text invalidates.

    Furniture and footnotes were established against the extraction backend's
    text. Where OCR has replaced that text, they describe lines that no longer
    exist, and a stale line index is worse than no label: the structure parser
    skips by index, so it would drop whichever line now happens to sit there.

    The two are not repairable in the same way. Furniture is recurrence across
    page *text* and is simply re-detected over the new readings. Footnotes are
    established from type size and position on the page -- geometry an OCR engine
    does not produce -- so for a re-read page they are cleared and the page says
    so. Losing the amendment-note separation on a scanned page is a real cost,
    and it is smaller than mislabelling arbitrary lines as footnotes.
    """
    run = ocr_engine.ocr_document(
        extraction, document.pdf_path, reader=reader,
        document_language=getattr(language_assessment, "content_language", None),
    )
    if not run.accepted:
        return run

    for page in extraction.pages:
        if page.text_source != "ocr":
            continue
        # The reading changed, so the verdicts taken over the old one are re-taken.
        page.language = language.classify_page(page)
        page.non_english_lines = language.non_english_lines(page)
        if page.footnotes:
            page.footnotes = []
            page.warnings.append(
                "footnote blocks were detected in the original text layer and "
                "dropped when OCR replaced it: they are located by type size and "
                "position, which an OCR engine does not report"
            )

    claimed = [
        {i for block in page.footnotes
         for i in range(block.line_start, block.line_end + 1)}
        for page in extraction.pages
    ]
    detected = furniture.detect(
        [p.selected_text for p in extraction.pages], skip_indices=claimed)
    for page in extraction.pages:
        page.furniture = detected.get(page.page_number, [])
    return run


def write_outputs(result: ProcessedDocument, data_dir: Path) -> Path:
    """Write ``document.json`` and ``pages.json`` atomically."""
    document = result.document
    extraction = result.extraction
    structure = result.structure
    directory = output_dir(data_dir, document.document_id)
    directory.mkdir(parents=True, exist_ok=True)

    pages_payload = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "document_id": document.document_id,
        "page_count": extraction.page_count,
        "note": (
            "'text' is exactly what the extraction backend produced, unmodified. "
            "'furniture' and 'footnotes' mark lines that look like running "
            "headers/footers/page numbers and footnote blocks; they are labelled "
            "here, not removed from 'text'."
        ),
        "pages": [page.to_dict() for page in extraction.pages],
    }
    document_payload = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "processed_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "source": {
            "provider": "India Code",
            **document.to_dict(),
        },
        "extraction": extraction.to_dict(),
        "content_language": result.language.to_dict() if result.language else None,
        "extraction_quality": result.quality.to_dict() if result.quality else None,
        "ocr_decision": result.ocr_decision.to_dict() if result.ocr_decision else None,
        "ocr_run": result.ocr_run.to_dict() if result.ocr_run else None,
        "eligible_for_indexing": result.eligible_for_indexing,
        "structure": structure.to_dict(),
        "artifacts": {
            "pages": config.PAGES_FILENAME,
            # The exact source file, relative to data/. With the sha256 above it
            # pins the *version* of the law this text represents.
            "raw_pdf": document.pdf_relpath,
        },
    }

    atomic_write_text(
        directory / config.PAGES_FILENAME,
        json.dumps(pages_payload, indent=2, ensure_ascii=False),
    )
    atomic_write_text(
        directory / config.DOCUMENT_FILENAME,
        json.dumps(document_payload, indent=2, ensure_ascii=False),
    )
    return directory
