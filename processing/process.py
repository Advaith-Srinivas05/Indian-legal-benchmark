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

from . import __version__, config, language, ocr, quality
from .backends import PdfBackend
from .errors import ProcessingError
from .extract import extract_document
from .models import CorpusDocument, ProcessedDocument
from .structure import parse_structure

log = logging.getLogger(__name__)


def output_dir(data_dir: Path, document_id: str) -> Path:
    return Path(data_dir) / config.PROCESSED_SUBDIR / document_id


def process_document(
    document: CorpusDocument,
    data_dir: Path,
    *,
    backend: Optional[PdfBackend] = None,
    detect_tables: bool = True,
    write: bool = True,
) -> ProcessedDocument:
    """Extract, parse and (optionally) write one document.

    Never raises for a per-document problem: a failure is returned as a
    :class:`~processing.models.ProcessedDocument` with ``ok=False`` and the error
    recorded, so a run over 100 documents reports 100 outcomes rather than
    stopping at the first unreadable PDF.
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
        structure = parse_structure(extraction.pages, metadata_title=document.title)
        # Order matters: quality reads the parsed structure (section numbering is
        # an independent witness to text quality), and the OCR decision reads
        # both quality and language.
        language_assessment = language.assess_pages(
            extraction.pages, metadata_language=document.language)
        quality_assessment = quality.assess(extraction.pages, structure=structure)
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
