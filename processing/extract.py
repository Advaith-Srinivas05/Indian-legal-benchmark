"""PDF -> pages, with every classification backed by recorded evidence.

This stage answers two independent questions, and keeping them independent is
the whole point of the design:

``pdf_type``
    Where did these pages come from? ``text_based`` (typeset, real fonts),
    ``scanned`` (images of pages), or ``mixed``.

``text_extraction_status``
    Did usable text come out? ``ok``, ``partial``, or ``requires_ocr``.

Collapsing the two would hide the case this corpus is full of: a *scanned*
gazette that carries an OCR text layer. Its extraction status is ``ok`` — text
came out — while its type is ``scanned`` and its text may be nonsense. Reporting
both, plus a separate OCR-quality suspicion flag, is what lets a later phase
decide whether to trust it, re-OCR it, or exclude it.

Nothing is discarded here. Rejected table candidates keep their rejection
reason, furniture is labelled rather than stripped, and page text is stored
verbatim.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from . import (
    config,
    footnotes,
    furniture,
    orientation as orientation_module,
    quality,
    tables as tables_module,
)
from .backends import PdfBackend, RawDocument, RawPage, RawTable, default_backend
from .errors import PDFOpenError
from .models import ExtractedDocument, ExtractedTable, PageText
from .textutils import alpha_count, is_blank, quality_signals, split_lines, union_area


def extract_document(
    path: Path,
    document_id: str,
    *,
    backend: Optional[PdfBackend] = None,
    detect_tables: bool = True,
) -> ExtractedDocument:
    """Read one PDF into an :class:`ExtractedDocument`.

    Raises :class:`~processing.errors.PDFOpenError` when the file cannot be read
    at all; a page-level problem is recorded as a warning instead, so one bad
    page never costs the rest of the document.
    """
    backend = backend or default_backend()
    started = time.monotonic()
    raw = backend.read(Path(path), detect_tables=detect_tables)
    if raw.page_count == 0 or not raw.pages:
        raise PDFOpenError(f"{path} contains no pages.")

    pages = [_build_page(raw_page) for raw_page in raw.pages]
    _attach_orientation(pages)
    # Footnotes first: position plus type size is stronger evidence than
    # repetition, so a running-foot rule must not get to claim those lines
    # before the footnote detector has seen them.
    _attach_footnotes(pages)
    _attach_furniture(pages)
    toc_pages = _detect_contents_pages(pages)
    tables_module.classify(pages, toc_pages=toc_pages)

    document = ExtractedDocument(
        document_id=document_id,
        page_count=len(pages),
        pages=pages,
        pdf_type="",
        text_extraction_status="",
        backend=raw.backend,
        pdf_metadata=raw.metadata,
        is_encrypted=raw.is_encrypted,
        toc_pages=sorted(toc_pages),
        warnings=list(raw.warnings),
    )
    _classify_document(document)
    document.extraction_seconds = time.monotonic() - started
    return document


# --- Page level -----------------------------------------------------------------


def _build_page(raw: RawPage) -> PageText:
    text = raw.text
    lines = split_lines(text)
    alpha = alpha_count(text)
    page_area = max(0.0, raw.width) * max(0.0, raw.height)
    covered = union_area(raw.image_bboxes, (0.0, 0.0, raw.width, raw.height))
    ratio = covered / page_area if page_area > 0 else 0.0

    signals = quality_signals(text)
    has_text = alpha >= config.MIN_PAGE_ALPHA_CHARS
    is_image_backed = ratio >= config.IMAGE_BACKED_AREA_RATIO
    # Text drawn as vector outlines: no text layer, nothing to OCR, and a
    # thousand paths where the words should be. Distinguished from a genuinely
    # blank page by the path count alone.
    is_vector_outlined = (
        not has_text
        and not is_image_backed
        and raw.drawing_path_count >= config.VECTOR_OUTLINE_MIN_PATHS
    )
    page = PageText(
        page_number=raw.page_number,
        text=text,
        char_count=len(text),
        alpha_char_count=alpha,
        line_count=len(lines),
        width=raw.width,
        height=raw.height,
        image_count=len(raw.image_bboxes),
        image_area_ratio=ratio,
        is_image_backed=is_image_backed,
        drawing_path_count=raw.drawing_path_count,
        is_vector_outlined=is_vector_outlined,
        has_text=has_text,
        is_empty=is_blank(text),
        text_quality_suspect=has_text and quality.page_is_suspect(text),
        quality_signals=signals,
        tables=[_build_table(raw.page_number, t) for t in raw.tables],
        line_metrics=list(raw.line_metrics),
        orientation=orientation_module.assess_page(
            raw.page_number, raw.rotation, raw.line_directions),
        warnings=list(raw.warnings),
    )
    if page.is_sideways:
        page.warnings.append(
            f"{page.orientation.sideways_line_ratio:.0%} of this page's lines run "
            f"vertically (declared /Rotate {page.orientation.declared_rotation}); "
            "the page is sideways and must be rotated before OCR can read it"
        )
    if is_vector_outlined:
        page.warnings.append(
            f"text appears to be drawn as vector outlines ("
            f"{raw.drawing_path_count} paths, no text layer, no page image); "
            "this page must be rasterised before OCR can read it"
        )
    elif not page.has_text and not page.is_image_backed:
        # Neither typeset text nor a scan nor outlined text: a genuinely blank
        # page, or a font this backend cannot decode. Worth naming rather than
        # silently counting as "no text".
        page.warnings.append(
            "no usable text and no page-covering image; "
            f"{page.alpha_char_count} letters, {page.image_count} image(s), "
            f"{raw.drawing_path_count} drawing path(s)"
        )
    return page


def _build_table(page_number: int, raw: RawTable) -> ExtractedTable:
    """Wrap a backend candidate. Retention is decided in :mod:`processing.tables`.

    The decision lives there because the strongest evidence against a candidate
    is contextual — a grid on an arrangement-of-sections page is not a table —
    and that context is not available until the page has been read.
    """
    rows = raw.rows
    row_count = raw.row_count
    col_count = raw.col_count
    cells = row_count * col_count
    filled = sum(1 for row in rows for cell in row if cell and str(cell).strip())
    return ExtractedTable(
        page_number=page_number,
        bbox=raw.bbox,
        row_count=row_count,
        col_count=col_count,
        filled_ratio=filled / cells if cells else 0.0,
        rows=rows,
        detector="pymupdf.find_tables",
        retained=False,
        rejected_reason=None,
    )


def _attach_furniture(pages: list[PageText]) -> None:
    claimed = [
        {i for block in page.footnotes
         for i in range(block.line_start, block.line_end + 1)}
        for page in pages
    ]
    detected = furniture.detect([p.text for p in pages], skip_indices=claimed)
    for page in pages:
        page.furniture = detected.get(page.page_number, [])


def _attach_orientation(pages: list[PageText]) -> None:
    """Settle the cross-page part of orientation: which way is forward here.

    Per-page orientation is decided in :func:`_build_page`; only the
    direction-consistency check needs to see the whole document, because "this
    page runs backwards" is a statement about the other pages.
    """
    flagged = orientation_module.flag_direction_inconsistency(
        [p.orientation for p in pages if p.orientation is not None]
    )
    inconsistent = {o.page_number for o in flagged if o.direction_inconsistent}
    for page in pages:
        if page.page_number in inconsistent:
            page.warnings.append(
                "this page's text runs in the opposite direction to the rest of "
                "the document, which on this corpus means a broken character map "
                "(the extracted characters are not the ones on the page)"
            )


def _attach_footnotes(pages: list[PageText]) -> None:
    detected = footnotes.detect(pages)
    for page in pages:
        page.footnotes = detected.get(page.page_number, [])


def _detect_contents_pages(pages: list[PageText]) -> set[int]:
    """Which pages are an arrangement-of-sections / contents listing.

    Established here, rather than only in the structure parser, because table
    retention depends on it: the commonest false-positive table in the first
    benchmark was a two-column contents listing.
    """
    from .structure import build_line_stream, detect_toc_pages    # noqa: PLC0415

    return detect_toc_pages(build_line_stream(pages))


# --- Document level -------------------------------------------------------------


def _classify_document(document: ExtractedDocument) -> None:
    pages = document.pages
    total = len(pages)
    image_backed = sum(1 for p in pages if p.is_image_backed)
    outlined = sum(1 for p in pages if p.is_vector_outlined)
    with_text = sum(1 for p in pages if p.has_text)
    empty = sum(1 for p in pages if p.is_empty)
    suspect = sum(1 for p in pages if p.text_quality_suspect)

    image_ratio = image_backed / total
    outlined_ratio = outlined / total
    text_ratio = with_text / total

    # Vector-outlined is tested before the image ratios, because on those pages
    # the image ratio is zero and the old rules would call the document
    # ``text_based`` — which is exactly the mistake this class exists to stop.
    if outlined_ratio >= config.VECTOR_OUTLINE_PAGE_RATIO:
        pdf_type = "vector_outlined"
    elif image_ratio >= config.SCANNED_PAGE_RATIO:
        pdf_type = "scanned"
    elif image_ratio <= config.TEXT_BASED_PAGE_RATIO and outlined_ratio <= config.TEXT_BASED_PAGE_RATIO:
        pdf_type = "text_based"
    else:
        pdf_type = "mixed"

    if text_ratio >= config.TEXT_OK_PAGE_RATIO:
        status = "ok"
    elif text_ratio <= config.REQUIRES_OCR_PAGE_RATIO:
        status = "requires_ocr"
    else:
        status = "partial"

    suspect_ratio = suspect / with_text if with_text else 0.0
    quality_suspect = suspect_ratio >= config.QUALITY_SUSPECT_PAGE_RATIO

    document.pdf_type = pdf_type
    document.text_extraction_status = status
    document.orientation = orientation_module.summarise(
        [p.orientation for p in pages if p.orientation is not None], total
    )
    document.classification_evidence = {
        "page_count": total,
        "image_backed_pages": image_backed,
        "image_backed_ratio": round(image_ratio, 4),
        "vector_outlined_pages": outlined,
        "vector_outlined_ratio": round(outlined_ratio, 4),
        "pages_with_text": with_text,
        "pages_with_text_ratio": round(text_ratio, 4),
        "empty_pages": empty,
        "text_quality_suspect_pages": suspect,
        "text_quality_suspect_ratio": round(suspect_ratio, 4),
        "text_quality_suspect": quality_suspect,
        "thresholds": {
            "min_page_alpha_chars": config.MIN_PAGE_ALPHA_CHARS,
            "image_backed_area_ratio": config.IMAGE_BACKED_AREA_RATIO,
            "scanned_page_ratio": config.SCANNED_PAGE_RATIO,
            "text_based_page_ratio": config.TEXT_BASED_PAGE_RATIO,
            "text_ok_page_ratio": config.TEXT_OK_PAGE_RATIO,
            "requires_ocr_page_ratio": config.REQUIRES_OCR_PAGE_RATIO,
            "vector_outline_min_paths": config.VECTOR_OUTLINE_MIN_PATHS,
            "vector_outline_page_ratio": config.VECTOR_OUTLINE_PAGE_RATIO,
        },
    }

    sideways = document.orientation.get("sideways_pages", 0)
    inconsistent = document.orientation.get("direction_inconsistent_pages", 0)
    if sideways:
        document.warnings.append(
            f"{sideways}/{total} pages are set sideways; their text layer is "
            "unreliable and they must be rotated before OCR"
        )
    if inconsistent:
        document.warnings.append(
            f"{inconsistent}/{total} pages extract with their writing direction "
            "reversed relative to the rest of the document, which indicates a "
            "broken character map on those pages"
        )
    if outlined:
        document.warnings.append(
            f"{outlined}/{total} pages draw their text as vector outlines; those "
            "pages have no text layer and no image, and must be rasterised "
            "before OCR can read them"
        )
    if status == "requires_ocr":
        document.warnings.append(
            f"no usable text layer ({with_text}/{total} pages had text); requires OCR"
        )
    elif status == "partial":
        document.warnings.append(
            f"text layer covers only {with_text}/{total} pages; the rest require OCR"
        )
    if quality_suspect:
        document.warnings.append(
            f"text layer quality is suspect on {suspect}/{with_text} text pages "
            "(likely OCR noise); do not cite without review"
        )
    if empty:
        document.warnings.append(f"{empty}/{total} pages extracted as empty")
