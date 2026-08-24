"""Builders for the small PDFs the processing tests run against.

The processing tests need real PDFs — a text page, a page that is an image of a
page, a page with a ruled table — but committing binary fixtures would make them
unreadable and unauditable. These builders generate them at test time from a few
lines of declarative input instead, so what a fixture *contains* is visible in
the test that uses it.

Everything here is offline and deterministic. PyMuPDF is used to write as well
as to read; the tests exercise our extraction and classification logic, not the
library's fidelity to itself.

**Keep fixture text ASCII.** These builders use the base-14 Helvetica, which has
no em dash: writing one produces ``?`` on extraction, and embedding a system
font instead would make the tests depend on which fonts a machine happens to
have. That is not a limitation worth working around here, because the structure
parser does not read PDFs — it reads
:class:`~processing.models.PageText`. Section-heading punctuation, em dashes and
every other typographic detail are therefore tested directly on real strings in
``test_processing_structure.py``, and these PDFs test what only a PDF can test:
page boundaries, image coverage, ruled tables, and failure handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import pymupdf

A4 = (595.0, 842.0)
_FONT = "helv"
_MARGIN = 40.0


def _textbox(page, text: str, *, fontsize: float = 9.0) -> None:
    rect = pymupdf.Rect(_MARGIN, _MARGIN, page.rect.width - _MARGIN,
                        page.rect.height - _MARGIN)
    page.insert_textbox(rect, text, fontsize=fontsize, fontname=_FONT)


def _page_image(width: float, height: float, shade: int = 235) -> pymupdf.Pixmap:
    """A flat grey rectangle standing in for a scan of a page."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, int(width), int(height)))
    pix.set_rect(pix.irect, (shade, shade, shade))
    return pix


def text_pdf(path: Path, pages: Sequence[str], *, fontsize: float = 9.0) -> Path:
    """A normal typeset PDF: one page per string, no images."""
    document = pymupdf.open()
    for body in pages:
        page = document.new_page(width=A4[0], height=A4[1])
        if body:
            _textbox(page, body, fontsize=fontsize)
    document.save(path)
    document.close()
    return Path(path)


def scanned_pdf(
    path: Path, page_count: int = 3, *, ocr_text: Optional[Sequence[str]] = None
) -> Path:
    """Pages that are a full-page image, optionally with an OCR text layer.

    ``ocr_text=None`` gives an image-only scan — the ``requires_ocr`` case.
    Supplying text gives the more dangerous case this corpus is full of: a
    scanned page that *does* yield text, so nothing looks broken.
    """
    document = pymupdf.open()
    pixmap = _page_image(*A4)
    for index in range(page_count):
        page = document.new_page(width=A4[0], height=A4[1])
        page.insert_image(page.rect, pixmap=pixmap)
        if ocr_text:
            _textbox(page, ocr_text[index % len(ocr_text)])
    document.save(path)
    document.close()
    return Path(path)


def mixed_pdf(path: Path, *, text_pages: Sequence[str], image_pages: int = 2) -> Path:
    """Typeset pages followed by scanned pages, in one file."""
    document = pymupdf.open()
    for body in text_pages:
        page = document.new_page(width=A4[0], height=A4[1])
        _textbox(page, body)
    pixmap = _page_image(*A4)
    for _ in range(image_pages):
        page = document.new_page(width=A4[0], height=A4[1])
        page.insert_image(page.rect, pixmap=pixmap)
    document.save(path)
    document.close()
    return Path(path)


def table_pdf(path: Path, rows: Sequence[Sequence[str]], *, heading: str = "") -> Path:
    """A page with a ruled grid, which is what a table detector keys on."""
    document = pymupdf.open()
    page = document.new_page(width=A4[0], height=A4[1])
    if heading:
        page.insert_text((_MARGIN, _MARGIN), heading, fontsize=10, fontname=_FONT)

    top = _MARGIN + 30.0
    row_height = 22.0
    columns = max(len(row) for row in rows)
    width = (page.rect.width - 2 * _MARGIN) / columns

    for index in range(len(rows) + 1):
        y = top + index * row_height
        page.draw_line(pymupdf.Point(_MARGIN, y),
                       pymupdf.Point(_MARGIN + columns * width, y))
    for index in range(columns + 1):
        x = _MARGIN + index * width
        page.draw_line(pymupdf.Point(x, top),
                       pymupdf.Point(x, top + len(rows) * row_height))
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            page.insert_text(
                (_MARGIN + column_index * width + 4, top + row_index * row_height + 15),
                cell, fontsize=9, fontname=_FONT,
            )
    document.save(path)
    document.close()
    return Path(path)


def vector_outlined_pdf(
    path: Path, page_count: int = 3, *, text_pages: Sequence[str] = ()
) -> Path:
    """Pages whose "text" is drawn as vector paths — no text layer, no image.

    Reproduces the shape of the UP Stamp Rules, 1942, where a converter replaced
    every glyph with its outline. Each page draws about a thousand short strokes
    where the words should be, which is exactly what the real document does. It
    matters that these pages carry *no* image: that is why the old classifier
    called the document ``text_based`` and reported two thirds of its pages as
    empty.

    *text_pages* prepends ordinary typeset pages, for building the mixed case.
    """
    document = pymupdf.open()
    for body in text_pages:
        page = document.new_page(width=A4[0], height=A4[1])
        _textbox(page, body)
    for _ in range(page_count):
        page = document.new_page(width=A4[0], height=A4[1])
        y = 60.0
        for _row in range(20):
            x = 50.0
            for _glyph in range(30):
                page.draw_line(pymupdf.Point(x, y), pymupdf.Point(x + 6, y))
                page.draw_line(pymupdf.Point(x, y), pymupdf.Point(x, y - 8))
                x += 9.0
            y += 18.0
    document.save(path)
    document.close()
    return Path(path)


def footnoted_pdf(
    path: Path,
    body: str,
    footnote: str,
    *,
    page_count: int = 3,
    body_size: float = 11.0,
    footnote_size: float = 8.0,
) -> Path:
    """Pages of body text with a smaller-set footnote block at the foot.

    Position and size are the two facts footnote detection rests on, so the
    fixture has to be a real page layout rather than a marker in a string.
    """
    document = pymupdf.open()
    for number in range(1, page_count + 1):
        page = document.new_page(width=A4[0], height=A4[1])
        # The body varies per page. Identical body text on every page would be
        # caught by running-head detection, which is correct behaviour for a
        # running head and wrong for this fixture.
        page.insert_textbox(
            pymupdf.Rect(_MARGIN, _MARGIN, page.rect.width - _MARGIN, 600),
            body.replace("{n}", str(number)), fontsize=body_size, fontname=_FONT,
        )
        page.insert_textbox(
            pymupdf.Rect(_MARGIN, 640, page.rect.width - _MARGIN,
                         page.rect.height - _MARGIN),
            footnote.replace("{n}", str(number)),
            fontsize=footnote_size, fontname=_FONT,
        )
    document.save(path)
    document.close()
    return Path(path)


def blank_pdf(path: Path, page_count: int = 2) -> Path:
    """Pages with nothing on them at all."""
    document = pymupdf.open()
    for _ in range(page_count):
        document.new_page(width=A4[0], height=A4[1])
    document.save(path)
    document.close()
    return Path(path)


def corrupt_pdf(path: Path) -> Path:
    """A file that is not a PDF, to exercise the open-failure path."""
    Path(path).write_bytes(b"%PDF-1.4\nthis is not a pdf\n")
    return Path(path)


def with_running_head(pages: Iterable[str], head: str, *, numbered: bool = True) -> list[str]:
    """Prefix each page with a running head and a page number.

    Used to build documents whose furniture the detector is supposed to find.
    """
    out = []
    for index, body in enumerate(pages, start=1):
        prefix = f"{head}\n{index}\n" if numbered else f"{head}\n"
        out.append(prefix + body)
    return out
