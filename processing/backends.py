"""PDF reading backends — the only place a PDF library is imported.

Everything above this module works on :class:`RawPage` objects, which is what
makes the rest of the package testable without a PDF and swappable if a
different library turns out to extract India Code's older scans better.

A backend's job is deliberately small: hand back, per page, exactly what the
library saw — text, page size, image placements, table candidates — and make no
judgement about it. All classification (``text_based``/``scanned``/``mixed``,
table quality, OCR-layer suspicion) happens in :mod:`processing.extract` on this
neutral representation.

The default backend is PyMuPDF. It was chosen over pdfminer/pdfplumber and pypdf
for three reasons that matter at 19,802 documents / 39 GB: it is roughly an
order of magnitude faster, it reports image placement per page (which is how a
scanned page is recognised without OCRing it), and its reading order on India
Code's single-column typesetting is good. Its table finder is *not* trusted —
see :mod:`processing.extract` for the quality filter applied to what it returns.

Note on licensing: PyMuPDF is AGPL-3.0. That is fine for an academic project but
is the reason the backend is isolated behind this interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Protocol

from .errors import BackendUnavailableError, PDFOpenError

#: Bounding box: ``(x0, y0, x1, y1)`` in PDF points, origin top-left.
BBox = tuple[float, float, float, float]


@dataclass
class RawTable:
    """A table candidate exactly as the backend reported it. Unfiltered."""

    bbox: BBox
    rows: list[list[Optional[str]]] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def col_count(self) -> int:
        return max((len(r) for r in self.rows), default=0)


@dataclass
class LineMetric:
    """Where one line sits on the page, and how big it is set.

    Indexed by position in ``RawPage.text.split("\\n")``. Footnotes are told
    from body text by exactly these two facts — they sit at the foot of the page
    and are set smaller — so the geometry has to survive out of the backend.
    ``RawPage.metrics_aligned`` records whether the correspondence with the text
    lines could be established; when it could not, the metrics are dropped
    rather than guessed at.
    """

    line_index: int
    y0: float
    y1: float
    size: float


@dataclass
class RawPage:
    """One page as the backend saw it, before any interpretation."""

    page_number: int                   # 1-based
    text: str
    width: float
    height: float
    image_bboxes: list[BBox] = field(default_factory=list)
    tables: list[RawTable] = field(default_factory=list)
    #: Straight-line and rectangle vector paths drawn on the page. Large counts
    #: with no text and no image mean the glyphs were converted to outlines.
    drawing_path_count: int = 0
    line_metrics: list[LineMetric] = field(default_factory=list)
    metrics_aligned: bool = False
    #: The page's ``/Rotate`` value. A viewer applies it before displaying the
    #: page, so it has to be known before a writing direction can be read as
    #: "sideways on screen" — see :mod:`processing.orientation`.
    rotation: int = 0
    #: Writing direction ``(dx, dy)`` per extracted line, in *file* coordinates
    #: and before ``/Rotate`` is applied. Unlike :attr:`line_metrics` this does
    #: not need to align with the plain-text line list: orientation is judged
    #: from the distribution over the page, not per line, so it survives the
    #: cases where the two extractions disagree about line breaks.
    line_directions: list[tuple[float, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class RawDocument:
    """A whole PDF as the backend saw it."""

    page_count: int
    pages: list[RawPage]
    metadata: dict = field(default_factory=dict)
    is_encrypted: bool = False
    backend: str = ""
    warnings: list[str] = field(default_factory=list)


class PdfBackend(Protocol):
    """What :mod:`processing.extract` needs from a PDF library."""

    name: str

    def read(self, path: Path, *, detect_tables: bool = True) -> RawDocument:
        """Read every page of *path*. Raises :class:`PDFOpenError` on failure."""
        ...


class PyMuPDFBackend:
    """PyMuPDF (``pymupdf``/``fitz``) implementation of :class:`PdfBackend`."""

    def __init__(self) -> None:
        self._pymupdf = _import_pymupdf()
        self.name = f"pymupdf {getattr(self._pymupdf, '__version__', 'unknown')}"

    def read(self, path: Path, *, detect_tables: bool = True) -> RawDocument:
        pymupdf = self._pymupdf
        path = Path(path)
        try:
            document = pymupdf.open(path)
        except Exception as exc:                        # library raises many types
            raise PDFOpenError(f"Cannot open {path}: {exc}") from exc

        warnings: list[str] = []
        try:
            is_encrypted = bool(document.needs_pass)
            if is_encrypted:
                # An encrypted PDF yields no pages; say so instead of silently
                # reporting a zero-page document.
                raise PDFOpenError(f"{path} is password protected.")
            metadata = _clean_metadata(document.metadata or {})
            pages = list(self._read_pages(document, detect_tables, warnings))
            return RawDocument(
                page_count=document.page_count,
                pages=pages,
                metadata=metadata,
                is_encrypted=False,
                backend=self.name,
                warnings=warnings,
            )
        finally:
            document.close()

    def _read_pages(self, document, detect_tables: bool, warnings: list[str]) -> Iterator[RawPage]:
        for index in range(document.page_count):
            number = index + 1
            page_warnings: list[str] = []
            try:
                page = document.load_page(index)
            except Exception as exc:
                # One broken page must not cost us the other 200. Emit a
                # placeholder so page numbering — and therefore every page
                # citation after it — stays correct.
                warnings.append(f"page {number}: could not be loaded ({exc})")
                yield RawPage(
                    page_number=number, text="", width=0.0, height=0.0,
                    warnings=[f"load failed: {exc}"],
                )
                continue

            rect = page.rect
            content = self._read_content(page, page_warnings)
            text, image_bboxes, metrics, aligned, directions = content
            drawings = self._ruled_path_count(page)
            tables = (
                self._read_tables(page, drawings, page_warnings) if detect_tables else []
            )
            yield RawPage(
                page_number=number,
                text=text,
                width=float(rect.width),
                height=float(rect.height),
                image_bboxes=image_bboxes,
                tables=tables,
                drawing_path_count=drawings,
                line_metrics=metrics,
                metrics_aligned=aligned,
                rotation=int(getattr(page, "rotation", 0) or 0),
                line_directions=directions,
                warnings=page_warnings,
            )

    @staticmethod
    def _read_content(
        page, page_warnings: list[str]
    ) -> tuple[str, list[BBox], list[LineMetric], bool, list[tuple[float, float]]]:
        """Text, image placements and per-line geometry, from one dict pass.

        Image *placements* (where a picture sits on the page) are what identify a
        scan, and they are not the same as ``get_images()``, which lists the
        file's image resources without saying whether or how large they were
        drawn.

        The line geometry is matched to the plain-text lines by comparing the
        strings, not by assuming the two extractions agree. PyMuPDF builds both
        from the same structure so they normally do, but a citation built on a
        mis-aligned index would point at the wrong line — so a mismatch drops the
        metrics and says so through ``metrics_aligned`` rather than shipping a
        guess.
        """
        try:
            content = page.get_text("dict")
        except Exception as exc:
            page_warnings.append(f"text extraction failed: {exc}")
            return "", [], [], False, []

        image_bboxes: list[BBox] = []
        dict_lines: list[tuple[str, float, float, float]] = []
        directions: list[tuple[float, float]] = []
        for block in content.get("blocks", []):
            if block.get("type") == 1:
                if block.get("bbox"):
                    x0, y0, x1, y1 = block["bbox"]
                    image_bboxes.append((float(x0), float(y0), float(x1), float(y1)))
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                text_of_line = "".join(span.get("text", "") for span in spans)
                bbox = line.get("bbox") or (0.0, 0.0, 0.0, 0.0)
                size = max((float(span.get("size", 0.0)) for span in spans), default=0.0)
                dict_lines.append((text_of_line, float(bbox[1]), float(bbox[3]), size))
                if text_of_line.strip():
                    # Blank lines carry no writing direction worth counting; a
                    # page of empty spans would otherwise vote on its own
                    # orientation.
                    dx, dy = line.get("dir") or (1.0, 0.0)
                    directions.append((float(dx), float(dy)))

        try:
            text = page.get_text("text")
        except Exception as exc:
            page_warnings.append(f"text extraction failed: {exc}")
            return "", image_bboxes, [], False, directions

        metrics, aligned = _align_metrics(text, dict_lines)
        if dict_lines and not aligned:
            page_warnings.append(
                "line geometry could not be matched to the extracted text; "
                "footnote detection is skipped on this page"
            )
        return text, image_bboxes, metrics, aligned, directions

    @staticmethod
    def _ruled_path_count(page) -> int:
        """How many straight-line/rectangle vector paths the page draws.

        Two consumers. It identifies **vector-outlined** pages — a thousand
        paths with no text and no image means the glyphs were converted to
        outlines — and it gates table detection.

        ``get_drawings()`` costs about 1 ms per page against 40–600 ms for
        ``find_tables()``, so this is the gate that makes table detection
        affordable over a 39 GB corpus. It is also a *quality* gate: a page with
        no ruled lines can only produce a table through the detector's
        text-alignment fallback, and that fallback is what turns indented
        sub-clauses into two-column "tables".

        The cost is that whitespace-aligned tables — common in India Code's
        older scans, which have no vector content at all — are not detected. The
        benchmark reports that rather than papering over it.
        """
        try:
            drawings = page.get_drawings()
        except Exception:
            return 0
        count = 0
        for path in drawings:
            for item in path.get("items", []):
                if item and item[0] in ("l", "re"):
                    count += 1
        return count

    @classmethod
    def _read_tables(cls, page, ruled_paths: int, page_warnings: list[str]) -> list[RawTable]:
        from . import config                              # noqa: PLC0415

        if ruled_paths < config.TABLE_MIN_RULED_PATHS:
            return []
        try:
            found = page.find_tables()
        except Exception as exc:
            page_warnings.append(f"table detection failed: {exc}")
            return []
        tables: list[RawTable] = []
        for table in getattr(found, "tables", []):
            try:
                rows = [list(row) for row in table.extract()]
            except Exception as exc:
                page_warnings.append(f"table extraction failed: {exc}")
                continue
            bbox = tuple(float(v) for v in table.bbox)     # type: ignore[assignment]
            tables.append(RawTable(bbox=bbox, rows=rows))
        return tables


def _align_metrics(
    text: str, dict_lines: list[tuple[str, float, float, float]]
) -> tuple[list[LineMetric], bool]:
    """Match per-line geometry to the plain-text line list, or give up.

    ``get_text("text")`` ends with a newline, so its split produces one more
    (empty) entry than there are real lines; that trailing entry is the only
    difference tolerated. Any other divergence means the two extractions
    disagree about line breaks and the indices cannot be trusted.
    """
    if not dict_lines:
        return [], False
    text_lines = text.split("\n")
    if len(text_lines) < len(dict_lines):
        return [], False
    for index, (line_text, _, _, _) in enumerate(dict_lines):
        if text_lines[index] != line_text:
            return [], False
    if any(line.strip() for line in text_lines[len(dict_lines):]):
        return [], False
    metrics = [
        LineMetric(line_index=index, y0=y0, y1=y1, size=size)
        for index, (_, y0, y1, size) in enumerate(dict_lines)
    ]
    return metrics, True


def _clean_metadata(metadata: dict) -> dict:
    """Keep the PDF metadata fields that help explain an extraction result.

    ``producer``/``creator`` in particular are the strongest available hint at
    *how* a document was made — India Code's scanned gazettes and its
    Word-exported acts are told apart by this field alone.
    """
    keys = ("format", "title", "author", "subject", "creator", "producer",
            "creationDate", "modDate")
    return {k: metadata.get(k) or None for k in keys}


def _import_pymupdf():
    try:
        import pymupdf                                   # noqa: PLC0415
    except ImportError:                                  # pragma: no cover
        try:
            import fitz as pymupdf                       # noqa: PLC0415
        except ImportError as exc:
            raise BackendUnavailableError(
                "PyMuPDF is required for PDF extraction but is not installed. "
                "Install it with:  pip install -r requirements.txt"
            ) from exc
    return pymupdf


def default_backend() -> PdfBackend:
    """The backend used unless a caller supplies its own (tests do)."""
    return PyMuPDFBackend()
