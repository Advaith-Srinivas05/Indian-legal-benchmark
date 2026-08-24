"""Lightweight structural validation of downloaded PDFs.

We intentionally avoid a heavyweight PDF parser at the download stage. The goal
here is only to reject obvious non-PDFs (e.g. an HTML error page served with a
200) and clearly truncated files, so the corpus does not fill with junk.

Deeper validation / text-extractability checks belong to the later *processing*
phase and can use a real PDF library (pypdf, pdfminer, ...) without changing
this module.
"""

from __future__ import annotations

from pathlib import Path

from . import config
from .errors import CorruptPDFError, NotPDFError

_PDF_MAGIC = b"%PDF-"


def validate_pdf(path: Path, *, content_type: str = "", source_url: str = "") -> None:
    """Validate that *path* contains a plausible PDF.

    Raises :class:`NotPDFError` if it is clearly not a PDF (e.g. an HTML page),
    or :class:`CorruptPDFError` if it looks truncated.
    """
    size = path.stat().st_size
    if size < config.MIN_PDF_BYTES:
        raise CorruptPDFError(
            f"Downloaded file from {source_url or path} is only {size} bytes; "
            "not a valid PDF."
        )

    with open(path, "rb") as handle:
        header = handle.read(8)
        # Read the tail to look for the end-of-file marker.
        tail_len = min(2048, size)
        handle.seek(-tail_len, 2)
        tail = handle.read(tail_len)

    if not header.startswith(_PDF_MAGIC):
        snippet = header[:5]
        raise NotPDFError(
            f"Response from {source_url or path} is not a PDF "
            f"(content-type={content_type!r}, starts with {snippet!r})."
        )

    if b"%%EOF" not in tail:
        # Header is fine but the EOF marker is missing: most likely truncated.
        raise CorruptPDFError(
            f"PDF from {source_url or path} is missing its %%EOF marker; "
            "it may be truncated."
        )
