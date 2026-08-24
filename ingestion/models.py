"""Typed data structures passed between the ingestion stages.

These deliberately model *India Code's* structure (handles, bitstreams, Dublin
Core) at the edge, and a normalised, source-agnostic download result at the
core, so that a future non-India-Code source can reuse the storage/manifest
layers without change.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional


class Outcome(str, enum.Enum):
    """What happened to a single requested document during a run."""

    NEW = "new"                # first time we have seen this document
    UNCHANGED = "unchanged"    # already stored, hash identical
    UPDATED = "updated"        # already stored, PDF content changed
    DRY_RUN = "dry_run"        # would have been fetched, but --dry-run
    FAILED = "failed"          # an error occurred (see .message)


@dataclass
class BitstreamRef:
    """A single downloadable file attached to an India Code item."""

    url: str
    filename: str
    sequence: Optional[int] = None
    is_primary: bool = False
    #: Visible text of the file's link (on India Code: the title in the file's
    #: own language) and any neighbouring explicit language label. Kept because
    #: they are the raw evidence behind :attr:`language`.
    link_text: Optional[str] = None
    language_label: Optional[str] = None
    #: Language established from India Code metadata (``"en"``/``"hi"``/...).
    #: ``None`` means *undetermined* — such a file is never downloaded.
    language: Optional[str] = None
    #: Which rule established :attr:`language` (see :mod:`ingestion.language`).
    language_source: Optional[str] = None
    #: Human-readable justification for :attr:`language`.
    language_evidence: Optional[str] = None
    downloaded: bool = False


@dataclass
class SubordinateRow:
    """One row of an act page's Rules/Regulations table.

    India Code does not publish subordinate legislation as DSpace items: it is
    listed on the parent act's page with explicit ``Files(Eng)`` and
    ``Files(Hindi)`` columns, and served from ``/ViewFileUploaded?…``.
    """

    document_type: str                 # "rule" | "regulation"
    tab_label: str                     # "Rules" | "Regulations"
    year: Optional[str] = None
    description: Optional[str] = None
    hindi_description: Optional[str] = None
    english_url: Optional[str] = None
    english_filename: Optional[str] = None
    english_column_label: Optional[str] = None
    hindi_url: Optional[str] = None
    hindi_column_label: Optional[str] = None


@dataclass
class SubordinateInfo:
    """Provenance for a document that hangs off a parent act rather than
    standing on its own as a DSpace item."""

    document_type: str                 # "rule" | "regulation"
    description: Optional[str] = None
    parent_handle: Optional[str] = None
    parent_title: Optional[str] = None
    parent_url: Optional[str] = None
    india_code_path: Optional[str] = None


@dataclass
class ParsedItem:
    """Everything reliably extracted from an India Code landing page."""

    source_url: str
    handle: str                       # e.g. "123456789/1372"
    handle_id: str                    # e.g. "1372"
    title: Optional[str] = None       # English DC.title
    primary_pdf_url: Optional[str] = None
    bitstreams: list[BitstreamRef] = field(default_factory=list)
    #: Set when the user asked for one specific file URL (``/bitstream/…`` or
    #: ``/ViewFileUploaded?…``): language selection is then confined to that
    #: file instead of the whole item.
    requested_bitstream_url: Optional[str] = None
    #: Set when this item is a Rule/Regulation attached to a parent act.
    subordinate: Optional[SubordinateInfo] = None
    #: The URL exactly as the caller supplied it, preserved for metadata.
    original_url: Optional[str] = None
    #: Cleaned, human-meaningful fields (only keys we could determine).
    metadata: dict = field(default_factory=dict)
    #: Raw captures, preserved verbatim for traceability / future re-parsing.
    dublin_core: dict = field(default_factory=dict)
    metadata_table: dict = field(default_factory=dict)


@dataclass
class FetchResult:
    """What one HTTP fetch of a PDF actually produced.

    ``final_url`` and ``redirects`` are kept apart from the requested URL
    because they are not the same thing for Rules/Regulations: a
    ``/ViewFileUploaded`` request is answered by India Code's file server after
    a redirect, and the redirect target has to be repaired before it can be
    followed (see :func:`ingestion.utils.repair_location`). The requested URL
    remains the citable India Code one; this records where the bytes came from.
    """

    content_type: str
    bytes: int
    http_status: int
    final_url: Optional[str] = None
    redirects: list[str] = field(default_factory=list)
    #: Set when India Code's ``/ViewFileUploaded`` lookup could not be resolved
    #: and the bytes came from the address it would have redirected to instead
    #: (see :func:`ingestion.indiacode.showfile_url`). ``None`` on the normal
    #: path, so the metadata records plainly whether a fallback was needed.
    fallback_url: Optional[str] = None


@dataclass
class DownloadResult:
    """Outcome of processing one requested URL. Consumed by the CLI summary."""

    source_url: str
    outcome: Outcome
    document_id: Optional[str] = None
    category: Optional[str] = None
    sha256: Optional[str] = None
    pdf_path: Optional[str] = None
    message: str = ""
    #: Exception class name behind a ``FAILED`` outcome (e.g. ``"FetchError"``).
    #: Kept so a batch can tell a transient failure worth retrying from a
    #: permanent one, and so failure reports say *what* went wrong.
    error_type: Optional[str] = None
    #: HTTP status behind a ``FAILED`` :class:`~ingestion.errors.FetchError`,
    #: when the server answered at all. ``None`` means no response came back
    #: (connection error, timeout) — which *is* worth retrying, unlike a 404.
    http_status: Optional[int] = None
    bytes: int = 0
