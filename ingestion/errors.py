"""Exception hierarchy for the ingestion pipeline.

Every error the downloader raises deliberately derives from
:class:`IngestionError` so that the CLI can catch a single base type, log it
per-document and continue with the next URL instead of aborting the whole run.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class for all ingestion errors."""


class InvalidURLError(IngestionError):
    """The supplied URL is not a usable India Code document URL."""


class FetchError(IngestionError):
    """An HTTP request failed (network error, timeout, non-2xx status).

    ``status_code`` carries the HTTP status when the server answered at all. It
    is what lets a batch tell a transient failure (429, 5xx, no response) from a
    permanent one (404) instead of retrying a missing file three times.
    """

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class HostNotAllowedError(FetchError):
    """A download redirected off India Code.

    Separate from :class:`FetchError` because it is a *security* stop, not a
    transport problem: it must never be worked around by trying another URL.
    """


class NotPDFError(IngestionError):
    """The server responded but the payload is not a PDF."""


class CorruptPDFError(IngestionError):
    """The downloaded bytes look like a truncated or malformed PDF."""


class MetadataError(IngestionError):
    """Required metadata could not be determined and cannot be invented."""


class LanguageError(IngestionError):
    """No provably-English file exists, or its language is ambiguous.

    This corpus is English-only. Raised instead of downloading a file whose
    language India Code's metadata does not establish — a file is never assumed
    to be English merely because nothing marked it as Hindi.
    """


class ManifestError(IngestionError):
    """The manifest file is unreadable or structurally invalid."""


class DocumentTypeError(IngestionError):
    """An unknown/unsupported document type was requested."""
