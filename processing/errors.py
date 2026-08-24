"""Exception hierarchy for the processing phase.

Mirrors :mod:`ingestion.errors`: one base class the benchmark can catch
per-document so a single unreadable PDF is *recorded as a failure* rather than
aborting a run over the whole sample.
"""

from __future__ import annotations


class ProcessingError(Exception):
    """Base class for all processing errors."""


class BackendUnavailableError(ProcessingError):
    """The PDF extraction backend is not installed.

    Kept separate from a per-document failure: nothing can be processed at all,
    so the run should stop and say what to install rather than record 100
    identical failures.
    """


class PDFOpenError(ProcessingError):
    """The PDF could not be opened (missing, encrypted, structurally broken)."""


class PageExtractionError(ProcessingError):
    """A page could not be read. Recorded per page; the rest still extract."""


class CorpusError(ProcessingError):
    """The manifest/inventory could not be read, or a document it names is
    missing from disk."""


class SamplingError(ProcessingError):
    """A representative sample could not be built from the corpus."""


class RunLockError(ProcessingError):
    """Another full-corpus run already holds the lock for this data directory.

    Kept separate from a per-document failure for the same reason
    :class:`BackendUnavailableError` is: nothing can be processed, and the fix
    is an action by the operator rather than a retry.
    """
