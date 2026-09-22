"""Typed failures. A bare exception never reaches the corpus runner."""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for everything this package raises on purpose."""


class SourceError(BenchmarkError):
    """A processed document is missing, unreadable, or in a shape we do not understand."""


class NotEligibleError(BenchmarkError):
    """The document was quarantined by processing and is not part of the corpus."""


class LineStreamMismatchError(BenchmarkError):
    """The rebuilt line stream disagrees with the one the structure was parsed on.

    Every unit addresses its text by index into that stream, so a mismatch means
    every offset computed from it would point at the wrong text.
    """


class SpanMismatchError(BenchmarkError):
    """A provision's character span does not reproduce the provision's text.

    This is the corpus's central guarantee failing. It is never tolerated.
    """
