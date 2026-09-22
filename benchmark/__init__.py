"""The benchmark: a published corpus of Indian statutory law, and its ground truth.

Every gold evidence location this package produces is a character span in a
published canonical text file, never an identifier of one pipeline's chunks. That
is what lets any retrieval system, chunked however it likes, be scored against
the same answer key.

Nothing here opens a PDF or writes under ``data/processed/``. The corpus is
derived from the processing output alone.
"""

__version__ = "0.1.0"
