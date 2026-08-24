"""The global download manifest (``data/manifest.json``).

The manifest is the single index of every document the pipeline has stored,
keyed by ``document_id``. It records enough to make runs *idempotent* and to
detect *content changes* by SHA-256 without re-reading every PDF:

* ``sha256`` of the current primary PDF (fast change detection),
* where the PDF and its ``metadata.json`` live (paths relative to ``data/``),
* first/last timestamps and a compact version history.

All writes are atomic.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from . import config
from .errors import ManifestError
from .utils import atomic_write_text, utcnow_iso

log = logging.getLogger(__name__)


class Manifest:
    """Load/modify/save the manifest. Keys are ``document_id`` strings."""

    def __init__(self, path: Path, data: dict):
        self.path = path
        self._data = data
        # A batch run mutates the manifest from several worker threads and would
        # otherwise rewrite the whole file once per document (quadratic I/O over
        # 20k documents). The lock makes concurrent upserts safe; ``defer_saves``
        # turns per-document saves into periodic checkpoints.
        self._lock = threading.RLock()
        self._deferred = False
        self._dirty = False

    # -- construction -----------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        """Load an existing manifest or return a fresh, empty one."""
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ManifestError(f"Manifest at {path} is unreadable: {exc}") from exc
            if not isinstance(data, dict) or "documents" not in data:
                raise ManifestError(f"Manifest at {path} has an unexpected structure.")
        else:
            data = {
                "schema_version": config.MANIFEST_SCHEMA_VERSION,
                "generated_at": utcnow_iso(),
                "documents": {},
            }
        return cls(path, data)

    # -- access -----------------------------------------------------------------

    @property
    def documents(self) -> dict:
        return self._data["documents"]

    def get(self, document_id: str) -> Optional[dict]:
        with self._lock:
            return self._data["documents"].get(document_id)

    def find_by_sha256(self, sha256: str) -> Optional[dict]:
        """Return any stored document whose *current* PDF matches this hash."""
        with self._lock:
            for entry in self._data["documents"].values():
                if entry.get("sha256") == sha256:
                    return entry
        return None

    # -- mutation ---------------------------------------------------------------

    def upsert(self, entry: dict) -> None:
        """Insert or replace a document entry keyed by its ``document_id``."""
        document_id = entry["document_id"]
        with self._lock:
            self._data["documents"][document_id] = entry

    @contextmanager
    def defer_saves(self):
        """Batch mode: turn :meth:`save` into a no-op until :meth:`flush`.

        Callers checkpoint explicitly (and the ``finally`` here always flushes),
        so an interrupted run loses at most the entries since the last
        checkpoint. Those documents are simply re-downloaded on the next run —
        the PDF on disk is replaced by identical bytes, never duplicated.
        """
        with self._lock:
            previous, self._deferred = self._deferred, True
        try:
            yield self
        finally:
            with self._lock:
                self._deferred = previous
            self.flush()

    def save(self) -> None:
        """Persist atomically, refreshing the top-level timestamp."""
        with self._lock:
            if self._deferred:
                self._dirty = True
                return
            self._write()

    def flush(self) -> None:
        """Persist now, even in :meth:`defer_saves` mode."""
        with self._lock:
            if self._deferred and not self._dirty:
                return
            self._write()

    def _write(self) -> None:
        self._data["generated_at"] = utcnow_iso()
        self._data.setdefault("schema_version", config.MANIFEST_SCHEMA_VERSION)
        text = json.dumps(self._data, indent=2, ensure_ascii=False, sort_keys=False)
        atomic_write_text(self.path, text)
        self._dirty = False
        log.debug("Manifest saved with %d documents -> %s", len(self.documents), self.path)
