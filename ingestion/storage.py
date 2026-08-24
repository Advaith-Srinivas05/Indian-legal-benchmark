"""On-disk corpus layout: directories, per-document files and versioning.

Layout produced (see the project spec)::

    data/
    +-- raw/indiacode/<category>/<document_id>/
    |       <document_id>.pdf          # current primary PDF
    |       metadata.json              # per-document metadata (this stage)
    |       versions/                  # only created when a doc is updated
    |           <document_id>.<sha8>.<timestamp>.pdf
    +-- processed/                     # reserved for later phases
    +-- chunks/                        # reserved for later phases
    +-- embeddings/                    # reserved for later phases
    +-- manifest.json

The store never *overwrites* an existing legal PDF with different content: on a
content change the previous file is archived under ``versions/`` first, so
historical law is preserved (a hard requirement for this domain).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from . import config
from .utils import atomic_replace, atomic_write_text, short_hash, utcnow_iso

log = logging.getLogger(__name__)


class DocumentStore:
    """Owns the ``data/`` tree and places document artefacts within it."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    # -- directory setup --------------------------------------------------------

    def ensure_layout(self) -> None:
        """Create the full data directory skeleton if it does not yet exist."""
        for category in config.CATEGORIES:
            (self.data_dir / config.RAW_SUBDIR / category).mkdir(parents=True, exist_ok=True)
        for sub in (config.PROCESSED_SUBDIR, config.CHUNKS_SUBDIR, config.EMBEDDINGS_SUBDIR):
            (self.data_dir / sub).mkdir(parents=True, exist_ok=True)

    # -- path helpers -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / config.MANIFEST_FILENAME

    def document_dir(self, category: str, document_id: str) -> Path:
        return self.data_dir / config.RAW_SUBDIR / category / document_id

    def pdf_path(self, category: str, document_id: str) -> Path:
        return self.document_dir(category, document_id) / f"{document_id}.pdf"

    def metadata_path(self, category: str, document_id: str) -> Path:
        return self.document_dir(category, document_id) / config.METADATA_FILENAME

    def part_path(self, category: str, document_id: str) -> Path:
        """Temp path a download streams into before validation/placement."""
        return self.document_dir(category, document_id) / f"{document_id}.pdf.part"

    def relpath(self, path: Path) -> str:
        """POSIX-style path relative to ``data/`` for storing in manifest/meta."""
        return path.relative_to(self.data_dir).as_posix()

    # -- placement --------------------------------------------------------------

    def place_new_pdf(self, part: Path, category: str, document_id: str) -> Path:
        """Atomically move a freshly-downloaded ``.part`` into its final path."""
        final = self.pdf_path(category, document_id)
        final.parent.mkdir(parents=True, exist_ok=True)
        atomic_replace(part, final)
        return final

    def archive_current_pdf(self, category: str, document_id: str, old_sha256: str) -> Path:
        """Move the current PDF into ``versions/`` before writing a new one.

        Returns the archive path. This is what guarantees an update never
        destroys the previously-stored legal text.
        """
        current = self.pdf_path(category, document_id)
        versions_dir = self.document_dir(category, document_id) / config.VERSIONS_DIRNAME
        versions_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow_iso().replace(":", "").replace("-", "")
        archive = versions_dir / f"{document_id}.{short_hash(old_sha256, 8)}.{stamp}.pdf"
        atomic_replace(current, archive)
        return archive

    def write_metadata(self, category: str, document_id: str, metadata: dict) -> Path:
        """Write ``metadata.json`` atomically next to the PDF."""
        path = self.metadata_path(category, document_id)
        text = json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=False)
        atomic_write_text(path, text)
        return path
