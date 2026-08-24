"""What there is to process: the manifest, read as a corpus.

The project spec requires successful documents to be discovered *through the existing
manifest* rather than by walking ``data/raw/``, and that is what this module
does. The manifest is the record of what the ingestion phase actually stored;
a PDF on disk that the manifest does not list is not part of the corpus, and
finding one is a bug worth surfacing rather than quietly processing.

The discovery inventory is joined in for two fields the manifest does not carry
and that the benchmark's sampling needs: the act **year** and the **state/UT**
a document belongs to. The join is by ``document_id``, which
:mod:`ingestion.utils` makes stable across both artefacts. It is optional:
without the inventory the corpus still loads, with those fields left ``None``.

Nothing here opens a PDF.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from ingestion import config as ingestion_config
from ingestion.manifest import Manifest

from .errors import CorpusError
from .models import CorpusDocument

log = logging.getLogger(__name__)


class Corpus:
    """The set of successfully-downloaded documents, with their provenance."""

    def __init__(self, data_dir: Path, documents: list[CorpusDocument]):
        self.data_dir = Path(data_dir)
        self.documents = documents

    def __len__(self) -> int:
        return len(self.documents)

    def __iter__(self) -> Iterator[CorpusDocument]:
        return iter(self.documents)

    @classmethod
    def load(
        cls,
        data_dir: Path = ingestion_config.DEFAULT_DATA_DIR,
        *,
        require_inventory: bool = False,
    ) -> "Corpus":
        """Load the corpus from ``<data_dir>/manifest.json``.

        Set *require_inventory* when the caller depends on year/jurisdiction (the
        benchmark sampler does), so a missing inventory fails loudly instead of
        silently producing an unstratified sample.
        """
        data_dir = Path(data_dir)
        manifest_path = data_dir / ingestion_config.MANIFEST_FILENAME
        if not manifest_path.exists():
            raise CorpusError(
                f"No manifest at {manifest_path}. Phase 1 (ingestion) must have run "
                "before anything can be processed."
            )
        manifest = Manifest.load(manifest_path)
        inventory = _load_inventory(data_dir, required=require_inventory)

        documents = []
        for document_id, entry in manifest.documents.items():
            extra = inventory.get(document_id, {})
            documents.append(_to_document(data_dir, document_id, entry, extra))
        documents.sort(key=lambda d: d.document_id)
        log.info("Loaded %d documents from %s", len(documents), manifest_path)
        return cls(data_dir, documents)

    # -- access -----------------------------------------------------------------

    def get(self, document_id: str) -> Optional[CorpusDocument]:
        for document in self.documents:
            if document.document_id == document_id:
                return document
        return None

    def by_category(self, category: str) -> list[CorpusDocument]:
        return [d for d in self.documents if d.category == category]

    def present(self) -> list[CorpusDocument]:
        """Documents whose PDF is actually on disk."""
        return [d for d in self.documents if d.pdf_path.exists()]

    def missing(self) -> list[CorpusDocument]:
        """Manifest entries whose PDF is not on disk.

        An empty list is the expected result. A non-empty one means the corpus
        and its index disagree, which processing must not paper over.
        """
        return [d for d in self.documents if not d.pdf_path.exists()]


def _to_document(
    data_dir: Path, document_id: str, entry: dict, extra: dict
) -> CorpusDocument:
    relpath = entry.get("pdf_relpath")
    if not relpath:
        raise CorpusError(f"Manifest entry {document_id} has no pdf_relpath.")
    return CorpusDocument(
        document_id=document_id,
        category=entry.get("category") or "",
        document_type=entry.get("document_type") or "",
        title=entry.get("title"),
        pdf_path=data_dir / relpath,
        pdf_relpath=relpath,
        sha256=entry.get("sha256") or "",
        bytes=int(entry.get("bytes") or 0),
        source_url=entry.get("source_url"),
        primary_pdf_url=entry.get("primary_pdf_url"),
        handle=entry.get("handle"),
        language=entry.get("language"),
        language_source=entry.get("language_source"),
        jurisdiction=extra.get("jurisdiction"),
        year=_as_year(extra.get("year")),
        enactment_date=extra.get("enactment_date"),
        parent_title=entry.get("parent_title"),
        collection_name=extra.get("collection_name"),
    )


def _as_year(value) -> Optional[int]:
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1500 <= year <= 2200 else None


def _load_inventory(data_dir: Path, *, required: bool) -> dict[str, dict]:
    path = (
        data_dir
        / ingestion_config.DISCOVERY_SUBDIR
        / ingestion_config.INVENTORY_FILENAME
    )
    if not path.exists():
        if required:
            raise CorpusError(
                f"No discovery inventory at {path}. It carries the act year and the "
                "state/UT each document belongs to, which representative sampling "
                "needs."
            )
        log.warning("No inventory at %s; year and jurisdiction will be unset.", path)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorpusError(f"Inventory at {path} is unreadable: {exc}") from exc
    records = payload.get("documents", []) if isinstance(payload, dict) else payload
    return {record["document_id"]: record for record in records if record.get("document_id")}
