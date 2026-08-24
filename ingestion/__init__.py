"""Indian legal corpus ingestion package.

Phase 1 of the project (see the project spec): download and organise authoritative
Indian legal PDFs from India Code into the ``data/`` directory, together with
per-document metadata, content hashes and a global manifest.

This package deliberately separates *downloading* (this phase) from later
*processing* stages (PDF text extraction, legal-hierarchy parsing,
section-aware chunking, embeddings, vector-DB insertion). The downloader
produces a stable, well-described corpus on disk that those later stages can
consume without the downloader having to be rewritten.

The corpus is **English-only**: see :mod:`ingestion.language`.

Public entry points::

    python -m ingestion.discover                                  # build the inventory
    python -m ingestion.download --url "<URL>" --type central_act  # fetch documents
"""

__version__ = "0.1.0"
