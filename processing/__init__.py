"""Indian legal corpus processing package (Phase 2: extraction + structure).

Phase 1 (:mod:`ingestion`) produced an immutable corpus of India Code PDFs on
disk, described by ``data/manifest.json``. This package turns those PDFs into
structured, page-provenanced legal text. It is deliberately a *separate*
package: it reads the manifest and the raw PDFs, and never writes into
``data/raw/``.

The stages, each in its own small module::

    manifest + inventory   ->  processing.corpus     (what there is to process)
              |
    deterministic sample   ->  processing.sample     (~100 representative docs)
              |
    PDF -> pages           ->  processing.extract    (via processing.backends)
              |
    repeated headers etc.  ->  processing.furniture  (detected, never deleted)
              |
    pages -> legal units   ->  processing.structure  (conservative hierarchy)
              |
    write per-document     ->  processing.process    (data/processed/indiacode/)
              |
              +--------------------------+
              |                          |
    aggregate + report         resumable corpus run
    on a ~100-doc sample       over all 19,802 PDFs
     -> processing.benchmark    -> processing.run

:mod:`processing.benchmark` and :mod:`processing.run` are the two drivers, and
they are deliberately separate. The benchmark *measures* a deterministic sample
and retains every result so it can render worked examples; the runner *produces*
the corpus and retains nothing, resuming from a status journal
(``data/processing_status.jsonl``) instead. Both call the same
:func:`processing.process.process_document`; neither reimplements it.

Nothing here embeds, indexes, chunks or generates, and nothing here runs OCR:
:mod:`processing.ocr` decides what would need it and stops.

Entry points::

    python -m processing.benchmark            # sample ~100 docs and report
    python -m processing.run --dry-run        # plan the full-corpus run
    python -m processing.run --workers 4      # process all 19,802, resumably
"""

__version__ = "0.1.0"
