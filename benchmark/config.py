"""Every constant the benchmark build depends on, with the reason for it."""

from __future__ import annotations

from pathlib import Path

from processing import config as processing_config

# --- Inputs ------------------------------------------------------------------------

PROCESSED_SUBDIR = processing_config.PROCESSED_SUBDIR
DOCUMENT_FILENAME = processing_config.DOCUMENT_FILENAME
PAGES_FILENAME = processing_config.PAGES_FILENAME
INVENTORY_RELPATH = Path("discovery") / "indiacode_inventory.json"

#: ``document.json`` / ``pages.json`` versions this package understands. A newer
#: file is refused rather than half-read.
SUPPORTED_PROCESSING_SCHEMA_VERSIONS = (1, 2)

#: Page fields that files written before 2026-08-25 lack. Their absence is
#: tolerated and reported per document, never silently defaulted away.
OPTIONAL_PAGE_FIELDS = ("text_source", "ocr", "quality")

# --- Outputs -----------------------------------------------------------------------

#: The canonical corpus, under the data directory. Derived and gitignored.
CORPUS_SUBDIR = Path("corpus")
TEXT_DIRNAME = "text"
META_DIRNAME = "meta"
STRUCTURE_DIRNAME = "structure"
DOCUMENTS_FILENAME = "documents.jsonl"
CHECKSUMS_FILENAME = "CHECKSUMS.txt"
BUILD_REPORT_FILENAME = "build_report.json"
DUPLICATES_FILENAME = "duplicates.json"
PROVISION_EQUIVALENTS_FILENAME = "provision_equivalents.jsonl"

#: Bumped whenever the shape of anything under ``corpus/`` changes. Gold
#: evidence records name the corpus version they were computed against.
CORPUS_SCHEMA_VERSION = 1

# --- Evidence ----------------------------------------------------------------------

#: Units that can be gold evidence. Their text and their line span agree exactly
#: (measured: 9,275 of 9,275 sections on a 500-document sample). Containers such
#: as chapters store heading-only text and are never evidence.
PROVISION_UNIT_TYPES = frozenset({"section", "article"})

#: Detectors that cannot be confused with a numbered list. The India Code house
#: style ``12. Punishment for murder.—Whoever …`` and its Article equivalent.
#: ``numbered_heading_line`` (a number and heading alone on a line) is exactly
#: what a list item looks like, and ``article_marker`` matches any line opening
#: with the word Article — both are kept out of the high tier.
STRONG_DETECTORS = frozenset({"numbered_heading_dash", "article_heading_dash"})

#: Evidence-confidence tiers, in the order they are reported.
CONFIDENCE_TIERS = ("high", "medium", "low")
