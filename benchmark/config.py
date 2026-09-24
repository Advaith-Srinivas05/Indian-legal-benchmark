"""Every constant the benchmark build depends on, with the reason for it."""

from __future__ import annotations

from pathlib import Path

# --- Inputs ------------------------------------------------------------------------

#: Where the corpus is built *from*. Only the build path reads these, so the
#: import is optional: the scoring half of this package is copied into other
#: projects, where ``processing`` does not exist. The fallbacks are checked
#: against the real values by ``tests/test_benchmark_portable.py`` whenever
#: ``processing`` is importable, so the two cannot drift apart unnoticed.
try:
    from processing import config as processing_config

    PROCESSED_SUBDIR = processing_config.PROCESSED_SUBDIR
    DOCUMENT_FILENAME = processing_config.DOCUMENT_FILENAME
    PAGES_FILENAME = processing_config.PAGES_FILENAME
except ModuleNotFoundError:                     # scoring only, package copied out
    PROCESSED_SUBDIR = Path("processed") / "indiacode"
    DOCUMENT_FILENAME = "document.json"
    PAGES_FILENAME = "pages.json"
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

# --- Gold evidence pool and sampling ---------------------------------------------

#: Construction workspace under the data directory: not published, gitignored.
BUILD_SUBDIR = Path("benchmark_build")
POOL_FILENAME = "evidence_pool.jsonl"
POOL_REPORT_FILENAME = "evidence_report.json"

#: Only born-digital PDFs supply gold. A scanned PDF's own text layer is OCR done
#: by someone else and never checked; it carries the same "103 read as 108" risk
#: as ours, invisibly to the tier rule.
GOLD_TEXT_SOURCES = frozenset({"born_digital"})
GOLD_MIN_CHARS = 150
#: The headline budget is 8,000 characters; a gold span must fit well inside it.
#: The corpus's longest "provision" is a 428,087-character parse failure.
GOLD_MAX_CHARS = 6000
GOLD_STUB_MAX_WORDS = 12
#: Exclusion reasons, in the order they are tested and reported.
GOLD_EXCLUSION_ORDER = ("tier", "ambiguous", "title_conflict", "text_source", "structure",
                        "too_short", "too_long", "stub")

#: Share of a sample drawn from each category. Not proportional to the pool,
#: which state acts dominate (two thirds): Central Acts are the law most people
#: ask about, and rules and regulations must be represented at all.
DEFAULT_ALLOCATION = {"central_acts": 0.30, "state_acts": 0.40, "rules": 0.20, "regulations": 0.10}
DEFAULT_SAMPLE_SIZE = 1000
#: Drawn samples are tracked in git: authoring works from them, and a question's
#: provenance names the sample row it came from.
SAMPLES_DIR = Path(__file__).resolve().parent / "data" / "samples"
SAMPLE_SCHEMA_VERSION = 1
