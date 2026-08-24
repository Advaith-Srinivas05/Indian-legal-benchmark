"""Thresholds and layout constants for the processing phase.

Every number here is a *classification threshold* that the benchmark exists to
validate. They are gathered in one dependency-free module so a benchmark run
can report the exact values it used, and so tuning them never means hunting
through the extraction code.

Reuses :mod:`ingestion.config` for anything about the corpus itself (data
directory, categories, manifest filename) rather than restating it.
"""

from __future__ import annotations

from pathlib import Path

from ingestion import config as ingestion_config

# --- On-disk layout -------------------------------------------------------------

#: Per-document processing output: ``data/processed/indiacode/<document_id>/``.
#: Mandated by the project spec; kept strictly separate from ``data/raw/``.
PROCESSED_SUBDIR = ingestion_config.PROCESSED_SUBDIR / "indiacode"
DOCUMENT_FILENAME = "document.json"
PAGES_FILENAME = "pages.json"

#: Benchmark artefacts live outside ``processed/`` so that directory stays
#: exactly one sub-directory per document.
BENCHMARK_SUBDIR = Path("benchmark") / "pdf_extraction"
SAMPLE_FILENAME = "sample.json"
REPORT_JSON_FILENAME = "report.json"
REPORT_MARKDOWN_FILENAME = "report.md"
EXAMPLES_DIRNAME = "examples"

# --- Full-corpus run (processing.run) -------------------------------------------
#
# The corpus runner keeps its three artefacts at the top of ``data/``, beside
# ``manifest.json`` and ``download_status.jsonl``, because they describe the
# corpus as a whole rather than any one document — and because
# ``data/processed/`` must stay exactly one sub-directory per document.

#: Append-only, one JSON object per processing attempt. This is what makes a
#: 19,802-document run resumable: it is consulted before a document is
#: processed and appended to after it. Last record for a ``document_id`` wins.
PROCESSING_STATUS_FILENAME = "processing_status.jsonl"
#: The machine-readable corpus-wide summary, rebuilt from the journal.
PROCESSING_REPORT_FILENAME = "processing_report.json"
#: Held for the duration of a run so two runners cannot write one journal.
PROCESSING_LOCK_FILENAME = "processing_status.lock"

#: Bumped when the journal record shape changes. A record written under a
#: different version is not trusted for resume; the document is reprocessed.
JOURNAL_SCHEMA_VERSION = 1
#: Bumped when the report shape changes.
PROCESSING_REPORT_SCHEMA_VERSION = 1

#: The controlled vocabulary of per-document outcomes.
#:
#: ``SUCCESS``      processed, output written, eligible for legal indexing.
#: ``SUCCESS_OCR``  processed, output written, and the stored text came from an
#:                  OCR engine. **Reserved and never emitted today** — the
#:                  pipeline decides OCR (:mod:`processing.ocr`) and never runs
#:                  it, so nothing on disk is OCR-derived. It exists so the
#:                  journal, the aggregator and the report already have a place
#:                  for OCR output when that step is built, instead of the
#:                  vocabulary having to change under an existing journal.
#: ``QUARANTINED``  processed, output written, **not** eligible for indexing.
#:                  The reasons are recorded on the record; nothing is deleted.
#: ``FAILED``       the PDF could not be processed. No output was written; the
#:                  document stays retryable.
#: ``SKIPPED``      deliberately not processed. Today the only cause is a
#:                  manifest entry whose PDF is not on disk.
PROCESSING_STATUSES = (
    "SUCCESS", "SUCCESS_OCR", "QUARANTINED", "FAILED", "SKIPPED")

#: Statuses that mean ``document.json`` and ``pages.json`` were written. Only
#: these may be skipped on resume, and only after the output itself verifies.
PROCESSING_COMPLETE_STATUSES = ("SUCCESS", "SUCCESS_OCR", "QUARANTINED")

#: How hard the runner checks that output a journal record claims to exist is
#: really there and really complete. See :func:`processing.run.verify_output`.
VERIFY_LEVELS = ("fast", "standard", "full")
DEFAULT_VERIFY_LEVEL = "standard"

#: Concurrent documents. Four was the benchmark's setting and the figure the
#: 15-25 hour estimate is extrapolated from.
RUN_DEFAULT_WORKERS = 4

#: Documents between progress blocks.
RUN_PROGRESS_EVERY = 25

#: Seconds one document may take before the runner stops waiting for it, records
#: it ``FAILED`` with ``error_stage: "timeout"`` and carries on. ``0`` disables
#: the check.
#:
#: This is a safety net for an unattended overnight run, not a performance
#: target, so it is deliberately generous. At the benchmark's ~0.45
#: worker-seconds per page, the corpus's largest documents (18 are over 100 MB,
#: the largest 300 MB) can legitimately need 450-1,350 s; a tighter bound would
#: abandon real work and report it as a failure.
#:
#: **A Python thread cannot be killed.** The document is therefore *abandoned*,
#: not stopped: its worker thread runs on until it finishes on its own. What the
#: timeout buys is that the run continues, the journal is written, and the
#: report is produced instead of a night being spent waiting on one PDF. The
#: document stays retryable, because ``FAILED`` always does.
RUN_DOCUMENT_TIMEOUT_SECONDS = 1800.0

#: How often the run loop wakes to collect finished documents and to look for
#: ones that have overrun. Short enough that a timeout is noticed promptly,
#: long enough that the polling itself costs nothing.
RUN_POLL_SECONDS = 1.0

#: Free space the run requires before it starts, and the headroom below which a
#: running batch stops rather than fill the disk. The corpus is expected to
#: produce 5-10 GB of output (48.1 MB for the 100-document benchmark), so 20 GB
#: is roughly a 2x margin.
RUN_MIN_FREE_BYTES = 20 * 1024 ** 3
RUN_MIN_HEADROOM_BYTES = 5 * 1024 ** 3

#: Failure rows carried in the report itself, so the common case needs no second
#: file. The journal holds every one of them.
RUN_REPORT_FAILURE_SAMPLE = 50

#: Human validation set: the artefacts a reviewer works from, and the labels
#: they produce. Separate from ``benchmark/`` because this is the one directory
#: whose contents are *not* generated by the system judging itself.
VALIDATION_SUBDIR = Path("validation")
VALIDATION_SELECTION_FILENAME = "selection.json"
VALIDATION_LABELS_TEMPLATE = "labels.csv"
VALIDATION_REVIEW_FILENAME = "review.html"
VALIDATION_PAGES_DIRNAME = "pages"
VALIDATION_SCORE_JSON = "score.json"
VALIDATION_SCORE_MARKDOWN = "score.md"

#: Target size of the validation set. Small enough to be reviewed by a person in
#: one sitting, large enough that a failure mode appearing twice is not noise.
VALIDATION_PAGE_COUNT = 50

#: Pages any single document may contribute. Without a cap the round-robin fills
#: from the 764-page Customs Tariff Act, because a long document satisfies more
#: modes and outranks short ones by sheer surface area.
VALIDATION_MAX_PAGES_PER_DOCUMENT = 3

#: Resolution the review images are rendered at. 150 dpi is legible enough to
#: read statutory body text on screen without making a 50-page review set
#: unwieldy; OCR is run separately at the engine's own dpi.
VALIDATION_RENDER_DPI = 150

PROCESSING_SCHEMA_VERSION = 1

# --- Page classification --------------------------------------------------------

#: A page needs at least this many *letters* before we call its text usable.
#: Below it, a page is "no usable text" — a bare page number or a caption is
#: not text extraction succeeding.
MIN_PAGE_ALPHA_CHARS = 100

#: Fraction of the page area covered by images before we call the page a scan
#: of a page rather than a page with a picture on it. India Code's scanned
#: gazettes place one image over the whole page (ratio ~1.0); a diagram inside
#: an otherwise typeset page sits far below this.
IMAGE_BACKED_AREA_RATIO = 0.5

# --- Document classification ----------------------------------------------------
#
# ``pdf_type`` answers "where did these pages come from?" and
# ``text_extraction_status`` answers "did we get usable text out?". They are
# deliberately independent: a scanned gazette carrying an OCR layer is a
# *scanned* document whose text extraction nevertheless succeeded, and saying
# so is more useful than collapsing both facts into one label.

#: At/above this fraction of image-backed pages the document is ``scanned``.
SCANNED_PAGE_RATIO = 0.9
#: At/below this fraction the document is ``text_based``. Between the two it is
#: ``mixed``.
TEXT_BASED_PAGE_RATIO = 0.1

#: At/above this fraction of pages carrying usable text, extraction is ``ok``.
TEXT_OK_PAGE_RATIO = 0.9
#: At/below this fraction, the document needs OCR before it is usable at all.
REQUIRES_OCR_PAGE_RATIO = 0.1

PDF_TYPES = ("text_based", "scanned", "mixed", "vector_outlined")
TEXT_EXTRACTION_STATUSES = ("ok", "partial", "requires_ocr")

#: OCR actions that mean the text currently on disk is **not** what should be
#: indexed — either because it is corrupt or because most of it is missing.
#:
#: Found while building the human validation set: eight documents were marked
#: eligible for indexing while simultaneously carrying one of these actions. The
#: Nagaland Agriculture Produce Marketing regulations yielded text on 2 of their
#: 68 pages and were eligible; the UP Stamp Rules on 49 of 148. The gate read
#: quality, language and orientation, and never asked whether extraction had
#: actually produced a document.
#:
#: This is a consistency requirement rather than a threshold: the pipeline
#: already computes and reports both facts, and they contradicted each other.
OCR_ACTIONS_BLOCKING_INDEX = (
    "ocr_required", "ocr_recommended", "rasterize_then_ocr")

# --- Vector-outlined pages ------------------------------------------------------
#
# A distinct third failure mode, found in the first benchmark run: a converter
# has replaced every glyph with its outline, so the page draws roughly a thousand
# vector paths, carries no text and carries no image. It is worse than a scan —
# there is no text layer *and* nothing to OCR without rasterising first — and
# before this was recognised it was classified ``text_based`` with two thirds of
# its pages silently empty (the UP Stamp Rules, 1942).

#: Drawing paths a page must have before "no text, no image" reads as outlined
#: text rather than as a blank page. Real prose set as outlines runs into the
#: hundreds of paths; a ruled form or a letterhead sits far below.
VECTOR_OUTLINE_MIN_PATHS = 50
#: Fraction of a document's pages that must look outlined before the *document*
#: is classified ``vector_outlined``.
VECTOR_OUTLINE_PAGE_RATIO = 0.5

# --- Extraction quality ---------------------------------------------------------
#
# A scanned page with a *bad* OCR layer is the most dangerous case for this
# project: text is present, so nothing looks broken, but the words are wrong and
# a citation drawn from it would be a fabricated quotation of real law.
#
# The first benchmark's single-signal heuristic caught about 2 of the 8
# documents a stronger check identified, so this is now a *panel* of signals
# scored together (see :mod:`processing.quality`). Every threshold below was
# read off the first benchmark's 100 documents: the "healthy" figures are the
# range over documents with no extraction problem, the "bad" figures are the
# range over the eight known-bad ones. They are empirical separators on one
# sample, not calibrated probabilities.

#: Words of this length or longer are examined for the "no vowel" signal.
GARBLE_MIN_WORD_LENGTH = 4

#: Mean word length. Healthy documents: 4.34–5.49. Known-bad: 1.68–3.77.
QUALITY_MEAN_WORD_LENGTH_MIN = 4.20
#: Share of one-letter words. Healthy: max 0.097. Known-bad: 0.20–0.60.
QUALITY_SINGLE_CHAR_RATE_MAX = 0.12
#: Share of words with capitals inside them ("publiL", "authorilY"). Healthy:
#: max 0.0022 — this barely occurs in real typesetting. Known-bad: 0.014–0.095.
QUALITY_MIXED_CASE_RATE_MAX = 0.010
#: Share of tokens mixing letters and digits ("t1diO"). Healthy: max 0.028.
QUALITY_ALNUM_MIX_RATE_MAX = 0.035
#: Share of tokens that are recognisable English words. Healthy: min 0.32.
QUALITY_COMMON_WORD_RATE_MIN = 0.28
#: Share of characters that are letters. Healthy: min 0.56.
QUALITY_ALPHA_RATIO_MIN = 0.50
#: Share of characters that are neither alphanumeric, whitespace, nor
#: punctuation legal drafting uses.
QUALITY_SYMBOL_RATIO_MAX = 0.06
#: Share of long words with no vowel.
QUALITY_VOWELLESS_RATIO_MAX = 0.15
#: Below this many words, the ratios above carry no evidence and the assessment
#: is ``questionable`` on grounds of insufficiency rather than of corruption.
#: Set low enough that a one-page repeal act is still measured rather than
#: quarantined for being short.
QUALITY_MIN_WORDS = 60

#: Score bands. The score is the share of the signal panel a document passes,
#: weighted (see :mod:`processing.quality`); it is a *summary of the reasons*,
#: not a probability.
QUALITY_GOOD_SCORE = 0.80
QUALITY_QUESTIONABLE_SCORE = 0.55
QUALITY_LEVELS = ("good", "questionable", "bad")

#: Fraction of a document's text pages that must be individually suspect before
#: page-level inconsistency counts against the document.
QUALITY_SUSPECT_PAGE_RATIO = 0.30

# --- Content-level language validation -------------------------------------------
#
# India Code's own file labels are evidence but not proof: the first benchmark
# found a Devanagari document served under ``Files(Eng)`` and a mirrored
# Gujarati scan, both of which the ingestion phase accepted correctly under its
# own rules. Language must therefore also be established from the *extracted
# content* before a document is eligible for legal indexing.
#
# The discriminator is deliberately not "contains Devanagari": the two documents
# above OCR'd into Latin glyphs and contain none, while perfectly good English
# acts quote Devanagari titles. What separates them is whether the *readable*
# words are English.

CONTENT_LANGUAGES = ("en", "non_en", "uncertain")

#: Clean words (alphabetic, >= 2 letters, consistently cased) needed before a
#: language call means anything. Below it the answer is ``uncertain``.
LANGUAGE_MIN_CLEAN_WORDS = 150
#: …unless the evidence is decisive. A one-page repeal act is short by nature,
#: not doubtful, so it may settle the question on fewer words — but only at a
#: clearly higher bar, so that weak evidence never gets in through this door.
LANGUAGE_SHORT_MIN_CLEAN_WORDS = 60
LANGUAGE_SHORT_ENGLISH_RATE = 0.30
#: Share of clean words that are English function words. Healthy documents on
#: the first benchmark: 0.28–0.63 (median 0.47). The Devanagari document scored
#: 0.025 and the Gujarati scan 0.015. English under heavy OCR damage still
#: scored 0.20, which is why the ``en`` bar sits at 0.15 — corrupted English
#: must not be mistaken for another language.
LANGUAGE_ENGLISH_RATE = 0.15
#: At or below this, the readable words are not English.
LANGUAGE_NON_ENGLISH_RATE = 0.08
#: Distinct English function words required alongside the rate, so a document
#: cannot pass on one word repeated.
LANGUAGE_MIN_DISTINCT_FUNCTION_WORDS = 12
#: Share of *letters* in a non-Latin script before the document is non-English
#: on script grounds alone. Well above the handful of Devanagari characters an
#: English act carries when it quotes a Hindi title.
LANGUAGE_NON_LATIN_LETTER_RATIO = 0.25

# A whole-document average hides a document that is part English and part not:
# the non-Latin letter ratio over a whole act is diluted by every English page in
# it. So pages are also counted individually — but on **script alone**.
#
# Counting pages that fail the full per-page language assessment was tried and
# is wrong: it reported 33 of the 222 pages of the Wild Life (Protection) Act,
# 1972 as not English, because they are the schedules of protected species and
# contain no English function words at all. See
# :func:`processing.language.page_language_profile`.

#: Letters a page needs before its script is worth counting. A part-title page
#: carrying "SCHEDULE II" is not evidence of anything.
LANGUAGE_PAGE_MIN_LETTERS = 200
#: Share of a document's measurable pages that may be decisively non-English
#: before the document as a whole stops being established English.
LANGUAGE_MIXED_NON_EN_PAGE_RATIO = 0.15

# --- Page orientation -------------------------------------------------------------
#
# A page can be sideways without the PDF saying so. The Rajasthan Legislative
# Assembly Secretariat recruitment rules declare ``/Rotate 0`` and set half their
# lines at 90° to the page; the OCR layer on those pages was produced in the
# other orientation and extracts with its characters in reverse order ("Jo siseq
# oy3 uQ" for "On the basis of"). It reads as text and it is worthless.
#
# The measurable signal is the share of a page's lines running vertically on
# screen — see :mod:`processing.orientation` for why that is an axis test rather
# than a direction test.

#: Lines a page needs before its geometry means anything. A title page with four
#: lines set in a decorative block would otherwise decide the question.
ORIENTATION_MIN_LINES = 8
#: Share of a page's lines that must run vertically before the page is sideways.
#: High on purpose: a rotated table set beside upright prose is a *mixed* page,
#: and calling it sideways would rotate the prose out of true.
ORIENTATION_SIDEWAYS_LINE_RATIO = 0.70
#: Share of a document's pages that must be sideways before the document is
#: flagged. One rotated fold-out schedule in a 300-page act is normal typesetting.
ORIENTATION_SUSPECT_PAGE_RATIO = 0.10
#: Tesseract's orientation-and-script detection reports a confidence. Below this
#: the page is left alone: rotating a page that did not need it is worse than
#: not rotating one that did, because the second is still detectable afterwards.
#:
#: Measured on this corpus, OSD returns 10–17 on a page it can read at 200 or
#: 300 dpi and collapses to 0.14 — or fails outright — at 72 dpi. The threshold
#: separates those two populations by an order of magnitude, and the practical
#: consequence is the one worth remembering: **OSD needs the page rendered at
#: 200 dpi or better.** Below that it does not answer wrongly, it stops
#: answering, which is the failure mode to prefer.
ORIENTATION_MIN_OSD_CONFIDENCE = 1.0

# --- Repeated headers / footers / page numbers ----------------------------------

#: Lines this far from the top/bottom of a page are candidates for furniture.
FURNITURE_ZONE_LINES = 3
#: A document needs at least this many pages before repetition means anything.
FURNITURE_MIN_PAGES = 3
#: A candidate line must recur on at least this fraction of pages.
FURNITURE_MIN_PAGE_RATIO = 0.6
#: Furniture candidates longer than this are left alone: a long recurring line
#: is more likely to be real repeated legal text than a running head.
FURNITURE_MAX_CHARS = 120

# --- Tables ---------------------------------------------------------------------

#: Ruled vector paths (lines and rectangles) a page must draw before the table
#: finder is run on it at all. Two reasons, one of each kind:
#:
#: *Cost* — ``find_tables()`` takes 40–600 ms per page against about 1 ms for the
#: drawings check, and at 19,802 documents that difference is the whole run.
#:
#: *Quality* — with no ruled lines the finder falls back to text alignment, and
#: that fallback is what reports indented sub-clauses as a two-column table.
#:
#: A 2x2 grid needs three horizontal and three vertical rules, so four is a
#: floor that no real table falls below.
TABLE_MIN_RULED_PATHS = 4

#: PyMuPDF's table finder over-reports on ordinary prose. The first benchmark
#: retained 615 candidates of which an audit found only ~140 plausibly real; the
#: rest were arrangement-of-sections listings and footnote blocks that happen to
#: occupy two columns. These thresholds are the tightened replacement, and the
#: policy is now explicitly a *lower bound*: a real table that fails them is
#: reported as a rejected candidate with its reason, which is recoverable, while
#: a contents listing admitted as a table is not.
TABLE_MIN_ROWS = 3
TABLE_MIN_COLS = 2
#: Fraction of the candidate's cells that must contain text.
TABLE_MIN_FILLED_RATIO = 0.70
#: Every column must be this full. A column that is empty, or nearly so, means
#: the "table" is really one column of text with a stray gutter beside it.
TABLE_MIN_COLUMN_FILL = 0.60
#: Rows must agree on how many columns they have; a listing whose rows wander
#: between one and three cells is not a grid.
TABLE_MIN_ROW_CONSISTENCY = 0.75
#: Share of a candidate's first-column cells that may look like contents
#: entries ("12. Short title.") before the candidate is rejected as a listing.
TABLE_MAX_CONTENTS_ENTRY_RATIO = 0.34

# --- Footnotes ------------------------------------------------------------------
#
# India Code prints amendment history as numbered footnotes at the foot of the
# page. Extracted linearly they land inside the section above them, so a section's
# text ends with "1. Ins. by Act 30 of 1965, s. 3" as though it were law. They are
# separated out here — never deleted — and kept as their own provenanced units.

#: A footnote block must *end* below this fraction of the page height —
#: footnotes are at the foot of the page. It is the end rather than the start
#: because a long amendment note can run to two thirds of a page.
FOOTNOTE_MIN_PAGE_FRACTION = 0.55
#: And be set smaller than the document's body text by at least this factor.
FOOTNOTE_MAX_SIZE_RATIO = 0.92
#: Lines a footnote block may run to before we stop believing it is one.
FOOTNOTE_MAX_LINES = 60

# --- Structure parsing ----------------------------------------------------------

#: A heading captured from a ``12. Short title.—`` style line is rejected above
#: this length: legal section headings are short, and a long "heading" means the
#: pattern matched a sentence that merely began with a number.
MAX_HEADING_CHARS = 200

#: A run of this many consecutive numbered lines with no body text is treated as
#: an arrangement-of-sections / contents listing rather than as sections.
TOC_RUN_LENGTH = 5

# --- Benchmark sampling ---------------------------------------------------------

#: Target sample size for the extraction benchmark (the project spec: "approximately
#: 100 representative documents").
BENCHMARK_SAMPLE_SIZE = 100

#: How the 100 are split across the four corpus categories. Not proportional to
#: the corpus (state acts are 51% of it): the benchmark has to say something
#: useful about Central Acts and Regulations too, and those are small
#: populations. Sums to :data:`BENCHMARK_SAMPLE_SIZE`.
CATEGORY_QUOTAS = {
    "central_acts": 25,
    "state_acts": 30,
    "rules": 25,
    "regulations": 20,
}

#: Era bands (by act year) so old and recent law are both represented. India
#: Code's oldest item is from 1793 and typesetting conventions changed several
#: times since.
ERA_BANDS = (
    ("pre_1900", None, 1899),
    ("1900_1949", 1900, 1949),
    ("1950_1990", 1950, 1990),
    ("1991_2010", 1991, 2010),
    ("2011_plus", 2011, None),
)
ERA_UNKNOWN = "year_unknown"

#: Size bands (by stored PDF bytes) so short and large documents are both
#: represented. The top band is where India Code's scanned gazette reproductions
#: live, which is how the sample reaches scanned/mixed documents without anyone
#: having to pre-judge which documents those are.
SIZE_BANDS = (
    ("tiny", 0, 100 * 1024),
    ("small", 100 * 1024, 500 * 1024),
    ("medium", 500 * 1024, 2 * 1024 ** 2),
    ("large", 2 * 1024 ** 2, 8 * 1024 ** 2),
    ("huge", 8 * 1024 ** 2, None),
)
