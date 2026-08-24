"""The human validation set: pages a person checks, so the gates can be scored.

Why this exists
---------------
Every threshold in :mod:`processing.config` was read off the same 100 documents
the benchmark then classifies. The benchmark's headline number — 78 of 100
documents eligible for indexing — is therefore the pipeline's opinion of itself,
produced by rules fitted to the very documents being judged. It cannot be
evidence that the rules are right, and no amount of re-running it can make it
so.

What is missing is an outside opinion. This module builds one: about 50 pages,
chosen deterministically to span the known failure modes, rendered as images
beside their extracted text so a person can answer one question per page —
**does the extracted text faithfully represent what is printed on this page?**

What this module does and does not do
-------------------------------------
It **builds artefacts**. It does not label them, and it must not: a ground truth
this system generated about itself would be the same circularity in a new file.
:mod:`processing.validation_score` scores the labels once a person has written
them, and reports honestly that nothing has been validated until they do.

The review artefact deliberately shows the reviewer a **rendered image of the
page**, never the PDF. Asking someone to open 50 binary files in a viewer,
find the right page, and hold it beside a JSON blob is how a review gets
abandoned halfway through, and a half-finished review is worse than none because
its coverage is silently biased towards the first documents in the list.

On the selection
----------------
Coverage is by *failure mode*, not by document. The question is not "are these
50 pages representative of the corpus" — they are deliberately not, being far
richer in damage than the corpus average. The question is "for each way this
pipeline can be wrong, do we have a page that would show it". A stratified
random sample of 50 pages from 5,461 would contain roughly one vector-outlined
page and no Articles at all.

Ordering within each mode is by ``sha256(document_id#page_number)``, so the set
is reproducible and re-running this never quietly swaps a page a reviewer has
already labelled.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ingestion import config as ingestion_config
from ingestion.utils import atomic_write_text, sha256_bytes, utcnow_iso

from . import __version__, config, language, orientation
from .errors import ProcessingError

log = logging.getLogger("processing.validation")


# --- The page universe ------------------------------------------------------------


@dataclass
class PageRecord:
    """One page of one processed document, with everything the system said."""

    document_id: str
    page_number: int
    document: dict                     # the document.json payload
    page: dict                         # the pages.json entry
    #: Legal units that *begin* on this page, by unit type.
    units: dict = field(default_factory=dict)

    @property
    def source(self) -> dict:
        return self.document.get("source", {})

    @property
    def extraction(self) -> dict:
        return self.document.get("extraction", {})

    @property
    def quality(self) -> dict:
        return self.document.get("extraction_quality", {})

    @property
    def content_language(self) -> dict:
        return self.document.get("content_language", {})

    @property
    def ocr_decision(self) -> dict:
        return self.document.get("ocr_decision", {})

    @property
    def page_orientation(self) -> dict:
        return self.page.get("orientation") or {}

    @property
    def retained_tables(self) -> list:
        return [t for t in self.page.get("tables", []) if t.get("retained")]


def load_pages(processed_dir: Path, document_ids: Iterable[str]) -> list[PageRecord]:
    """Read every page of the named processed documents.

    Reads only what :mod:`processing.process` already wrote. No PDF is opened
    here: selection is a decision about recorded classifications, and keeping it
    that way means the selection can be re-derived, argued with and diffed
    without a 39 GB corpus to hand.
    """
    records: list[PageRecord] = []
    for document_id in document_ids:
        directory = Path(processed_dir) / document_id
        document_path = directory / config.DOCUMENT_FILENAME
        pages_path = directory / config.PAGES_FILENAME
        if not document_path.exists() or not pages_path.exists():
            log.warning("No processed output for %s; skipping", document_id)
            continue
        document = json.loads(document_path.read_text(encoding="utf-8"))
        pages = json.loads(pages_path.read_text(encoding="utf-8"))
        units_by_page = _units_by_page(document.get("structure") or {})
        for page in pages.get("pages", []):
            records.append(PageRecord(
                document_id=document_id,
                page_number=page["page_number"],
                document=document,
                page=page,
                units=units_by_page.get(page["page_number"], {}),
            ))
    return records


def _units_by_page(structure: dict) -> dict[int, dict]:
    """Index the parsed legal units by the page each one begins on.

    By ``page_start`` rather than by span: a reviewer checking that section 103
    was read correctly needs the page where its number and heading are printed,
    not the fourth page of its text.
    """
    index: dict[int, dict] = {}

    def visit(unit: dict) -> None:
        page = unit.get("page_start")
        if page is not None:
            bucket = index.setdefault(page, {})
            bucket.setdefault(unit.get("unit_type", "?"), []).append({
                "number": unit.get("number"),
                "heading": unit.get("heading"),
                "detected_by": unit.get("detected_by"),
            })
        for child in unit.get("children", []) or []:
            visit(child)

    for unit in structure.get("units", []) or []:
        visit(unit)
    for footnote in structure.get("footnotes", []) or []:
        page = footnote.get("page_number")
        if page is not None:
            index.setdefault(page, {}).setdefault("footnote", []).append(
                {"number": None, "heading": None, "detected_by": "footnote"})
    return index


# --- Failure modes ----------------------------------------------------------------
#
# Each mode is a question the validation set has to be able to answer. They
# overlap on purpose — a page is usually several of these at once — and a page is
# recorded with *every* mode it satisfies, not only the one that selected it, so
# the coverage table reflects what a reviewer will actually be looking at.


def _doc_type(record: PageRecord) -> str:
    return record.extraction.get("pdf_type", "")


def _quality(record: PageRecord) -> str:
    return record.quality.get("classification", "")


def _year(record: PageRecord) -> Optional[int]:
    return record.source.get("year")


def _page_non_latin_ratio(record: PageRecord) -> float:
    profile = language.script_profile(record.page.get("text") or "")
    letters = profile["letters"]
    if letters < config.LANGUAGE_PAGE_MIN_LETTERS:
        return 0.0
    latin = profile["by_script"].get("LATIN", 0)
    unknown = profile["by_script"].get("UNKNOWN", 0)
    return (letters - latin - unknown) / letters


#: ``(name, description, predicate)``. Order is the round-robin order, so the
#: rarest and most consequential modes come first: if the set runs out of slots,
#: it must not be the vector-outlined page that gets dropped.
FAILURE_MODES: tuple[tuple[str, str, Callable[[PageRecord], bool]], ...] = (
    ("vector_outlined",
     "text drawn as vector outlines — no text layer and no image",
     lambda r: bool(r.page.get("is_vector_outlined"))),
    ("broken_character_map",
     "writing direction reversed against the rest of the document",
     lambda r: bool(r.page_orientation.get("direction_inconsistent"))),
    ("rotated_page",
     "page set sideways, or carrying a declared /Rotate",
     lambda r: (r.page_orientation.get("orientation") == "sideways"
                or bool(r.page_orientation.get("declared_rotation")))),
    ("non_english",
     "content the system says is not English",
     lambda r: (r.content_language.get("content_language") == "non_en"
                or _page_non_latin_ratio(r) >= config.LANGUAGE_NON_LATIN_LETTER_RATIO)),
    ("uncertain_language",
     "content whose language the system could not establish",
     lambda r: r.content_language.get("content_language") == "uncertain"),
    ("article",
     "a page where an Article begins",
     lambda r: "article" in r.units),
    ("no_text_layer",
     "a page that yielded no usable text at all",
     lambda r: not r.page.get("has_text") and not r.page.get("is_vector_outlined")),
    ("poor_ocr_layer",
     "a scan whose existing OCR layer the system calls bad",
     lambda r: (_doc_type(r) in ("scanned", "mixed") and _quality(r) == "bad"
                and bool(r.page.get("has_text")))),
    ("questionable_extraction",
     "text the system is not willing to vouch for",
     lambda r: (_quality(r) == "questionable"
                or bool(r.page.get("text_quality_suspect")))),
    ("mixed_document",
     "a page from a document that is part typeset and part scanned",
     lambda r: _doc_type(r) == "mixed"),
    ("good_ocr_layer",
     "a scan whose existing OCR layer the system calls good",
     lambda r: (_doc_type(r) in ("scanned", "mixed") and _quality(r) == "good"
                and bool(r.page.get("has_text")))),
    ("table",
     "a page with a table the system retained",
     lambda r: bool(r.retained_tables)),
    ("footnote",
     "a page with a footnote block separated out of the body text",
     lambda r: bool(r.page.get("footnotes"))),
    ("scanned_page",
     "a page that is an image of a page",
     lambda r: bool(r.page.get("is_image_backed"))),
    ("old_document",
     "law from before 1950, in older typesetting",
     lambda r: (_year(r) or 9999) < 1950 and bool(r.page.get("has_text"))),
    ("modern_document",
     "law from 2011 onwards",
     lambda r: (_year(r) or 0) >= 2011 and bool(r.page.get("has_text"))),
    ("section",
     "a page where a numbered Section begins",
     lambda r: "section" in r.units),
    ("subsection",
     "a page where a subsection begins",
     lambda r: "subsection" in r.units),
    ("clause",
     "a page where a clause begins",
     lambda r: "clause" in r.units),
    ("born_digital_clean",
     "typeset text the system is confident about — the control group",
     lambda r: (_doc_type(r) == "text_based" and _quality(r) == "good"
                and bool(r.page.get("has_text"))
                and not r.page.get("text_quality_suspect")
                and r.page_orientation.get("orientation") == "upright")),
)

MODE_NAMES = tuple(name for name, _, _ in FAILURE_MODES)
MODE_DESCRIPTIONS = {name: description for name, description, _ in FAILURE_MODES}


def modes_of(record: PageRecord) -> list[str]:
    """Every failure mode this page satisfies."""
    return [name for name, _, predicate in FAILURE_MODES if predicate(record)]


# --- Selection --------------------------------------------------------------------


@dataclass
class ValidationPage:
    """One page selected for human review."""

    document_id: str
    page_number: int
    selected_for: str                  # the mode whose turn chose this page
    modes: list[str]
    rank: str
    title: Optional[str] = None
    category: str = ""
    year: Optional[int] = None
    jurisdiction: Optional[str] = None
    pdf_relpath: str = ""

    @property
    def slug(self) -> str:
        return f"{self.document_id[:60]}-p{self.page_number:04d}"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "selected_for": self.selected_for,
            "selected_because": MODE_DESCRIPTIONS.get(self.selected_for, ""),
            "modes": self.modes,
            "rank": self.rank,
            "title": self.title,
            "category": self.category,
            "year": self.year,
            "jurisdiction": self.jurisdiction,
            "pdf_relpath": self.pdf_relpath,
            "slug": self.slug,
        }


def _rank(document_id: str, page_number: int) -> str:
    return sha256_bytes(f"{document_id}#{page_number}".encode("utf-8"))


def select(
    records: list[PageRecord],
    *,
    count: int = config.VALIDATION_PAGE_COUNT,
    max_per_document: int = config.VALIDATION_MAX_PAGES_PER_DOCUMENT,
) -> list[ValidationPage]:
    """Choose the validation pages, deterministically, mode by mode.

    Round-robin rather than quota-per-mode: the modes have wildly different
    populations (one vector-outlined page in the whole sample, thousands of
    clean ones), and a fixed quota would either fail to fill or crowd out the
    rare cases. Going round in order takes one page per mode per pass and stops
    when the set is full, which spends the scarce slots on the scarce modes.

    *max_per_document* stops the round-robin filling from whichever document is
    longest: a 764-page act satisfies more modes than a 9-page rule and would
    otherwise supply a third of the set.
    """
    candidates: dict[str, list[PageRecord]] = {name: [] for name in MODE_NAMES}
    all_modes: dict[tuple[str, int], list[str]] = {}
    for record in records:
        satisfied = modes_of(record)
        if not satisfied:
            continue
        all_modes[(record.document_id, record.page_number)] = satisfied
        for name in satisfied:
            candidates[name].append(record)

    for name in MODE_NAMES:
        candidates[name].sort(key=lambda r: _rank(r.document_id, r.page_number))

    chosen: list[ValidationPage] = []
    seen: set[tuple[str, int]] = set()
    per_document: dict[str, int] = {}
    cursors = {name: 0 for name in MODE_NAMES}

    progressed = True
    while len(chosen) < count and progressed:
        progressed = False
        for name in MODE_NAMES:
            if len(chosen) >= count:
                break
            index = cursors[name]
            pool = candidates[name]
            while index < len(pool):
                record = pool[index]
                index += 1
                key = (record.document_id, record.page_number)
                if key in seen:
                    continue
                if per_document.get(record.document_id, 0) >= max_per_document:
                    continue
                seen.add(key)
                per_document[record.document_id] = \
                    per_document.get(record.document_id, 0) + 1
                chosen.append(ValidationPage(
                    document_id=record.document_id,
                    page_number=record.page_number,
                    selected_for=name,
                    modes=all_modes[key],
                    rank=_rank(record.document_id, record.page_number),
                    title=record.source.get("title"),
                    category=record.source.get("category", ""),
                    year=record.source.get("year"),
                    jurisdiction=record.source.get("jurisdiction"),
                    pdf_relpath=record.source.get("pdf_relpath", ""),
                ))
                progressed = True
                break
            cursors[name] = index

    chosen.sort(key=lambda p: (p.document_id, p.page_number))
    return chosen


def coverage(selected: list[ValidationPage]) -> dict:
    """How many selected pages exhibit each mode, and which modes went unfilled.

    An unfilled mode is a finding: it means the 100-document benchmark contains
    no example of it, so this validation set cannot say anything about that part
    of the pipeline and the final report must not imply otherwise.
    """
    counts = {name: 0 for name in MODE_NAMES}
    for page in selected:
        for name in page.modes:
            counts[name] = counts.get(name, 0) + 1
    return {
        "pages": len(selected),
        "documents": len({p.document_id for p in selected}),
        "by_mode": counts,
        "modes_with_no_page": [name for name, n in counts.items() if n == 0],
        "selected_for": {
            name: sum(1 for p in selected if p.selected_for == name)
            for name in MODE_NAMES
        },
    }


# --- Building the review artefacts --------------------------------------------------


def validation_dir(data_dir: Path) -> Path:
    return Path(data_dir) / config.VALIDATION_SUBDIR


def render_page_image(
    pdf_path: Path, page_number: int, dpi: int = config.VALIDATION_RENDER_DPI
) -> bytes:
    """A PNG of the page **as a viewer would display it**.

    As displayed, not as corrected: the reviewer's job includes judging whether
    the system was right about the orientation, and handing them a page this
    code has already straightened would take that question away from them.
    """
    import pymupdf                                         # noqa: PLC0415

    document = pymupdf.open(pdf_path)
    try:
        index = max(0, min(page_number - 1, document.page_count - 1))
        return document.load_page(index).get_pixmap(dpi=dpi).tobytes("png")
    finally:
        document.close()


def needs_ocr_reference(record: PageRecord) -> bool:
    """Whether an OCR pass would tell the reviewer anything they do not have.

    Running it on a clean born-digital page produces a second opinion nobody
    needs and 50 more seconds of review. Running it on a bad scan is the whole
    question: is the text on disk bad because the page is unreadable, or because
    the existing OCR layer is bad and a re-run would fix it? Those are different
    verdicts and the reviewer cannot tell them apart without seeing both.
    """
    if record.page.get("is_vector_outlined"):
        return True
    if not record.page.get("has_text"):
        return True
    if record.page.get("is_image_backed"):
        return True
    if _quality(record) in ("bad", "questionable"):
        return True
    if record.page.get("text_quality_suspect"):
        return True
    orientation_state = record.page_orientation
    return (orientation_state.get("orientation") == "sideways"
            or bool(orientation_state.get("declared_rotation")))


def ocr_reference(pdf_path: Path, page_number: int, engine) -> dict:
    """OCR the page with the selected engine, correcting orientation first.

    Tesseract was chosen over RapidOCR and EasyOCR on the 25-page engine
    comparison (1.35 s/page at 200 dpi against 6.57 and 19.04, with 40% more
    words recovered and none of RapidOCR's lost word spacing), so this is the
    engine whose output the corpus would actually be rebuilt with — which makes
    it the one worth putting in front of a reviewer.
    """
    from .ocr_eval import render_page                      # noqa: PLC0415

    image = render_page(pdf_path, page_number, 200)
    upright, detected = orientation.upright_image(image)
    try:
        text = engine(upright)
        error = None
    except Exception as exc:                               # engines fail variously
        text, error = "", f"{type(exc).__name__}: {exc}"
    return {
        "engine": "tesseract@200",
        "text": text,
        "error": error,
        "rotation_applied": detected.rotate_degrees if detected.known else None,
        "rotation_source": detected.source,
        "rotation_confidence": round(detected.confidence, 2),
    }


# --- The label vocabulary -----------------------------------------------------------
#
# One source of truth, shared by the review sheet, the CSV template, LABELS.md and
# the scorer. A label whose meaning drifts between the sheet a person read and the
# code that scores it would silently corrupt every number in the final report.

#: ``(label, one-line meaning, what it implies for indexing)``.
LABELS: tuple[tuple[str, str, str], ...] = (
    ("GOOD",
     "The extracted text faithfully represents the page. Every section number, "
     "figure, date, name and negation is right. Line breaks and hyphenation may "
     "differ from the print.",
     "safe to index and quote"),
    ("MINOR_ERROR",
     "Wrong in ways that do not change legal meaning: a garbled running head, a "
     "dropped page number, a mangled ornament, a misread character inside a word "
     "that is still unambiguous.",
     "safe to index; not safe to quote character-for-character"),
    ("BAD",
     "Readable, but the legal meaning is altered or lost. Any one of the "
     "legal-critical errors below is sufficient on its own, however good the rest "
     "of the page looks.",
     "must not be indexed"),
    ("UNUSABLE",
     "Nothing worth keeping came out: an empty page, pure noise, or so little of "
     "the page recovered that there is nothing to judge.",
     "must not be indexed; needs OCR or is a genuine gap"),
    ("NON_ENGLISH",
     "The page is substantively in a language other than English, whether or not "
     "the extraction of it is accurate.",
     "out of scope for this corpus, whatever its quality"),
    ("NEEDS_REVIEW",
     "The reviewer cannot decide — the source page is itself illegible, or the "
     "page needs a subject-matter expert. An honest answer, not a fallback.",
     "quarantined pending a second opinion"),
)

LABEL_NAMES = tuple(name for name, _, _ in LABELS)

#: Errors that make a page BAD no matter how well the rest of it reads. This list
#: is the operative part of the whole exercise: the failure this project must not
#: have is text that *looks* fine and states something the statute does not.
LEGAL_CRITICAL_ERRORS: tuple[tuple[str, str], ...] = (
    ("wrong_section_number",
     "a section or Article number differs from the print (103 read as 108, 65A as 65)"),
    ("wrong_subsection_number",
     "a subsection or clause marker differs — (2)(a) read as (2)(d)"),
    ("missing_clause",
     "a clause, sub-clause or item present on the page is absent from the text"),
    ("altered_negation",
     "a 'not', 'no', 'nor', 'shall not', 'unless' or 'except' added, dropped or reversed"),
    ("wrong_date",
     "any date differs, including in an amendment note or a commencement provision"),
    ("wrong_number",
     "any figure differs — a penalty, a period, a percentage, a currency amount"),
    ("wrong_name",
     "a name of a person, place, office, court or enactment differs"),
    ("missing_proviso",
     "a proviso printed on the page is absent from the text"),
    ("missing_explanation",
     "an Explanation printed on the page is absent from the text"),
    ("meaning_changing_word_split",
     "words merged or split so that the sense changes ('no where' for 'nowhere')"),
    ("ocr_substitution",
     "a character substitution that yields a different real word ('lakh'/'lakhs', "
     "'may'/'many', 'fine'/'five')"),
    ("missing_line",
     "a printed line of substantive text is absent from the extraction"),
    ("wrong_page_order",
     "text from elsewhere on the page, or from another page, appears out of order "
     "in a way that changes what a provision says"),
)

ERROR_NAMES = tuple(name for name, _ in LEGAL_CRITICAL_ERRORS)

#: How a human label maps onto the system's binary "may this be indexed" gate.
#: Used only by the scorer, and stated here so the mapping is reviewable rather
#: than buried in the confusion-matrix code.
HUMAN_ACCEPTABLE = ("GOOD", "MINOR_ERROR")
HUMAN_REJECT = ("BAD", "UNUSABLE", "NON_ENGLISH")


# --- Review artefacts ---------------------------------------------------------------


def render_labels_template(entries: list[dict]) -> str:
    """The CSV a reviewer fills in. One row per page, empty where they decide.

    CSV because it opens in a spreadsheet and can be edited without a tool being
    built first; the scorer reads it back and never writes to it, so a reviewer's
    work can never be overwritten by re-running the builder.
    """
    import csv                                             # noqa: PLC0415
    import io                                              # noqa: PLC0415

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "document_id", "page_number", "label", "legal_critical_errors", "notes",
        "system_extraction_quality", "system_content_language",
        "system_orientation", "system_pdf_type", "system_ocr_action",
        "system_eligible_for_indexing", "selected_for",
    ])
    for entry in entries:
        page = entry["page"]
        system = entry["system"]
        writer.writerow([
            page["document_id"], page["page_number"],
            "",                                            # label — the reviewer
            "",                                            # legal_critical_errors
            "",                                            # notes
            system["extraction_quality"], system["content_language"],
            system["page_orientation"], system["pdf_type"], system["ocr_action"],
            system["eligible_for_indexing"], page["selected_for"],
        ])
    return buffer.getvalue()


def render_labels_document() -> str:
    """``LABELS.md`` — what each label means, and what makes a page BAD."""
    lines = [
        "# Validation labels",
        "",
        "One label per page. The question being answered is always the same:",
        "",
        "> **Does the extracted text faithfully represent what is printed on "
        "this page?**",
        "",
        "Not *is the page a good page*, and not *did the system try hard*. A page "
        "can be a filthy 1890s scan and still be labelled `GOOD` if the text that "
        "came out of it says what the page says.",
        "",
        "## The labels",
        "",
        "| label | meaning | consequence |",
        "| --- | --- | --- |",
    ]
    for name, meaning, consequence in LABELS:
        lines.append(f"| `{name}` | {meaning} | {consequence} |")

    lines += [
        "",
        "## What makes a page `BAD`",
        "",
        "**A page that reads well and states something the statute does not is "
        "`BAD`.** This is the failure this whole exercise exists to detect, and it "
        "is the one that looks like success. Fluency is not evidence of accuracy: "
        "OCR damage in legal text characteristically produces plausible words in "
        "the wrong places.",
        "",
        "Any *one* of the following is sufficient, however good the rest of the "
        "page looks:",
        "",
        "| error | what it looks like |",
        "| --- | --- |",
    ]
    for name, description in LEGAL_CRITICAL_ERRORS:
        lines.append(f"| `{name}` | {description} |")

    lines += [
        "",
        "Record every one you find in the `legal_critical_errors` column, "
        "semicolon-separated, using the names in the left column.",
        "",
        "## What is *not* an error",
        "",
        "These are extraction artefacts, not defects in the text, and marking them "
        "would drown the signal that matters:",
        "",
        "- line breaks falling in different places than in the print;",
        "- a hyphenated word rejoined, or left split, across a line break;",
        "- running heads, running feet and page numbers appearing inline (they are "
        "  detected and labelled separately, deliberately not deleted);",
        "- footnotes appearing after the body text rather than at the foot;",
        "- differences in whitespace, indentation or column spacing;",
        "- typographic substitutions that do not change a word — a straight quote "
        "  for a curly one, a hyphen for an en dash.",
        "",
        "## How to review one page",
        "",
        "1. Read the **page image** on the left. That is the source of truth.",
        "2. Read the **extracted text** on the right. That is what the corpus "
        "   currently holds for this page.",
        "3. Check the legal-critical list against the two, in the order given "
        "   above — numbers and negations first, because they are the errors "
        "   least visible on a skim and most damaging downstream.",
        "4. Where an **OCR reference** is shown, it is what re-running OCR would "
        "   produce. It does not change the label — the label is about the text "
        "   the corpus holds today — but note in `notes` if OCR would clearly fix "
        "   the page, because that decides whether it is worth re-processing.",
        "5. Write the label. If you cannot decide, `NEEDS_REVIEW` is a real "
        "   answer; guessing is not.",
        "",
        "## Do not trust the system columns",
        "",
        "The CSV carries the system's own classifications so that disagreements "
        "can be scored afterwards. They are **not** hints. If you find yourself "
        "agreeing with a column rather than with the page, blank the row and come "
        "back to it: the entire value of this exercise is that these labels were "
        "produced independently of the classifier being tested.",
        "",
    ]
    return "\n".join(lines)


def _escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def render_review_html(payload: dict) -> str:
    """The review sheet: page image on the left, what the system made of it on
    the right.

    A single local page. Images are referenced relatively rather than inlined as
    data URIs, which keeps the file small enough to open instantly and lets the
    reviewer zoom one page in an image viewer when a scan is marginal.
    """
    entries = payload["entries"]
    documents = len({e["page"]["document_id"] for e in entries})
    out = [
        "<!-- Generated by processing.validation. Edit labels.csv, not this file. -->",
        "<meta charset='utf-8'>",
        f"<title>Validation review - {len(entries)} pages</title>",
        _REVIEW_CSS,
        "<h1>Human validation set</h1>",
        f"<p class='meta'>{len(entries)} pages from {documents} documents. "
        f"Generated {_escape(payload['generated_at'])} by "
        f"{_escape(payload['processor'])}. Read <code>LABELS.md</code> first; "
        "record your labels in <code>labels.csv</code>.</p>",
        "<div class='warn'><b>The page image is the source of truth.</b> "
        "The system's own classifications are shown so that disagreements can be "
        "scored afterwards &mdash; they are not hints, and agreeing with them "
        "rather than with the page destroys the only thing this exercise "
        "measures.</div>",
        "<h2>Labels</h2>",
        "<table class='legend'><tr><th>label</th><th>meaning</th></tr>",
    ]
    for name, meaning, _consequence in LABELS:
        out.append(f"<tr><td><code>{name}</code></td>"
                   f"<td>{_escape(meaning)}</td></tr>")
    out.append("</table>")

    out.append("<h2>A page that reads well but changes legal meaning is "
               "<code>BAD</code></h2>")
    out.append("<ul class='errors'>")
    for name, description in LEGAL_CRITICAL_ERRORS:
        out.append(f"<li><code>{name}</code> &mdash; {_escape(description)}</li>")
    out.append("</ul>")

    for index, entry in enumerate(entries, start=1):
        out.append(_render_entry_html(index, len(entries), entry))
    return "\n".join(out)


def _render_entry_html(index: int, total: int, entry: dict) -> str:
    page = entry["page"]
    system = entry["system"]
    signals = entry["signals"]
    identifier = f"{page['document_id']}#{page['page_number']}"
    others = [m for m in page["modes"] if m != page["selected_for"]]
    also = ", ".join(f"<code>{_escape(m)}</code>" for m in others) or "&mdash;"
    year = page["year"] if page["year"] is not None else "unknown"

    parts = [
        f"<section id='{_escape(page['slug'])}'>",
        f"<h3>{index}/{total} &middot; "
        f"{_escape(page['title'] or page['document_id'])} &mdash; "
        f"page {page['page_number']}</h3>",
        "<p class='meta'>",
        f"<code>{_escape(identifier)}</code><br>",
        f"{_escape(page['category'])} &middot; "
        f"{_escape(page['jurisdiction'] or 'n/a')} &middot; year {_escape(year)}<br>",
        f"selected for <b>{_escape(page['selected_for'])}</b> &mdash; "
        f"{_escape(MODE_DESCRIPTIONS.get(page['selected_for'], ''))}<br>",
        f"also exhibits: {also}",
        "</p>",
        "<div class='cols'>",
        "<div class='col imgcol'>",
        f"<img src='{config.VALIDATION_PAGES_DIRNAME}/{_escape(page['slug'])}.png' "
        f"alt='page {page['page_number']} as printed' loading='lazy'>",
        "<p class='cap'>The page as a viewer displays it. Source of truth.</p>",
        "</div>",
        "<div class='col'>",
        "<h4>Extracted text &mdash; what the corpus holds for this page</h4>",
        f"<pre>{_escape(entry['extracted_text']) or '(nothing was extracted)'}</pre>",
    ]

    ocr = entry.get("ocr")
    if ocr:
        rotation = ocr.get("rotation_applied")
        note = (f"rotated {rotation}&deg; first (OSD confidence "
                f"{ocr.get('rotation_confidence')})"
                if rotation else "no rotation applied")
        ocr_text = _escape(ocr.get("text") or "") or "(no output)"
        parts += [
            f"<h4>OCR reference &mdash; <code>{_escape(ocr['engine'])}</code>, "
            f"{note}</h4>",
            "<p class='cap'>What re-running OCR would produce. This does "
            "<b>not</b> change the label, which is about the text the corpus "
            "holds today &mdash; but say so in <code>notes</code> if it would "
            "clearly fix the page.</p>",
            f"<pre class='ocr'>{ocr_text}</pre>",
        ]

    parts += ["<h4>What the system says</h4>", "<table class='sys'>"]
    for label, value in (
        ("pdf_type", system["pdf_type"]),
        ("text_extraction_status", system["text_extraction_status"]),
        ("extraction_quality",
         f"{system['extraction_quality']} (score {system['quality_score']})"),
        ("content_language",
         f"{system['content_language']} (English function words "
         f"{system['function_word_rate']})"),
        ("page orientation", system["page_orientation"]),
        ("ocr_action", system["ocr_action"]),
        ("eligible_for_indexing", system["eligible_for_indexing"]),
    ):
        parts.append(f"<tr><th>{_escape(label)}</th><td>{_escape(value)}</td></tr>")
    parts.append("</table>")

    if signals.get("units"):
        parts.append("<h4>Legal units the parser says begin on this page</h4><ul>")
        for unit_type, units in signals["units"].items():
            numbers = ", ".join(_escape(u["number"] or "-") for u in units[:12])
            parts.append(f"<li><code>{_escape(unit_type)}</code> "
                         f"({len(units)}): {numbers}</li>")
        parts.append("</ul>")
    if signals.get("tables"):
        parts.append(f"<p class='cap'>{signals['tables']} table(s) retained on "
                     "this page.</p>")
    if signals.get("footnotes"):
        parts.append(f"<p class='cap'>{signals['footnotes']} footnote block(s) "
                     "separated out of the body text.</p>")
    if signals.get("warnings"):
        parts.append("<h4>Page warnings</h4><ul>")
        parts += [f"<li>{_escape(w)}</li>" for w in signals["warnings"]]
        parts.append("</ul>")

    labels = " &middot; ".join(f"<code>{n}</code>" for n in LABEL_NAMES)
    parts += [
        "<h4>Your judgement</h4>",
        "<p class='judgement'>Record in <code>labels.csv</code>, the row for "
        f"<code>{_escape(page['document_id'])}</code> page "
        f"<code>{page['page_number']}</code>:<br>"
        f"<b>label</b> = {labels}<br>"
        "<b>legal_critical_errors</b> = semicolon-separated, from the list at the "
        "top<br><b>notes</b> = anything the label does not carry</p>",
        "</div>", "</div>", "</section>",
    ]
    return "\n".join(parts)


_REVIEW_CSS = """<style>
:root { color-scheme: light dark; }
body { font: 15px/1.55 -apple-system, Segoe UI, Roboto, sans-serif;
       max-width: 1600px; margin: 0 auto; padding: 1.5rem; }
h1 { margin-bottom: .2rem; }
h3 { margin-top: 2.5rem; border-top: 3px solid currentColor; padding-top: .8rem; }
h4 { margin: 1.2rem 0 .3rem; font-size: .95rem; text-transform: uppercase;
     letter-spacing: .04em; opacity: .75; }
.meta { opacity: .8; font-size: .9rem; }
.warn { border-left: 4px solid #c60; padding: .6rem .9rem; margin: 1rem 0;
        background: rgba(204,102,0,.09); }
.cols { display: flex; gap: 1.5rem; align-items: flex-start; }
.col { flex: 1 1 0; min-width: 0; }
.imgcol { position: sticky; top: 1rem; }
img { max-width: 100%; height: auto; border: 1px solid rgba(128,128,128,.5); }
pre { white-space: pre-wrap; word-break: break-word; font-size: 13px;
      background: rgba(128,128,128,.10); padding: .7rem; border-radius: 4px;
      max-height: 30rem; overflow: auto; }
pre.ocr { background: rgba(0,128,128,.10); }
table { border-collapse: collapse; margin: .4rem 0; }
td, th { border: 1px solid rgba(128,128,128,.4); padding: .25rem .55rem;
         text-align: left; vertical-align: top; font-size: .9rem; }
table.legend td:first-child, table.sys th { white-space: nowrap; }
ul.errors { columns: 2; font-size: .9rem; }
.cap { font-size: .85rem; opacity: .75; margin: .3rem 0; }
.judgement { border: 2px dashed currentColor; padding: .7rem; border-radius: 4px; }
code { font-size: .9em; }
</style>"""


# --- Building the set ---------------------------------------------------------------


def build_entry(record: PageRecord, page: ValidationPage) -> dict:
    """Everything the review sheet needs about one page, gathered in one place."""
    quality = record.quality
    content_language = record.content_language
    page_orientation = record.page_orientation
    return {
        "page": page.to_dict(),
        "extracted_text": record.page.get("text") or "",
        "system": {
            "pdf_type": record.extraction.get("pdf_type", ""),
            "text_extraction_status":
                record.extraction.get("text_extraction_status", ""),
            "extraction_quality": quality.get("classification", ""),
            "quality_score": round(quality.get("score") or 0.0, 3),
            "quality_failed_checks": quality.get("failed_checks", []),
            "content_language": content_language.get("content_language", ""),
            "function_word_rate":
                (content_language.get("signals") or {}).get("function_word_rate"),
            "page_orientation": page_orientation.get("orientation", "unknown"),
            "page_declared_rotation": page_orientation.get("declared_rotation", 0),
            "page_direction_inconsistent":
                bool(page_orientation.get("direction_inconsistent")),
            "ocr_action": record.ocr_decision.get("action", ""),
            "ocr_text_source": record.ocr_decision.get("text_source", ""),
            "eligible_for_indexing":
                bool(record.document.get("eligible_for_indexing")),
        },
        "signals": {
            "char_count": record.page.get("char_count", 0),
            "alpha_char_count": record.page.get("alpha_char_count", 0),
            "has_text": bool(record.page.get("has_text")),
            "is_image_backed": bool(record.page.get("is_image_backed")),
            "is_vector_outlined": bool(record.page.get("is_vector_outlined")),
            "is_empty": bool(record.page.get("is_empty")),
            "text_quality_suspect": bool(record.page.get("text_quality_suspect")),
            "tables": len(record.retained_tables),
            "footnotes": len(record.page.get("footnotes") or []),
            "units": record.units,
            "warnings": record.page.get("warnings") or [],
        },
    }


def render_page_markdown(entry: dict) -> str:
    """A text-only view of one page, for review away from a browser."""
    page, system = entry["page"], entry["system"]
    lines = [
        f"# {page['title']} — page {page['page_number']}",
        "",
        f"- document: `{page['document_id']}`",
        f"- image: `{config.VALIDATION_PAGES_DIRNAME}/{page['slug']}.png`",
        f"- selected for: **{page['selected_for']}** — "
        f"{MODE_DESCRIPTIONS.get(page['selected_for'], '')}",
        f"- also exhibits: {', '.join(page['modes']) or '—'}",
        f"- system: pdf_type `{system['pdf_type']}`, quality "
        f"`{system['extraction_quality']}`, language "
        f"`{system['content_language']}`, orientation "
        f"`{system['page_orientation']}`, ocr_action `{system['ocr_action']}`, "
        f"eligible `{system['eligible_for_indexing']}`",
        "",
        "## Extracted text — what the corpus holds for this page",
        "",
        "```text",
        entry["extracted_text"].rstrip() or "(nothing was extracted)",
        "```",
        "",
    ]
    ocr = entry.get("ocr")
    if ocr:
        lines += [
            f"## OCR reference — {ocr['engine']}"
            + (f", rotated {ocr['rotation_applied']}°"
               if ocr.get("rotation_applied") else ""),
            "",
            "```text",
            (ocr.get("text") or "").rstrip() or "(no output)",
            "```",
            "",
        ]
    lines += [
        "## Your judgement",
        "",
        f"Record in `{config.VALIDATION_LABELS_TEMPLATE}`: label = one of "
        + " / ".join(LABEL_NAMES) + ".",
        "",
    ]
    return "\n".join(lines)


def build(
    data_dir: Path,
    *,
    count: int = config.VALIDATION_PAGE_COUNT,
    with_ocr: bool = True,
    render: bool = True,
) -> dict:
    """Select the pages, render them, and write everything a reviewer needs.

    *with_ocr* and *render* exist so the selection can be re-derived and diffed
    cheaply — the expensive halves are rasterising 50 pages and OCRing about half
    of them, and neither affects which pages are chosen.
    """
    data_dir = Path(data_dir)
    report_path = (data_dir / config.BENCHMARK_SUBDIR / config.REPORT_JSON_FILENAME)
    if not report_path.exists():
        raise ProcessingError(
            f"No benchmark report at {report_path}; run "
            "`python -m processing.benchmark` first."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    document_ids = [row["document_id"] for row in report.get("documents", [])
                    if row.get("ok")]
    records = load_pages(data_dir / config.PROCESSED_SUBDIR, document_ids)
    if not records:
        raise ProcessingError(
            "No processed pages found. Run `python -m processing.benchmark` "
            "without --no-write so that data/processed/ is populated."
        )
    log.info("Universe: %d pages across %d documents",
             len(records), len(document_ids))

    selected = select(records, count=count)
    log.info("Selected %d pages across %d documents",
             len(selected), len({p.document_id for p in selected}))

    by_key = {(r.document_id, r.page_number): r for r in records}
    directory = validation_dir(data_dir)
    pages_dir = directory / config.VALIDATION_PAGES_DIRNAME
    pages_dir.mkdir(parents=True, exist_ok=True)

    engine = None
    if with_ocr:
        try:
            from .ocr_eval import _pytesseract_loader     # noqa: PLC0415

            engine = _pytesseract_loader()
        except Exception as exc:
            log.warning("No OCR reference will be included: %s", exc)

    entries: list[dict] = []
    for index, page in enumerate(selected, start=1):
        record = by_key[(page.document_id, page.page_number)]
        entry = build_entry(record, page)
        pdf_path = data_dir / page.pdf_relpath
        if render:
            image = render_page_image(pdf_path, page.page_number)
            (pages_dir / f"{page.slug}.png").write_bytes(image)
        if engine is not None and needs_ocr_reference(record):
            entry["ocr"] = ocr_reference(pdf_path, page.page_number, engine)
        atomic_write_text(pages_dir / f"{page.slug}.md",
                          render_page_markdown(entry))
        entries.append(entry)
        log.info("  … %d/%d pages prepared", index, len(selected))

    payload = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "benchmark_report": report.get("generated_at"),
        "labels": [{"name": n, "meaning": m, "consequence": c}
                   for n, m, c in LABELS],
        "legal_critical_errors": [{"name": n, "description": d}
                                  for n, d in LEGAL_CRITICAL_ERRORS],
        "failure_modes": [{"name": n, "description": d}
                          for n, d, _ in FAILURE_MODES],
        "coverage": coverage(selected),
        "ocr_engine": "tesseract@200" if engine is not None else None,
        "entries": entries,
        "status": "AWAITING_HUMAN_LABELS",
        "note": (
            "Selection and artefacts only. No page here has been validated: the "
            "labels are written by a person into "
            f"{config.VALIDATION_LABELS_TEMPLATE}, and "
            "processing.validation_score scores them afterwards. Nothing in this "
            "file is evidence that the pipeline is correct."
        ),
    }

    atomic_write_text(directory / config.VALIDATION_SELECTION_FILENAME,
                      json.dumps(payload, indent=2, ensure_ascii=False))
    atomic_write_text(directory / config.VALIDATION_REVIEW_FILENAME,
                      render_review_html(payload))
    atomic_write_text(directory / "LABELS.md", render_labels_document())
    labels_path = directory / config.VALIDATION_LABELS_TEMPLATE
    if labels_path.exists():
        # Never overwrite work a reviewer has already done.
        log.warning("%s already exists and was left untouched; the new template "
                    "is at %s.new", labels_path, labels_path.name)
        atomic_write_text(Path(str(labels_path) + ".new"),
                          render_labels_template(entries))
    else:
        atomic_write_text(labels_path, render_labels_template(entries))
    return payload


# --- CLI ------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m processing.validation",
        description=(
            "Build a deterministic ~50-page human validation set from the "
            "existing benchmark, so the quality gates can be scored against "
            "something other than themselves. Selects, renders and writes "
            "artefacts; labels nothing."
        ),
    )
    parser.add_argument("--data-dir", type=Path,
                        default=ingestion_config.DEFAULT_DATA_DIR)
    parser.add_argument("--pages", type=int, default=config.VALIDATION_PAGE_COUNT,
                        help=f"Pages to select (default: "
                             f"{config.VALIDATION_PAGE_COUNT}).")
    parser.add_argument("--no-ocr", action="store_true",
                        help="Skip the OCR reference column.")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rasterising the page images (selection only).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        payload = build(args.data_dir, count=args.pages,
                        with_ocr=not args.no_ocr, render=not args.no_render)
    except (ProcessingError, OSError) as exc:
        log.error("%s", exc)
        return 2

    directory = validation_dir(args.data_dir)
    coverage_report = payload["coverage"]
    print()
    print("=" * 62)
    print("  HUMAN VALIDATION SET — BUILT, NOT VALIDATED")
    print("=" * 62)
    print(f"  pages selected          : {coverage_report['pages']:6d}")
    print(f"  documents represented   : {coverage_report['documents']:6d}")
    print(f"  failure modes covered   : "
          f"{len(MODE_NAMES) - len(coverage_report['modes_with_no_page']):6d}"
          f" / {len(MODE_NAMES)}")
    if coverage_report["modes_with_no_page"]:
        print(f"  modes with NO page      : "
              f"{', '.join(coverage_report['modes_with_no_page'])}")
    print("-" * 62)
    for name in MODE_NAMES:
        print(f"  {name:26s} {coverage_report['by_mode'][name]:4d} page(s)")
    print("=" * 62)
    print()
    print(f"Review sheet : {directory / config.VALIDATION_REVIEW_FILENAME}")
    print(f"Labels       : {directory / 'LABELS.md'}")
    print(f"Fill in      : {directory / config.VALIDATION_LABELS_TEMPLATE}")
    print()
    print("Nothing has been validated yet. Label the pages, then run:")
    print("  python -m processing.validation_score")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
