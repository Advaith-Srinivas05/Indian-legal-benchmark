"""Extraction-quality assessment: is this text usable as legal evidence?

The first benchmark used a single heuristic — long vowel-free words and stray
symbols — and it caught roughly two of the eight documents a stronger check
identified. The failure was not the threshold but the design: that heuristic
only fires on *catastrophic* noise, while the case that actually matters is text
that reads as plausible at a glance and is wrong in detail::

    Any publiL' SerVi!ll1 or authorilY found guilty, by the Slate COlllmis~i()n
    at lhl: time uf deeiding any eomplaint or ;lppeal…

That is the National Food Security Act, 2013. Pages, characters and section
numbers all come out; nothing looks broken; and a provision quoted from it would
be a fabricated quotation of real law.

So quality is judged by a **panel** of independent signals, each of which
degrades differently, scored together. No single one can condemn a document and
no single one can clear it. The signals fall into four families:

*Shape of the text* — mean word length, share of one-letter words, share of
characters that are letters. Damaged OCR shatters words into fragments, and this
is the family that separates the two populations most cleanly.

*Impossible-looking tokens* — capitals inside a word ("authorilY"), digits inside
a word ("t1diO"), vowel-free long words, symbols legal drafting does not use.
These barely occur in real typesetting at all.

*Vocabulary* — share of tokens that are recognisable English legal words. Real
statutory prose is dense in them.

*Consistency* — how many of the document's pages are individually suspect, and
whether the section numbering that was parsed out of it is plausible. A document
can be sound overall and have three ruined pages; that is worth knowing
separately from a document that is uniformly poor.

On the score
------------
The score is the weighted share of the panel a document passes. It is a compact
summary of the reasons, and it is **not** a calibrated probability: the
thresholds were read off one 100-document sample (healthy documents versus eight
known-bad ones) and would move on a different corpus. The ``reasons`` list is
the substantive output; the number exists to sort by.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import config
from .textutils import quality_signals
from .vocabulary import word_validity
from .wordlists import COMMON_WORDS

#: Runs of letters. Hyphens and apostrophes split rather than join, because a
#: hyphenated compound is two correctly-cased words: without this, the Wildlife
#: Protection Act's schedules of bird names ("White-browed", "Jerdon's") read as
#: 1.5% mis-cased words and a sound Central Act is flagged as damaged.
_WORD = re.compile(r"[A-Za-z]+")
_TOKEN = re.compile(r"\S+")
_HAS_LETTER = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"\d")

#: Each signal's weight in the score. Shape signals carry the most because they
#: separated the two populations most cleanly on the benchmark sample; the
#: vocabulary signal carries less because a legitimate schedule of place names
#: or a tariff table can score low on it while being perfectly extracted.
_WEIGHTS = {
    "mean_word_length": 2.0,
    "single_char_rate": 2.0,
    "mixed_case_rate": 2.0,
    "alnum_mix_rate": 1.5,
    "alpha_ratio": 1.0,
    "common_word_rate": 1.0,
    "symbol_ratio": 1.0,
    "vowelless_ratio": 1.0,
    "page_consistency": 1.0,
    "section_number_plausibility": 0.5,
}


@dataclass
class QualityAssessment:
    """A quality verdict with its score, signals and human-readable reasons."""

    classification: str                       # good | questionable | bad
    score: float
    signals: dict = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    #: True when there was too little text for the ratios to mean anything, so
    #: the verdict reflects insufficiency rather than measured corruption.
    insufficient_text: bool = False

    @property
    def is_usable(self) -> bool:
        return self.classification == "good"

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "score": round(self.score, 4),
            "insufficient_text": self.insufficient_text,
            "failed_checks": self.failed_checks,
            "reasons": self.reasons,
            "signals": self.signals,
            "note": (
                "Heuristic panel score, not a calibrated probability. Thresholds "
                "were read off the 100-document extraction benchmark."
            ),
        }


def text_signals(text: str) -> dict:
    """Measure one block of text. Pure, and the same for a page or a document."""
    words = [w.lower() for w in _WORD.findall(text)]
    raw_words = _WORD.findall(text)
    tokens = _TOKEN.findall(text)
    characters = len(text)
    word_count = len(words)

    mixed_case = sum(
        1 for w in raw_words
        if len(w) >= 4 and not w.isupper() and not w.islower() and not w.istitle()
    )
    alnum_mix = sum(
        1 for t in tokens
        if len(t) >= 4 and _HAS_LETTER.search(t) and _HAS_DIGIT.search(t)
    )
    legacy = quality_signals(text)

    return {
        "characters": characters,
        "word_count": word_count,
        "token_count": len(tokens),
        "mean_word_length": round(
            sum(len(w) for w in words) / word_count, 3) if word_count else 0.0,
        "single_char_rate": round(
            sum(1 for w in words if len(w) == 1) / word_count, 4) if word_count else 0.0,
        "mixed_case_rate": round(mixed_case / word_count, 4) if word_count else 0.0,
        "alnum_mix_rate": round(alnum_mix / len(tokens), 4) if tokens else 0.0,
        "alpha_ratio": round(
            sum(1 for c in text if c.isalpha()) / characters, 4) if characters else 0.0,
        "printable_ratio": round(
            sum(1 for c in text if c.isprintable() or c.isspace()) / characters, 4
        ) if characters else 0.0,
        "replacement_char_count": text.count("�"),
        "common_word_rate": round(
            sum(1 for w in words if w in COMMON_WORDS) / word_count, 4
        ) if word_count else 0.0,
        "symbol_ratio": legacy["symbol_ratio"],
        "vowelless_ratio": legacy["vowelless_ratio"],
    }


def page_is_suspect(text: str) -> bool:
    """Whether one page's text fails the shape checks on its own.

    Used for the consistency signal and for page-level indexability, and only on
    pages with enough text for the ratios to mean anything.

    Note the asymmetry: a page too short to judge is **not** suspect, because
    this answers "is there evidence against this page", not "is this page
    sound". :func:`page_is_judgeable` is the question to ask before trusting a
    ``False``, and :func:`indexable_pages` asks it.
    """
    signals = text_signals(text)
    if signals["word_count"] < config.QUALITY_PAGE_MIN_WORDS:
        return False
    return (
        signals["mean_word_length"] < config.QUALITY_MEAN_WORD_LENGTH_MIN
        or signals["single_char_rate"] > config.QUALITY_SINGLE_CHAR_RATE_MAX
        or signals["mixed_case_rate"] > config.QUALITY_MIXED_CASE_RATE_MAX
    )


def page_is_judgeable(text: str) -> bool:
    """Whether this page carries enough English words to judge its quality."""
    return len(_WORD.findall(text)) >= config.QUALITY_PAGE_MIN_WORDS


def indexable_pages(pages: Iterable, assessment, *, page_text=None) -> list:
    """The pages whose text passes the page-level checks.

    **Not wired to eligibility, deliberately.** ``process_document`` records each
    page's verdict and does not use this to decide indexability. Read
    :attr:`processing.models.ProcessedDocument.eligible_for_indexing` before
    reaching for it.

    It was built to salvage the sound pages of a document quarantined on quality
    -- 36% of the judgeable English pages in the pilot's quarantined set pass
    these checks -- and reverted when the pages it admits turned out to be
    corrupted in ways the checks do not see: ``thereil``, ``fixecj``,
    ``concerngd``, ``recognuon``. These three checks were designed as one
    contributor to a weighted document-level panel, where the aggregate and the
    page-consistency ratio do the discriminating. Promoted to an admission gate
    they admit text that is not quotable as law.

    Measured against pages from documents that were never quarantined, the
    admitted pages are worse on every discriminating signal, and the clearest of
    them -- English common-word rate, median 0.54 against 0.63 -- overlaps so
    heavily that no threshold separates the populations: 0.50 rejects a third of
    the admitted pages and an eighth of the trusted ones.

    Kept because the per-page verdict is worth recording, and because a real
    word-validity signal -- which needs a dictionary this project does not carry,
    and labels it has declined to produce -- would slot in exactly here.

    Three cases:

    *Judgeable and sound* — indexable.

    *Judgeable and suspect* — not indexable, whatever the document's average.

    *Too short to judge* — a cover, a part-title, a page of signatures. Indexable
    only if the document's judged pages came out ``good``. An unjudgeable page
    carries no evidence of its own, so it inherits the company it keeps rather
    than being waved through: the alternative admits the empty pages of a
    document that is unreadable everywhere it can be read.

    A document whose quality could not be established at all -- no judgeable page
    anywhere -- has no indexable pages. Nothing is established about it, and
    ``insufficient_text`` says exactly that.

    *page_text* extracts the text to judge from a page; it defaults to the page's
    selected reading. :mod:`processing.process` passes the English-only lines,
    for the same reason :func:`assess` is given them: a translation left in the
    pool reads as extraction damage.
    """
    if page_text is None:
        def page_text(page):
            return getattr(page, "selected_text", None) or getattr(page, "text", "") or ""

    page_list = list(pages)
    if getattr(assessment, "insufficient_text", False):
        return []

    document_is_good = getattr(assessment, "classification", "") == "good"
    indexable = []
    for page in page_list:
        text = page_text(page)
        if not page_is_judgeable(text):
            if document_is_good:
                indexable.append(page)
            continue
        if not page_is_suspect(text):
            indexable.append(page)
    return indexable


def page_verdict(text: str) -> dict:
    """This page's own quality verdict, for the page record.

    ``good`` / ``suspect`` / ``unjudged`` — never a score. The score is a
    document-level summary of a weighted panel; a page runs the three shape
    checks and nothing else, and reporting a number here would imply the two are
    comparable.
    """
    if not page_is_judgeable(text):
        return {
            "verdict": "unjudged",
            "reason": (
                f"fewer than {config.QUALITY_PAGE_MIN_WORDS} English words: too "
                "little text to judge, which is not the same as bad text"
            ),
        }
    if not page_is_suspect(text):
        return {"verdict": "good", "reason": "the page-level shape checks pass"}
    signals = text_signals(text)
    failed = []
    if signals["mean_word_length"] < config.QUALITY_MEAN_WORD_LENGTH_MIN:
        failed.append(f"mean word length {signals['mean_word_length']}")
    if signals["single_char_rate"] > config.QUALITY_SINGLE_CHAR_RATE_MAX:
        failed.append(f"{signals['single_char_rate']:.1%} single-letter words")
    if signals["mixed_case_rate"] > config.QUALITY_MIXED_CASE_RATE_MAX:
        failed.append(f"{signals['mixed_case_rate']:.2%} words with capitals inside")
    return {"verdict": "suspect", "reason": "; ".join(failed)}


def section_number_plausibility(structure) -> Optional[float]:
    """How orderly the parsed section numbering is, if any was parsed.

    Badly OCR'd digits produce sections numbered 8, 3, 77, 1 — the numbering is
    an independent witness to text quality, and one that does not depend on the
    words at all. Returns ``None`` when the document has too few sections for
    the measure to say anything, so it can be dropped from the panel rather than
    counted as a failure.
    """
    if structure is None:
        return None
    numbers = []
    for unit in structure.all_units():
        if unit.unit_type not in ("section", "article") or not unit.number:
            continue
        match = re.match(r"(\d{1,4})", unit.number)
        if match:
            numbers.append(int(match.group(1)))
    if len(numbers) < 5:
        return None
    ascending = sum(1 for a, b in zip(numbers, numbers[1:]) if b >= a)
    return ascending / (len(numbers) - 1)


def assess(
    pages: Iterable,
    *,
    structure=None,
    text: Optional[str] = None,
) -> QualityAssessment:
    """Assess the extraction quality of one document.

    Accepts the extracted pages (so page-level consistency can be measured) and
    optionally the parsed structure (for the section-numbering signal). *text*
    overrides the pages, for testing a block of text directly.
    """
    page_list = list(pages)
    if text is None:
        text = "\n".join(getattr(p, "text", "") or "" for p in page_list)

    signals = text_signals(text)
    # Recorded, never scored. The panel's checks measure word *shape*, and this
    # measures whether the words are words -- the signal DECISIONS D21 said was
    # missing. It deliberately contributes nothing to `score` or `classification`:
    # imposing it corpus-wide would quarantine a quarter of the documents already
    # accepted (25th percentile 0.937). One caller consults it, on the one path
    # where a document is admitted on unreviewed OCR. See config and D28.
    signals["word_validity"] = word_validity(text)
    checks: list[tuple[str, bool, str]] = []       # (name, passed, description)

    def check(name: str, passed: bool, description: str) -> None:
        checks.append((name, passed, description))

    if signals["word_count"] < config.QUALITY_MIN_WORDS:
        signals["page_count"] = len(page_list)
        return QualityAssessment(
            classification="questionable",
            score=0.0,
            signals=signals,
            failed_checks=["insufficient_text"],
            reasons=[
                f"only {signals['word_count']} words extracted (need "
                f"{config.QUALITY_MIN_WORDS} to judge); quality cannot be "
                "established, which is not the same as the text being bad"
            ],
            insufficient_text=True,
        )

    check("mean_word_length",
          signals["mean_word_length"] >= config.QUALITY_MEAN_WORD_LENGTH_MIN,
          f"mean word length {signals['mean_word_length']} "
          f"(healthy >= {config.QUALITY_MEAN_WORD_LENGTH_MIN})")
    check("single_char_rate",
          signals["single_char_rate"] <= config.QUALITY_SINGLE_CHAR_RATE_MAX,
          f"{signals['single_char_rate']:.1%} of words are a single letter "
          f"(healthy <= {config.QUALITY_SINGLE_CHAR_RATE_MAX:.0%})")
    check("mixed_case_rate",
          signals["mixed_case_rate"] <= config.QUALITY_MIXED_CASE_RATE_MAX,
          f"{signals['mixed_case_rate']:.2%} of words have capitals inside them "
          f"(healthy <= {config.QUALITY_MIXED_CASE_RATE_MAX:.1%})")
    check("alnum_mix_rate",
          signals["alnum_mix_rate"] <= config.QUALITY_ALNUM_MIX_RATE_MAX,
          f"{signals['alnum_mix_rate']:.1%} of tokens mix letters and digits "
          f"(healthy <= {config.QUALITY_ALNUM_MIX_RATE_MAX:.1%})")
    check("alpha_ratio",
          signals["alpha_ratio"] >= config.QUALITY_ALPHA_RATIO_MIN,
          f"{signals['alpha_ratio']:.1%} of characters are letters "
          f"(healthy >= {config.QUALITY_ALPHA_RATIO_MIN:.0%})")
    check("common_word_rate",
          signals["common_word_rate"] >= config.QUALITY_COMMON_WORD_RATE_MIN,
          f"{signals['common_word_rate']:.1%} of words are recognisable English "
          f"legal vocabulary (healthy >= {config.QUALITY_COMMON_WORD_RATE_MIN:.0%})")
    check("symbol_ratio",
          signals["symbol_ratio"] <= config.QUALITY_SYMBOL_RATIO_MAX,
          f"{signals['symbol_ratio']:.1%} of characters are symbols legal text "
          f"does not use (healthy <= {config.QUALITY_SYMBOL_RATIO_MAX:.0%})")
    check("vowelless_ratio",
          signals["vowelless_ratio"] <= config.QUALITY_VOWELLESS_RATIO_MAX,
          f"{signals['vowelless_ratio']:.1%} of long words have no vowel "
          f"(healthy <= {config.QUALITY_VOWELLESS_RATIO_MAX:.0%})")

    suspect_pages = [p for p in page_list if page_is_suspect(getattr(p, "text", "") or "")]
    measurable = [
        p for p in page_list
        if len(_WORD.findall(getattr(p, "text", "") or "")) >= 60
    ]
    suspect_ratio = len(suspect_pages) / len(measurable) if measurable else 0.0
    signals["suspect_page_count"] = len(suspect_pages)
    signals["measurable_page_count"] = len(measurable)
    signals["suspect_page_ratio"] = round(suspect_ratio, 4)
    if measurable:
        check("page_consistency",
              suspect_ratio <= config.QUALITY_SUSPECT_PAGE_RATIO,
              f"{len(suspect_pages)} of {len(measurable)} measurable pages fail "
              f"the shape checks individually")

    plausibility = section_number_plausibility(structure)
    if plausibility is not None:
        signals["section_number_ascending_ratio"] = round(plausibility, 4)
        check("section_number_plausibility", plausibility >= 0.6,
              f"only {plausibility:.0%} of consecutive parsed section numbers "
              "ascend, which is what mis-read digits look like")

    if signals["replacement_char_count"]:
        check("replacement_characters", False,
              f"{signals['replacement_char_count']} Unicode replacement "
              "characters in the text")
    if signals["printable_ratio"] < 0.99:
        check("printable_ratio", False,
              f"{1 - signals['printable_ratio']:.1%} of characters are "
              "non-printable")

    total_weight = sum(_WEIGHTS.get(name, 1.0) for name, _, _ in checks)
    passed_weight = sum(
        _WEIGHTS.get(name, 1.0) for name, passed, _ in checks if passed)
    score = passed_weight / total_weight if total_weight else 0.0

    failed = [name for name, passed, _ in checks if not passed]
    reasons = [description for _, passed, description in checks if not passed]
    if not failed:
        reasons = ["every extraction-quality check passed"]

    if score >= config.QUALITY_GOOD_SCORE:
        classification = "good"
    elif score >= config.QUALITY_QUESTIONABLE_SCORE:
        classification = "questionable"
    else:
        classification = "bad"

    # Two shape failures together are decisive whatever the weighted score says:
    # text whose words are both fragmented and mis-cased is not recoverable by
    # any downstream stage.
    decisive = {"mean_word_length", "single_char_rate", "mixed_case_rate"}
    if len(decisive & set(failed)) >= 2 and classification == "good":
        classification = "questionable"
    if len(decisive & set(failed)) >= 3:
        classification = "bad"

    signals["page_count"] = len(page_list)
    signals["thresholds"] = {
        "mean_word_length_min": config.QUALITY_MEAN_WORD_LENGTH_MIN,
        "single_char_rate_max": config.QUALITY_SINGLE_CHAR_RATE_MAX,
        "mixed_case_rate_max": config.QUALITY_MIXED_CASE_RATE_MAX,
        "alnum_mix_rate_max": config.QUALITY_ALNUM_MIX_RATE_MAX,
        "common_word_rate_min": config.QUALITY_COMMON_WORD_RATE_MIN,
        "alpha_ratio_min": config.QUALITY_ALPHA_RATIO_MIN,
        "good_score": config.QUALITY_GOOD_SCORE,
        "questionable_score": config.QUALITY_QUESTIONABLE_SCORE,
    }
    return QualityAssessment(
        classification=classification,
        score=score,
        signals=signals,
        failed_checks=failed,
        reasons=reasons,
    )


def worst_pages(pages: Iterable, limit: int = 5) -> list[int]:
    """Page numbers of the most suspect pages, for a reviewer to look at first."""
    scored = []
    for page in pages:
        text = getattr(page, "text", "") or ""
        words = _WORD.findall(text)
        if len(words) < 60:
            continue
        signals = text_signals(text)
        scored.append((signals["mean_word_length"], getattr(page, "page_number", 0)))
    scored.sort()
    return [number for _, number in scored[:limit]]


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0
