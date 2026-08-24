"""Content-level language validation.

The project spec makes the corpus English-only, and the ingestion phase enforces that
from India Code's own metadata. The first extraction benchmark showed that is
necessary but **not sufficient**: two documents served under ``Files(Eng)``
turned out to be a Devanagari rules notification and a mirrored Gujarati scan.
Ingestion was not wrong — it applied its rules to the evidence it had — but the
evidence was the publisher's label, not the document.

So a document is eligible for downstream legal indexing only when *both* hold:

* India Code's metadata identifies it as English (established in Phase 1), and
* its extracted content independently supports that (established here).

What this is not
----------------
It is not a "contains Devanagari" rule. That rule fails in both directions:

* both offending documents OCR'd into **Latin** glyphs and contain no
  Devanagari at all, so the rule would clear them; and
* perfectly good English acts carry Devanagari when they quote a Hindi title —
  the Passports Act's metadata does exactly this — so the rule would reject
  them.

The discriminator is whether the **readable** words are English. Function words
are used for that because they survive OCR damage better than anything else in
the language: short, high-frequency, and largely absent from a transliteration
of another language. Tokens too damaged to be words at all are excluded from the
denominator, so English under heavy scan damage still scores well above a
transliteration — which is the distinction that matters, since corrupted English
is a *quality* problem, not a language one.

Three outcomes, and ``uncertain`` is a real answer rather than a hedge: a
two-line document, or one whose text is too broken to read, genuinely does not
establish its language, and quarantining it is correct. Only ``en`` is eligible
for indexing.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

from . import config
from .wordlists import FUNCTION_WORDS

#: Runs of letters. Hyphens and apostrophes split a token rather than joining
#: it, because a hyphenated compound is two correctly-cased words —
#: "White-browed" is not evidence of anything wrong, and treating it as one
#: token makes it look mis-cased.
_WORD = re.compile(r"[A-Za-z]+")

_SCRIPT_PREFIXES = (
    "DEVANAGARI", "BENGALI", "GURMUKHI", "GUJARATI", "ORIYA", "TAMIL",
    "TELUGU", "KANNADA", "MALAYALAM", "ARABIC", "CYRILLIC", "GREEK",
    "CJK", "HIRAGANA", "KATAKANA", "HANGUL", "HEBREW", "THAI", "SINHALA",
)


@dataclass
class LanguageAssessment:
    """The content-level language verdict, with the evidence behind it."""

    content_language: str                     # en | non_en | uncertain
    eligible_for_indexing: bool
    signals: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "content_language": self.content_language,
            "eligible_for_indexing": self.eligible_for_indexing,
            "signals": self.signals,
            "reasons": self.reasons,
        }


def script_profile(text: str) -> dict:
    """Distribution of the text's *letters* across writing systems.

    Letters only: punctuation and digits are shared between scripts and would
    dilute the ratio, and it is the ratio that has to survive an English act
    quoting a Devanagari title without being called Hindi.
    """
    counts: dict[str, int] = {}
    total = 0
    for character in text:
        if not character.isalpha():
            continue
        total += 1
        try:
            name = unicodedata.name(character)
        except ValueError:
            counts["UNKNOWN"] = counts.get("UNKNOWN", 0) + 1
            continue
        if name.startswith("LATIN"):
            counts["LATIN"] = counts.get("LATIN", 0) + 1
            continue
        for prefix in _SCRIPT_PREFIXES:
            if name.startswith(prefix):
                counts[prefix] = counts.get(prefix, 0) + 1
                break
        else:
            counts["OTHER"] = counts.get("OTHER", 0) + 1
    return {"letters": total, "by_script": counts}


def clean_words(text: str) -> list[str]:
    """Tokens readable enough to be evidence about language.

    A token qualifies when it is at least two letters and consistently cased —
    all lower, all upper, or capitalised. That admits ordinary prose, proper
    nouns and Latin abbreviations set in capitals ("SUO MOTU", "ULTRA VIRES"),
    and excludes the mid-word capitals and stray glyphs OCR damage produces.
    """
    words = []
    for match in _WORD.finditer(text):
        token = match.group()
        if len(token) < 2:
            continue
        if token.islower() or token.isupper() or token.istitle():
            words.append(token.lower())
    return words


def assess_text(text: str) -> LanguageAssessment:
    """Judge the language of one block of extracted text."""
    profile = script_profile(text)
    letters = profile["letters"]
    by_script = profile["by_script"]
    latin = by_script.get("LATIN", 0)
    non_latin = letters - latin - by_script.get("UNKNOWN", 0)
    non_latin_ratio = (non_latin / letters) if letters else 0.0

    words = clean_words(text)
    function_hits = [w for w in words if w in FUNCTION_WORDS]
    rate = (len(function_hits) / len(words)) if words else 0.0
    distinct = len(set(function_hits))

    dominant_other = max(
        ((name, count) for name, count in by_script.items()
         if name not in ("LATIN", "UNKNOWN")),
        key=lambda item: item[1],
        default=None,
    )

    signals = {
        "letters": letters,
        "latin_letter_ratio": round(latin / letters, 4) if letters else 0.0,
        "non_latin_letter_ratio": round(non_latin_ratio, 4),
        "scripts": by_script,
        "clean_word_count": len(words),
        "function_word_rate": round(rate, 4),
        "distinct_function_words": distinct,
        "thresholds": {
            "min_clean_words": config.LANGUAGE_MIN_CLEAN_WORDS,
            "english_rate": config.LANGUAGE_ENGLISH_RATE,
            "non_english_rate": config.LANGUAGE_NON_ENGLISH_RATE,
            "min_distinct_function_words": config.LANGUAGE_MIN_DISTINCT_FUNCTION_WORDS,
            "non_latin_letter_ratio": config.LANGUAGE_NON_LATIN_LETTER_RATIO,
        },
    }
    reasons: list[str] = []

    # A script other than Latin carrying the bulk of the letters settles it
    # outright, and names the script so the call can be checked.
    if non_latin_ratio >= config.LANGUAGE_NON_LATIN_LETTER_RATIO and letters >= 200:
        script = dominant_other[0].lower() if dominant_other else "non-Latin"
        reasons.append(
            f"{non_latin_ratio:.0%} of letters are {script}, above the "
            f"{config.LANGUAGE_NON_LATIN_LETTER_RATIO:.0%} limit"
        )
        return LanguageAssessment("non_en", False, signals, reasons)

    decisive_short = (
        len(words) >= config.LANGUAGE_SHORT_MIN_CLEAN_WORDS
        and rate >= config.LANGUAGE_SHORT_ENGLISH_RATE
        and distinct >= config.LANGUAGE_MIN_DISTINCT_FUNCTION_WORDS
    )
    if len(words) < config.LANGUAGE_MIN_CLEAN_WORDS and not decisive_short:
        reasons.append(
            f"only {len(words)} readable words (need "
            f"{config.LANGUAGE_MIN_CLEAN_WORDS}, or "
            f"{config.LANGUAGE_SHORT_MIN_CLEAN_WORDS} with clearly English "
            "wording); the language cannot be established from this text"
        )
        return LanguageAssessment("uncertain", False, signals, reasons)
    if decisive_short and len(words) < config.LANGUAGE_MIN_CLEAN_WORDS:
        # A one-page repeal act is short by nature, not doubtful. It can still
        # settle the question — but only on clearly stronger evidence than a
        # long document needs, which is what the higher rate demands.
        reasons.append(
            f"short document ({len(words)} readable words) but decisively "
            f"English: {rate:.1%} function words, {distinct} distinct"
        )
        return LanguageAssessment("en", True, signals, reasons)

    if rate <= config.LANGUAGE_NON_ENGLISH_RATE:
        reasons.append(
            f"only {rate:.1%} of readable words are English function words "
            f"(at or below the {config.LANGUAGE_NON_ENGLISH_RATE:.0%} floor); "
            "the substantive text is not English"
        )
        return LanguageAssessment("non_en", False, signals, reasons)

    if rate >= config.LANGUAGE_ENGLISH_RATE and \
            distinct >= config.LANGUAGE_MIN_DISTINCT_FUNCTION_WORDS:
        reasons.append(
            f"{rate:.1%} of readable words are English function words "
            f"({distinct} distinct), over {len(words)} readable words"
        )
        if non_latin_ratio > 0:
            reasons.append(
                f"{non_latin_ratio:.1%} of letters are non-Latin, which is "
                "consistent with quoted titles rather than non-English text"
            )
        return LanguageAssessment("en", True, signals, reasons)

    reasons.append(
        f"English function words are {rate:.1%} of readable words with "
        f"{distinct} distinct — between the {config.LANGUAGE_NON_ENGLISH_RATE:.0%} "
        f"and {config.LANGUAGE_ENGLISH_RATE:.0%} bounds, so the text neither "
        "confirms nor rules out English"
    )
    return LanguageAssessment("uncertain", False, signals, reasons)


def page_language_profile(pages: Iterable) -> dict:
    """Count the pages written in a script other than Latin.

    A pooled document average hides a document that is part English and part
    not: the non-Latin letter ratio over a whole act is diluted by every English
    page in it, so a substantial Hindi section can sit under the document-level
    threshold and the act is cleared as English.

    **Only script is used here, and that is a deliberate narrowing.** The obvious
    alternative — running the full per-page assessment and counting the pages
    that come back ``non_en`` — was tried and is wrong. It reported 33 of the 222
    pages of the Wild Life (Protection) Act, 1972 as not English, because those
    pages are the schedules of protected species::

        43.  Leopard              Panthera pardus
        44.  Leopard Cat          Prionailurus bengalensis

    Two columns of names, no sentences, an English function-word rate of exactly
    zero — and unambiguously part of an English statute. Tariff schedules and
    lists of place names fail the same way. A list carries no evidence about
    language, and a rule that quarantines a Central Act for containing one is
    worse than no rule.

    Script does not have that failure mode: a page of Latin species names is
    Latin, and a page of Devanagari is not.

    What this therefore does **not** catch is another language transliterated
    into Latin glyphs, or a Devanagari page behind a broken character map that
    extracts as Latin nonsense. Those are real and they are covered elsewhere —
    the first by the document-level function-word check, the second by
    :func:`processing.orientation.flag_direction_inconsistency`, which is what
    finds the 64 Devanagari pages in the Madhya Pradesh Goods and Services Tax
    Act, 2017.
    """
    measurable = 0
    non_english: list[int] = []
    for page in pages:
        text = getattr(page, "text", "") or ""
        profile = script_profile(text)
        letters = profile["letters"]
        if letters < config.LANGUAGE_PAGE_MIN_LETTERS:
            continue                        # a cover or a part-title page
        measurable += 1
        latin = profile["by_script"].get("LATIN", 0)
        unknown = profile["by_script"].get("UNKNOWN", 0)
        if (letters - latin - unknown) / letters >= \
                config.LANGUAGE_NON_LATIN_LETTER_RATIO:
            non_english.append(getattr(page, "page_number", 0))
    ratio = len(non_english) / measurable if measurable else 0.0
    return {
        "measurable_pages": measurable,
        "non_english_pages": len(non_english),
        "non_english_page_numbers": non_english[:50],
        "non_english_page_ratio": round(ratio, 4),
        "mixed_language": ratio >= config.LANGUAGE_MIXED_NON_EN_PAGE_RATIO,
        "basis": "non_latin_script",
        "thresholds": {
            "page_min_letters": config.LANGUAGE_PAGE_MIN_LETTERS,
            "non_latin_letter_ratio": config.LANGUAGE_NON_LATIN_LETTER_RATIO,
            "mixed_non_en_page_ratio": config.LANGUAGE_MIXED_NON_EN_PAGE_RATIO,
        },
    }


def assess_pages(pages: Iterable, *, metadata_language: str | None = None) -> LanguageAssessment:
    """Judge a whole document from its extracted pages.

    *metadata_language* is India Code's own determination, carried forward from
    ingestion. It is recorded and it gates eligibility — a document India Code
    does not call English is never eligible whatever its content looks like —
    but it can no longer clear a document on its own.
    """
    page_list = list(pages)
    text = "\n".join(getattr(page, "text", "") or "" for page in page_list)
    assessment = assess_text(text)
    assessment.signals["metadata_language"] = metadata_language

    profile = page_language_profile(page_list)
    assessment.signals["page_language"] = profile
    if profile["mixed_language"] and assessment.content_language == "en":
        # Only a document the pooled check *cleared* can be downgraded here. One
        # already found non-English stays non-English: "mixed" would be a weaker
        # and less accurate statement about it, and the page profile of a
        # wholly non-English document trivially satisfies this rule.
        #
        # The downgrade is to ``uncertain`` rather than to ``non_en``, because
        # the document *is* substantially English and calling it another language
        # would be as wrong as calling it English. What is true is that its
        # language is not uniform — so it is not established English, and under
        # the project spec, unestablished means quarantined.
        assessment.content_language = "uncertain"
        assessment.eligible_for_indexing = False
        assessment.reasons.append(
            f"{profile['non_english_pages']} of {profile['measurable_pages']} "
            f"measurable pages ({profile['non_english_page_ratio']:.0%}) are "
            "decisively not English, so this document is mixed-language even "
            "though its text pooled together reads as English"
        )
    if metadata_language and metadata_language != "en":
        assessment.eligible_for_indexing = False
        assessment.reasons.append(
            f"India Code metadata records this document as {metadata_language!r}, "
            "not English"
        )
    elif not metadata_language:
        assessment.eligible_for_indexing = False
        assessment.reasons.append(
            "no India Code language determination is recorded for this document"
        )
    return assessment
