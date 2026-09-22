"""Word-level helpers shared by the question checks.

Every check that compares a question with its evidence must agree on what a
"word" is, or a question is rejected on one notion and written on another.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

#: Words that carry no retrieval signal. Measured against this corpus when it
#: served a keyword index: dropping them took one query from 2,345 ms to 61 ms
#: without changing its results.
STOPWORDS = frozenset(
    "a and any as be by for in is may of on or shall such the to under".split())

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_SPACE = re.compile(r"\s+")

#: A quantity is a *fact* only with a unit. A bare number matches the provision's
#: own number, which any answer satisfies by citing it.
QUANTITY = re.compile(
    r"\b(?:\d[\d,]*(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|fifteen|twenty|thirty|forty|fifty|sixty|ninety|hundred|thousand)"
    r"\s*(?:per\s*cent|percent|%|rupees|rs\.?|years?|months?|weeks?|days?|hours?"
    r"|lakhs?|crores?|kilograms?|kg|metres?|litres?)\b",
    re.IGNORECASE)


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text or "")]


def content_words(text: str) -> list[str]:
    """Tokens that carry signal: no stopwords, nothing one character long."""
    return [w for w in tokens(text) if len(w) > 1 and w not in STOPWORDS]


def containment(query_terms: Iterable[str], source_terms: Iterable[str]) -> float:
    """Share of *query_terms* that also appear in *source_terms*.

    Asymmetric on purpose: a short question lifted word for word out of a long
    provision has a Jaccard similarity near zero with it and containment 1.0.
    """
    query = set(query_terms)
    if not query:
        return 0.0
    return len(query & set(source_terms)) / len(query)


def leakage(question: str, evidence_text: str) -> float:
    """How much of a question is made of its evidence's own words."""
    return containment(content_words(question), content_words(evidence_text))


def normalise(text: str) -> str:
    """Case-folded, Unicode-folded, whitespace-collapsed — for "does X appear in Y"."""
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text or "")).strip().casefold()


def appears_in(fact: str, text: str) -> bool:
    """Whether *fact* occurs in *text* as a whole phrase, ignoring case and wrapping."""
    needle = normalise(fact)
    if not needle:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", normalise(text)) is not None
