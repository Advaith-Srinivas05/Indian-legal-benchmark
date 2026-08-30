"""Whether a document's words are words, and where the reference list came from.

The signal this project did not have
------------------------------------
``docs/DECISIONS.md`` D21 recorded why page-level extraction quality was built
and reverted: the page-level shape checks admit text that is visibly damaged
(``"thereil"``, ``"concerngd"``, ``"recognuon"``) because they measure *shape* --
stray capitals, single-letter words, symbol rates -- and damaged English keeps
the shape of English. The note ends by naming what would separate the two
populations: **a word-validity signal, which COMMON_WORDS (307 words) is not.**

This module is that signal. It answers one question -- what share of a text's
words are real words -- and it is deliberately nothing more.

Where the reference list comes from
-----------------------------------
There is no English dictionary in this project's dependencies and the test suite
must run offline, so the list is derived from the corpus itself, from the subset
of it that **cannot** contain OCR damage: documents whose PDF carried a real text
layer (``pdf_type == "text_based"``), which are eligible, quality ``good``,
English, and which contributed **zero** OCR pages. A word earns its place by
appearing in at least 20 of 3,000 such documents.

That threshold is what makes the list trustworthy. Extraction damage is
idiosyncratic -- ``"Clticf"``, ``"Cheptcr"`` and ``"Kotkeiiti"`` are artefacts of
one scan of one book -- so a damaged word cannot reach twenty independent
born-digital documents. Genuine legal vocabulary, including Indian legal and
place-name vocabulary that no English dictionary would carry (``aadhaar``,
``panchayat``, ``zilla``, ``ryotwari``), reaches it easily. A general dictionary
would have been worse at this job, not better.

What it is measured on
----------------------
Words of four characters or more, lowercased. Shorter tokens are excluded
because they are dominated by section numbers, list markers and initials, where
being "not a word" means nothing.

What it must not become
-----------------------
A per-page gate. That is the mistake D21 already recorded: quality is judged
over a document because a page is too small a sample to judge. This is a
document-level measurement and the thresholds in :mod:`processing.config` are
calibrated as one.

It is also not a language test. A page of Hindi scores near zero here, but
:func:`processing.language.mangled_script` is what should catch that, and it
reports *why*. A low validity rate means "these are not words", not "this is not
English".
"""

from __future__ import annotations

import functools
import re
from pathlib import Path

#: Tokens short enough that being absent from the list means nothing.
MIN_WORD_LENGTH = 4

_WORD = re.compile(r"[A-Za-z]{%d,}" % MIN_WORD_LENGTH)

VOCABULARY_PATH = Path(__file__).parent / "data" / "english_legal_vocabulary.txt"


@functools.lru_cache(maxsize=1)
def vocabulary() -> frozenset[str]:
    """The reference word list, read once and cached."""
    words = set()
    with VOCABULARY_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    return frozenset(words)


def word_validity(text: str) -> dict:
    """Share of *text*'s long words that appear in the reference vocabulary.

    Returns the rate, the counts behind it, and whether there was enough text to
    measure at all. ``measurable`` is ``False`` below
    :data:`processing.config.QUALITY_WORD_VALIDITY_MIN_WORDS`, and a caller must
    treat that as "unknown" rather than as a pass or a fail -- the project fails
    conservatively when evidence is ambiguous.
    """
    from . import config

    words = [w.lower() for w in _WORD.findall(text)]
    known = sum(1 for w in words if w in vocabulary())
    measurable = len(words) >= config.QUALITY_WORD_VALIDITY_MIN_WORDS
    return {
        "rate": round(known / len(words), 4) if words else 0.0,
        "words": len(words),
        "known_words": known,
        "measurable": measurable,
    }
