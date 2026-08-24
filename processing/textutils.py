"""Pure text helpers shared by extraction, furniture detection and parsing.

Everything here is deterministic and free of PDF concepts, which is what lets
the structure parser and the furniture detector be tested on plain strings
rather than on generated PDFs.

One convention runs through the whole package: a page's lines are
``page.text.split("\\n")`` and a *line index* always indexes that list. It is
never a stripped, filtered or re-ordered list — otherwise an index recorded in
``pages.json`` would not point at the line a human sees when reading the page
text, and the audit trail would be worthless.
"""

from __future__ import annotations

import re
import unicodedata

_VOWELS = frozenset("aeiouyAEIOUY")

#: Digit runs are folded so that "Page 12" and "Page 13" compare equal — a page
#: number is furniture precisely because its digits change.
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z]+")

#: Punctuation legitimately dense in legal text; not counted as noise.
#: Deliberately excludes ``~ < > | \ ` ^ @ # $ _``, which legal drafting does not
#: use and which are exactly what a poor OCR layer produces from smudged type.
_LEGAL_PUNCTUATION = set(".,;:()[]{}'\"/-–—―‐’‘“”…§¶&%*!?°× \t\r\n")

#: A page whose only content is a number, a roman numeral, or "Page 3 of 40".
_PAGE_NUMBER_LINE = re.compile(
    r"""^\s*
        (?:page\s*)?                       # optional "Page"
        (?:\d{1,4}|[ivxlcdm]{1,7})         # arabic or roman
        (?:\s*(?:of|/)\s*\d{1,4})?         # "of 40" / "/40"
        \s*[.\-–—]?\s*
        $""",
    re.IGNORECASE | re.VERBOSE,
)


def split_lines(text: str) -> list[str]:
    """Split page text into the canonical line list used package-wide."""
    return text.split("\n")


def normalise_line(line: str, *, fold_digits: bool = True) -> str:
    """Fold a line to a form comparable across pages.

    Case and whitespace runs are always normalised away, and Unicode is
    NFKC-folded so a ligature or a full-width digit from an OCR layer does not
    make an otherwise identical running head look different.

    ``fold_digits`` additionally collapses digit runs to ``#``, which makes
    "Page 12" and "Page 13" compare equal. That is what page-number detection
    needs and what running-head detection must *not* have: with digits folded,
    ordinary body text that happens to contain a number ("in the year 1967")
    starts matching across pages. Callers pick the comparison they mean.
    """
    folded = unicodedata.normalize("NFKC", line)
    folded = _WHITESPACE.sub(" ", folded).strip().casefold()
    return _DIGITS.sub("#", folded) if fold_digits else folded


def is_page_number_line(line: str) -> bool:
    """Whether a line is nothing but a page number."""
    stripped = line.strip()
    if not stripped:
        return False
    return bool(_PAGE_NUMBER_LINE.match(stripped))


def alpha_count(text: str) -> int:
    """Number of alphabetic characters — the measure of "is there text here?".

    Counting letters rather than characters keeps a page of dot leaders, form
    rules or stray punctuation from passing as extracted text.
    """
    return sum(1 for ch in text if ch.isalpha())


def is_blank(text: str) -> bool:
    """Whether a page/line contains nothing but whitespace."""
    return not text.strip()


def quality_signals(text: str) -> dict:
    """Cheap signals that text may be a bad OCR layer rather than real text.

    This is the most consequential thing the benchmark can flag. A scanned
    gazette carrying a poor OCR layer looks *successful* — pages, characters and
    section numbers all come out — but the words are wrong, so a section quoted
    from it would be a fabricated quotation of real law. Two signals, both
    cheap, both language-agnostic enough for legal English:

    ``vowelless_ratio``
        Fraction of longer alphabetic words containing no vowel. Real English
        has almost none; OCR noise ("~t1diO", "tQ ltA ti") is full of them.

    ``symbol_ratio``
        Fraction of characters that are neither alphanumeric, whitespace, nor
        punctuation that legal text actually uses. Mis-recognised glyphs land
        here.

    Returns the raw ratios and counts. Deciding what they *mean* is
    :mod:`processing.extract`'s job, against the thresholds in
    :mod:`processing.config`.
    """
    from . import config

    words = _WORD.findall(text)
    long_words = [w for w in words if len(w) >= config.GARBLE_MIN_WORD_LENGTH]
    vowelless = sum(1 for w in long_words if not (set(w) & _VOWELS))

    total_chars = len(text)
    symbols = sum(
        1 for ch in text
        if not ch.isalnum() and ch not in _LEGAL_PUNCTUATION and not ch.isspace()
    )
    return {
        "word_count": len(words),
        "long_word_count": len(long_words),
        "vowelless_word_count": vowelless,
        "vowelless_ratio": round(vowelless / len(long_words), 4) if long_words else 0.0,
        "symbol_count": symbols,
        "symbol_ratio": round(symbols / total_chars, 4) if total_chars else 0.0,
    }


def union_area(rects, clip: tuple[float, float, float, float]) -> float:
    """Area covered by *rects* within *clip*, counting overlaps once.

    Used to decide whether images cover a page. Overlapping placements are
    common (a scan laid down as several strips), and summing their areas would
    report >100% coverage and make the threshold meaningless.

    Exact, via coordinate compression — the rectangle counts here are small
    (a handful per page), so the O(n^2) grid costs nothing.
    """
    cx0, cy0, cx1, cy1 = clip
    if cx1 <= cx0 or cy1 <= cy0:
        return 0.0

    clipped = []
    for x0, y0, x1, y1 in rects:
        nx0, ny0 = max(min(x0, x1), cx0), max(min(y0, y1), cy0)
        nx1, ny1 = min(max(x0, x1), cx1), min(max(y0, y1), cy1)
        if nx1 > nx0 and ny1 > ny0:
            clipped.append((nx0, ny0, nx1, ny1))
    if not clipped:
        return 0.0

    xs = sorted({v for r in clipped for v in (r[0], r[2])})
    ys = sorted({v for r in clipped for v in (r[1], r[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            for rx0, ry0, rx1, ry1 in clipped:
                if rx0 <= x0 and rx1 >= x1 and ry0 <= y0 and ry1 >= y1:
                    area += (x1 - x0) * (y1 - y0)
                    break
    return area
