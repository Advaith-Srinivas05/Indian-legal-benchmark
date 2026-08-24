"""Repairing lost word spacing in OCR output.

Why this exists
---------------
The first engine comparison measured the share of characters that are spaces,
and RapidOCR came out at 10.5% against 15.5% for the text layer already in the
PDFs. That gap is not cosmetic. PP-OCR's recogniser reads one *line* crop at a
time with a CTC head that is not trained to emit a space symbol reliably, so
words at the end of a confident run get glued together::

    the power to revise the valuationandassessmentconferred
    bysection65of theMunicipalActand thepowertoamend

Every word in that passage was read correctly. What was lost is the token
boundaries — and tokens are the unit both BM25 and a dense embedder work in, so
a document in this state is close to unretrievable by the words it contains.
On the 25 evaluation pages RapidOCR produced 6.3% tokens of 12+ characters
against 2.6% for the embedded layer and 2.1% for EasyOCR: the defect is
specific to this engine, not to OCR in general.

What this module does about it
------------------------------
It inserts spaces. It never deletes a character, never substitutes one, and
never reorders anything, so the repair is auditable against the raw OCR output
and cannot fabricate a word that the engine did not read.

Three rules, applied in order, each individually conservative:

1. **Case boundaries.** ``theMunicipalAct`` -> ``the Municipal Act``. A lowercase
   letter followed by an uppercase one does not occur inside a word in statutory
   English; it is a lost space essentially every time.

2. **Digit/letter boundaries.** ``section65of`` -> ``section 65 of``. Guarded so
   that section numbering survives: a letter run must be at least
   :data:`MIN_LETTERS_AROUND_DIGITS` long before it may be split off a digit run,
   which keeps ``65A``, ``1st`` and ``2nd`` intact.

3. **Lexicon segmentation.** ``thepowertoamend`` -> ``the power to amend``, over
   the same word lists the language and quality checks use
   (:mod:`processing.wordlists`). The lexicon is 307 words and English is not, so
   a covering may leave one run unrecognised — but only at an end, never in the
   middle, and never as the only thing holding the covering together. See
   :func:`_segment` for why those constraints are the whole design: without
   them a small lexicon turns ``permission`` into ``per missi on``, which is
   worse than leaving the words glued.

The recall this gives up is deliberate. ``valuationandassessmentconferred``
stays glued because nothing in the lexicon can account for enough of it.
Leaving a token unsplit costs a retrieval opportunity; splitting one wrongly
puts a word into the corpus that was never on the page, and this is a corpus
whose whole purpose is to be quotable.

Nothing here is applied to text extracted from a PDF's own text layer. It runs
on OCR output only, and :func:`respace` reports what it changed so a run can be
measured rather than trusted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .wordlists import COMMON_WORDS

#: Tokens shorter than this are left alone. A short unknown token is far more
#: likely to be an abbreviation, a name or a citation fragment than two words
#: run together, and splitting it would cost more than the space it recovers.
GLUED_MIN_LENGTH = 8

#: A segment the lexicon does not know must be at least this long to be believed.
UNKNOWN_MIN_LENGTH = 4

#: Letters that must sit beside a digit run before the boundary may be split.
#: Three keeps ``65A``, ``1st``, ``2nd``, ``3rd`` and ``4th`` whole while still
#: splitting ``section65`` and ``clause3``. A shorter run is split anyway when it
#: is itself a lexicon word, which is what recovers the ``of`` in ``65of``.
MIN_LETTERS_AROUND_DIGITS = 3

#: Known words a covering must contain before its boundaries are believed. Two,
#: so that a single function word can never be peeled off a real word that
#: happens to start with one: ``information`` -> ``in formation``,
#: ``therefore`` -> ``the refore``, ``withholdwritten`` -> ``with holdwritten``
#: are all rejected by this rule alone.
MIN_KNOWN_SEGMENTS = 2

#: Length a token must reach before a covering that leaves part of it
#: unrecognised is considered at all. Below it, only a covering made entirely of
#: known words is admitted — which is what keeps ordinary long words such as
#: ``whosoever`` out of the segmenter's reach.
PARTIAL_MIN_LENGTH = 12

#: Share of a glued token's characters that a covering with an unrecognised run
#: must account for with words the lexicon actually knows.
#:
#: This is the guard against the last shape of damage the structural rules do not
#: catch: an unrecognised run that happens to *end* where two short known words
#: begin. ``withholdwritten`` covers as ``withholdwr`` + ``it`` + ``ten`` — an
#: unknown run at the edge, two known words, every rule above satisfied, and the
#: word destroyed. At 33% known coverage it fails here.
#:
#: It applies only to coverings that leave a run unrecognised. A covering made
#: entirely of known words is already fully accounted for — ``appointedbythe`` is
#: ``appointed`` + ``by`` + ``the``, all three in the lexicon, and is split.
MIN_KNOWN_COVERAGE = 0.40

#: Longest token the segmenter will look at. The search is quadratic in the
#: token length and a 200-character run is OCR debris rather than lost spacing.
MAX_SEGMENT_LENGTH = 60

_LOWER_UPPER = re.compile(r"(?<=[a-z])(?=[A-Z])")
_ALPHA_RUN = re.compile(r"[A-Za-z]+")
_TOKEN_CORE = re.compile(r"^(?P<lead>[^\w]*)(?P<core>.*?)(?P<trail>[^\w]*)$", re.DOTALL)


@dataclass
class RespaceResult:
    """Repaired text plus what it took, so the repair can be audited."""

    text: str
    spaces_inserted: int = 0
    tokens_changed: int = 0
    tokens_examined: int = 0
    #: ``original -> repaired`` for every token that changed, capped by the
    #: caller. The evidence for "this run improved the text" is these pairs, not
    #: the counter above them.
    examples: list[tuple[str, str]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return self.spaces_inserted > 0

    def to_dict(self, example_limit: int = 20) -> dict:
        return {
            "spaces_inserted": self.spaces_inserted,
            "tokens_changed": self.tokens_changed,
            "tokens_examined": self.tokens_examined,
            "examples": [
                {"before": before, "after": after}
                for before, after in self.examples[:example_limit]
            ],
        }


def respace(text: str, *, lexicon: frozenset[str] = COMMON_WORDS) -> RespaceResult:
    """Insert the spaces an OCR recogniser dropped between words.

    Line structure is preserved exactly: this splits within whitespace-separated
    tokens and touches nothing else, so page and line provenance still hold
    afterwards.
    """
    if not text:
        return RespaceResult(text=text)

    result = RespaceResult(text="")
    out_lines: list[str] = []
    for line in text.split("\n"):
        pieces: list[str] = []
        for token in line.split(" "):
            repaired = _repair_token(token, lexicon, result)
            pieces.append(repaired)
        out_lines.append(" ".join(pieces))
    result.text = "\n".join(out_lines)
    return result


def _repair_token(token: str, lexicon: frozenset[str], result: RespaceResult) -> str:
    """Apply the three rules to one whitespace-delimited token."""
    if len(token) < GLUED_MIN_LENGTH:
        return token

    match = _TOKEN_CORE.match(token)
    lead, core, trail = match.group("lead"), match.group("core"), match.group("trail")
    if len(core) < GLUED_MIN_LENGTH:
        return token

    result.tokens_examined += 1
    if core.lower() in lexicon:
        return token

    parts = [core]
    parts = _split_all(parts, _split_case)
    parts = _split_all(parts, lambda part: _split_digit_boundaries(part, lexicon))
    parts = _split_all(parts, lambda part: _segment(part, lexicon))

    repaired_core = " ".join(parts)
    if repaired_core == core:
        return token

    inserted = repaired_core.count(" ") - core.count(" ")
    result.spaces_inserted += inserted
    result.tokens_changed += 1
    repaired = f"{lead}{repaired_core}{trail}"
    result.examples.append((token, repaired))
    return repaired


def _split_all(parts: list[str], rule) -> list[str]:
    out: list[str] = []
    for part in parts:
        out.extend(rule(part))
    return out


# --- Rule 1: case boundaries ------------------------------------------------------


def _split_case(part: str) -> list[str]:
    """``theMunicipalAct`` -> ``the`` ``Municipal`` ``Act``.

    A one-character fragment means the boundary was not a lost space — an
    initial, a roman numeral, ``McX`` — so the whole split is abandoned rather
    than applied selectively.
    """
    pieces = _LOWER_UPPER.split(part)
    if len(pieces) == 1:
        return [part]
    if any(len(piece) < 2 for piece in pieces):
        return [part]
    return pieces


# --- Rule 2: digit/letter boundaries ----------------------------------------------


def _split_digit_boundaries(
    part: str, lexicon: frozenset[str] = COMMON_WORDS
) -> list[str]:
    """``section65of`` -> ``section`` ``65`` ``of``, but ``65A`` stays ``65A``.

    The two boundaries are not symmetrical, because in statutory text letters
    *before* and *after* a number mean different things.

    Letters running **into** a number are a word that lost its space:
    ``section65``, ``clause3``, ``Act30``. Cut when there are enough of them to
    be a word, or when they are one.

    Letters running **out of** a number usually belong to the number:
    ``s. 65A``, ``Article 21A``, ``1st``, ``2nd``. Splitting one changes a
    citation — ``65A`` is not ``65`` — so the cut is made only when the letters
    are *entirely* a word the lexicon knows, which recovers the ``of`` in
    ``65of`` and leaves ``65Aofthe`` and ``1stand2ndof`` alone. Measuring the
    length of that run instead would break exactly those two, because the run is
    long only because more words are glued to the end of it.
    """
    if not any(character.isdigit() for character in part):
        return [part]

    runs = re.findall(r"\d+|[A-Za-z]+|[^\dA-Za-z]+", part)
    if len(runs) < 2:
        return [part]

    pieces: list[str] = []
    current = runs[0]
    for index in range(1, len(runs)):
        previous, run = runs[index - 1], runs[index]
        boundary = (
            previous[0].isdigit() != run[0].isdigit()
            and previous[0].isalnum()
            and run[0].isalnum()
        )
        if previous[0].isalpha():           # letters running into a number
            cut = (len(previous) >= MIN_LETTERS_AROUND_DIGITS
                   or previous.lower() in lexicon)
        else:                               # letters running out of a number
            cut = run.lower() in lexicon
        if boundary and cut:
            pieces.append(current)
            current = run
        else:
            current += run
    pieces.append(current)
    return [piece for piece in pieces if piece]


# --- Rule 3: lexicon segmentation --------------------------------------------------


def _segment(part: str, lexicon: frozenset[str]) -> list[str]:
    """Split a glued alphabetic run into lexicon words, or leave it alone.

    Two shapes of covering are admitted, and no others:

    * the whole run is a sequence of words the lexicon knows —
      ``thepowertoamend`` -> ``the power to amend``; or
    * a sequence of known words with **one** unrecognised run attached at one
      end — ``appointedbythe`` -> ``appointed by the``,
      ``Actshallbeexercisedbyasub`` -> ``Act shall be exercisedbyasub``.

    What is excluded is the shape that a 307-word lexicon otherwise produces in
    quantity: known words carved out of the *middle* of an unrecognised one.
    Allowed to do that, the segmenter turns ``withholdwritten`` into ``with
    holdwr it ten`` and ``permission`` into ``per missi on`` — every piece
    defensible on its own and the word destroyed. A word may be *followed* or
    *preceded* by something the lexicon cannot name; it may not be interrupted
    by one.

    Three further guards, each closing a specific hole:

    * a covering must contain at least two known words, so a single function word
      cannot be peeled off a real word that merely begins with one —
      ``information`` -> ``in formation``, ``therefore`` -> ``the refore``,
      ``withholdwritten`` -> ``with holdwritten``;
    * a covering with an unknown run must reach :data:`PARTIAL_MIN_LENGTH`, so
      an ordinary long word such as ``whosoever`` is never a candidate; and
    * :data:`MIN_KNOWN_COVERAGE` of the characters must be accounted for by
      known words, which is what rejects ``withholdwr`` + ``it`` + ``ten`` — an
      unknown run at an edge, two known words, every other rule satisfied, and
      the word destroyed.
    """
    if len(part) < GLUED_MIN_LENGTH:
        return [part]
    if not part.isalpha():
        # A hyphenated or bracketed token — ``asub-committee``,
        # ``1[StateGovernment]`` — is segmented run by run, with the punctuation
        # left exactly where it was. Without this the commonest glued form in
        # this corpus, a run-together phrase ending in a hyphenated compound,
        # would be skipped entirely.
        return [_ALPHA_RUN.sub(
            lambda match: " ".join(_segment(match.group(), lexicon)), part)]
    if part.lower() in lexicon or len(part) > MAX_SEGMENT_LENGTH:
        return [part]

    whole = _known_cover(part, lexicon)
    if whole and len(whole) >= MIN_KNOWN_SEGMENTS:
        return whole
    if len(part) < PARTIAL_MIN_LENGTH:
        return [part]

    best: Optional[list[str]] = None
    for cut in range(UNKNOWN_MIN_LENGTH, len(part) - UNKNOWN_MIN_LENGTH + 1):
        for pieces in (
            _partial_cover([part[:cut]], part[cut:], lexicon),      # unknown first
            _partial_cover(None, part[:cut], lexicon, tail=part[cut:]),  # unknown last
        ):
            if pieces is None:
                continue
            if best is None or _known_characters(pieces, lexicon) >                     _known_characters(best, lexicon):
                best = pieces
    if best is None:
        return [part]
    if _known_characters(best, lexicon) / len(part) < MIN_KNOWN_COVERAGE:
        return [part]
    return best


def _partial_cover(
    head: Optional[list[str]],
    covered: str,
    lexicon: frozenset[str],
    *,
    tail: Optional[str] = None,
) -> Optional[list[str]]:
    """A covering of known words with one unknown run at the front or the back."""
    words = _known_cover(covered, lexicon)
    if words is None or len(words) < MIN_KNOWN_SEGMENTS:
        return None
    return (head or []) + words + ([tail] if tail else [])


def _known_cover(text: str, lexicon: frozenset[str]) -> Optional[list[str]]:
    """Cover *text* entirely with lexicon words, preferring longer words.

    ``None`` when no such covering exists, which is the common case and the
    reason this can be trusted: it never approximates.
    """
    length = len(text)
    # best[i] is the covering of text[:i] with the fewest, longest words.
    best: list[Optional[list[str]]] = [None] * (length + 1)
    best[0] = []
    for end in range(1, length + 1):
        for start in range(end):
            if best[start] is None:
                continue
            word = text[start:end]
            if word.lower() not in lexicon:
                continue
            candidate = best[start] + [word]
            if best[end] is None or len(candidate) < len(best[end]):
                best[end] = candidate
    return best[length]


def _known_characters(pieces: list[str], lexicon: frozenset[str]) -> int:
    return sum(len(piece) for piece in pieces if piece.lower() in lexicon)


# --- Measurement -------------------------------------------------------------------


def glued_token_rate(text: str, min_length: int = 12) -> float:
    """Share of alphabetic tokens at least *min_length* characters long.

    The simplest available proxy for lost spacing, and the one the engine
    comparison reports: real English legal prose runs about 2.5% by this measure
    and unrepaired RapidOCR output about 6%.
    """
    tokens = _ALPHA_RUN.findall(text)
    if not tokens:
        return 0.0
    return round(sum(1 for t in tokens if len(t) >= min_length) / len(tokens), 4)
