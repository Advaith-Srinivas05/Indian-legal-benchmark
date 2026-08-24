"""The regular expressions that identify Indian legal structure.

Kept in their own module because they are the part of the parser most likely to
need tuning against real documents, and because every one of them is a claim
about how Indian legislation is typeset — a claim that deserves to be readable
and testable on its own.

Two section styles dominate India Code:

``12. Punishment for murder.—Whoever commits murder shall …``
    Number, a short heading, then an em dash introducing the body. This is the
    house style of India Code's own re-typeset Central Acts and is by far the
    most reliable signal, because the dash cannot be produced by an ordinary
    numbered list.

``12. Punishment for murder``  (heading alone on its line, body below)
    Common in state legislation and in older documents. Much weaker: it is
    indistinguishable from a numbered list item or a contents entry, so it is
    only used when the dashed style is absent (see :mod:`processing.structure`).

Everything here matches at the *start of a line*. Nothing here decides anything;
:mod:`processing.structure` does, using the sequence and the context.
"""

from __future__ import annotations

import re

#: ``12``, ``12A``, ``12-A``, ``376AB``. Amending acts insert lettered sections
#: rather than renumber, so the letter suffix is not optional decoration.
SECTION_NUMBER = r"\d{1,4}(?:\s?-\s?[A-Z]{1,3}|[A-Z]{1,3})?"

#: Roman or arabic or a short letter — Part II, Chapter 5, Chapter VIA.
DIVISION_NUMBER = r"(?:[IVXLCDM]{1,7}|\d{1,3})(?:\s?-\s?[A-Z]{1,3}|[A-Z]{1,3})?"

#: The separator between a section heading and its body. India Code prints an
#: em dash; OCR of older gazettes yields hyphen runs and ``.c--`` style noise,
#: so a short hyphen run counts too.
_DASH = r"(?:[‐-―−]+|-{1,3})"

# --- Top-level divisions --------------------------------------------------------

PART_RE = re.compile(
    rf"^\s*PART\s+(?P<number>{DIVISION_NUMBER})\b\s*[.:—–-]?\s*(?P<heading>.*)$",
    re.IGNORECASE,
)

CHAPTER_RE = re.compile(
    rf"^\s*CHAPTER\s+(?P<number>{DIVISION_NUMBER})\b\s*[.:—–-]?\s*(?P<heading>.*)$",
    re.IGNORECASE,
)

#: ``THE FIRST SCHEDULE``, ``SCHEDULE II``, ``SCHEDULE``. The ordinal may precede
#: or follow the word.
SCHEDULE_RE = re.compile(
    r"^\s*(?:THE\s+)?"
    r"(?:(?P<ordinal>FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
    r"ELEVENTH|TWELFTH|\d{1,2}(?:ST|ND|RD|TH))\s+)?"
    r"SCHEDULES?\b"
    r"(?:\s*(?P<number>[IVXLCDM]{1,7}|\d{1,3}))?"
    r"\s*[.:—–-]?\s*(?P<heading>.*)$",
    re.IGNORECASE,
)

APPENDIX_RE = re.compile(
    r"^\s*APPENDIX\b(?:\s*(?P<number>[IVXLCDM]{1,7}|\d{1,3}|[A-Z]))?"
    r"\s*[.:—–-]?\s*(?P<heading>.*)$",
    re.IGNORECASE,
)

# --- Sections -------------------------------------------------------------------

#: Tier 1 — number, heading, dash. High confidence.
SECTION_DASH_RE = re.compile(
    rf"^\s*(?:\[\s*)?(?P<number>{SECTION_NUMBER})\s*\.\s*"
    rf"(?P<heading>\S.{{0,198}}?)\s*\.\s*{_DASH}\s*"
)

#: Tier 2 — number and heading alone on the line. Low confidence; used only when
#: tier 1 finds nothing (see :mod:`processing.structure`).
SECTION_LINE_RE = re.compile(
    rf"^\s*(?:\[\s*)?(?P<number>{SECTION_NUMBER})\s*\.\s+(?P<heading>\S.{{0,198}}?)\s*\.?\s*$"
)

# --- Articles -------------------------------------------------------------------
#
# The Constitution and the Portuguese-derived civil codes still in force in Goa
# number their provisions as Articles, not Sections. They are a different unit of
# law, cited differently, and folding them into Sections would produce citations
# to provisions that do not exist — so they are matched on the literal word and
# never inferred from numbering alone.

#: ``Article 14 – Conflict of rights – Whoever in exercise of his own right …``
#: The heading sits between two dashes, which is how the Goa codes are set.
ARTICLE_DASH_RE = re.compile(
    rf"^\s*(?:ARTICLE|Article|Art\.)\s+(?P<number>{SECTION_NUMBER})\s*"
    rf"(?:\.|º)?\s*{_DASH}\s*(?P<heading>\S.{{0,198}}?)\s*\.?\s*{_DASH}\s*"
)

#: ``Article 21.`` / ``ARTICLE 370`` / ``Art. 5`` — the word plus a number,
#: whatever follows.
ARTICLE_RE = re.compile(
    rf"^\s*(?:ARTICLE|Article|Art\.)\s+(?P<number>{SECTION_NUMBER})\s*"
    rf"(?:\.|º)?\s*[.:—–-]?\s*(?P<heading>.*)$"
)

# --- Inside a section -----------------------------------------------------------

#: ``(1)``, ``(a)``, ``(iv)``, ``(1A)``. What kind of unit it is depends on the
#: label *and* on what came before it — see
#: :func:`processing.structure.classify_label`.
PAREN_LABEL_RE = re.compile(r"^\s*\(\s*(?P<label>[0-9A-Za-z]{1,6})\s*\)\s*(?=\S)")

#: A subsection introduced immediately after the section heading dash, on the
#: same line: ``1. Short title.—(1) This Act may be called …``.
INLINE_FIRST_LABEL_RE = re.compile(r"^\s*\(\s*(?P<label>[0-9A-Za-z]{1,6})\s*\)\s*(?=\S)")

PROVISO_RE = re.compile(r"^\s*(?:\[\s*)?Provided\s+(?:that|further|also|however)\b", re.IGNORECASE)

EXPLANATION_RE = re.compile(
    r"^\s*(?:\[\s*)?Explanation\s*(?P<number>[0-9IVX]{0,4})\s*[.:—–-]",
    re.IGNORECASE,
)

# --- Contents listings ----------------------------------------------------------

#: Headings that name a contents listing outright.
TOC_MARKER_RE = re.compile(
    r"^\s*(?:ARRANGEMENT\s+OF\s+(?:SECTIONS?|RULES?|REGULATIONS?|CLAUSES?|CHAPTERS?|PARAGRAPHS?)"
    r"|TABLE\s+OF\s+CONTENTS?"
    r"|CONTENTS?"
    r"|INDEX"
    r"|SECTIONS?)\s*[.:]?\s*$",
    re.IGNORECASE,
)

#: A contents entry: a number and a short caption, no sentence body.
TOC_ENTRY_RE = re.compile(
    rf"^\s*{SECTION_NUMBER}\s*\.\s+\S.{{0,118}}$"
)

#: A bare number on its own line — how a two-column contents page extracts.
TOC_BARE_NUMBER_RE = re.compile(rf"^\s*{SECTION_NUMBER}\s*\.?\s*$")

# --- Preamble -------------------------------------------------------------------

#: The enacting formula. Its presence is the only thing that makes us call text
#: a preamble; we never label leading text a preamble just because it is leading.
PREAMBLE_RE = re.compile(
    r"(?:^|\s)(?:An\s+Act\s+to\b|An\s+Act\s+further\s+to\b|WHEREAS\b|BE\s+it\s+enacted\b"
    r"|In\s+exercise\s+of\s+the\s+powers\s+conferred\b)",
    re.IGNORECASE,
)
