"""Extracted pages -> legal hierarchy, conservatively.

The governing rule from the project spec is *do not invent hierarchy*. Everything here
follows from it:

* A unit is emitted only when a pattern in :mod:`processing.patterns` matched
  actual text, and the unit records **which** pattern in ``detected_by``.
* When no structure can be identified, the result is an empty
  :class:`~processing.models.DocumentStructure` with
  ``confidence="unstructured"`` — not a guess. The page text is untouched and
  still fully available in ``pages.json``.
* Contents/arrangement-of-sections listings are identified and excluded from the
  section index, because their entries look exactly like section headings and
  would otherwise double every section in the document. The pages are recorded
  in ``toc_pages`` so the exclusion can be checked, and their text is not
  removed from anywhere.
* Every unit carries the page range it came from, because a section that cannot
  be pointed back at a page of a specific PDF cannot be cited.

The parser works on a flat *line stream* built from the pages with their
detected furniture skipped. Each line remembers its page number and its index
within that page's untouched text, so any unit can be traced back to the exact
lines it was built from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from . import config, patterns
from .furniture import content_lines
from .models import DocumentStructure, LegalUnit, PageText
from .textutils import normalise_line

#: Nesting depth of the units that live inside a section. Used to decide which
#: open unit a new one closes.
_DEPTH = {"subsection": 1, "clause": 2, "subclause": 3}

_ROMAN_CHARS = frozenset("ivxlcdm")
#: Roman numerals in the order sub-clauses use them, for sequence checks.
_ROMAN_SEQUENCE = (
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
)


@dataclass
class Line:
    """One line of the document, with everything needed to cite it back."""

    index: int              # position in the document-wide line stream
    page_number: int
    page_line_index: int    # index into the untouched page text's line list
    text: str


@dataclass
class _Anchor:
    """A structural marker found on a line, before the hierarchy is assembled."""

    line: int
    unit_type: str
    number: Optional[str]
    heading: Optional[str]
    detected_by: str


# --- Entry point ----------------------------------------------------------------


def parse_structure(
    pages: Iterable[PageText],
    *,
    metadata_title: Optional[str] = None,
) -> DocumentStructure:
    """Identify the legal hierarchy of one document.

    *metadata_title* is India Code's own title for the document (from the
    ingestion manifest). It is authoritative and is preferred over anything
    guessed from the page image; the PDF is only consulted to confirm it.
    """
    pages = list(pages)
    lines = build_line_stream(pages)
    structure = DocumentStructure()
    if not lines:
        structure.warnings.append("no text lines to parse; structure not attempted")
        _finalise(structure, lines)
        return structure

    toc_pages = detect_toc_pages(lines)
    structure.toc_pages = sorted(toc_pages)
    page_count = len({line.page_number for line in lines})
    if page_count and len(toc_pages) > 0.3 * page_count:
        # Either the document really is mostly a listing (an index volume, a
        # schedule of forms), or the contents heuristic has over-fired and real
        # sections are being suppressed. Both are worth a human look.
        structure.warnings.append(
            f"{len(toc_pages)}/{page_count} pages were treated as a contents "
            "listing; sections on those pages are not indexed"
        )

    anchors = _find_anchors(lines, toc_pages, structure)
    structure.units = _assemble(anchors, lines)
    structure.preamble = _find_preamble(lines, anchors, toc_pages)
    structure.title, structure.title_source = _find_title(lines, metadata_title, toc_pages)
    structure.footnotes = [
        block for page in pages for block in (getattr(page, "footnotes", []) or [])
    ]
    _finalise(structure, lines)
    return structure


def build_line_stream(pages: Iterable[PageText]) -> list[Line]:
    """Flatten pages into one line stream, skipping furniture and footnotes.

    Both are *labelled* on the page, never removed from it: the page text in
    ``pages.json`` is untouched, and this stream is what the parser reads. Taking
    footnotes out here is what stops a provision's text from ending with the
    amendment note printed below it.

    Blank lines are dropped from the stream (they carry no structure) but the
    surviving lines keep their original per-page index, so the mapping back to
    the verbatim page text stays exact.
    """
    stream: list[Line] = []
    for page in pages:
        skip = set()
        for block in getattr(page, "footnotes", []) or []:
            skip.update(range(block.line_start, block.line_end + 1))
        # Lines in another script. A translation printed beside the English text
        # is not legal hierarchy in English, and parsing it as such emitted
        # Devanagari LegalUnits -- on one real document, 8 of its 9 sections.
        for mark in getattr(page, "non_english_lines", []) or []:
            skip.add(mark["line_index"])
        for page_line_index, text in content_lines(page.text, page.furniture):
            if page_line_index in skip or not text.strip():
                continue
            stream.append(
                Line(
                    index=len(stream),
                    page_number=page.page_number,
                    page_line_index=page_line_index,
                    text=text,
                )
            )
    return stream


# --- Contents listings ----------------------------------------------------------


def _is_contents_entry(text: str) -> bool:
    """A numbered line that is a *listing* entry rather than a section.

    The discriminator is the body. ``12. Punishment for murder.—Whoever …``
    carries the provision itself and is a section wherever it appears; ``12.
    Punishment for murder.`` carries only a caption and is what a contents page
    is made of. Without this the arrangement-of-sections heuristic would fire on
    ordinary body pages and suppress every section on them.
    """
    if patterns.SECTION_DASH_RE.match(text):
        return False
    return bool(
        patterns.TOC_ENTRY_RE.match(text) or patterns.TOC_BARE_NUMBER_RE.match(text)
    )


def detect_toc_pages(lines: list[Line]) -> set[int]:
    """Pages that are an arrangement-of-sections / contents listing.

    Two independent signals, either sufficient:

    1. The page carries a heading that names it (``ARRANGEMENT OF SECTIONS``,
       ``CONTENTS``, ``INDEX``, a bare ``SECTIONS``).
    2. The page is mostly numbered entries with no body — either
       ``12. Short title and extent.`` captions, or bare ``12.`` numbers, which
       is how a two-column contents page extracts when the captions land in a
       separate column.

    Getting this wrong in the permissive direction costs real sections, so both
    signals require the *page*, not a line, to look like a listing.
    """
    by_page: dict[int, list[Line]] = {}
    for line in lines:
        by_page.setdefault(line.page_number, []).append(line)

    toc: set[int] = set()
    for page_number, page_lines in by_page.items():
        texts = [line.text for line in page_lines]
        if any(patterns.TOC_MARKER_RE.match(text) for text in texts):
            toc.add(page_number)
            continue
        entries = sum(1 for text in texts if _is_contents_entry(text))
        if entries >= config.TOC_RUN_LENGTH and entries >= 0.5 * len(texts):
            toc.add(page_number)
    return toc


# --- Anchors --------------------------------------------------------------------


def _is_division_heading(line: str, heading: str) -> bool:
    """Whether a ``PART``/``CHAPTER``/``SCHEDULE`` match is really a heading.

    ``CHAPTER V`` on its own line is a heading. ``Chapter V of the Act shall
    apply to …`` is a sentence that merely starts with the word, and labelling
    it a chapter would invent a division that does not exist. Headings are
    short, are not left mid-clause, and — when they carry a caption at all —
    capitalise it.
    """
    stripped = line.strip()
    if len(stripped) > config.MAX_HEADING_CHARS:
        return False
    if stripped.endswith((",", ";")):
        return False
    caption = heading.strip()
    if caption and caption[0].islower():
        return False
    return True


def _caption_from_next_line(lines: list[Line], index: int) -> Optional[str]:
    """The caption of a division whose heading line carries only its number.

    India Code sets divisions on two lines::

        CHAPTER V
        OF THE LIABILITY OF THE OWNER

    Taking the second line is safe only when it looks like a caption and not
    like the start of the text: short, set in capitals, and not itself a
    structural marker.
    """
    if index + 1 >= len(lines):
        return None
    candidate = lines[index + 1].text.strip()
    if not candidate or len(candidate) > config.MAX_HEADING_CHARS:
        return None
    if (
        patterns.SECTION_DASH_RE.match(candidate)
        or patterns.SECTION_LINE_RE.match(candidate)
        or patterns.ARTICLE_RE.match(candidate)
        or patterns.CHAPTER_RE.match(candidate)
        or patterns.PART_RE.match(candidate)
        or patterns.PAREN_LABEL_RE.match(candidate)
    ):
        return None
    letters = [c for c in candidate if c.isalpha()]
    if not letters or sum(1 for c in letters if c.isupper()) / len(letters) < 0.8:
        return None
    return _clean(candidate)


def _find_anchors(
    lines: list[Line], toc_pages: set[int], structure: DocumentStructure
) -> list[_Anchor]:
    """Locate every structural marker, then decide which style to trust."""
    divisions: list[_Anchor] = []
    articles: list[_Anchor] = []
    dashed: list[_Anchor] = []
    plain: list[_Anchor] = []

    for line in lines:
        if line.page_number in toc_pages:
            continue
        text = line.text

        match = patterns.PART_RE.match(text)
        if match and _is_division_heading(text, match.group("heading")):
            heading = _clean(match.group("heading"))
            detected = "part_heading"
            if not heading:
                heading = _caption_from_next_line(lines, line.index)
                detected = "part_heading_caption_below" if heading else detected
            divisions.append(_Anchor(line.index, "part", match.group("number"),
                                     heading, detected))
            continue

        match = patterns.CHAPTER_RE.match(text)
        if match and _is_division_heading(text, match.group("heading")):
            heading = _clean(match.group("heading"))
            detected = "chapter_heading"
            if not heading:
                heading = _caption_from_next_line(lines, line.index)
                detected = "chapter_heading_caption_below" if heading else detected
            divisions.append(_Anchor(line.index, "chapter", match.group("number"),
                                     heading, detected))
            continue

        match = patterns.SCHEDULE_RE.match(text)
        if match and _is_division_heading(text, match.group("heading")):
            number = match.group("number") or match.group("ordinal")
            heading = _clean(match.group("heading"))
            detected = "schedule_heading"
            if not heading:
                heading = _caption_from_next_line(lines, line.index)
                detected = "schedule_heading_caption_below" if heading else detected
            divisions.append(_Anchor(line.index, "schedule", _clean(number),
                                     heading, detected))
            continue

        match = patterns.APPENDIX_RE.match(text)
        if match and _is_division_heading(text, match.group("heading")):
            heading = _clean(match.group("heading"))
            detected = "appendix_heading"
            if not heading:
                heading = _caption_from_next_line(lines, line.index)
                detected = "appendix_heading_caption_below" if heading else detected
            divisions.append(_Anchor(line.index, "appendix", _clean(match.group("number")),
                                     heading, detected))
            continue

        # Articles are matched on the literal word, never inferred from
        # numbering: an Article is a different unit of law from a Section and
        # citing one as the other would be wrong.
        match = patterns.ARTICLE_DASH_RE.match(text)
        if match:
            articles.append(_Anchor(line.index, "article",
                                    _normalise_number(match.group("number")),
                                    _clean(match.group("heading")), "article_heading_dash"))
            continue

        match = patterns.ARTICLE_RE.match(text)
        if match:
            heading = _clean(match.group("heading"))
            # "Article 348 of the Constitution provides …" is a sentence that
            # mentions an Article, not the start of one. A real marker is
            # followed by nothing, by a separator, or by a capitalised caption.
            if heading and heading[0].islower():
                continue
            if heading and len(text.strip()) > config.MAX_HEADING_CHARS // 2:
                # The line runs on into the provision's text; that text is the
                # body, not a heading, and calling it one would be inventing.
                heading = None
            articles.append(_Anchor(line.index, "article",
                                    _normalise_number(match.group("number")),
                                    heading, "article_marker"))
            continue

        match = patterns.SECTION_DASH_RE.match(text)
        if match:
            dashed.append(_Anchor(line.index, "section", _normalise_number(match.group("number")),
                                  _clean(match.group("heading")), "numbered_heading_dash"))
            continue

        match = patterns.SECTION_LINE_RE.match(text)
        if match and len(text.strip()) <= config.MAX_HEADING_CHARS:
            plain.append(_Anchor(line.index, "section", _normalise_number(match.group("number")),
                                 _clean(match.group("heading")), "numbered_heading_line"))

    provisions, vocabulary = _choose_provision_style(articles, dashed, plain, structure)
    structure.unit_vocabulary = vocabulary
    provisions = _drop_out_of_sequence(provisions, structure)
    return sorted(divisions + provisions, key=lambda a: a.line)


def _choose_provision_style(
    articles: list[_Anchor], dashed: list[_Anchor], plain: list[_Anchor],
    structure: DocumentStructure,
) -> tuple[list[_Anchor], Optional[str]]:
    """Pick one provision style rather than merging several.

    A document numbers its provisions one way. Accepting several at once mixes
    the weak pattern's false positives — numbered lists, stray captions — into
    an otherwise reliable result, and in a document numbered by Article the
    weak pattern matches every numbered item *inside* an article.

    Precedence: Articles when the document says "Article"; otherwise the dashed
    section heading, which cannot be produced by an ordinary numbered list;
    otherwise the weak line pattern, whose use is reported as a warning.

    Returns the chosen anchors and the vocabulary the document was found to use.
    """
    if len(articles) >= 3:
        chosen = list(articles)
        if len(dashed) >= 3:
            # A document can carry both: Goa's codes state Articles and
            # occasionally set a Section in the dashed house style.
            chosen += dashed
        elif dashed:
            structure.warnings.append(
                f"{len(dashed)} dashed section candidate(s) ignored: this document "
                f"numbers its provisions as Articles ({len(articles)} found)"
            )
        if plain:
            structure.warnings.append(
                f"{len(plain)} line-style section candidates ignored: this document "
                f"numbers its provisions as Articles ({len(articles)} found), and "
                "numbered lines inside an Article are its list items, not sections"
            )
        return chosen, "article"

    if len(dashed) >= 3:
        if plain:
            structure.warnings.append(
                f"{len(plain)} line-style section candidates ignored: this document "
                f"uses the dashed heading style ({len(dashed)} found)"
            )
        return dashed, "section"

    if len(plain) >= 3:
        structure.warnings.append(
            "sections identified from the weaker 'number + heading on its own line' "
            "pattern; no dashed headings were present"
        )
        return plain, "section"

    if articles:
        return articles, "article"
    if dashed:
        return dashed, "section"
    if plain:
        structure.warnings.append(
            f"only {len(plain)} line-style section candidate(s); too few to be "
            "confident this document is section-structured"
        )
        return plain, "section"
    return [], None


def _section_sort_key(number: Optional[str]) -> tuple[int, str]:
    if not number:
        return (0, "")
    match = re.match(r"(\d+)\s*-?\s*([A-Z]*)", number)
    if not match:
        return (0, number)
    return (int(match.group(1)), match.group(2))


def _drop_out_of_sequence(
    provisions: list[_Anchor], structure: DocumentStructure
) -> list[_Anchor]:
    """Apply the ordering filter within each provision vocabulary.

    Sections and Articles run in their own sequences, so a document carrying
    both must not have them interleaved before the ordering check — Article 5
    following section 200 is not a break in the numbering.
    """
    by_type: dict[str, list[_Anchor]] = {}
    for anchor in provisions:
        by_type.setdefault(anchor.unit_type, []).append(anchor)
    kept: list[_Anchor] = []
    for anchors in by_type.values():
        kept.extend(_drop_out_of_sequence_within(anchors, structure))
    return sorted(kept, key=lambda a: a.line)


def _drop_out_of_sequence_within(
    sections: list[_Anchor], structure: DocumentStructure
) -> list[_Anchor]:
    """Keep the longest run of non-decreasing provision numbers.

    Sections in an Act run in order. A candidate that breaks the order is
    normally a numbered list inside another section's text, a form field, or a
    stray caption. Dropping it removes it from the *index* only — the text stays
    in ``pages.json`` and in the enclosing section's text, so nothing is lost.

    The filter refuses to run when it would drop a large share of candidates:
    that would mean the premise (this document numbers its sections in order) is
    wrong, and the honest response is to keep everything and say so.
    """
    if len(sections) < 3:
        return sections

    keys = [_section_sort_key(a.number) for a in sections]
    best_length = [1] * len(sections)
    previous = [-1] * len(sections)
    for i in range(len(sections)):
        for j in range(i):
            if keys[j] <= keys[i] and best_length[j] + 1 > best_length[i]:
                best_length[i] = best_length[j] + 1
                previous[i] = j
    end = max(range(len(sections)), key=lambda i: best_length[i])
    kept_indices = []
    while end != -1:
        kept_indices.append(end)
        end = previous[end]
    kept_indices.reverse()

    dropped = len(sections) - len(kept_indices)
    if dropped == 0:
        return sections
    if dropped > 0.3 * len(sections):
        structure.warnings.append(
            f"section numbers are not in ascending order ({dropped} of "
            f"{len(sections)} candidates break the sequence); no candidates were "
            "dropped, but the section index may contain false positives"
        )
        return sections

    dropped_numbers = [
        sections[i].number for i in range(len(sections)) if i not in set(kept_indices)
    ]
    structure.warnings.append(
        f"{dropped} out-of-sequence section candidate(s) excluded from the index "
        f"(numbers: {', '.join(str(n) for n in dropped_numbers[:10])}"
        f"{'…' if dropped > 10 else ''}); their text is unchanged in pages.json"
    )
    return [sections[i] for i in kept_indices]


# --- Hierarchy assembly ---------------------------------------------------------


def _assemble(anchors: list[_Anchor], lines: list[Line]) -> list[LegalUnit]:
    """Turn a flat anchor list into a Part > Chapter > Section tree."""
    if not anchors:
        return []

    top: list[LegalUnit] = []
    current_part: Optional[LegalUnit] = None
    current_chapter: Optional[LegalUnit] = None
    in_schedules = False

    for position, anchor in enumerate(anchors):
        end = anchors[position + 1].line - 1 if position + 1 < len(anchors) else len(lines) - 1
        unit_type = anchor.unit_type

        if unit_type in ("schedule", "appendix"):
            # A schedule ends the body of the Act: sections after it belong to
            # the schedule's own text, not to the preceding chapter.
            unit = _make_unit(anchor, lines, anchor.line, end, heading_only=False)
            top.append(unit)
            current_part = current_chapter = None
            in_schedules = True
            continue

        if unit_type == "part":
            unit = _make_unit(anchor, lines, anchor.line, end, heading_only=True)
            top.append(unit)
            current_part, current_chapter = unit, None
            in_schedules = False
            continue

        if unit_type == "chapter":
            unit = _make_unit(anchor, lines, anchor.line, end, heading_only=True)
            (current_part.children if current_part else top).append(unit)
            current_chapter = unit
            in_schedules = False
            continue

        # A provision: a section or an article. They sit at the same level of
        # the hierarchy and differ in what they are called, which is why the
        # unit type is carried through rather than normalised away.
        unit = _make_unit(anchor, lines, anchor.line, end, heading_only=False)
        unit.children = parse_section_body(unit, lines, anchor.line, end)
        if in_schedules and top:
            top[-1].children.append(unit)
        elif current_chapter is not None:
            current_chapter.children.append(unit)
        elif current_part is not None:
            current_part.children.append(unit)
        else:
            top.append(unit)

    _extend_container_ranges(top)
    return top


def _make_unit(
    anchor: _Anchor, lines: list[Line], start: int, end: int, *, heading_only: bool
) -> LegalUnit:
    """Build a unit spanning lines ``start..end`` of the stream.

    ``heading_only`` containers (Part, Chapter) carry only their heading line as
    text: their content is their child sections, and duplicating every section's
    text into its chapter would multiply the output size for no gain. Sections
    and schedules carry their full text, because that is the unit a citation and
    a future chunk will be built from.
    """
    end = max(start, min(end, len(lines) - 1))
    body_end = start if heading_only else end
    span = lines[start:body_end + 1]
    return LegalUnit(
        unit_type=anchor.unit_type,
        number=anchor.number,
        heading=anchor.heading or None,
        text="\n".join(line.text for line in span),
        page_start=lines[start].page_number,
        page_end=lines[body_end].page_number,
        line_start=start,
        line_end=body_end,
        detected_by=anchor.detected_by,
    )


def _extend_container_ranges(units: list[LegalUnit]) -> None:
    """Make a container's page range cover its children.

    A Chapter's heading sits on one page but the chapter runs for twenty. The
    citable range is the whole span, so it is taken from the children once they
    are attached.
    """
    for unit in units:
        if not unit.children:
            continue
        _extend_container_ranges(unit.children)
        unit.page_end = max(unit.page_end, max(c.page_end for c in unit.children))
        unit.line_end = max(unit.line_end, max(c.line_end for c in unit.children))


# --- Inside a section -----------------------------------------------------------


def classify_label(
    label: str, *, open_types: list[str], last_clause: Optional[str],
    last_subclause: Optional[str],
) -> tuple[Optional[str], str]:
    """Decide what a ``(x)`` label introduces. Returns ``(unit_type, reason)``.

    Digits are unambiguous — ``(1)``, ``(2A)`` is always a subsection. Letters
    are not: ``(i)``, ``(v)`` and ``(x)`` are both the ninth/twenty-second/
    twenty-fourth clause letters and the first/fifth/tenth roman sub-clause
    numerals, and Indian drafting uses both conventions in the same document.

    The ambiguity is resolved by sequence rather than by preference: a label
    that continues the clause letters already seen is a clause; a label that
    continues (or opens) the roman sub-clause run inside an open clause is a
    sub-clause. When neither sequence explains it, we fall back on nesting
    context and say so in the reason, so the call is visible in the output.
    """
    if label[0].isdigit():
        return "subsection", "numeric label"

    lowered = label.lower()
    is_roman = bool(lowered) and set(lowered) <= _ROMAN_CHARS

    if not is_roman:
        return "clause", "alphabetic label"

    # Roman-looking. Does it continue the alphabetic clause run?
    if last_clause and len(lowered) == 1 and len(last_clause) == 1:
        if ord(lowered) == ord(last_clause.lower()) + 1:
            return "clause", f"continues clause sequence after ({last_clause})"

    if "clause" in open_types:
        if last_subclause:
            expected = _next_roman(last_subclause)
            if expected and lowered == expected:
                return "subclause", f"continues roman sequence after ({last_subclause})"
        if lowered == "i":
            return "subclause", "opens a roman sequence inside a clause"
        return "subclause", "roman label inside an open clause"

    if lowered == "i" and not last_clause:
        return "clause", "opens a lettered sequence at (i)"
    return "clause", "roman label with no enclosing clause"


def _next_roman(current: str) -> Optional[str]:
    try:
        position = _ROMAN_SEQUENCE.index(current.lower())
    except ValueError:
        return None
    if position + 1 >= len(_ROMAN_SEQUENCE):
        return None
    return _ROMAN_SEQUENCE[position + 1]


def parse_section_body(
    section: LegalUnit, lines: list[Line], start: int, end: int
) -> list[LegalUnit]:
    """Parse subsections, clauses, sub-clauses, provisos and explanations.

    Units are opened at a marker and closed when a marker of the same or a
    shallower depth appears, so the nesting follows the document rather than an
    assumed shape. Provisos and explanations attach to whatever unit is
    innermost at the point they appear — which is what they qualify.

    Text is apportioned, not duplicated: each unit's ``text`` runs from its
    marker to the next marker, so a subsection's text stops where its first
    clause begins. The whole, uncut text of the section is on the section unit
    itself, which is the level a citation and a future chunk are built at.
    """
    children: list[LegalUnit] = []
    stack: list[LegalUnit] = []
    last_clause: Optional[str] = None
    last_subclause: Optional[str] = None
    pending: list[tuple[LegalUnit, int]] = []      # (unit, first line index)

    def attach(unit: LegalUnit) -> None:
        (stack[-1].children if stack else children).append(unit)

    def close_to(depth: int) -> None:
        nonlocal last_clause, last_subclause
        while stack and _DEPTH.get(stack[-1].unit_type, 99) >= depth:
            stack.pop()
        if depth <= _DEPTH["clause"]:
            last_subclause = None
        if depth <= _DEPTH["subsection"]:
            last_clause = None

    for index in range(start, min(end, len(lines) - 1) + 1):
        line = lines[index]
        text = line.text
        # The section's own first line: its body may begin after the heading
        # dash, so look for a label there rather than at the line start.
        if index == start:
            match = (
                patterns.SECTION_DASH_RE.match(text)
                or patterns.ARTICLE_DASH_RE.match(text)
            )
            remainder = text[match.end():] if match else ""
            label_match = patterns.INLINE_FIRST_LABEL_RE.match(remainder)
            if not label_match:
                continue
            label = label_match.group("label")
            offset = match.end() if match else 0
            unit_type, reason = classify_label(
                label, open_types=[u.unit_type for u in stack],
                last_clause=last_clause, last_subclause=last_subclause,
            )
        else:
            label_match = patterns.PAREN_LABEL_RE.match(text)
            if label_match:
                label = label_match.group("label")
                unit_type, reason = classify_label(
                    label, open_types=[u.unit_type for u in stack],
                    last_clause=last_clause, last_subclause=last_subclause,
                )
                offset = 0
            else:
                unit_type = None
                if patterns.EXPLANATION_RE.match(text):
                    unit_type, reason, label, offset = "explanation", "explanation marker", None, 0
                elif patterns.PROVISO_RE.match(text):
                    unit_type, reason, label, offset = "proviso", "proviso marker", None, 0
                if unit_type is None:
                    continue

        if unit_type in _DEPTH:
            close_to(_DEPTH[unit_type])
        unit = LegalUnit(
            unit_type=unit_type,
            number=label,
            heading=None,
            text=text[offset:] if offset else text,
            page_start=line.page_number,
            page_end=line.page_number,
            line_start=index,
            line_end=index,
            detected_by=reason,
        )
        attach(unit)
        pending.append((unit, index))
        if unit_type == "clause":
            last_clause, last_subclause = label, None
        elif unit_type == "subclause":
            last_subclause = label
        if unit_type in _DEPTH:
            stack.append(unit)

    _close_spans(pending, lines, end)
    return children


def _close_spans(pending: list[tuple[LegalUnit, int]], lines: list[Line], end: int) -> None:
    """Extend each unit's text/page range to where the next unit begins.

    A subsection runs until the next marker, however many lines and pages later
    that is. Without this every unit would be one line long and its page range
    would be wrong for anything that straddles a page break — exactly the case
    page provenance exists for.
    """
    end = min(end, len(lines) - 1)
    for position, (unit, start_index) in enumerate(pending):
        stop = pending[position + 1][1] - 1 if position + 1 < len(pending) else end
        stop = max(start_index, min(stop, end))
        if stop == start_index:
            continue
        tail = "\n".join(lines[i].text for i in range(start_index + 1, stop + 1))
        unit.text = f"{unit.text}\n{tail}" if tail else unit.text
        unit.line_end = stop
        unit.page_end = lines[stop].page_number
    # A parent's range must cover the children that were attached to it.
    for unit, _ in reversed(pending):
        if unit.children:
            unit.page_end = max(unit.page_end, max(c.page_end for c in unit.children))
            unit.line_end = max(unit.line_end, max(c.line_end for c in unit.children))


# --- Preamble and title ---------------------------------------------------------


def _find_preamble(
    lines: list[Line], anchors: list[_Anchor], toc_pages: set[int]
) -> Optional[LegalUnit]:
    """The enacting text before the first section, when there is one.

    Only emitted when an enacting formula is actually present. Leading text is
    not a preamble merely by being leading — on India Code that leading text is
    just as often a cover page or a gazette masthead.
    """
    first_anchor = anchors[0].line if anchors else len(lines)
    start = None
    for index in range(min(first_anchor, len(lines))):
        if lines[index].page_number in toc_pages:
            continue
        if patterns.PREAMBLE_RE.search(lines[index].text):
            start = index
            break
    if start is None:
        return None
    end = max(start, first_anchor - 1)
    span = [line for line in lines[start:end + 1] if line.page_number not in toc_pages]
    if not span:
        return None
    return LegalUnit(
        unit_type="preamble",
        number=None,
        heading=None,
        text="\n".join(line.text for line in span),
        page_start=span[0].page_number,
        page_end=span[-1].page_number,
        line_start=start,
        line_end=end,
        detected_by="enacting_formula",
    )


def _find_title(
    lines: list[Line], metadata_title: Optional[str], toc_pages: set[int]
) -> tuple[Optional[str], Optional[str]]:
    """The document title, preferring India Code's own metadata.

    India Code states the title of every document it publishes and the ingestion
    phase already recorded it. Re-deriving one from the page image could only
    make it worse, so the PDF is used to *confirm* the recorded title, never to
    replace it. Deriving a title from the page is a last resort and is labelled
    as such.
    """
    head = [line for line in lines[:40] if line.page_number not in toc_pages]
    if metadata_title:
        target = normalise_line(metadata_title)
        if target.startswith("the "):
            target = target[4:]
        for line in head:
            if target and target in normalise_line(line.text):
                return metadata_title, "india_code_metadata_confirmed_on_page"
        return metadata_title, "india_code_metadata"

    for line in head:
        text = line.text.strip()
        letters = [c for c in text if c.isalpha()]
        if 10 <= len(text) <= config.MAX_HEADING_CHARS and letters:
            if sum(1 for c in letters if c.isupper()) / len(letters) > 0.8:
                return text, "pdf_first_page_heading"
    return None, None


# --- Finalisation ---------------------------------------------------------------


def _finalise(structure: DocumentStructure, lines: list[Line]) -> None:
    counts: dict[str, int] = {}
    for unit in structure.all_units():
        counts[unit.unit_type] = counts.get(unit.unit_type, 0) + 1
    if structure.preamble:
        counts["preamble"] = 1
    counts["footnote"] = len(structure.footnotes)
    counts["line_count"] = len(lines)
    structure.counts = counts

    sections = counts.get("section", 0) + counts.get("article", 0)
    divisions = counts.get("chapter", 0) + counts.get("part", 0) + counts.get("schedule", 0)
    if sections >= 3:
        structure.confidence = "structured"
    elif sections or divisions:
        structure.confidence = "partial"
    else:
        structure.confidence = "unstructured"
        structure.warnings.append(
            "no legal structure could be identified; the extracted page text is "
            "preserved as-is and no hierarchy was guessed"
        )


def _clean(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = " ".join(value.split()).strip(" .:–—-")
    return cleaned or None


def _normalise_number(number: str) -> str:
    """``10 - A`` and ``10A`` are the same section; store one spelling."""
    return re.sub(r"\s*-\s*", "-", " ".join(number.split()))
