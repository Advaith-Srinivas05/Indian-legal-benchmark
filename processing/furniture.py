"""Detection of running headers, footers and page numbers.

The rule the project spec sets is that page furniture may be *detected* but never
blindly deleted, and that any removal rule must be testable. This module
therefore only ever **labels** lines: it returns, per page, which line indices
look like furniture and why. ``pages.json`` keeps the page text exactly as the
backend produced it, and the structure parser skips the labelled lines. Anyone
can re-read the original text and check the call.

The rule is repetition in the top or bottom zone of a page, applied two ways:

*Running heads and feet* must recur **literally**: "THE GAZETTE OF INDIA
EXTRAORDINARY" is the same string on every page it heads.

*Page numbers* must recur **with digit runs folded**, since their whole nature
is that the digits change — but only for lines that are nothing *but* a page
number ("7", "Page 3 of 40"). Folding digits for every line would be too
permissive: body text containing a number would start matching itself across
pages, and real legal text would be labelled furniture.

Both need the line to be short and to appear on most pages, and a document with
only a page or two is left alone entirely — with two pages, "recurs on most
pages" is not evidence of anything.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from . import config
from .models import FurnitureLine
from .textutils import is_page_number_line, normalise_line, split_lines


def _zone_lines(
    lines: list[str], zone_size: int, claimed: Optional[set[int]] = None
) -> tuple[list[int], list[int]]:
    """Indices of the first and last *zone_size* non-blank lines of a page.

    Non-blank rather than absolute positions: a leading empty line (common when
    a page begins with vertical space) must not hide the running head behind it.
    Lines already *claimed* — by footnote detection — are passed over, so the
    running foot is looked for above the notes rather than inside them.
    """
    claimed = claimed or set()
    filled = [
        i for i, line in enumerate(lines) if line.strip() and i not in claimed
    ]
    if not filled:
        return [], []
    top = filled[:zone_size]
    bottom = filled[-zone_size:]
    return top, bottom


def detect(
    page_texts: Iterable[str],
    *,
    zone_size: Optional[int] = None,
    min_pages: Optional[int] = None,
    min_page_ratio: Optional[float] = None,
    max_chars: Optional[int] = None,
    skip_indices: Optional[list[set[int]]] = None,
) -> dict[int, list[FurnitureLine]]:
    """Label furniture lines. Returns ``{page_number: [FurnitureLine, ...]}``.

    Pages are numbered from 1, matching :class:`processing.models.PageText`.
    A document with fewer than *min_pages* pages gets an empty result: with two
    pages, "recurs on most pages" carries no evidence at all.

    *skip_indices* gives, per page, line indices already claimed by a stronger
    rule — in practice the footnote blocks, which are found first because
    position-plus-type-size is better evidence than repetition. Without it a
    document whose amendment note happens to repeat would have its footnotes
    relabelled as a running foot.
    """
    zone_size = config.FURNITURE_ZONE_LINES if zone_size is None else zone_size
    min_pages = config.FURNITURE_MIN_PAGES if min_pages is None else min_pages
    min_page_ratio = (
        config.FURNITURE_MIN_PAGE_RATIO if min_page_ratio is None else min_page_ratio
    )
    max_chars = config.FURNITURE_MAX_CHARS if max_chars is None else max_chars

    pages = list(page_texts)
    if len(pages) < min_pages:
        return {}

    # (rule, zone, comparison key) -> the concrete occurrences it had
    occurrences: dict[tuple[str, str, str], list[tuple[int, int, str]]] = defaultdict(list)
    for page_index, text in enumerate(pages):
        lines = split_lines(text)
        claimed = skip_indices[page_index] if skip_indices else set()
        top, bottom = _zone_lines(lines, zone_size, claimed)
        for zone, indices in (("header", top), ("footer", bottom)):
            for line_index in indices:
                line = lines[line_index]
                if len(line.strip()) > max_chars:
                    continue
                literal = normalise_line(line, fold_digits=False)
                if literal:
                    occurrences[("literal", zone, literal)].append(
                        (page_index + 1, line_index, line))
                if is_page_number_line(line):
                    folded = normalise_line(line, fold_digits=True)
                    occurrences[("page_number", zone, folded)].append(
                        (page_index + 1, line_index, line))

    threshold = max(min_pages, int(round(min_page_ratio * len(pages))))
    result: dict[int, list[FurnitureLine]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    # Page numbers first: a line that is both a running head and a page number
    # is more usefully labelled the latter.
    ordered = sorted(occurrences.items(), key=lambda kv: kv[0][0] != "page_number")
    for (rule, zone, key), hits in ordered:
        # Count *pages*, not occurrences: a line repeated three times on one
        # page is not a running head.
        page_hits = {page for page, _, _ in hits}
        if len(page_hits) < threshold:
            continue
        for page_number, line_index, line in hits:
            if (page_number, line_index) in seen:
                continue
            seen.add((page_number, line_index))
            result[page_number].append(
                FurnitureLine(
                    line_index=line_index,
                    text=line,
                    kind="page_number" if rule == "page_number" else zone,
                    reason=(
                        f"{zone} zone line {key!r} recurs on "
                        f"{len(page_hits)}/{len(pages)} pages"
                        f"{' (digits folded)' if rule == 'page_number' else ''}"
                    ),
                )
            )

    for lines_found in result.values():
        lines_found.sort(key=lambda f: f.line_index)
    return dict(result)


def content_lines(text: str, furniture: Iterable[FurnitureLine]) -> list[tuple[int, str]]:
    """The page's lines minus its furniture, as ``(line_index, line)`` pairs.

    The original index travels with every line so a legal unit parsed from here
    can still be pointed back at the exact line of the untouched page text.
    """
    skip = {f.line_index for f in furniture}
    return [(i, line) for i, line in enumerate(split_lines(text)) if i not in skip]
