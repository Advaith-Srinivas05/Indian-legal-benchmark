"""Detection and separation of footnote blocks.

India Code prints amendment history as numbered footnotes at the foot of the
page. Extracted linearly they land inside whichever section the page ended on,
so in the first benchmark the Code of Civil Procedure's section 1 ended with::

    1. This Act has been amended in its application to Assam by Assam Acts 2 of
    1941 and 3 of 1953; to Tamil Nadu by Madras Act 34 of 1950 …

read as though it were part of the provision. That is not a formatting blemish:
a chunk built from that section, quoted back to a user, would present editorial
apparatus as statutory text.

Footnotes are **separated, not deleted**. They become their own provenanced
units — the amendment history is genuinely useful, and the project spec forbids
silently discarding extracted text either way — and the page text in
``pages.json`` is untouched.

Confidence comes from agreement between three independent facts, which is what
keeps the detector from eating a genuine numbered list at the end of a page:

*Position* — the block starts in the bottom third of the page.
*Size* — it is set smaller than the page's body text.
*Form* — it opens with a footnote marker (``1.``, ``2.``, ``*``) and its content
looks like an amendment note (``Ins. by``, ``Subs. by``, ``w.e.f.``, ``vide``)
or the marker sequence restarts from 1 at the foot of the page.

Position and size come from the backend's per-line geometry. When that geometry
could not be aligned to the text lines, this module returns nothing rather than
falling back to the weaker text-only signals — a wrongly separated line is worse
than an unseparated one.
"""

from __future__ import annotations

import re
from statistics import median
from typing import Iterable, Optional

from . import config
from .models import FootnoteBlock

#: A footnote marker at the start of a line: ``1.`` ``12)`` ``*`` ``**``.
_MARKER = re.compile(r"^\s*(?:(?P<number>\d{1,3})\s*[.)]|(?P<star>\*{1,3}))\s+(?=\S)")

#: The vocabulary of an India Code amendment note.
_AMENDMENT_CUE = re.compile(
    r"\b(?:ins\.|subs\.|sub\.|omitted|substituted|inserted|rep\.|repealed|"
    r"w\.e\.f\.|vide|see\s+now|now\s+see|renumbered|added\s+by|"
    r"act\s+\d+\s+of\s+\d{4}|s\.\s*\d+|ss\.\s*\d+|notifn\.|notification\s+no)",
    re.IGNORECASE,
)


def document_body_size(pages: Iterable) -> Optional[float]:
    """The size this *document* sets its body text in.

    Measured across every page rather than per page, because a per-page measure
    fails on exactly the pages that matter. India Code's Code of Civil Procedure
    has pages where the amendment note runs to two thirds of the page: there the
    per-page median *is* the footnote size, the footnotes then fail a
    "smaller than the body" test, and nothing is separated. Over a whole
    document the body text is the commonest size by a wide margin.

    Only lines in the upper part of each page are counted — the region footnotes
    do not occupy — so a document whose notes are voluminous still measures its
    body correctly. Counted by line and rounded to a tenth of a point, so a
    handful of oversized headings cannot move it.
    """
    counts: dict[float, int] = {}
    fallback: dict[float, int] = {}
    for page in pages:
        height = getattr(page, "height", 0.0) or 0.0
        cutoff = height * config.FOOTNOTE_MIN_PAGE_FRACTION
        for metric in getattr(page, "line_metrics", None) or []:
            if metric.size <= 0:
                continue
            key = round(metric.size, 1)
            fallback[key] = fallback.get(key, 0) + 1
            if height and metric.y0 < cutoff:
                counts[key] = counts.get(key, 0) + 1
    chosen = counts or fallback
    if not chosen:
        return None
    # Commonest size; on a tie prefer the larger, since body text is never the
    # smaller of two equally common sizes on a page that has footnotes.
    return max(chosen.items(), key=lambda item: (item[1], item[0]))[0]


def detect_page(
    text: str,
    metrics,
    *,
    page_height: float,
    body_size: Optional[float] = None,
    furniture_indices: Optional[set[int]] = None,
) -> list[tuple[int, int]]:
    """Find the footnote block on one page. Returns ``(start, end)`` line indices.

    Both ends are inclusive and index ``text.split("\\n")``, so a caller can map
    a block back to the exact lines of the untouched page text.

    The block is found as the run of smaller-set type that runs to the foot of
    the page — footnotes are, by definition, at the foot — opened by a footnote
    marker and reading like amendment notes. Requiring the run to reach the last
    line of the page is what keeps a genuine numbered list in the middle of a
    page from being taken, and requiring larger type somewhere above it is what
    keeps a page that is entirely footnote continuation from being swallowed.
    """
    if not metrics or page_height <= 0:
        return []
    lines = text.split("\n")
    furniture_indices = furniture_indices or set()
    body = body_size or _page_body_size(metrics)
    if not body:
        return []

    by_index = {m.line_index: m for m in metrics}
    size_limit = body * config.FOOTNOTE_MAX_SIZE_RATIO

    content = [
        index for index in sorted(by_index)
        if index not in furniture_indices and lines[index].strip()
    ]
    if not content:
        return []

    # The trailing run of small type.
    run_start = None
    for index in reversed(content):
        if by_index[index].size > size_limit:
            break
        run_start = index
    if run_start is None:
        return []
    end = content[-1]

    # The block has to end at the foot of the page…
    if by_index[end].y0 < page_height * config.FOOTNOTE_MIN_PAGE_FRACTION:
        return []
    # …and there has to be body-sized type above it, or this is a continuation
    # page rather than a page with footnotes on it.
    if not any(by_index[i].size > size_limit for i in content if i < run_start):
        return []

    run = [i for i in content if run_start <= i <= end]
    if len(run) > config.FOOTNOTE_MAX_LINES:
        return []

    markers = [(i, _MARKER.match(lines[i])) for i in run]
    marked = [(i, m) for i, m in markers if m]
    if not marked:
        return []
    start = marked[0][0]
    if start != run_start:
        # The run opens with continuation text from the previous page's note;
        # the block still starts where the small type does.
        start = run_start

    block_text = " ".join(lines[i] for i in run)
    numbers = [int(m.group("number")) for _, m in marked if m.group("number")]
    looks_like_notes = bool(_AMENDMENT_CUE.search(block_text))
    restarts_at_one = bool(numbers) and numbers[0] == 1
    if not (looks_like_notes or restarts_at_one):
        return []

    return [(start, end)]


def _page_body_size(metrics) -> Optional[float]:
    sizes = [m.size for m in metrics if m.size > 0]
    return median(sizes) if sizes else None


def detect(pages: Iterable) -> dict[int, list[FootnoteBlock]]:
    """Find footnote blocks across a document's pages.

    Returns ``{page_number: [FootnoteBlock, ...]}``. Pages whose line geometry
    could not be aligned contribute nothing — a wrongly separated line is worse
    than an unseparated one.
    """
    pages = list(pages)
    body_size = document_body_size(pages)
    found: dict[int, list[FootnoteBlock]] = {}
    for page in pages:
        metrics = getattr(page, "line_metrics", None)
        if not metrics:
            continue
        furniture_indices = {f.line_index for f in getattr(page, "furniture", [])}
        blocks = detect_page(
            page.text, metrics,
            page_height=page.height,
            body_size=body_size,
            furniture_indices=furniture_indices,
        )
        if not blocks:
            continue
        lines = page.text.split("\n")
        found[page.page_number] = [
            FootnoteBlock(
                page_number=page.page_number,
                line_start=start,
                line_end=end,
                text="\n".join(
                    lines[i] for i in range(start, end + 1)
                    if i not in furniture_indices
                ),
                detected_by="foot_of_page_smaller_type_with_marker",
            )
            for start, end in blocks
        ]
    return found
