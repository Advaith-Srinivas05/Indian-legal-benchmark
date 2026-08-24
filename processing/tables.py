"""Which table candidates are actually tables.

The first benchmark retained 615 candidates. An audit of them found that only
about 140 had every column populated over three or more rows; the rest were
arrangement-of-sections listings and footnote blocks that happen to occupy two
columns, admitted because the only tests were "two rows, two columns, half the
cells filled". A contents listing reported as a table is worse than a missed
table: a later stage would treat it as data.

The policy is therefore explicitly a **conservative lower bound**. A candidate
has to look like a grid — enough rows, columns that are actually populated, rows
that agree on their shape — and must not look like the two things that
masquerade as one. Every rejection keeps the candidate and records why, so the
detector's behaviour stays measurable instead of disappearing into a count.

Retention is decided here rather than in :mod:`processing.extract` because the
strongest evidence is contextual: a grid on an arrangement-of-sections page is
not a table, and which pages those are is only known once the structure has been
parsed. The function is idempotent, so extraction may call it without that
context and processing may call it again with it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from . import config, patterns

#: Amendment-note vocabulary. A "table" whose cells are mostly this is the
#: footnote block at the foot of an India Code page, not tabular data.
_FOOTNOTE_CUE = re.compile(
    r"\b(?:ins\.|subs\.|omitted|substituted|inserted|rep\.|repealed|w\.e\.f\.|"
    r"vide|see\s+now|s\.\s*\d|sec\.\s*\d|act\s+\d+\s+of\s+\d{4})",
    re.IGNORECASE,
)


def _cells(rows: list[list[Optional[str]]]) -> list[list[str]]:
    return [[(cell or "").strip() for cell in row] for row in rows]


def _column_fill(rows: list[list[str]], columns: int) -> list[float]:
    if not rows or not columns:
        return []
    fills = []
    for index in range(columns):
        filled = sum(1 for row in rows if index < len(row) and row[index])
        fills.append(filled / len(rows))
    return fills


def _row_consistency(rows: list[list[str]]) -> float:
    """Share of rows that agree with the commonest row shape.

    A grid has the same number of populated cells on nearly every row. A listing
    that wanders between one and three cells does not, and that difference is
    what separates a real two-column table from a two-column page of prose.
    """
    if not rows:
        return 0.0
    shapes = Counter(sum(1 for cell in row if cell) for row in rows)
    return shapes.most_common(1)[0][1] / len(rows)


def _contents_entry_ratio(rows: list[list[str]]) -> float:
    """Share of first-column cells that read as contents entries.

    Only the "number plus caption" form counts. A column of *bare* numbers is
    the opposite signal — a serial-number column is the hallmark of a real
    schedule — and counting it here would reject every table in the corpus that
    starts with "Sl. No.".
    """
    first_column = [row[0] for row in rows if row and row[0]]
    if not first_column:
        return 0.0
    hits = sum(1 for cell in first_column if patterns.TOC_ENTRY_RE.match(cell))
    return hits / len(first_column)


def _footnote_cue_ratio(rows: list[list[str]]) -> float:
    populated = [cell for row in rows for cell in row if cell]
    if not populated:
        return 0.0
    return sum(1 for cell in populated if _FOOTNOTE_CUE.search(cell)) / len(populated)


def evaluate(table, *, on_contents_page: bool = False) -> tuple[bool, Optional[str]]:
    """Decide one candidate. Returns ``(retained, rejection reason)``."""
    rows = _cells(table.rows)
    row_count = len(rows)
    column_count = max((len(row) for row in rows), default=0)

    if row_count < config.TABLE_MIN_ROWS:
        return False, f"only {row_count} row(s); a grid needs {config.TABLE_MIN_ROWS}"
    if column_count < config.TABLE_MIN_COLS:
        return False, f"only {column_count} column(s)"

    total_cells = row_count * column_count
    filled = sum(1 for row in rows for cell in row if cell)
    filled_ratio = filled / total_cells if total_cells else 0.0
    if filled_ratio < config.TABLE_MIN_FILLED_RATIO:
        return False, (
            f"only {filled_ratio:.0%} of cells contain text "
            f"(need {config.TABLE_MIN_FILLED_RATIO:.0%})"
        )

    fills = _column_fill(rows, column_count)
    if fills and min(fills) < config.TABLE_MIN_COLUMN_FILL:
        return False, (
            f"one column is only {min(fills):.0%} populated "
            f"(need {config.TABLE_MIN_COLUMN_FILL:.0%}); this is one column of "
            "text with a gutter beside it, not a grid"
        )

    consistency = _row_consistency(rows)
    if consistency < config.TABLE_MIN_ROW_CONSISTENCY:
        return False, (
            f"rows disagree about their shape (only {consistency:.0%} share the "
            "commonest cell count)"
        )

    # Footnotes before contents: a numbered amendment note ("1. Ins. by Act 30
    # of 1965") satisfies both shapes, and naming the more specific one makes
    # the rejection reason useful rather than merely correct.
    footnote_ratio = _footnote_cue_ratio(rows)
    if footnote_ratio >= 0.5:
        return False, (
            f"{footnote_ratio:.0%} of cells are amendment notes "
            "('Ins. by …', 'w.e.f. …'); this is a footnote block"
        )

    contents_ratio = _contents_entry_ratio(rows)
    if contents_ratio > config.TABLE_MAX_CONTENTS_ENTRY_RATIO:
        return False, (
            f"{contents_ratio:.0%} of first-column cells read as contents entries "
            "('12. Short title.'); this is an arrangement-of-sections listing"
        )

    if on_contents_page:
        return False, "this page is an arrangement-of-sections / contents listing"

    return True, None


def classify(pages: Iterable, *, toc_pages: Optional[set[int]] = None) -> dict:
    """Apply :func:`evaluate` to every candidate on every page.

    Mutates the tables in place and returns a summary. Idempotent: calling it
    again with better context (the set of contents pages) simply re-decides.
    """
    toc_pages = toc_pages or set()
    retained = 0
    rejected: Counter = Counter()
    for page in pages:
        on_contents_page = page.page_number in toc_pages
        for table in page.tables:
            keep, reason = evaluate(table, on_contents_page=on_contents_page)
            table.retained = keep
            table.rejected_reason = reason
            if keep:
                retained += 1
            else:
                rejected[_reason_key(reason)] += 1
    return {
        "retained": retained,
        "rejected": sum(rejected.values()),
        "rejected_by_reason": dict(rejected.most_common()),
        "policy": (
            "Conservative lower bound: a candidate must look like a populated "
            "grid and must not look like a contents listing or a footnote block. "
            "Rejected candidates are kept with their reason."
        ),
    }


def _reason_key(reason: Optional[str]) -> str:
    if not reason:
        return "unknown"
    if "row(s)" in reason:
        return "too_few_rows"
    if "column(s)" in reason:
        return "too_few_columns"
    if "of cells contain text" in reason:
        return "too_empty"
    if "populated" in reason:
        return "empty_column"
    if "disagree about their shape" in reason:
        return "inconsistent_rows"
    if "contents entries" in reason:
        return "contents_listing"
    if "contents listing" in reason:
        return "on_contents_page"
    if "footnote block" in reason:
        return "footnote_block"
    return "other"
