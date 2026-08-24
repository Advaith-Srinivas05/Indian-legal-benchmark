"""Deterministic, representative sampling of the corpus for the benchmark.

The project spec asks for approximately 100 documents that are both *deterministic* and
*representative*, and for the exact sample to be recorded. Two properties are
therefore non-negotiable:

**Deterministic.** No RNG and no clock. Ordering within a stratum is by
``sha256(document_id)``, which is stable, uniformly distributed and independent
of manifest ordering — so re-running the sampler on the same corpus selects
exactly the same 100 documents, and a benchmark number can be compared against
a later one.

**Representative.** Documents are stratified on three axes that are all readable
from the ingestion artefacts, so nothing has to be inferred from a PDF before it
is chosen:

``category``
    Central Acts / State Acts / Rules / Regulations, by fixed quota rather than
    in proportion to the corpus. State Acts are 51% of the corpus, but a sample
    that was half state acts would say almost nothing about the other three.

``era`` (act year)
    Typesetting conventions changed repeatedly between 1793 and 2026, and the
    extraction problems differ accordingly.

``size`` (stored PDF bytes)
    Short and large documents behave differently. The top band is also, in
    practice, where India Code's scanned gazette reproductions live — which is
    how the sample reaches scanned and mixed documents *without anyone
    pre-judging which documents those are*. Whether they are actually scanned is
    something the benchmark measures, not something the sampler assumes.

Jurisdiction is not a stratum but a tie-break: when several documents could fill
a cell, the one from the least-used state/UT wins, so the sample spans many
jurisdictions instead of clustering in the two largest.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Iterable, Optional

from . import config
from .errors import SamplingError
from .models import CorpusDocument


def era_band(year: Optional[int]) -> str:
    """Which era band a year falls in (see :data:`processing.config.ERA_BANDS`)."""
    if year is None:
        return config.ERA_UNKNOWN
    for name, low, high in config.ERA_BANDS:
        if (low is None or year >= low) and (high is None or year <= high):
            return name
    return config.ERA_UNKNOWN


def size_band(size_bytes: int) -> str:
    """Which size band a stored PDF falls in."""
    for name, low, high in config.SIZE_BANDS:
        if size_bytes >= low and (high is None or size_bytes < high):
            return name
    return config.SIZE_BANDS[-1][0]


def document_rank(document_id: str) -> str:
    """Stable, corpus-order-independent sort key for a document."""
    return hashlib.sha256(document_id.encode("utf-8")).hexdigest()


def _cell_order_key(cell: tuple[str, str]) -> tuple[int, int]:
    eras = [name for name, _, _ in config.ERA_BANDS] + [config.ERA_UNKNOWN]
    sizes = [name for name, _, _ in config.SIZE_BANDS]
    era, size = cell
    return (eras.index(era) if era in eras else len(eras),
            sizes.index(size) if size in sizes else len(sizes))


def scale_quotas(quotas: dict[str, int], target: int) -> dict[str, int]:
    """Rescale the category quotas to a different sample size.

    ``--sample-size 20`` has to keep the same *shape* as the full run, or a
    smaller benchmark would silently become a different benchmark. Largest
    remainder, so the parts still sum to the target exactly, and every category
    that had a quota keeps at least one document.
    """
    total = sum(quotas.values())
    if total == target or total == 0:
        return quotas
    exact = {name: quota * target / total for name, quota in quotas.items()}
    scaled = {name: max(1, int(value)) for name, value in exact.items()}
    remainder = target - sum(scaled.values())
    order = sorted(quotas, key=lambda name: (-(exact[name] - int(exact[name])), name))
    index = 0
    while remainder > 0 and order:
        scaled[order[index % len(order)]] += 1
        remainder -= 1
        index += 1
    while remainder < 0:
        # Over-allocated by the max(1, ...) floor; take back from the largest.
        largest = max(scaled, key=lambda name: (scaled[name], name))
        if scaled[largest] <= 1:
            break
        scaled[largest] -= 1
        remainder += 1
    return scaled


def select(
    documents: Iterable[CorpusDocument],
    *,
    target: Optional[int] = None,
    quotas: Optional[dict[str, int]] = None,
) -> list[dict]:
    """Choose the benchmark sample. Returns one record per selected document.

    Each record carries the document *and* why it was chosen — its category,
    era/size cell and rank within that cell — so the sample is auditable rather
    than a bare list of ids.
    """
    target = config.BENCHMARK_SAMPLE_SIZE if target is None else target
    quotas = scale_quotas(dict(config.CATEGORY_QUOTAS if quotas is None else quotas),
                          target)

    pool = [d for d in documents]
    if not pool:
        raise SamplingError("The corpus is empty; there is nothing to sample.")

    selected: list[dict] = []
    for category in sorted(quotas):
        candidates = [d for d in pool if d.category == category]
        selected.extend(_select_from_category(category, candidates, quotas[category]))
    taken = {record["document_id"] for record in selected}

    # A category smaller than its quota leaves the sample short. Top it up in
    # deterministic order so the sample still reaches its target size, and say
    # so in the record.
    if len(selected) < target:
        remaining = sorted(
            (d for d in pool if d.document_id not in taken),
            key=lambda d: document_rank(d.document_id),
        )
        for document in remaining[: target - len(selected)]:
            selected.append(_record(document, rank=-1, reason="quota_top_up"))

    selected.sort(key=lambda r: (r["category"], r["era_band"], r["size_band"],
                                 r["document_id"]))
    return selected


def _select_from_category(
    category: str, candidates: list[CorpusDocument], quota: int
) -> list[dict]:
    """Fill one category's quota by round-robin over its era/size cells."""
    if quota <= 0 or not candidates:
        return []

    cells: dict[tuple[str, str], list[CorpusDocument]] = defaultdict(list)
    for document in candidates:
        cells[(era_band(document.year), size_band(document.bytes))].append(document)
    for bucket in cells.values():
        bucket.sort(key=lambda d: document_rank(d.document_id))

    order = sorted(cells, key=_cell_order_key)
    jurisdictions: Counter = Counter()
    chosen: list[dict] = []

    while len(chosen) < quota:
        progressed = False
        for cell in order:
            if len(chosen) >= quota:
                break
            bucket = cells[cell]
            index = _next_index(bucket, jurisdictions)
            if index is None:
                continue
            # Removing the pick is what advances the cell: a document can never
            # be selected twice, whatever the jurisdiction preference did.
            document = bucket.pop(index)
            progressed = True
            jurisdictions[document.jurisdiction or "unknown"] += 1
            chosen.append(
                _record(
                    document,
                    rank=len(chosen),
                    reason=f"stratum {category}/{cell[0]}/{cell[1]}",
                )
            )
        if not progressed:
            break
    return chosen


def _next_index(
    bucket: list[CorpusDocument], jurisdictions: Counter
) -> Optional[int]:
    """Pick from *bucket*: least-used jurisdiction first, then hash order.

    ``bucket`` is already in hash order, so ``min`` over it with a
    (jurisdiction usage, position) key is deterministic.
    """
    if not bucket:
        return None
    return min(
        range(len(bucket)),
        key=lambda i: (jurisdictions[bucket[i].jurisdiction or "unknown"], i),
    )


def _record(document: CorpusDocument, *, rank: int, reason: str) -> dict:
    return {
        "document_id": document.document_id,
        "category": document.category,
        "document_type": document.document_type,
        "title": document.title,
        "jurisdiction": document.jurisdiction,
        "year": document.year,
        "bytes": document.bytes,
        "sha256": document.sha256,
        "pdf_relpath": document.pdf_relpath,
        "era_band": era_band(document.year),
        "size_band": size_band(document.bytes),
        "selection_rank": rank,
        "selection_reason": reason,
        "selection_key": document_rank(document.document_id)[:16],
    }


def describe(records: list[dict]) -> dict:
    """Coverage summary of a sample: how many of each category/era/size/state."""
    return {
        "size": len(records),
        "by_category": dict(Counter(r["category"] for r in records)),
        "by_era": dict(Counter(r["era_band"] for r in records)),
        "by_size_band": dict(Counter(r["size_band"] for r in records)),
        "by_jurisdiction": dict(Counter(r["jurisdiction"] or "unknown" for r in records)),
        "distinct_jurisdictions": len({r["jurisdiction"] or "unknown" for r in records}),
        "year_range": [
            min((r["year"] for r in records if r["year"]), default=None),
            max((r["year"] for r in records if r["year"]), default=None),
        ],
        "bytes_range": [
            min((r["bytes"] for r in records), default=None),
            max((r["bytes"] for r in records), default=None),
        ],
    }
