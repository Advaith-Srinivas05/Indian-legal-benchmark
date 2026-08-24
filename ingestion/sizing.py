"""Estimate the corpus's disk footprint from HTTP headers alone.

The inventory records *which* PDF to download but nothing about how large it
is, so the only honest way to size the corpus short of fetching it is to ask
the server. This module samples the inventory and reads ``Content-Length`` off
response **headers**; no PDF body is ever transferred, and nothing is written
to ``data/raw/``.

Two request shapes are needed because India Code answers HEAD inconsistently:

* ``/bitstream/…`` (Acts) usually answers HEAD directly with
  ``Content-Type: application/pdf`` and ``Content-Length``. A minority answer
  HEAD with a bare ``302`` carrying no ``Location`` at all.
* ``/ViewFileUploaded?…`` (Rules/Regulations) *always* answers HEAD that way,
  so the redirect cannot be followed from a HEAD alone.

Where HEAD is useless the probe falls back to a **streaming GET whose body is
never read**: ``stream=True`` returns once the headers arrive, and the
connection is released without a single call to ``iter_content``, so
``Content-Length`` is learned without transferring the PDF. Redirects are
resolved by hand and re-checked against the same India Code host allow-list the
downloader uses, so the size measured is the size the download would fetch.

Sampling is stratified by document type and, within a type, spread round-robin
across jurisdictions, so 35 States are represented rather than whichever ones
happen to sort first. Each type is extrapolated from its own sample, which
keeps the estimate unbiased no matter how the sample was allocated between
types.
"""

from __future__ import annotations

import logging
import math
import random
import statistics
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional, Sequence
from urllib.parse import urljoin, urlparse

import requests

from . import config
from .http_client import RateLimiter, SessionPool
# The same repair the downloader applies, so a probe measures exactly the file a
# download would fetch.
from .utils import repair_location, utcnow_iso

log = logging.getLogger(__name__)

#: Probe outcomes.
OK = "ok"
#: Reached the server, but it would not say how big the file is.
UNKNOWN = "unknown"
#: Could not reach the file at all (HTTP error, timeout, bad redirect).
ERROR = "error"

_PDF_CONTENT_TYPES = ("application/pdf", "application/x-pdf", "application/octet-stream")


@dataclass
class Probe:
    """The result of asking the server how large one document is."""

    document_id: str
    document_type: str
    jurisdiction: str
    pdf_url_kind: str
    url: str
    status: str = UNKNOWN
    content_length: Optional[int] = None
    content_type: Optional[str] = None
    http_status: Optional[int] = None
    final_url: Optional[str] = None
    #: How the size was obtained: ``head`` or ``redirect+head``.
    method: Optional[str] = None
    #: Why a probe is ``unknown``/``error``.
    reason: Optional[str] = None
    #: Why HEAD was abandoned, when the headers-only GET had to take over.
    fallback_reason: Optional[str] = None


# --- sampling -------------------------------------------------------------------


def stratified_sample(
    documents: Sequence[dict],
    *,
    per_type: int,
    seed: int = 20260817,
) -> list[dict]:
    """Pick up to *per_type* documents from each document type.

    Within a type the picks are dealt round-robin across jurisdictions, so a
    type spanning 35 States contributes all 35 rather than a random draw that
    over-weights the largest ones. Ordering within a jurisdiction is shuffled
    with a fixed *seed*, which makes the whole sample reproducible.
    """
    by_type: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for document in documents:
        document_type = (document.get("document_type") or "unknown").strip()
        jurisdiction = (document.get("jurisdiction") or "unknown").strip()
        by_type[document_type][jurisdiction].append(document)

    rng = random.Random(seed)
    sample: list[dict] = []
    for document_type in sorted(by_type):
        buckets = []
        for jurisdiction in sorted(by_type[document_type]):
            group = list(by_type[document_type][jurisdiction])
            rng.shuffle(group)
            buckets.append(group)
        taken = 0
        while taken < per_type and any(buckets):
            for bucket in buckets:
                if not bucket:
                    continue
                sample.append(bucket.pop())
                taken += 1
                if taken >= per_type:
                    break
            buckets = [b for b in buckets if b]
    return sample


# --- probing --------------------------------------------------------------------


def _host_allowed(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(
        host == suffix or host.endswith("." + suffix)
        for suffix in config.ALLOWED_DOWNLOAD_HOST_SUFFIXES
    )


def _looks_like_pdf(content_type: str | None) -> bool:
    return any(kind in (content_type or "").lower() for kind in _PDF_CONTENT_TYPES)


def _content_length(response: requests.Response) -> Optional[int]:
    raw = response.headers.get("Content-Length")
    if raw is None:
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        return None
    return value if value > 0 else None


#: How many redirect hops the probe will resolve by hand before giving up.
MAX_HOPS = 3


def probe_size(
    session: requests.Session,
    url: str,
    *,
    kind: str,
    limiter: Optional[RateLimiter] = None,
    **identity,
) -> Probe:
    """Ask the server for one document's size without fetching its body."""
    probe = Probe(
        document_id=identity.get("document_id", ""),
        document_type=identity.get("document_type", ""),
        jurisdiction=identity.get("jurisdiction", ""),
        pdf_url_kind=kind,
        url=url,
    )
    try:
        if kind == "viewfileuploaded":
            # HEAD here is always a Location-less 302; go straight to the
            # header-only GET that can actually see the redirect.
            _probe_stream(session, url, probe, limiter, hops=0)
        else:
            _probe_head(session, url, probe, limiter, hops=0)
    except requests.RequestException as exc:
        probe.status = ERROR
        probe.reason = f"{type(exc).__name__}: {exc}"
    return probe


def _probe_head(
    session: requests.Session,
    url: str,
    probe: Probe,
    limiter: Optional[RateLimiter],
    *,
    hops: int,
) -> None:
    """HEAD *url*, falling back to a header-only GET when HEAD is no use."""
    if limiter:
        limiter.wait()
    response = session.head(url, allow_redirects=True, timeout=config.REQUEST_TIMEOUT)
    probe.http_status = response.status_code
    probe.final_url = response.url
    probe.content_type = response.headers.get("Content-Type")
    probe.method = "head" if hops == 0 else "redirect+head"
    if response.status_code >= 400:
        probe.status = ERROR
        probe.reason = f"HTTP {response.status_code} on HEAD"
        return
    if not _host_allowed(response.url):
        probe.status = ERROR
        probe.reason = f"redirected off India Code to {response.url!r}"
        return
    if response.status_code in (301, 302, 303, 307, 308):
        # A redirect requests could not follow: India Code answers HEAD for some
        # bitstreams with a 302 that carries no Location at all. The same URL
        # answers a GET properly, so ask again for headers only.
        _fall_back(
            session, url, probe, limiter, hops,
            f"HTTP {response.status_code} on HEAD with no followable Location",
        )
        return
    length = _content_length(response)
    if length is None:
        _fall_back(session, url, probe, limiter, hops, "no Content-Length header on HEAD")
        return
    if not _looks_like_pdf(probe.content_type):
        probe.status = UNKNOWN
        probe.reason = f"response is not a PDF (Content-Type={probe.content_type!r})"
        return
    probe.content_length = length
    probe.status = OK
    probe.reason = None


def _fall_back(
    session: requests.Session,
    url: str,
    probe: Probe,
    limiter: Optional[RateLimiter],
    hops: int,
    why: str,
) -> None:
    """HEAD could not answer; retry for headers only and remember why."""
    probe.status = UNKNOWN
    probe.reason = why
    probe.fallback_reason = why
    _probe_stream(session, url, probe, limiter, hops=hops)


def _probe_stream(
    session: requests.Session,
    url: str,
    probe: Probe,
    limiter: Optional[RateLimiter],
    *,
    hops: int,
) -> None:
    """Read a response's headers with a GET whose body is never touched.

    ``stream=True`` means requests returns as soon as the headers arrive, and
    the ``with`` block releases the connection without a single call to
    ``iter_content``/``.content`` — so the size is learned from
    ``Content-Length`` and the PDF body is not transferred. Redirects are
    resolved by hand (``allow_redirects=False``) because that is the only way
    to see India Code's ``/ViewFileUploaded`` ``Location`` header at all.
    """
    if hops >= MAX_HOPS:
        probe.status = ERROR
        probe.reason = f"more than {MAX_HOPS} redirects"
        return
    if limiter:
        limiter.wait()
    with session.get(
        url, allow_redirects=False, stream=True, timeout=config.REQUEST_TIMEOUT
    ) as response:
        probe.http_status = response.status_code
        probe.final_url = response.url
        location = response.headers.get("Location")
        if response.status_code >= 400:
            probe.status = ERROR
            probe.reason = f"HTTP {response.status_code} on {url[:80]}"
            return
        if response.status_code < 300:
            probe.content_type = response.headers.get("Content-Type")
            probe.method = "stream-headers" if hops == 0 else "redirect+stream-headers"
            length = _content_length(response)
            if length is None:
                probe.status = UNKNOWN
                probe.reason = "no Content-Length header"
            elif not _looks_like_pdf(probe.content_type):
                probe.status = UNKNOWN
                probe.reason = f"response is not a PDF (Content-Type={probe.content_type!r})"
            else:
                probe.content_length = length
                probe.status = OK
                probe.reason = None
            return
    if not location:
        probe.status = UNKNOWN
        probe.reason = f"HTTP {probe.http_status} without a Location header"
        return
    # Relative Locations are legal; resolve against the request URL rather than
    # assembling the target by hand. The query string is otherwise passed
    # through as the server wrote it.
    target = urljoin(url, repair_location(location))
    if not _host_allowed(target):
        probe.status = ERROR
        probe.reason = f"redirect leaves India Code: {target!r}"
        return
    _probe_head(session, target, probe, limiter, hops=hops + 1)


def probe_documents(
    documents: Sequence[dict],
    *,
    workers: int = 4,
    delay: float = 0.25,
    progress: Optional[Callable[[int, int], None]] = None,
) -> list[Probe]:
    """Probe every document in *documents* concurrently but politely."""
    limiter = RateLimiter(delay)
    pool = SessionPool()
    probes: list[Probe] = []
    total = len(documents)

    def work(document: dict) -> Probe:
        return probe_size(
            pool.get(),
            (document.get("english_pdf_url") or "").strip(),
            kind=(document.get("pdf_url_kind") or "").strip(),
            limiter=limiter,
            document_id=document.get("document_id") or "",
            document_type=document.get("document_type") or "",
            jurisdiction=document.get("jurisdiction") or "",
        )

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(work, d): d for d in documents}
        for done, future in enumerate(as_completed(futures), 1):
            probes.append(future.result())
            if progress:
                progress(done, total)
    return probes


# --- statistics -----------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (``q`` in 0..100) of *values*."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def describe(sizes: Sequence[int]) -> dict:
    """Mean/median/percentile summary of a set of file sizes, in bytes."""
    if not sizes:
        return {"count": 0}
    ordered = sorted(sizes)
    return {
        "count": len(ordered),
        "total_bytes": sum(ordered),
        "mean_bytes": statistics.fmean(ordered),
        "median_bytes": statistics.median(ordered),
        "p90_bytes": percentile(ordered, 90),
        "p95_bytes": percentile(ordered, 95),
        "p99_bytes": percentile(ordered, 99),
        "min_bytes": ordered[0],
        "max_bytes": ordered[-1],
        "stdev_bytes": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
    }


def _standard_error(sizes: Sequence[int], population: int) -> float:
    """Standard error of the sample mean, with a finite-population correction."""
    n = len(sizes)
    if n < 2:
        return 0.0
    se = statistics.stdev(sizes) / math.sqrt(n)
    if population > 1 and n <= population:
        se *= math.sqrt((population - n) / (population - 1))
    return se


def estimate(
    population_by_type: dict[str, int],
    probes: Iterable[Probe],
) -> dict:
    """Extrapolate the corpus size from the probes, stratified by type.

    Each document type is scaled by its own sample mean, and the 95% interval
    is the usual ``1.96 × standard error`` widened by the population size, so
    the report says how firm the number is rather than only how big.
    """
    probes = list(probes)
    sized: dict[str, list[int]] = defaultdict(list)
    for probe in probes:
        if probe.status == OK and probe.content_length:
            sized[probe.document_type].append(probe.content_length)

    by_type: dict[str, dict] = {}
    total_bytes = 0.0
    total_variance = 0.0
    median_total = 0.0
    covered = 0
    uncovered: list[str] = []
    for document_type, population in sorted(population_by_type.items()):
        sizes = sized.get(document_type, [])
        stats = describe(sizes)
        entry = {"population": population, "sample": stats}
        if sizes:
            se = _standard_error(sizes, population)
            estimated = population * stats["mean_bytes"]
            entry["estimated_bytes"] = estimated
            entry["estimated_bytes_ci95"] = 1.96 * population * se
            entry["estimated_bytes_by_median"] = population * stats["median_bytes"]
            total_bytes += estimated
            total_variance += (population * se) ** 2
            median_total += population * stats["median_bytes"]
            covered += population
        else:
            entry["estimated_bytes"] = None
            entry["reason"] = "no successful probe for this document type"
            uncovered.append(document_type)
        by_type[document_type] = entry

    return {
        "population": sum(population_by_type.values()),
        "population_covered": covered,
        "types_without_a_sample": uncovered,
        "estimated_total_bytes": total_bytes if covered else None,
        "estimated_total_bytes_ci95": 1.96 * math.sqrt(total_variance) if covered else None,
        "estimated_total_bytes_by_median": median_total if covered else None,
        "by_document_type": by_type,
    }


def estimate_by_jurisdiction(
    population: dict[tuple[str, str], int],
    probes: Iterable[Probe],
    *,
    min_sample: int = 5,
) -> dict:
    """Per-jurisdiction estimate, falling back to the type mean when thin.

    *population* maps ``(jurisdiction, document_type)`` to a document count. A
    jurisdiction/type cell with fewer than *min_sample* probes borrows that
    type's overall mean, and the result says so, because a State represented by
    two files should not be reported as if it had been measured.
    """
    probes = list(probes)
    cell_sizes: dict[tuple[str, str], list[int]] = defaultdict(list)
    type_sizes: dict[str, list[int]] = defaultdict(list)
    for probe in probes:
        if probe.status == OK and probe.content_length:
            cell_sizes[(probe.jurisdiction, probe.document_type)].append(probe.content_length)
            type_sizes[probe.document_type].append(probe.content_length)

    result: dict[str, dict] = {}
    for (jurisdiction, document_type), count in population.items():
        sizes = cell_sizes.get((jurisdiction, document_type), [])
        measured = len(sizes) >= min_sample
        if measured:
            mean = statistics.fmean(sizes)
        elif type_sizes.get(document_type):
            mean = statistics.fmean(type_sizes[document_type])
        else:
            continue
        row = result.setdefault(
            jurisdiction,
            {"documents": 0, "estimated_bytes": 0.0, "sampled": 0, "fully_measured": True},
        )
        row["documents"] += count
        row["estimated_bytes"] += count * mean
        row["sampled"] += len(sizes)
        if not measured:
            row["fully_measured"] = False
    return dict(sorted(result.items(), key=lambda kv: -kv[1]["estimated_bytes"]))


def build_report(
    *,
    inventory_path: str,
    documents: Sequence[dict],
    sample: Sequence[dict],
    probes: Sequence[Probe],
    per_type: int,
    seed: int,
    workers: int,
    delay: float,
) -> dict:
    """Assemble the JSON document saved to ``data/discovery/size_estimate.json``."""
    population_by_type: dict[str, int] = defaultdict(int)
    population_by_cell: dict[tuple[str, str], int] = defaultdict(int)
    for document in documents:
        document_type = (document.get("document_type") or "unknown").strip()
        jurisdiction = (document.get("jurisdiction") or "unknown").strip()
        population_by_type[document_type] += 1
        population_by_cell[(jurisdiction, document_type)] += 1

    sizes = [p.content_length for p in probes if p.status == OK and p.content_length]
    by_status: dict[str, int] = defaultdict(int)
    by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_method: dict[str, int] = defaultdict(int)
    for probe in probes:
        by_status[probe.status] += 1
        by_kind[probe.pdf_url_kind][probe.status] += 1
        if probe.status == OK:
            by_method[probe.method or "unknown"] += 1

    failures = [
        asdict(p) for p in probes if p.status != OK
    ][:100]

    return {
        "schema_version": 1,
        "generated_at": utcnow_iso(),
        "method": (
            "HTTP header probe only: HEAD for /bitstream URLs; where the server "
            "answers HEAD with a Location-less 302 (always for /ViewFileUploaded, "
            "occasionally for /bitstream) a streaming GET is used to read the "
            "headers and the redirect target, and its body is released without "
            "ever being read. No PDF body was downloaded and no file was written "
            "under data/raw/."
        ),
        "inventory": {
            "path": inventory_path,
            "documents": len(documents),
            "by_document_type": dict(sorted(population_by_type.items())),
        },
        "sampling": {
            "strategy": "stratified by document_type, round-robin across jurisdictions",
            "per_type": per_type,
            "seed": seed,
            "workers": workers,
            "delay_seconds": delay,
            "sampled": len(sample),
            "probed": len(probes),
        },
        "outcomes": {
            "known_content_length": by_status.get(OK, 0),
            "unknown": by_status.get(UNKNOWN, 0),
            "error": by_status.get(ERROR, 0),
            "by_url_kind": {kind: dict(v) for kind, v in sorted(by_kind.items())},
            "sized_by_method": dict(sorted(by_method.items())),
        },
        "sample_statistics": describe(sizes),
        "sample_statistics_by_url_kind": {
            kind: describe(
                [
                    p.content_length
                    for p in probes
                    if p.pdf_url_kind == kind and p.status == OK and p.content_length
                ]
            )
            for kind in sorted({p.pdf_url_kind for p in probes})
        },
        "estimate": estimate(dict(population_by_type), probes),
        "estimate_by_jurisdiction": estimate_by_jurisdiction(dict(population_by_cell), probes),
        "failed_probes": failures,
        "probes": [asdict(p) for p in probes],
    }
