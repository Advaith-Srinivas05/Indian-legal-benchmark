"""Running the whole corpus through the pipeline: concurrency, retries, reports.

This adds nothing to what a single document does — every target still goes
through :func:`ingestion.pipeline.ingest_document` (resolve -> language gate ->
download -> PDF validation -> SHA-256 -> store -> metadata -> manifest ->
versioning). What it adds is what 20,000 of them need:

* a small, fixed worker pool and a **global** minimum interval between
  documents, so a public government site sees a handful of connections rather
  than a stampede;
* a bounded retry with exponential backoff for *transient* failures only —
  India Code intermittently drops the file block from an act page and
  occasionally 5xxs — while a permanent failure (not English, not a PDF, bad
  URL) fails once and is recorded;
* resumption by skipping documents already in the manifest, so an interrupted
  run continues instead of starting over;
* checkpointed manifest saves, because rewriting a 20k-entry manifest after
  every document is quadratic;
* a persistent, append-only failure report and periodic progress output.

Nothing here weakens validation: a document that fails any check is a failure,
never a stored file.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from . import config
from .http_client import RateLimiter, SessionPool
from .manifest import Manifest
from .models import DownloadResult, Outcome
from .pipeline import ingest_document
from .storage import DocumentStore
from .utils import utcnow_iso

log = logging.getLogger(__name__)

#: Failures worth another attempt: the network was unhappy, or India Code served
#: an act page without its file block (a known intermittent fault of the site).
#: Everything else — LanguageError, NotPDFError, CorruptPDFError,
#: InvalidURLError, DocumentTypeError — is a property of the document, and
#: retrying it would only repeat the same answer.
TRANSIENT_ERRORS = frozenset({"FetchError", "MetadataError", "ManifestError"})

#: HTTP statuses that mean "ask again later" rather than "this is the answer".
#: Any other 4xx is the server stating the file is not there, so repeating the
#: request only repeats the answer.
RETRYABLE_STATUSES = frozenset({408, 409, 425, 429})


def is_transient(result: DownloadResult) -> bool:
    """Whether *result* is worth another attempt."""
    if (result.error_type or "") not in TRANSIENT_ERRORS:
        return False
    status = result.http_status
    if status is None or status >= 500:
        return True
    return status in RETRYABLE_STATUSES

#: Stop the run rather than fill the disk completely.
MIN_HEADROOM_BYTES = 5 * 1024 ** 3


@dataclass
class Stats:
    """Live counters for one batch run."""

    total: int = 0
    processed: int = 0
    new: int = 0
    unchanged: int = 0
    updated: int = 0
    failed: int = 0
    skipped: int = 0
    retried: int = 0
    bytes: int = 0
    failures_by_type: dict = field(default_factory=dict)
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.monotonic() - self.started_at)

    @property
    def rate(self) -> float:
        """Documents per second, averaged over the run."""
        return self.processed / self.elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        remaining = self.total - self.processed
        if remaining <= 0 or self.processed == 0:
            return 0.0 if remaining <= 0 else None
        return remaining / self.rate

    def record(self, result: DownloadResult) -> None:
        self.processed += 1
        self.bytes += result.bytes or 0
        if result.outcome is Outcome.NEW:
            self.new += 1
        elif result.outcome is Outcome.UPDATED:
            self.updated += 1
        elif result.outcome is Outcome.UNCHANGED:
            self.unchanged += 1
        elif result.outcome is Outcome.FAILED:
            self.failed += 1
            key = result.error_type or "Unknown"
            self.failures_by_type[key] = self.failures_by_type.get(key, 0) + 1


class FailureReport:
    """Append-only JSONL record of every document that did not store.

    Append-only and flushed per line so the report survives an interrupted run
    and can be inspected while the batch is still going.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0

    def record(self, target, result: DownloadResult, attempts: int) -> None:
        row = {
            "timestamp": utcnow_iso(),
            "document_id": result.document_id or getattr(target, "document_id", None),
            "title": getattr(target, "title", None),
            "document_type": config.CATEGORY_TO_DOCUMENT_TYPE.get(target.category),
            "category": target.category,
            "url": target.url,
            "parent_url": target.parent_url,
            "error_type": result.error_type or "Unknown",
            "http_status": result.http_status,
            "error_message": result.message,
            "attempts": attempts,
        }
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


def format_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.2f} {unit}" if unit != "B" else f"{value:,.0f} B"
        value /= 1024
    return f"{value:.2f} TB"


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def render_progress(stats: Stats) -> str:
    """The periodic checkpoint block."""
    return "\n".join([
        f"[{stats.processed}/{stats.total}]",
        f"  Downloaded: {stats.new}",
        f"  Unchanged:  {stats.unchanged}",
        f"  Updated:    {stats.updated}",
        f"  Failed:     {stats.failed}",
        f"  Bytes:      {format_bytes(stats.bytes)}",
        f"  Rate:       {stats.rate * 60:.1f} docs/min"
        f"  ({format_bytes(stats.bytes / stats.elapsed)}/s)",
        f"  ETA:        {format_duration(stats.eta_seconds)}",
    ])


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def check_free_space(path: Path, required_bytes: int) -> tuple[bool, int]:
    """Is there room to start? Returns ``(ok, free_bytes)``."""
    path.mkdir(parents=True, exist_ok=True)
    free = free_bytes(path)
    return free >= required_bytes, free


def partition_targets(targets: Sequence, manifest: Manifest) -> tuple[list, list]:
    """Split into (to process, already in the manifest).

    Resumption is exactly this: a document the manifest already knows is left
    alone, so re-running after an interruption costs one dictionary lookup per
    finished document rather than a network round-trip.
    """
    todo, done = [], []
    for target in targets:
        document_id = getattr(target, "document_id", None)
        if document_id and manifest.get(document_id) is not None:
            done.append(target)
        else:
            todo.append(target)
    return todo, done


def run_batch(
    targets: Sequence,
    store: DocumentStore,
    manifest: Manifest,
    *,
    workers: int = 3,
    delay: float = 0.3,
    retries: int = 2,
    retry_backoff: float = 3.0,
    checkpoint_every: int = 50,
    progress_every: int = 250,
    failures: Optional[FailureReport] = None,
    on_progress: Optional[Callable[[Stats], None]] = None,
    stop: Optional[threading.Event] = None,
) -> Stats:
    """Ingest every target. Never raises for a per-document failure."""
    stats = Stats(total=len(targets))
    limiter = RateLimiter(delay)
    pool = SessionPool()
    stop = stop or threading.Event()
    lock = threading.Lock()
    state = {"since_checkpoint": 0}

    def work(target) -> tuple[object, DownloadResult, int]:
        attempts = 0
        result = None
        for attempt in range(1, retries + 2):
            if stop.is_set():
                break
            attempts = attempt
            limiter.wait()
            result = ingest_document(
                pool.get(), store, manifest, target.url, target.category,
                parent_url=target.parent_url, skip_existing=True,
            )
            if result.outcome is not Outcome.FAILED:
                return target, result, attempts
            if not is_transient(result) or attempt > retries:
                return target, result, attempts
            # Exponential backoff with jitter, so a struggling server is not
            # hit by every worker at the same instant.
            pause = retry_backoff * (2 ** (attempt - 1)) * (0.5 + random.random())
            log.warning(
                "Transient %s on %s (attempt %d/%d); retrying in %.1fs: %s",
                result.error_type, target.url[:90], attempt, retries + 1, pause,
                result.message[:160],
            )
            with lock:
                stats.retried += 1
            time.sleep(pause)
        return target, result, attempts

    try:
        with manifest.defer_saves():
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = [executor.submit(work, t) for t in targets]
                for future in as_completed(futures):
                    target, result, attempts = future.result()
                    if result is None:      # cancelled before it ran
                        continue
                    with lock:
                        stats.record(result)
                        if result.outcome is Outcome.FAILED and failures is not None:
                            failures.record(target, result, attempts)
                        state["since_checkpoint"] += 1
                        checkpoint = state["since_checkpoint"] >= checkpoint_every
                        if checkpoint:
                            state["since_checkpoint"] = 0
                        show = bool(progress_every) and stats.processed % progress_every == 0
                    if checkpoint:
                        manifest.flush()
                        if free_bytes(store.data_dir) < MIN_HEADROOM_BYTES:
                            log.error(
                                "Less than %s free on the target drive; stopping "
                                "the batch. Re-run after freeing space to continue.",
                                format_bytes(MIN_HEADROOM_BYTES),
                            )
                            stop.set()
                    if show and on_progress:
                        on_progress(stats)
                    if stop.is_set():
                        for pending in futures:
                            pending.cancel()
    finally:
        manifest.flush()
    return stats
