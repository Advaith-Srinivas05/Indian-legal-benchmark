"""Persistence and orchestration for the discovery phase.

Separated from :mod:`ingestion.discovery` (which knows how to *read* India
Code) the way :mod:`ingestion.storage` is separated from
:mod:`ingestion.indiacode`: this module knows how to *keep* what was found.

Files written under ``data/discovery/``::

    listings.json               cached browse results (phase 1)
    journal.jsonl               one line per inspected act (phase 2, append-only)
    indiacode_inventory.json    the deliverable: deduped entries + summary

Resumability rests on the journal. Each act appends exactly one line the moment
it is finished, so an interrupted run loses at most the act in flight; a
re-run reloads the journal, skips everything already in it and carries on. A
half-written final line (killed mid-write) is discarded on load rather than
being allowed to corrupt the run. ``listings.json`` and the inventory itself are
only ever written atomically, so a reader never sees a partial file.

Deduplication is by **English PDF URL**: that is exactly "one download job", so
the same document reached through two listings collapses to one entry while
every route to it is preserved under ``sources``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Optional

import requests

from . import config
from .discovery import (
    ActListing,
    ActResult,
    Collection,
    InventoryEntry,
    Outcome,
    Rejection,
    browse_collection,
    discover_collections,
    inspect_act,
)
from .errors import IngestionError
from .http_client import SessionPool, build_session
from .utils import atomic_write_text, utcnow_iso

log = logging.getLogger(__name__)


class InventoryStore:
    """Owns ``data/discovery/`` and every file the discovery phase writes."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.dir = self.data_dir / config.DISCOVERY_SUBDIR

    # -- paths ------------------------------------------------------------------

    @property
    def listings_path(self) -> Path:
        return self.dir / config.LISTINGS_FILENAME

    @property
    def journal_path(self) -> Path:
        return self.dir / config.JOURNAL_FILENAME

    @property
    def inventory_path(self) -> Path:
        return self.dir / config.INVENTORY_FILENAME

    def ensure_layout(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def relpath(self, path: Path) -> str:
        return path.relative_to(self.data_dir).as_posix()

    # -- phase 1: cached listings ----------------------------------------------

    def load_listings(self) -> Optional[tuple[list[Collection], list[ActListing]]]:
        """Return cached ``(collections, listings)``, or ``None`` if not browsed."""
        if not self.listings_path.exists():
            return None
        try:
            data = json.loads(self.listings_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Ignoring unreadable %s (%s); re-browsing.", self.listings_path, exc)
            return None
        if data.get("schema_version") != config.DISCOVERY_SCHEMA_VERSION:
            log.warning("Cached listings use an older schema; re-browsing.")
            return None
        collections = [Collection(**c) for c in data.get("collections", [])]
        listings = [ActListing(**item) for item in data.get("listings", [])]
        if not listings:
            return None
        return collections, listings

    def save_listings(self, collections: list[Collection], listings: list[ActListing]) -> None:
        payload = {
            "schema_version": config.DISCOVERY_SCHEMA_VERSION,
            "generated_at": utcnow_iso(),
            "collections": [asdict(c) for c in collections],
            "listings": [asdict(item) for item in listings],
        }
        atomic_write_text(self.listings_path, json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("Cached %d listings -> %s", len(listings), self.relpath(self.listings_path))

    # -- phase 2: append-only journal -------------------------------------------

    def load_journal(self) -> dict[str, ActResult]:
        """Replay the journal into ``{act handle: ActResult}``.

        A trailing partial line (process killed mid-append) is dropped with a
        warning instead of aborting the run — that is precisely the case
        resumability has to survive.
        """
        results: dict[str, ActResult] = {}
        if not self.journal_path.exists():
            return results
        with open(self.journal_path, "r", encoding="utf-8") as handle:
            lines = handle.readlines()
        for number, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if number == len(lines):
                    log.warning(
                        "Discarding incomplete final journal line %d (interrupted run).",
                        number,
                    )
                    continue
                log.warning("Skipping corrupt journal line %d.", number)
                continue
            try:
                results[record["handle"]] = ActResult(
                    handle=record["handle"],
                    outcome=record["outcome"],
                    entries=[InventoryEntry(**e) for e in record.get("entries", [])],
                    rejections=[Rejection(**r) for r in record.get("rejections", [])],
                    error=record.get("error"),
                )
            except (KeyError, TypeError) as exc:
                log.warning("Skipping unusable journal line %d (%s).", number, exc)
        if results:
            log.info("Resuming: %d acts already inspected.", len(results))
        return results

    def append_journal(self, result: ActResult) -> None:
        """Append one act's result. Flushed immediately so a kill loses nothing."""
        record = {
            "handle": result.handle,
            "outcome": result.outcome,
            "error": result.error,
            "entries": [asdict(e) for e in result.entries],
            "rejections": [asdict(r) for r in result.rejections],
            "at": utcnow_iso(),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.journal_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def reset_journal(self) -> None:
        self.journal_path.unlink(missing_ok=True)

    # -- the deliverable ---------------------------------------------------------

    def write_inventory(self, results: Iterable[ActResult], collections: list[Collection],
                        listings: list[ActListing], *, complete: bool) -> dict:
        """Deduplicate, summarise and write ``indiacode_inventory.json``."""
        entries, duplicates = deduplicate(results)
        summary = summarise(
            results, entries, collections, listings,
            duplicates=duplicates, complete=complete,
        )
        payload = {
            "schema_version": config.DISCOVERY_SCHEMA_VERSION,
            "generated_at": utcnow_iso(),
            "provider": "India Code",
            "source": config.BASE_URL,
            "scope": {
                "document_types": ["central_act", "state_act", "rule", "regulation"],
                "language": "en",
                "note": (
                    "English-only. An entry is present only when its English "
                    "status was positively established from India Code metadata."
                ),
            },
            "complete": complete,
            "summary": summary,
            "collections": [asdict(c) for c in collections],
            "documents": [asdict(e) for e in entries],
        }
        atomic_write_text(self.inventory_path, json.dumps(payload, indent=2, ensure_ascii=False))
        log.info(
            "Inventory written: %d documents -> %s",
            len(entries), self.relpath(self.inventory_path),
        )
        return payload


# --- deduplication --------------------------------------------------------------


def dedup_key(entry: InventoryEntry) -> str:
    """The identity of a download job: its English PDF URL, normalised.

    Host case and a stray trailing space (India Code emits both) must not make
    the same file look like two different documents.
    """
    return entry.english_pdf_url.strip().replace(" ", "%20")


def deduplicate(results: Iterable[ActResult]) -> tuple[list[InventoryEntry], int]:
    """Collapse entries that point at the same English PDF.

    The first entry wins; later ones contribute their ``sources`` so every
    India Code route to the document stays traceable. Returns the deduplicated
    entries and how many duplicate sightings were merged away.
    """
    merged: dict[str, InventoryEntry] = {}
    duplicates = 0
    for result in results:
        for entry in result.entries:
            key = dedup_key(entry)
            existing = merged.get(key)
            if existing is None:
                merged[key] = entry
                continue
            duplicates += 1
            known = {json.dumps(s, sort_keys=True) for s in existing.sources}
            for source in entry.sources:
                if json.dumps(source, sort_keys=True) not in known:
                    existing.sources.append(source)
            # Keep whichever sighting knew more about the document.
            for attribute in (
                "title", "short_title", "year", "act_number", "enactment_date",
                "ministry", "department", "india_code_act_id", "parent_document_id",
            ):
                if getattr(existing, attribute) in (None, "") and getattr(entry, attribute):
                    setattr(existing, attribute, getattr(entry, attribute))
    return list(merged.values()), duplicates


# --- summary --------------------------------------------------------------------


def summarise(results, entries, collections, listings, *, duplicates: int, complete: bool) -> dict:
    """Build the end-of-run report described in the discovery requirements."""
    results = list(results)
    by_type = Counter(e.document_type for e in entries)
    state_acts_by_state = Counter(
        e.jurisdiction for e in entries if e.document_type == "state_act"
    )
    rejections = [r for result in results for r in result.rejections]
    rejected_by_reason = Counter(r.reason for r in rejections)
    act_outcomes = Counter(result.outcome for result in results)

    return {
        "total_discovered": len(entries),
        "central_acts": by_type.get("central_act", 0),
        "state_acts": by_type.get("state_act", 0),
        "state_acts_by_state": dict(sorted(state_acts_by_state.items())),
        "rules": by_type.get("rule", 0),
        "regulations": by_type.get("regulation", 0),
        "english_confirmed": len(entries),
        "hindi_rejected": rejected_by_reason.get(Outcome.HINDI_REJECTED, 0),
        "ambiguous_rejected": rejected_by_reason.get(Outcome.AMBIGUOUS, 0),
        "no_pdf_available": rejected_by_reason.get(Outcome.NO_PDF, 0),
        "duplicates_merged": duplicates,
        "errors": act_outcomes.get(Outcome.ERROR, 0),
        "acts_inspected": len(results),
        "acts_listed": len(listings),
        "collections": len(collections),
        "act_outcomes": dict(sorted(act_outcomes.items())),
        "language_sources": dict(
            sorted(Counter(e.language_source for e in entries).items())
        ),
        "pdf_url_kinds": dict(sorted(Counter(e.pdf_url_kind for e in entries).items())),
        "complete": complete,
    }


# --- orchestration ---------------------------------------------------------------


#: One session per worker thread. Lives in http_client so the sizing pass can
#: share it; kept under the old name here for the callers below.
_SessionPool = SessionPool


def run_discovery(
    store: InventoryStore,
    *,
    collections_filter: Optional[list[str]] = None,
    limit: Optional[int] = None,
    workers: int = config.DISCOVERY_WORKERS,
    delay: float = config.DISCOVERY_DELAY_SECONDS,
    rebrowse: bool = False,
    restart: bool = False,
    recheck: bool = False,
    checkpoint_every: int = 250,
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Run both discovery phases and write the inventory. Resumable.

    No PDF is fetched: only listing pages and act landing pages are read.

    ``recheck`` re-queues every act that did not end in ``english_confirmed``.
    India Code intermittently serves an act page without its file block, which
    looks exactly like "this document has no PDF", so a cheap second look at
    the non-confirmed minority recovers documents a single pass would lose.
    """
    store.ensure_layout()
    if restart:
        log.info("--restart: discarding previous discovery progress.")
        store.reset_journal()
        store.listings_path.unlink(missing_ok=True)

    pool = _SessionPool()

    # --- phase 1: collections + listings (cached across runs) -------------------
    cached = None if rebrowse else store.load_listings()
    if cached is None:
        session = pool.get()
        collections = discover_collections(session)
        if collections_filter:
            wanted = {value.lower() for value in collections_filter}
            collections = [
                c for c in collections
                if c.name.lower() in wanted or c.handle_id in wanted
            ]
            if not collections:
                raise IngestionError(
                    f"No collection matched {collections_filter!r}."
                )
        listings: list[ActListing] = []
        for collection in collections:
            listings.extend(browse_collection(session, collection))
        store.save_listings(collections, listings)
    else:
        collections, listings = cached
        log.info(
            "Using cached listings: %d acts across %d collections "
            "(--rebrowse to refresh).", len(listings), len(collections),
        )
        if collections_filter:
            wanted = {value.lower() for value in collections_filter}
            keep = {
                c.handle for c in collections
                if c.name.lower() in wanted or c.handle_id in wanted
            }
            listings = [item for item in listings if item.collection_handle in keep]

    # --- phase 2: inspect each act (resumed from the journal) -------------------
    done = store.load_journal()
    if recheck:
        stale = [
            handle for handle, result in done.items()
            if result.outcome != Outcome.ENGLISH_CONFIRMED
        ]
        for handle in stale:
            del done[handle]
        log.info("--recheck: re-queueing %d acts that were not confirmed.", len(stale))
    pending = [item for item in listings if item.handle not in done]
    if limit is not None:
        pending = pending[:limit]

    log.info(
        "Inspecting %d act pages (%d already done, %d workers).",
        len(pending), len(done), workers,
    )

    completed = 0
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_inspect_one, pool, listing, delay): listing
                for listing in pending
            }
            try:
                for future in as_completed(futures):
                    listing = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # defensive: never kill the run
                        log.exception("Unexpected error inspecting %s", listing.url)
                        result = ActResult(listing.handle, Outcome.ERROR, error=str(exc))
                    done[result.handle] = result
                    store.append_journal(result)
                    completed += 1
                    if progress:
                        progress(completed, len(pending))
                    if checkpoint_every and completed % checkpoint_every == 0:
                        store.write_inventory(
                            done.values(), collections, listings, complete=False
                        )
            except KeyboardInterrupt:
                log.warning(
                    "Interrupted: %d/%d acts inspected. Progress is journalled; "
                    "re-run to continue.", completed, len(pending),
                )
                for future in futures:
                    future.cancel()
                store.write_inventory(done.values(), collections, listings, complete=False)
                raise

    complete = not [item for item in listings if item.handle not in done]
    return store.write_inventory(done.values(), collections, listings, complete=complete)


def _inspect_one(pool: _SessionPool, listing: ActListing, delay: float) -> ActResult:
    if delay:
        time.sleep(delay)
    return inspect_act(pool.get(), listing)
