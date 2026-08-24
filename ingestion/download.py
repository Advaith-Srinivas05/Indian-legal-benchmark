"""Command-line entry point for the India Code downloader.

Examples
--------
Download one central act::

    python -m ingestion.download --url "https://www.indiacode.nic.in/handle/123456789/1372" --type central_act

Download many documents listed in a JSON or CSV file::

    python -m ingestion.download --input examples/urls.example.json

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import __version__, bulk, config
from .errors import DocumentTypeError, IngestionError
from .http_client import build_session
from .manifest import Manifest
from .models import DownloadResult, Outcome
from .pipeline import ingest_document
from .storage import DocumentStore

log = logging.getLogger("ingestion")


@dataclass
class Target:
    """One document to download, plus what is needed to verify its language."""

    url: str
    category: str
    #: The act page a Rules/Regulations file is listed on (``None`` for acts).
    parent_url: Optional[str] = None
    document_id: Optional[str] = None
    title: Optional[str] = None


# --- input handling ------------------------------------------------------------


def resolve_category(doc_type: str | None) -> str:
    """Map a user ``--type``/input ``type`` value to a canonical category dir."""
    if not doc_type:
        raise DocumentTypeError(
            "A document type is required. Use --type or a 'type' field in the "
            f"input file. One of: {', '.join(sorted(set(config.TYPE_TO_CATEGORY)))}."
        )
    key = doc_type.strip().lower()
    if key not in config.TYPE_TO_CATEGORY:
        raise DocumentTypeError(
            f"Unknown document type {doc_type!r}. "
            f"Valid types: {', '.join(sorted(set(config.TYPE_TO_CATEGORY)))}."
        )
    return config.TYPE_TO_CATEGORY[key]


def load_input_file(path: Path, default_type: str | None) -> list[Target]:
    """Load download targets from a JSON or CSV file.

    JSON accepts either a list of ``{"url": ..., "type": ...}`` objects or an
    object with a top-level ``"documents"`` list. CSV must have ``url`` and
    (optionally) ``type`` columns. A row/object without a type falls back to
    ``default_type`` (from ``--type``).

    A row may also carry ``parent_url``: the act page a Rules/Regulations file
    is listed on, which is what makes that file's language verifiable.
    """
    if not path.exists():
        raise IngestionError(f"Input file not found: {path}")
    suffix = path.suffix.lower()
    rows: list[dict] = []
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("documents", [])
        if not isinstance(data, list):
            raise IngestionError(
                f"{path} must be a JSON list or an object with a 'documents' list."
            )
        rows = data
    elif suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        raise IngestionError(f"Unsupported input file type {suffix!r}; use .json or .csv.")

    targets: list[Target] = []
    for i, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise IngestionError(f"Entry #{i} in {path} is not an object/row.")
        url = (row.get("url") or "").strip()
        if not url:
            raise IngestionError(f"Entry #{i} in {path} is missing a 'url'.")
        category = resolve_category(row.get("type") or default_type)
        parent = (row.get("parent_url") or "").strip() or None
        targets.append(Target(url=url, category=category, parent_url=parent))
    return targets


def load_inventory(path: Path, types: list[str] | None = None) -> list[Target]:
    """Load download targets from a discovery inventory.

    The inventory is the natural driver for a corpus download: each document
    already carries the English PDF URL, its document type, and — for Rules and
    Regulations — the parent act page their language is verified against.
    """
    if not path.exists():
        raise IngestionError(
            f"Inventory not found: {path}. Run 'python -m ingestion.discover' first."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise IngestionError(f"Inventory at {path} is unreadable: {exc}") from exc
    documents = data.get("documents")
    if not isinstance(documents, list):
        raise IngestionError(f"{path} has no 'documents' list; is it an inventory?")

    wanted = {value.strip().lower() for value in types} if types else None
    targets: list[Target] = []
    for i, document in enumerate(documents, 1):
        document_type = (document.get("document_type") or "").strip()
        if wanted and document_type.lower() not in wanted:
            continue
        url = (document.get("english_pdf_url") or "").strip()
        if not url:
            raise IngestionError(f"Inventory entry #{i} has no 'english_pdf_url'.")
        targets.append(
            Target(
                url=url,
                category=resolve_category(document_type),
                # Only subordinate legislation needs its parent page; for an act
                # the india_code_url *is* its own landing page.
                parent_url=(document.get("india_code_url") or "").strip() or None
                if document.get("parent_handle") else None,
                document_id=document.get("document_id"),
                title=document.get("title"),
            )
        )
    return targets


# --- CLI -----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.download",
        description="Download and organise Indian legal PDFs (English only) from India Code.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_argument_group("sources")
    src.add_argument("--url", action="append", default=[], metavar="URL",
                     help="India Code handle, bitstream or ViewFileUploaded URL "
                          "(repeatable).")
    src.add_argument("--parent-url", metavar="URL",
                     help="For a ViewFileUploaded (Rules/Regulations) --url: the act "
                          "page it is listed on, where India Code states its language. "
                          "Looked up in the inventory when not given.")
    src.add_argument("--input", type=Path, metavar="FILE",
                     help="JSON or CSV file listing documents to download.")
    src.add_argument("--inventory", type=Path, nargs="?", const=True, metavar="FILE",
                     help="Download from a discovery inventory (default: "
                          "<data-dir>/discovery/indiacode_inventory.json).")
    src.add_argument("--only", action="append", default=[], metavar="TYPE",
                     help="With --inventory: keep only these document types "
                          "(repeatable), e.g. --only rule --only regulation.")
    src.add_argument("--limit", type=int, metavar="N",
                     help="Download at most N documents this run.")
    parser.add_argument("--type", dest="doc_type", metavar="TYPE",
                        help="Document type for --url (and default for --input rows): "
                             f"{', '.join(sorted(set(config.TYPE_TO_CATEGORY)))}.")
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="Root data directory (default: ./data).")

    batch = parser.add_argument_group("batch (corpus-scale runs)")
    batch.add_argument("--workers", type=int, default=1, metavar="N",
                       help="Concurrent documents (default: 1 = sequential). Keep "
                            "this small: India Code is a public government site.")
    batch.add_argument("--delay", type=float, default=0.0, metavar="SECONDS",
                       help="Minimum interval between documents, across all "
                            "workers (default: 0).")
    batch.add_argument("--retries", type=int, default=2, metavar="N",
                       help="Extra attempts for transient failures — network "
                            "errors, 429/5xx, an act page served without its file "
                            "block (default: 2). Permanent failures are not retried.")
    batch.add_argument("--checkpoint-every", type=int, default=50, metavar="N",
                       help="Save the manifest every N documents (default: 50).")
    batch.add_argument("--progress-every", type=int, default=250, metavar="N",
                       help="Print a progress block every N documents (default: 250).")
    batch.add_argument("--failure-log", type=Path, metavar="FILE",
                       help="Append failures as JSONL (default: "
                            "<data-dir>/download_failures.jsonl).")
    batch.add_argument("--min-free-gb", type=float, default=0.0, metavar="GB",
                       help="Refuse to start unless this much space is free on the "
                            "target drive.")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--all-bitstreams", action="store_true",
                           help="Also download additional *English* files attached to the "
                                "item. Non-English files are never downloaded: this corpus "
                                "is English-only.")
    behaviour.add_argument("--force", action="store_true",
                           help="Re-place the PDF even if content is unchanged.")
    behaviour.add_argument("--skip-existing", action="store_true",
                           help="Skip documents already in the manifest without downloading.")
    behaviour.add_argument("--dry-run", action="store_true",
                           help="Resolve and report actions without downloading.")

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    logging_group.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def inventory_path(args) -> Path:
    """Where the inventory lives: an explicit path, or the default location."""
    if isinstance(args.inventory, Path):
        return args.inventory
    return args.data_dir / config.DISCOVERY_SUBDIR / config.INVENTORY_FILENAME


def _collect_targets(args) -> list[Target]:
    """Build the ordered work list from CLI + input file + inventory."""
    targets: list[Target] = []
    if args.url:
        category = resolve_category(args.doc_type)
        targets.extend(
            Target(url=u, category=category, parent_url=args.parent_url)
            for u in args.url
        )
    if args.input:
        targets.extend(load_input_file(args.input, args.doc_type))
    if args.inventory:
        targets.extend(load_inventory(inventory_path(args), args.only or None))

    # A ViewFileUploaded URL given without its parent page can still be verified
    # if the inventory happens to know where India Code lists it.
    unresolved = [t for t in targets if t.parent_url is None and _needs_parent(t.url)]
    if unresolved and not args.inventory:
        _fill_parents_from_inventory(unresolved, inventory_path(args))

    if args.limit is not None:
        targets = targets[: args.limit]
    return targets


def _needs_parent(url: str) -> bool:
    return "viewfileuploaded" in url.lower()


def _fill_parents_from_inventory(targets: list[Target], path: Path) -> None:
    """Look up each file's parent act page in the inventory, if one exists."""
    if not path.exists():
        return
    try:
        known = {t.url: t for t in load_inventory(path)}
    except IngestionError as exc:
        log.debug("Could not consult the inventory for parent pages: %s", exc)
        return
    for target in targets:
        match = known.get(target.url)
        if match is not None and match.parent_url:
            target.parent_url = match.parent_url
            log.info(
                "Parent act page for %s taken from the inventory: %s",
                target.url[:80], match.parent_url,
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, args.quiet)

    if not args.url and not args.input and not args.inventory:
        log.error("Nothing to do: pass --url, --input and/or --inventory. See --help.")
        return 2

    try:
        targets = _collect_targets(args)
    except IngestionError as exc:
        log.error("%s", exc)
        return 2
    if not targets:
        log.error("No valid documents to download.")
        return 2

    store = DocumentStore(args.data_dir)
    store.ensure_layout()
    manifest = Manifest.load(store.manifest_path)

    if args.min_free_gb:
        required = int(args.min_free_gb * 1024 ** 3)
        ok, free = bulk.check_free_space(store.data_dir, required)
        if not ok:
            log.error(
                "Only %s free on the drive holding %s, but --min-free-gb asks for "
                "%s. Refusing to start; free up space and re-run.",
                bulk.format_bytes(free), store.data_dir.resolve(),
                bulk.format_bytes(required),
            )
            return 3
        log.info("Free space on the target drive: %s (required %s).",
                 bulk.format_bytes(free), bulk.format_bytes(required))

    if _is_batch(args):
        return _run_batch(args, store, manifest, targets)

    session = build_session()
    log.info("Ingesting %d document(s) into %s", len(targets), args.data_dir.resolve())
    results: list[DownloadResult] = []
    for target in targets:
        result = ingest_document(
            session, store, manifest, target.url, target.category,
            parent_url=target.parent_url,
            all_bitstreams=args.all_bitstreams, force=args.force,
            skip_existing=args.skip_existing, dry_run=args.dry_run,
        )
        results.append(result)

    return _report(results)


def _is_batch(args) -> bool:
    """Corpus-scale runs take the batch path; a handful of URLs does not."""
    return args.workers > 1 and not args.dry_run and not args.force


def _run_batch(args, store: DocumentStore, manifest: Manifest, targets: list[Target]) -> int:
    """Concurrent, resumable, checkpointed ingestion of a large target list."""
    todo, already = bulk.partition_targets(targets, manifest)
    failures = bulk.FailureReport(
        args.failure_log or (args.data_dir / "download_failures.jsonl")
    )
    log.info(
        "Batch: %d document(s) to ingest, %d already in the manifest (skipped). "
        "%d worker(s), >= %.2fs between documents, %d retries for transient errors.",
        len(todo), len(already), args.workers, args.delay, args.retries,
    )
    log.info("Failures are appended to %s", failures.path)

    stats = bulk.run_batch(
        todo, store, manifest,
        workers=args.workers, delay=args.delay, retries=args.retries,
        checkpoint_every=args.checkpoint_every, progress_every=args.progress_every,
        failures=failures,
        on_progress=lambda s: print(bulk.render_progress(s), flush=True),
    )
    stats.skipped = len(already)
    print(_batch_summary(stats, failures), flush=True)
    return 1 if stats.processed and stats.failed == stats.processed else 0


def _batch_summary(stats: bulk.Stats, failures: bulk.FailureReport) -> str:
    lines = [
        "---- Batch summary ----",
        f"  Attempted:  {stats.processed}",
        f"  Downloaded: {stats.new}",
        f"  Updated:    {stats.updated}",
        f"  Unchanged:  {stats.unchanged}",
        f"  Skipped:    {stats.skipped}   (already in the manifest)",
        f"  Failed:     {stats.failed}",
        f"  Retried:    {stats.retried}   (transient errors)",
        f"  Bytes:      {bulk.format_bytes(stats.bytes)}",
        f"  Elapsed:    {bulk.format_duration(stats.elapsed)}",
    ]
    if stats.failures_by_type:
        lines.append("  Failures by type:")
        for name, count in sorted(stats.failures_by_type.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {name:<20} {count}")
        lines.append(f"  Failure report: {failures.path}")
    return "\n".join(lines)


def _report(results: list[DownloadResult]) -> int:
    """Print a per-outcome summary and return an appropriate exit code."""
    counts = {outcome: 0 for outcome in Outcome}
    for result in results:
        counts[result.outcome] += 1

    log.info("---- Summary ----")
    for outcome in Outcome:
        if counts[outcome]:
            log.info("  %-9s %d", outcome.value + ":", counts[outcome])
    for result in results:
        if result.outcome is Outcome.FAILED:
            log.error("  FAILED %s -> %s", result.source_url, result.message)

    # Non-zero exit only if *every* target failed, so partial batches still
    # succeed usefully while surfacing total failure to scripts/CI.
    if results and counts[Outcome.FAILED] == len(results):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
