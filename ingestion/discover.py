"""Command-line entry point for corpus discovery.

Builds ``data/discovery/indiacode_inventory.json``: every Central Act, State
Act, Rule and Regulation on India Code whose **English** PDF could be positively
identified. No PDF is downloaded — discovery reads HTML and metadata only.

Examples
--------
Full inventory (resumable; safe to interrupt and re-run)::

    python -m ingestion.discover

Just one collection, and only the first 50 acts of it::

    python -m ingestion.discover --collection "Central Acts" --limit 50

See ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__, config
from .errors import IngestionError
from .inventory import InventoryStore, run_discovery

log = logging.getLogger("ingestion.discover")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.discover",
        description=(
            "Inventory the English Central Acts, State Acts, Rules and "
            "Regulations available from India Code. Downloads no PDFs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="Root data directory (default: ./data).")
    parser.add_argument("--collection", action="append", default=[], metavar="NAME",
                        help="Restrict to a collection by name or handle id "
                             "(repeatable), e.g. --collection \"Central Acts\".")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Inspect at most N act pages this run (for sampling).")

    behaviour = parser.add_argument_group("behaviour")
    behaviour.add_argument("--workers", type=int, default=config.DISCOVERY_WORKERS,
                           help=f"Concurrent page fetches (default: {config.DISCOVERY_WORKERS}).")
    behaviour.add_argument("--delay", type=float, default=config.DISCOVERY_DELAY_SECONDS,
                           metavar="SECONDS",
                           help="Pause before each request, per worker (politeness).")
    behaviour.add_argument("--rebrowse", action="store_true",
                           help="Refresh the cached collection listings.")
    behaviour.add_argument("--restart", action="store_true",
                           help="Discard previous progress and start over.")
    behaviour.add_argument("--recheck", action="store_true",
                           help="Re-inspect acts that were not confirmed English. India "
                                "Code intermittently serves an act page without its file "
                                "block, so a second look recovers real documents.")
    behaviour.add_argument("--checkpoint-every", type=int, default=250, metavar="N",
                           help="Rewrite the inventory every N acts (default: 250).")

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    logging_group.add_argument("-q", "--quiet", action="store_true",
                               help="Warnings and errors only.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _progress(done: int, total: int) -> None:
    if done % 100 == 0 or done == total:
        log.info("  … %d/%d act pages inspected (%.0f%%)", done, total, 100 * done / total)


def render_summary(payload: dict) -> str:
    """Render the end-of-run report."""
    summary = payload["summary"]
    lines = [
        "",
        "==================== DISCOVERY SUMMARY ====================",
        f"  total discovered (English confirmed) : {summary['total_discovered']:>8,}",
        "  ---------------------------------------------------------",
        f"  Central Acts                         : {summary['central_acts']:>8,}",
        f"  State Acts                           : {summary['state_acts']:>8,}",
        f"  Rules                                : {summary['rules']:>8,}",
        f"  Regulations                          : {summary['regulations']:>8,}",
        "  ---------------------------------------------------------",
        f"  English confirmed                    : {summary['english_confirmed']:>8,}",
        f"  Hindi rejected                       : {summary['hindi_rejected']:>8,}",
        f"  Ambiguous / rejected                 : {summary['ambiguous_rejected']:>8,}",
        f"  No PDF available                     : {summary['no_pdf_available']:>8,}",
        f"  Duplicates merged                    : {summary['duplicates_merged']:>8,}",
        f"  Errors                               : {summary['errors']:>8,}",
        "  ---------------------------------------------------------",
        f"  Act pages inspected                  : {summary['acts_inspected']:>8,}"
        f" / {summary['acts_listed']:,} listed",
        f"  Complete                             : {str(summary['complete']):>8}",
    ]
    if summary["state_acts_by_state"]:
        lines.append("")
        lines.append("  State Acts by state/UT:")
        for state, count in sorted(
            summary["state_acts_by_state"].items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"    {state:<36} {count:>6,}")
    lines.append("===========================================================")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, args.quiet)

    store = InventoryStore(args.data_dir)
    try:
        payload = run_discovery(
            store,
            collections_filter=args.collection or None,
            limit=args.limit,
            workers=args.workers,
            delay=args.delay,
            rebrowse=args.rebrowse,
            restart=args.restart,
            recheck=args.recheck,
            checkpoint_every=args.checkpoint_every,
            progress=_progress,
        )
    except KeyboardInterrupt:
        log.warning("Discovery interrupted; re-run to resume where it stopped.")
        return 130
    except IngestionError as exc:
        log.error("%s", exc)
        return 2

    print(render_summary(payload))
    print(f"\nInventory: {store.inventory_path}")
    if not payload["summary"]["complete"]:
        print("Run again to continue: some act pages have not been inspected yet.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
