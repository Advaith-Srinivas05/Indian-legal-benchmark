"""Command-line entry point for the header-only corpus size estimate.

Reads ``data/discovery/indiacode_inventory.json``, samples it, asks India Code
how large each sampled PDF is (headers only — no PDF body is fetched and
nothing is written under ``data/raw/``) and writes the result to
``data/discovery/size_estimate.json``.

Example::

    python -m ingestion.estimate_size --per-type 250 --workers 4 --delay 0.25
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import __version__, config, sizing
from .errors import IngestionError
from .utils import atomic_write_text

log = logging.getLogger("ingestion.estimate_size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ingestion.estimate_size",
        description=(
            "Estimate the corpus's disk footprint from HTTP headers. Downloads "
            "no PDF bytes and writes no PDFs."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR,
                        help="Root data directory (default: ./data).")
    parser.add_argument("--inventory", type=Path, metavar="FILE",
                        help="Inventory to sample (default: "
                             "<data-dir>/discovery/indiacode_inventory.json).")
    parser.add_argument("--output", type=Path, metavar="FILE",
                        help="Where to write the estimate (default: "
                             "<data-dir>/discovery/size_estimate.json).")
    parser.add_argument("--per-type", type=int, default=250, metavar="N",
                        help="Documents to sample per document type (default: 250).")
    parser.add_argument("--seed", type=int, default=20260817,
                        help="Sampling seed, so the estimate is reproducible.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Concurrent probes (default: 4).")
    parser.add_argument("--delay", type=float, default=0.25, metavar="SECONDS",
                        help="Minimum interval between requests overall (default: 0.25).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def inventory_path(args) -> Path:
    if args.inventory:
        return args.inventory
    return args.data_dir / config.DISCOVERY_SUBDIR / config.INVENTORY_FILENAME


def output_path(args) -> Path:
    if args.output:
        return args.output
    return args.data_dir / config.DISCOVERY_SUBDIR / config.SIZE_ESTIMATE_FILENAME


def load_documents(path: Path) -> list[dict]:
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
    return documents


def _mb(value) -> str:
    return "n/a" if value is None else f"{value / 1024 / 1024:,.2f} MB"


def _gb(value) -> str:
    return "n/a" if value is None else f"{value / 1024 ** 3:,.2f} GB"


def render_summary(report: dict) -> str:
    """Human-readable version of the JSON estimate."""
    lines: list[str] = []
    sampling, outcomes = report["sampling"], report["outcomes"]
    stats = report["sample_statistics"]
    estimate = report["estimate"]

    lines.append("---- Size estimate (headers only; no PDF downloaded) ----")
    lines.append(f"  inventory documents : {report['inventory']['documents']:,}")
    lines.append(f"  sampled             : {sampling['sampled']:,} "
                 f"({sampling['per_type']} per document type, seed {sampling['seed']})")
    lines.append(f"  known Content-Length: {outcomes['known_content_length']:,}")
    lines.append(f"  size unknown        : {outcomes['unknown']:,}")
    lines.append(f"  errors              : {outcomes['error']:,}")
    if stats.get("count"):
        lines.append("  --- sampled file sizes ---")
        for label, key in (
            ("mean", "mean_bytes"), ("median", "median_bytes"), ("p90", "p90_bytes"),
            ("p95", "p95_bytes"), ("p99", "p99_bytes"), ("min", "min_bytes"),
            ("max", "max_bytes"),
        ):
            lines.append(f"  {label:<20}: {_mb(stats[key])}")
    lines.append("  --- estimated corpus size ---")
    lines.append(f"  total (mean-based)  : {_gb(estimate['estimated_total_bytes'])} "
                 f"+/- {_gb(estimate['estimated_total_bytes_ci95'])} (95%)")
    lines.append(f"  total (median-based): {_gb(estimate['estimated_total_bytes_by_median'])}")
    for document_type, row in estimate["by_document_type"].items():
        lines.append(
            f"  {document_type:<20}: {_gb(row['estimated_bytes'])}"
            f"  ({row['population']:,} docs, n={row['sample'].get('count', 0)})"
        )
    jurisdictions = report.get("estimate_by_jurisdiction") or {}
    if jurisdictions:
        lines.append("  --- top jurisdictions ---")
        for name, row in list(jurisdictions.items())[:10]:
            flag = "" if row["fully_measured"] else "  (partly from the type mean)"
            lines.append(
                f"  {name:<20}: {_gb(row['estimated_bytes'])}"
                f"  ({row['documents']:,} docs){flag}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        source = inventory_path(args)
        documents = load_documents(source)
    except IngestionError as exc:
        log.error("%s", exc)
        return 2

    sample = sizing.stratified_sample(documents, per_type=args.per_type, seed=args.seed)
    log.info(
        "Probing %d of %d documents with %d worker(s), >= %.2fs between requests.",
        len(sample), len(documents), args.workers, args.delay,
    )

    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            log.info("  probed %d/%d", done, total)

    probes = sizing.probe_documents(
        sample, workers=args.workers, delay=args.delay, progress=progress,
    )
    report = sizing.build_report(
        inventory_path=str(source),
        documents=documents,
        sample=sample,
        probes=probes,
        per_type=args.per_type,
        seed=args.seed,
        workers=args.workers,
        delay=args.delay,
    )

    destination = output_path(args)
    atomic_write_text(destination, json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(render_summary(report))
    print(f"\nWritten to {destination}")

    if not report["outcomes"]["known_content_length"]:
        log.error(
            "No document returned a usable Content-Length; the server does not "
            "support header-only sizing. Not falling back to downloading PDFs."
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
