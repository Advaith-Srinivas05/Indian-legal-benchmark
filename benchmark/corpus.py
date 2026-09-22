"""Build and verify the canonical corpus under ``data/corpus/``.

Per document, three files are written in a fixed order — ``text/``, then
``structure/``, then ``meta/`` — and ``meta/`` is written last, so its existence
is the completion marker. A document that fails is not written at all: anything
partial is removed and the failure is reported with its reason. The run exits
non-zero if any eligible document failed.

Then ``documents.jsonl`` (one row per document), ``CHECKSUMS.txt`` (sha256 of
every published file) and ``build_report.json`` are rebuilt from what is on
disk. Nothing written depends on the clock or on worker scheduling, so two
builds of the same input are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Optional

from . import config
from .canonical import build_canonical
from .errors import BenchmarkError, NotEligibleError
from .provenance import build_meta, load_inventory, load_raw_metadata, structure_record
from .source import load_document, load_pages, require_eligible


# --- One document ------------------------------------------------------------------


def _paths(out_dir: Path, document_id: str) -> dict[str, Path]:
    return {
        "text": out_dir / config.TEXT_DIRNAME / f"{document_id}.txt",
        "structure": out_dir / config.STRUCTURE_DIRNAME / f"{document_id}.json",
        "meta": out_dir / config.META_DIRNAME / f"{document_id}.json",
    }


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    # newline="" — never translate "\n" to "\r\n" on Windows; offsets depend on it.
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(content)
    os.replace(tmp, path)


def _json(obj, *, compact: bool = False) -> str:
    if compact:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def build_document(data_dir: Path, out_dir: Path, document_id: str,
                   inventory_entry: Optional[dict] = None, force: bool = False) -> dict:
    """Build one document. Returns a result row; never raises for document faults."""
    paths = _paths(out_dir, document_id)
    if paths["meta"].exists() and not force:
        return {"document_id": document_id, "status": "already_built"}
    try:
        document = load_document(data_dir, document_id)
        require_eligible(document)
        pages, missing = load_pages(data_dir, document_id)
        canonical = build_canonical(document, pages)
        meta = build_meta(document, canonical, raw=load_raw_metadata(data_dir, document),
                          inventory_entry=inventory_entry, missing_page_fields=missing)
    except NotEligibleError:
        return {"document_id": document_id, "status": "not_eligible"}
    except (BenchmarkError, KeyError, ValueError, TypeError) as exc:
        for path in paths.values():
            path.unlink(missing_ok=True)
        return {"document_id": document_id, "status": "failed",
                "error_type": type(exc).__name__, "error": str(exc)}

    _write_atomic(paths["text"], canonical.text)
    _write_atomic(paths["structure"], _json(structure_record(canonical), compact=True))
    _write_atomic(paths["meta"], _json(meta))
    return {"document_id": document_id, "status": "built"}


def _build_one(args) -> dict:
    return build_document(*args)


# --- The corpus --------------------------------------------------------------------


def processed_ids(data_dir: Path) -> list[str]:
    root = Path(data_dir) / config.PROCESSED_SUBDIR
    return sorted(entry.name for entry in os.scandir(root) if entry.is_dir())


def build_corpus(data_dir: Path, out_dir: Path, *, workers: int = 1,
                 only: Optional[Iterable[str]] = None, force: bool = False,
                 progress: bool = False) -> dict:
    """Build every eligible document, then the corpus-level files. Returns the report."""
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    ids = sorted(set(only)) if only is not None else processed_ids(data_dir)
    inventory = load_inventory(data_dir)
    jobs = [(data_dir, out_dir, d, inventory.get(d), force) for d in ids]

    results: list[dict] = []
    started = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(pool.map(_build_one, jobs, chunksize=32), 1):
                results.append(row)
                if progress and i % 1000 == 0:
                    print(f"  {i}/{len(jobs)} documents, {time.time() - started:.0f}s", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            results.append(_build_one(job))
            if progress and i % 1000 == 0:
                print(f"  {i}/{len(jobs)} documents, {time.time() - started:.0f}s", flush=True)

    if any(r["status"] == "built" for r in results):
        # Duplicate findings describe the corpus as it was; a rebuild invalidates them.
        for name in (config.DUPLICATES_FILENAME, config.PROVISION_EQUIVALENTS_FILENAME):
            (out_dir / name).unlink(missing_ok=True)
    report = finalise(out_dir, results)
    if progress:
        print(f"  finished in {time.time() - started:.0f}s", flush=True)
    return report


def finalise(out_dir: Path, results: list[dict]) -> dict:
    """Rebuild ``documents.jsonl``, ``CHECKSUMS.txt`` and ``build_report.json``."""
    out_dir = Path(out_dir)
    meta_dir = out_dir / config.META_DIRNAME
    metas = []
    if meta_dir.exists():
        for path in sorted(meta_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                metas.append(json.load(handle))

    rows = [{
        "document_id": m["document_id"],
        "title": m["title"],
        "category": m["category"],
        "document_type": m["document_type"],
        "jurisdiction": m["jurisdiction"],
        "year": m["year"],
        "char_count": m["text"]["char_count"],
        "pages": m["text"]["page_count"],
        "provisions": m["provisions"]["count"],
        "confidence": m["provisions"]["confidence"],
        "structure_confidence": m["quality"]["structure_confidence"],
        "ocr_pages": len(m["quality"]["ocr_pages"]),
        "text_sha256": m["text"]["sha256"],
    } for m in metas]
    _write_atomic(out_dir / config.DOCUMENTS_FILENAME,
                  "".join(_json(r, compact=True) for r in rows))

    status: dict[str, int] = {}
    for r in results:
        status[r["status"]] = status.get(r["status"], 0) + 1
    tiers = {t: sum(m["provisions"]["confidence"][t] for m in metas) for t in config.CONFIDENCE_TIERS}
    by_category: dict[str, dict] = {}
    for m in metas:
        c = by_category.setdefault(m["category"], {"documents": 0, "provisions": 0,
                                                     **{t: 0 for t in config.CONFIDENCE_TIERS}})
        c["documents"] += 1
        c["provisions"] += m["provisions"]["count"]
        for t in config.CONFIDENCE_TIERS:
            c[t] += m["provisions"]["confidence"][t]

    report = {
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "documents": len(metas),
        "this_run": dict(sorted(status.items())),
        "failures": sorted((r for r in results if r["status"] == "failed"),
                           key=lambda r: r["document_id"]),
        "provisions": sum(m["provisions"]["count"] for m in metas),
        "confidence": tiers,
        "by_category": dict(sorted(by_category.items())),
        "characters": sum(m["text"]["char_count"] for m in metas),
        "pages": sum(m["text"]["page_count"] for m in metas),
        "documents_with_ocr_pages": sum(1 for m in metas if m["quality"]["ocr_pages"]),
        "documents_missing_page_fields": sum(1 for m in metas if m["quality"]["missing_page_fields"]),
        "documents_without_raw_metadata": sum(1 for m in metas if not m["quality"]["raw_metadata_found"]),
        "replaced_line_boundaries": sum(m["quality"]["replaced_line_boundaries"] for m in metas),
        "non_bmp_chars": sum(m["quality"]["non_bmp_chars"] for m in metas),
        "duplicate_provision_keys": sum(m["provisions"]["duplicate_keys"] for m in metas),
        "ambiguous_provisions": sum(m["provisions"]["ambiguous_keys"] for m in metas),
    }
    _write_atomic(out_dir / config.BUILD_REPORT_FILENAME, _json(report))
    write_checksums(out_dir)
    return report


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _published_files(out_dir: Path) -> list[Path]:
    files = []
    for sub in (config.TEXT_DIRNAME, config.META_DIRNAME, config.STRUCTURE_DIRNAME):
        d = out_dir / sub
        if d.exists():
            files.extend(p for p in d.iterdir() if p.is_file() and not p.name.endswith(".tmp"))
    files.append(out_dir / config.DOCUMENTS_FILENAME)
    for name in (config.DUPLICATES_FILENAME, config.PROVISION_EQUIVALENTS_FILENAME):
        if (out_dir / name).exists():
            files.append(out_dir / name)
    return sorted(files, key=lambda p: p.relative_to(out_dir).as_posix())


def write_checksums(out_dir: Path) -> None:
    lines = [f"{_sha256_file(p)}  {p.relative_to(out_dir).as_posix()}"
             for p in _published_files(out_dir)]
    _write_atomic(out_dir / config.CHECKSUMS_FILENAME, "\n".join(lines) + "\n")


# --- Verification of the published files --------------------------------------------


def verify_corpus(out_dir: Path) -> dict:
    """Re-check the published corpus using only the published files.

    Independent of the build: it never reads ``data/processed/``. Checks every
    checksum, that each text file matches its recorded hash and length, that the
    page map tiles the text exactly, and that every provision span reproduces
    its recorded text hash.
    """
    out_dir = Path(out_dir)
    problems: list[str] = []

    recorded = {}
    for line in (out_dir / config.CHECKSUMS_FILENAME).read_text(encoding="utf-8").splitlines():
        digest, rel = line.split("  ", 1)
        recorded[rel] = digest
    actual = {p.relative_to(out_dir).as_posix() for p in _published_files(out_dir)}
    if set(recorded) != actual:
        problems.append(f"CHECKSUMS.txt lists {len(recorded)} files, {len(actual)} are present")
    for rel, digest in recorded.items():
        path = out_dir / rel
        if path.exists() and _sha256_file(path) != digest:
            problems.append(f"checksum mismatch: {rel}")

    documents = provisions = 0
    for meta_path in sorted((out_dir / config.META_DIRNAME).glob("*.json")):
        documents += 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        did = meta["document_id"]
        with open(out_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
            text = handle.read()
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != meta["text"]["sha256"]:
            problems.append(f"{did}: text hash differs from meta")
        if len(text) != meta["text"]["char_count"]:
            problems.append(f"{did}: text length differs from meta")
        if len(text.splitlines()) != meta["text"]["line_count"] and text:
            problems.append(f"{did}: splitlines() disagrees with line_count")
        cursor = 0
        for page in meta["page_map"]:
            if page["char_start"] != cursor or page["char_end"] < page["char_start"]:
                problems.append(f"{did}: page map gap or overlap at page {page['page_number']}")
                break
            cursor = page["char_end"]
        if meta["page_map"] and cursor != len(text):
            problems.append(f"{did}: page map ends at {cursor}, text is {len(text)}")
        structure = json.loads((out_dir / config.STRUCTURE_DIRNAME / f"{did}.json").read_text(encoding="utf-8"))
        for p in structure["provisions"]:
            provisions += 1
            piece = text[p["char_start"]:p["char_end"]]
            if hashlib.sha256(piece.encode("utf-8")).hexdigest() != p["text_sha256"]:
                problems.append(f"{did}: provision {p['key']} span does not match its hash")
    return {"documents": documents, "provisions": provisions, "problems": problems}


# --- CLI ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark",
                                     description="Build and verify the canonical corpus.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=None,
                        help="Corpus directory (default: <data-dir>/corpus).")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="Build the canonical corpus (resumable).")
    build.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    build.add_argument("--only-from", type=Path, help="File with one document_id per line.")
    build.add_argument("--force", action="store_true", help="Rebuild documents already built.")
    dup = sub.add_parser("duplicates", help="Find copies of the same instrument and identical provisions.")
    dup.add_argument("--review-sample", type=Path, metavar="CSV",
                     help="Also write a stratified sample of edges for human review to this CSV.")
    sub.add_parser("evidence", help="Build the gold evidence pool into <data-dir>/benchmark_build/.")
    smp = sub.add_parser("sample", help="Draw a seeded, stratified sample of gold candidates.")
    smp.add_argument("--seed", type=int, required=True)
    smp.add_argument("--size", type=int, default=config.DEFAULT_SAMPLE_SIZE)
    smp.add_argument("--tag", help="Draw only provisions with this author tag (a top-up).")
    smp.add_argument("--exclude-sample", type=Path, metavar="SAMPLE",
                     help="Never draw an instrument or provision class already in this sample.")
    smp.add_argument("--verify", type=Path, metavar="SAMPLE",
                     help="Instead of drawing, check an existing sample against the corpus.")
    author = sub.add_parser("author", help="Question authoring and verification.")
    asub = author.add_subparsers(dest="author_command", required=True)
    new = asub.add_parser("new", help="Draft a question skeleton from a sample row.")
    new.add_argument("--sample", type=Path, required=True)
    new.add_argument("--index", type=int, required=True)
    new.add_argument("--category", required=True)
    asub.add_parser("check", help="Validate every question.")
    rev = asub.add_parser("review", help="Build the verification page.")
    rev.add_argument("--page", type=Path, default=None,
                     help="Output HTML (default: <data-dir>/benchmark_build/review.html).")
    ver = asub.add_parser("verdicts", help="Apply a verifier's exported verdicts.json.")
    ver.add_argument("file", type=Path)
    sub.add_parser("verify", help="Re-check the published corpus files.")
    args = parser.parse_args(argv)
    out = args.out or args.data_dir / config.CORPUS_SUBDIR

    if args.command == "build":
        only = None
        if args.only_from:
            only = [l.strip() for l in args.only_from.read_text(encoding="utf-8").splitlines() if l.strip()]
        report = build_corpus(args.data_dir, out, workers=args.workers, only=only,
                              force=args.force, progress=True)
        print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
        for f in report["failures"][:20]:
            print(f"FAILED {f['document_id']}: {f['error_type']}: {f['error']}")
        if report["failures"]:
            print(f"{len(report['failures'])} eligible document(s) failed and were not written.")
            return 1
        return 0

    if args.command == "duplicates":
        from .duplicates import write_duplicates, write_review_sample
        print(json.dumps(write_duplicates(out), indent=2))
        if args.review_sample:
            rows = write_review_sample(out, args.review_sample)
            print(f"wrote {rows} edges for review to {args.review_sample}")
        return 0

    build_dir = args.data_dir / config.BUILD_SUBDIR
    if args.command == "evidence":
        from .sample import write_pool
        print(json.dumps(write_pool(out, build_dir), indent=2))
        return 0

    if args.command == "sample":
        from .sample import verify_sample, write_sample
        if args.verify:
            problems = verify_sample(args.verify, out)
            print(f"{args.verify}: {len(problems)} problem(s)")
            for p in problems[:50]:
                print("  " + p)
            return 1 if problems else 0
        path = write_sample(out, build_dir, seed=args.seed, size=args.size, tag=args.tag,
                            exclude_sample=args.exclude_sample)
        sample = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(sample["by_category"], indent=2))
        print(f"wrote {len(sample['rows'])} candidates to {path}")
        return 0

    if args.command == "author":
        from . import authoring
        if args.author_command == "new":
            q = authoring.new_draft(out, args.sample, args.index, args.category)
            print(f"wrote {authoring.save_question(q)}")
            return 0
        if args.author_command == "check":
            report = authoring.check(out)
            print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=2))
            for qid, problems in report["problems"].items():
                for p in problems:
                    print(f"  {qid}: {p}")
            return 1 if report["with_problems"] else 0
        if args.author_command == "review":
            page = args.page or build_dir / "review.html"
            print(f"{authoring.build_review_page(out, page)} questions on {page}")
            return 0
        print(json.dumps(authoring.apply_verdicts(args.file), indent=2))
        return 0

    result = verify_corpus(out)
    print(f"verified {result['documents']} documents, {result['provisions']} provisions; "
          f"{len(result['problems'])} problem(s)")
    for p in result["problems"][:50]:
        print("  " + p)
    return 1 if result["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
