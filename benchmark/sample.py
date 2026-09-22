"""A seeded, stratified draw of gold candidates for question authors.

The draw is over **instruments**, not provisions, so a long code cannot swamp the
sample: within each category, instruments (a duplicate cluster, or a lone
document) are shuffled, and one provision is chosen from each in turn. Two
provisions whose text is word for word the same (one equivalence class) count
once. Only when a category runs out of instruments does it take a second
provision from one — and the report says so.

Everything depends on the seed alone. The sample records a fingerprint of the
corpus it was drawn from (the sha256 of ``CHECKSUMS.txt``), so a sample can
never be silently read against a different corpus.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

from . import config
from .evidence import build_pool


def corpus_fingerprint(corpus_dir: Path) -> str:
    return hashlib.sha256((Path(corpus_dir) / config.CHECKSUMS_FILENAME).read_bytes()).hexdigest()


def write_pool(corpus_dir: Path, build_dir: Path) -> dict:
    """Build the pool and write it, with its report, to the build directory."""
    from .corpus import _json, _write_atomic

    rows, report = build_pool(corpus_dir)
    report["corpus_fingerprint"] = corpus_fingerprint(corpus_dir)
    _write_atomic(Path(build_dir) / config.POOL_FILENAME, "".join(_json(r, compact=True) for r in rows))
    _write_atomic(Path(build_dir) / config.POOL_REPORT_FILENAME, _json(report))
    return report


def load_pool(build_dir: Path, corpus_dir: Path) -> list[dict]:
    """Read the written pool, refusing one built from a different corpus."""
    build_dir = Path(build_dir)
    report = json.loads((build_dir / config.POOL_REPORT_FILENAME).read_text(encoding="utf-8"))
    if report.get("corpus_fingerprint") != corpus_fingerprint(corpus_dir):
        raise ValueError("the evidence pool was built from a different corpus; "
                         "run `python -m benchmark evidence` again")
    with open(build_dir / config.POOL_FILENAME, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _instrument(row: dict) -> str:
    return row["cluster_id"] or row["document_id"]


def draw(pool: list[dict], *, seed: int, size: int,
         allocation: Optional[dict[str, float]] = None, tag: Optional[str] = None,
         exclude: Optional[list[dict]] = None) -> tuple[list[dict], dict]:
    """Draw *size* candidates. Returns (rows, per-category report).

    *tag* restricts the draw to provisions carrying that author tag — a top-up
    for a question category the main sample under-supplies. *exclude* is an
    earlier sample's rows: its instruments and equivalence classes are not drawn
    again, so a top-up never overlaps what it tops up. *allocation* may be
    ``"proportional"``: shares follow the filtered pool's instruments.
    """
    rng = random.Random(seed)
    excluded_instruments = {_instrument(r) for r in exclude or []}
    excluded_classes = {r["equivalence_class"] for r in exclude or [] if r["equivalence_class"]}
    pool = [r for r in pool
            if (tag is None or tag in r["tags"])
            and _instrument(r) not in excluded_instruments
            and r["equivalence_class"] not in excluded_classes]

    # Collapse word-for-word identical provisions to one representative each.
    seen_class: set[str] = set()
    unique: list[dict] = []
    for row in sorted(pool, key=lambda r: (r["category"] != "central_acts", r["document_id"], r["key"])):
        cls = row["equivalence_class"]
        if cls:
            if cls in seen_class:
                continue
            seen_class.add(cls)
        unique.append(row)

    if allocation == "proportional":
        counts: dict[str, set] = defaultdict(set)
        for row in unique:
            counts[row["category"]].add(_instrument(row))
        total = sum(len(v) for v in counts.values()) or 1
        allocation = {c: len(v) / total for c, v in sorted(counts.items())} or dict(config.DEFAULT_ALLOCATION)
    allocation = allocation or config.DEFAULT_ALLOCATION
    if abs(sum(allocation.values()) - 1.0) > 1e-9:
        raise ValueError(f"allocation shares sum to {sum(allocation.values())}, not 1")

    # One home category per instrument. A cluster can hold a Central Act and its
    # state-collection copies; without this it would be an instrument in both
    # strata and could be drawn twice. Central wins, then the first document.
    home: dict[str, str] = {}
    for row in sorted(unique, key=lambda r: (r["category"] != "central_acts", r["document_id"])):
        home.setdefault(_instrument(row), row["category"])
    by_instrument: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in unique:
        by_instrument[home[_instrument(row)]][_instrument(row)].append(row)

    targets = _targets(size, allocation)
    drawn: list[dict] = []
    report: dict[str, dict] = {}
    for category in sorted(targets):
        instruments = by_instrument.get(category, {})
        order = sorted(instruments)
        rng.shuffle(order)
        remaining = {k: sorted(instruments[k], key=lambda r: (r["document_id"], r["key"])) for k in order}
        picked: list[dict] = []
        rounds = 0
        while len(picked) < targets[category] and any(remaining.values()):
            rounds += 1
            for k in order:
                if len(picked) >= targets[category]:
                    break
                if remaining[k]:
                    picked.append(remaining[k].pop(rng.randrange(len(remaining[k]))))
        for row in picked:
            drawn.append({**row, "stratum": category})
        report[category] = {"target": targets[category], "drawn": len(picked),
                            "instruments_available": len(order), "rounds": rounds}

    for i, row in enumerate(drawn):
        row["sample_index"] = i
    return drawn, report


def _targets(size: int, allocation: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding, so the targets always sum to *size*."""
    raw = {c: size * s for c, s in allocation.items()}
    targets = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: (-(raw[c] - targets[c]), c))[:size - sum(targets.values())]:
        targets[c] += 1
    return targets


def write_sample(corpus_dir: Path, build_dir: Path, *, seed: int, size: int,
                 out_dir: Optional[Path] = None, allocation=None, tag: Optional[str] = None,
                 exclude_sample: Optional[Path] = None) -> Path:
    from .corpus import _json, _write_atomic

    pool = load_pool(build_dir, corpus_dir)
    exclude = None
    if exclude_sample:
        exclude = json.loads(Path(exclude_sample).read_text(encoding="utf-8"))["rows"]
    if tag and allocation is None:
        allocation = "proportional"
    rows, report = draw(pool, seed=seed, size=size, allocation=allocation, tag=tag, exclude=exclude)
    payload = {
        "sample_schema_version": config.SAMPLE_SCHEMA_VERSION,
        "seed": seed,
        "size": size,
        "tag": tag,
        "excludes_sample": Path(exclude_sample).name if exclude_sample else None,
        "allocation": allocation or config.DEFAULT_ALLOCATION,
        "corpus_fingerprint": corpus_fingerprint(corpus_dir),
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "pool_size": len(pool),
        "by_category": report,
        "rows": rows,
    }
    out_dir = Path(out_dir) if out_dir else config.SAMPLES_DIR
    path = out_dir / (f"sample-s{seed}-n{size}" + (f"-{tag}" if tag else "") + ".json")
    _write_atomic(path, _json(payload))
    return path


def verify_sample(sample_path: Path, corpus_dir: Path) -> list[str]:
    """Check every sampled row still resolves, exactly, in the corpus."""
    corpus_dir = Path(corpus_dir)
    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    problems = []
    if sample["corpus_fingerprint"] != corpus_fingerprint(corpus_dir):
        problems.append("sample was drawn from a different corpus (CHECKSUMS.txt differs)")
    texts: dict[str, str] = {}
    for row in sample["rows"]:
        did = row["document_id"]
        if did not in texts:
            meta = json.loads((corpus_dir / config.META_DIRNAME / f"{did}.json").read_text(encoding="utf-8"))
            with open(corpus_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
                texts[did] = handle.read()
            # Hash the file itself, not the meta record's claim about it.
            if hashlib.sha256(texts[did].encode("utf-8")).hexdigest() != row["document_text_sha256"]:
                problems.append(f"{did}: document text changed since the sample was drawn")
        piece = texts[did][row["char_start"]:row["char_end"]]
        if hashlib.sha256(piece.encode("utf-8")).hexdigest() != row["text_sha256"]:
            problems.append(f"{did} {row['key']}: span no longer reproduces its text")
    return problems
