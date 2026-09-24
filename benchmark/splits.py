"""The public dev / test split: 30 % dev, 70 % test, stratified, seeded.

``BENCHMARK_DESIGN.md`` §6. There is no training split — RAG systems are not
trained on this. ``dev`` is for development and tuning; ``test`` is what a
result reports, and a report must say which it used.

The split lives in its own file rather than in the question records: the
questions are verified artefacts, and re-splitting must never mean rewriting
them. It is written under ``benchmark/data/splits/`` and tracked in git, because
a result measured against a different split is not comparable to one measured
against this one.

Strata are ``(category, jurisdiction, evidence confidence)`` — the three axes the
design names. Within a stratum the assignment is a seeded shuffle, so a small
stratum cannot land entirely in one half by accident.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional, Sequence

from . import config
from .jsonio import dumps, write_atomic
from .questions import Corpus

SPLIT_SCHEMA_VERSION = 1
SPLITS = ("dev", "test")
DEV_SHARE = 0.30
SPLITS_DIR = Path(__file__).resolve().parent / "data" / "splits"


def stratum(q: dict, corpus: Optional[Corpus] = None) -> tuple[str, str, str]:
    """``(category, jurisdiction, confidence)`` — what the split balances."""
    locations = [loc for g in q["gold_evidence"] for loc in g["locations"]]
    if not locations:
        return (q["category"], "none", "none")
    first = locations[0]
    jurisdiction = q.get("jurisdiction_hint") or "India"
    if corpus is not None:
        meta = corpus.meta(first["document_id"])
        if meta:
            jurisdiction = meta.get("jurisdiction") or jurisdiction
    return (q["category"], jurisdiction, first.get("evidence_confidence") or "none")


def assign(questions: Sequence[dict], *, seed: int, corpus: Optional[Corpus] = None,
           dev_share: float = DEV_SHARE) -> dict[str, str]:
    """Question id → ``dev`` or ``test``. Deterministic in *seed* and the ids."""
    buckets: dict[tuple[str, str, str], list[str]] = {}
    for q in sorted(questions, key=lambda x: x["question_id"]):
        buckets.setdefault(stratum(q, corpus), []).append(q["question_id"])

    split: dict[str, str] = {}
    #: Fractional dev seats carry across strata, so 30 % holds over the set even
    #: when most strata are small enough to round to zero on their own.
    carry = 0.0
    for key in sorted(buckets):
        ids = buckets[key]
        rng = random.Random(f"{seed}|{'|'.join(key)}")
        shuffled = list(ids)
        rng.shuffle(shuffled)
        want = len(ids) * dev_share + carry
        take = int(want)
        carry = want - take
        for i, qid in enumerate(shuffled):
            split[qid] = "dev" if i < take else "test"
    return split


def write_split(questions: Sequence[dict], path: Path, *, seed: int,
                corpus: Optional[Corpus] = None, dev_share: float = DEV_SHARE) -> dict:
    """Write the split file and return it."""
    split = assign(questions, seed=seed, corpus=corpus, dev_share=dev_share)
    by_category: dict[str, dict[str, int]] = {}
    for q in questions:
        row = by_category.setdefault(q["category"], {"dev": 0, "test": 0})
        row[split[q["question_id"]]] += 1
    record = {
        "split_schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "dev_share": dev_share,
        "questions": len(questions),
        "counts": {s: sum(1 for v in split.values() if v == s) for s in SPLITS},
        "by_category": by_category,
        "assignment": {qid: split[qid] for qid in sorted(split)},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(path, dumps(record))
    return record


def load_split(path: Path) -> dict[str, str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["assignment"]


def select(questions: Sequence[dict], split_path: Optional[Path], which: str) -> list[dict]:
    """The questions in *which* split — ``all`` needs no split file."""
    if which == "all":
        return list(questions)
    if which not in SPLITS:
        raise ValueError(f"split must be one of {('all',) + SPLITS}")
    if split_path is None:
        raise ValueError("a split file is required to score a dev or test split")
    assignment = load_split(split_path)
    missing = [q["question_id"] for q in questions if q["question_id"] not in assignment]
    if missing:
        raise KeyError(f"{len(missing)} question(s) are not in the split file, "
                       f"starting with {missing[0]} — re-run 'split' after authoring")
    return [q for q in questions if assignment[q["question_id"]] == which]
