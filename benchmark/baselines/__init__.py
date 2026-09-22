"""Baselines shipped with the benchmark, so "better than nothing" is not the bar.

``EVALUATION_PROTOCOL.md`` §6 names four. Three are here:

* ``oracle`` — gold spans returned directly. The ceiling, and the test that the
  scorer is correct: it must score 1.0 recall at every budget.
* ``bm25-windows`` — BM25 over fixed-size windows of canonical text: the sparse
  floor.
* ``bm25-two-stage`` — document resolution, then search inside the chosen
  documents. The design the archived ``corpusdb/query.py`` implemented, whose
  notes warn it "may be surprisingly hard to beat on citation lookups".

The fourth, a **dense single-vector baseline, is not shipped in v1**. It needs a
named embedding model, and every model worth naming has to be downloaded; this
project reads local data only and adds no network dependency. The interface is
the four lines of :class:`benchmark.score.RetrievalSystem`, so a dense system is
a contribution, not a fork — but shipping one here would mean shipping an
untested claim, and the results report says plainly that the slot is empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from ..score import RetrievalSystem

#: Baseline name → what it establishes, in the order they are reported.
NAMES = ("oracle", "bm25-windows", "bm25-two-stage")


def load(name: str, *, corpus_dir: Path, questions: Optional[Sequence[dict]] = None,
         index_path: Optional[Path] = None, **kwargs) -> RetrievalSystem:
    """Build one baseline by name, with only what that baseline needs."""
    if name == "oracle":
        if questions is None:
            raise ValueError("the oracle baseline needs the question set")
        from .oracle import Oracle
        return Oracle(questions, **kwargs)
    if name in ("bm25-windows", "bm25-two-stage"):
        if index_path is None:
            raise ValueError(f"{name} needs --index (build it with 'python -m benchmark index')")
        from .bm25 import Bm25Index, Bm25TwoStage, Bm25Windows
        index = Bm25Index(index_path)
        return Bm25Windows(index, **kwargs) if name == "bm25-windows" else Bm25TwoStage(index, **kwargs)
    raise ValueError(f"unknown baseline {name!r}; one of {NAMES}")
