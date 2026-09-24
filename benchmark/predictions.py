"""The predictions file: how a system is scored without importing this package.

A participant runs their own pipeline, in their own language if they like,
writes one JSON file of the spans they would hand a reader, and scores it:

```json
{
  "system": "my-rag-v3",
  "split": "test",
  "budgets": [2000, 8000, 32000],
  "predictions": {
    "IN-STAT-0001": {
      "2000":  [["some-act__handle-123", 4120, 5600]],
      "8000":  [["some-act__handle-123", 4120, 5600], ["other-act__handle-9", 0, 900]],
      "32000": []
    }
  },
  "answers": {
    "IN-STAT-0001": {"text": "…", "abstained": false,
                     "citations": [{"document_id": "some-act__handle-123", "key": "section:8"}]}
  }
}
```

A span is ``[document_id, char_start, char_end]`` — half-open, into the
canonical text file of that document. **An empty list is a real answer**: it is
how a system abstains, and the unanswerable questions are where that earns
credit. ``answers`` is optional; retrieval-only submissions are valid.

Everything here is strict and loud. A prediction for a question that is not in
the split, a span that runs backwards, a budget the file does not cover — each
stops the run with the reason, because a silently-dropped prediction is a
score that means nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

from .score import Span

PREDICTIONS_SCHEMA_VERSION = 1


class PredictionsError(ValueError):
    """The predictions file cannot be scored as it stands, and why."""


def _spans(raw, where: str) -> list[Span]:
    if not isinstance(raw, list):
        raise PredictionsError(f"{where}: expected a list of spans, got {type(raw).__name__}")
    spans: list[Span] = []
    for i, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise PredictionsError(f"{where}[{i}]: a span is [document_id, char_start, char_end]")
        did, start, end = item
        if not isinstance(did, str) or not isinstance(start, int) or not isinstance(end, int):
            raise PredictionsError(f"{where}[{i}]: document_id must be a string and the offsets integers")
        if start < 0 or end < start:
            raise PredictionsError(f"{where}[{i}]: {start}..{end} is not a forward span")
        spans.append((did, start, end))
    return spans


def load(path: Path, questions: Sequence[dict], budgets: Sequence[int]) -> dict:
    """Read and check a predictions file against the questions being scored.

    Returns ``{"system", "run", "answers", "split", "missing"}``. A question with
    no entry at all counts as an abstention, and is reported in ``missing`` so a
    half-finished run cannot look like a modest one.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "predictions" not in payload:
        raise PredictionsError("the file must be an object with a 'predictions' member")

    known = {q["question_id"] for q in questions}
    predictions = payload["predictions"]
    if not isinstance(predictions, dict):
        raise PredictionsError("'predictions' must map question ids to budgets")
    unknown = sorted(set(predictions) - known)
    if unknown:
        raise PredictionsError(
            f"{len(unknown)} prediction(s) are not in the split being scored, "
            f"starting with {unknown[0]} — check --split")

    run: dict[str, dict[int, list[Span]]] = {}
    for qid, by_budget in predictions.items():
        if not isinstance(by_budget, dict):
            raise PredictionsError(f"{qid}: expected an object keyed by budget")
        row: dict[int, list[Span]] = {}
        for budget in budgets:
            if str(budget) in by_budget:
                row[budget] = _spans(by_budget[str(budget)], f"{qid}[{budget}]")
            elif budget in by_budget:                      # ints survive some writers
                row[budget] = _spans(by_budget[budget], f"{qid}[{budget}]")
            else:
                raise PredictionsError(
                    f"{qid}: no spans for budget {budget}. Either predict for every budget "
                    f"in {list(budgets)} or pass --budgets to score only the ones you have")
        run[qid] = row

    answers = payload.get("answers") or None
    if answers is not None:
        if not isinstance(answers, dict):
            raise PredictionsError("'answers' must map question ids to answer objects")
        stray = sorted(set(answers) - known)
        if stray:
            raise PredictionsError(f"{len(stray)} answer(s) are not in the split, starting with {stray[0]}")

    return {
        "system": payload.get("system") or Path(path).stem,
        "split": payload.get("split"),
        "run": run,
        "answers": answers,
        "missing": sorted(known - set(predictions)),
    }
