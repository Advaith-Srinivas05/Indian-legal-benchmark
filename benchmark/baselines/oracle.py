"""The oracle: return the gold spans. The ceiling, and the scorer's own test.

It must score 1.0 evidence recall at B = 8,000 and B = 32,000. If it does not,
the scorer is broken, not the system (``EVALUATION_PROTOCOL.md`` §6). That is
the single most important check in the project, and it is why this baseline
exists at all.

At B = 2,000 it scores less, and must: τ = 0.5 of a 6,000-character provision
does not fit in 2,000 characters. What the oracle scores *is* the ceiling at
that budget — no system can do better, and systems are read against it.

The oracle returns one location per group, best-covering group first, then the
remaining locations. It does not return *every* equivalent copy: the budget is
finite, and a system that hits one location in a group has hit the group.
"""

from __future__ import annotations

from typing import Sequence

from ..score import Span


class Oracle:
    name = "oracle"

    def __init__(self, questions: Sequence[dict], *, noisy_chars: int = 0) -> None:
        #: Keyed by question text: the scorer hands a system the question, nothing else.
        self._spans: dict[str, list[Span]] = {}
        for q in questions:
            self._spans[q["question"]] = self._plan(q, noisy_chars)

    @staticmethod
    def _plan(q: dict, noisy_chars: int) -> list[Span]:
        first: list[Span] = []
        rest: list[Span] = []
        #: ``required`` groups first: a multi-hop question is only answered when
        #: every one of them is covered, and the budget is spent in rank order.
        groups = ([g for g in q["gold_evidence"] if g["requirement"] == "required"]
                  + [g for g in q["gold_evidence"] if g["requirement"] != "required"])
        for group in groups:
            for i, loc in enumerate(group["locations"]):
                start, end = loc["char_start"], loc["char_end"]
                if noisy_chars:
                    start = max(0, start - noisy_chars)
                    end = end + noisy_chars
                (first if i == 0 else rest).append((loc["document_id"], start, end))
        return first + rest

    def retrieve(self, question: str, budget_chars: int) -> list[Span]:
        return list(self._spans.get(question, []))
