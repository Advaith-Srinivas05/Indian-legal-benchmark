"""The scorer: what a system's returned spans are worth against gold evidence.

Implements ``docs/EVALUATION_PROTOCOL.md``. Three properties matter more than
the arithmetic, and each has a test named after it:

* **The budget is the unit of comparison.** A system is asked for its best
  spans within *B* characters. Overlapping spans are merged before the budget
  is counted, so padding the list with near-duplicates buys nothing, and a
  chunker with 4,000-character chunks cannot smuggle ten times the context past
  a 400-character one.
* **Recall and precision are always reported together.** Returning a whole
  document must score high recall and near-zero precision; that is the
  anti-gaming property, and it is what stops "return the Act" being a strategy.
* **Categories are never averaged together.** They measure different things.
  :func:`score_run` emits per-category figures and no cross-category mean —
  there is deliberately no single number to quote.

The oracle baseline scores 1.0 recall at B = 8,000 and B = 32,000; if it ever
does not, the scorer is wrong, not the system (``EVALUATION_PROTOCOL.md`` §6).
At B = 2,000 the oracle is *below* 1.0 and must be: a gold provision may be
6,000 characters long, and half of it does not fit in a 2,000-character budget.
The oracle's score is the ceiling at each budget, and that is what a system is
read against.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Optional, Protocol, Sequence

from . import config, terms
from .questions import Corpus

#: ``(document_id, char_start, char_end)`` — a half-open span of canonical text.
Span = tuple[str, int, int]

SCORER_VERSION = 1
#: Reported at three budgets; 8,000 is the headline (``EVALUATION_PROTOCOL.md`` §2).
DEFAULT_BUDGETS = (2000, 8000, 32000)
#: Share of a gold location's characters a system must return to have hit it.
DEFAULT_TAU = 0.5
STRICTER_TAUS = (0.5, 0.8, 1.0)
#: A quoted string short enough to occur by accident is not evidence of grounding.
QUOTE_MIN_CHARS = 12
_QUOTE = re.compile(r'"([^"\n]{%d,})"|“([^”\n]{%d,})”' % (QUOTE_MIN_CHARS, QUOTE_MIN_CHARS))


class RetrievalSystem(Protocol):
    """The whole contract a participating system implements."""

    name: str

    def retrieve(self, question: str, budget_chars: int) -> list[Span]:
        """Ranked spans of canonical text, best first. Empty is a real answer."""


# --- Spans and the budget -------------------------------------------------------------


def _subtract(start: int, end: int, taken: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of ``[start, end)`` not already covered by *taken* (sorted, disjoint)."""
    pieces: list[tuple[int, int]] = []
    cursor = start
    for t_start, t_end in taken:
        if t_end <= cursor:
            continue
        if t_start >= end:
            break
        if t_start > cursor:
            pieces.append((cursor, min(t_start, end)))
        cursor = max(cursor, t_end)
        if cursor >= end:
            return pieces
    if cursor < end:
        pieces.append((cursor, end))
    return pieces


def _insert(taken: list[tuple[int, int]], start: int, end: int) -> None:
    taken.append((start, end))
    taken.sort()
    merged: list[tuple[int, int]] = []
    for s, e in taken:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    taken[:] = merged


class Retrieved:
    """What a system's ranked spans amount to once the budget is applied.

    ``pieces`` are disjoint ``(rank, document_id, start, end)`` in rank order:
    the characters the system actually spent its budget on. A span that repeats
    ground already covered contributes nothing and costs nothing — merging
    happens here, once, so every metric below sees the same characters.
    """

    def __init__(self, pieces: list[tuple[int, str, int, int]], used_chars: int,
                 returned_spans: int, budget: int) -> None:
        self.pieces = pieces
        self.used_chars = used_chars
        self.returned_spans = returned_spans
        self.budget = budget
        self.union: dict[str, list[tuple[int, int]]] = {}
        for _rank, did, start, end in pieces:
            _insert(self.union.setdefault(did, []), start, end)

    @property
    def documents(self) -> set[str]:
        return set(self.union)

    def is_empty(self) -> bool:
        return not self.pieces


def within_budget(spans: Sequence[Span], budget: int) -> Retrieved:
    """Walk the ranked list, keeping novel characters until *budget* is spent.

    The span that crosses the line is truncated rather than dropped: a system
    whose chunks are larger than the budget should be scored on the part of its
    best chunk that fits, not on nothing at all.
    """
    if budget < 0:
        raise ValueError("budget must not be negative")
    taken: dict[str, list[tuple[int, int]]] = {}
    pieces: list[tuple[int, str, int, int]] = []
    used = 0
    for rank, (did, start, end) in enumerate(spans, start=1):
        if end <= start:
            continue
        if used >= budget:
            break
        for piece_start, piece_end in _subtract(start, end, taken.setdefault(did, [])):
            room = budget - used
            if room <= 0:
                break
            piece_end = min(piece_end, piece_start + room)
            _insert(taken[did], piece_start, piece_end)
            pieces.append((rank, did, piece_start, piece_end))
            used += piece_end - piece_start
    return Retrieved(pieces, used, len(spans), budget)


def _overlap(union: list[tuple[int, int]], start: int, end: int) -> int:
    return sum(max(0, min(end, e) - max(start, s)) for s, e in union)


def coverage(retrieved: Retrieved, location: dict) -> float:
    """Share of one gold location's characters the system returned.

    Computed over the union of everything returned for that document, so a
    provision split across two adjacent retrieved chunks still counts — it must,
    or the metric would punish small chunks for being small.
    """
    length = location["char_end"] - location["char_start"]
    if length <= 0:
        return 0.0
    union = retrieved.union.get(location["document_id"], [])
    return _overlap(union, location["char_start"], location["char_end"]) / length


# --- One question ---------------------------------------------------------------------


def _groups(q: dict, requirement: str) -> list[dict]:
    return [g for g in q["gold_evidence"] if g["requirement"] == requirement]


def _group_hit(retrieved: Retrieved, group: dict, tau: float) -> bool:
    return any(coverage(retrieved, loc) >= tau for loc in group["locations"])


def _group_rank(spans: Sequence[Span], group: dict, tau: float, budget: int) -> Optional[int]:
    """Rank of the shortest prefix of the ranked list that hits *group*."""
    for k in range(1, len(spans) + 1):
        if _group_hit(within_budget(spans[:k], budget), group, tau):
            return k
    return None


def score_question(q: dict, spans: Sequence[Span], budget: int, *,
                   tau: float = DEFAULT_TAU, with_rank: bool = True) -> dict:
    """Retrieval metrics for one question at one budget. Unanswerable: abstention only.

    *with_rank* buys MRR, which costs a pass over every prefix of the ranked
    list; the stricter-tau tables do not report it and do not pay for it.
    """
    retrieved = within_budget(spans, budget)
    result: dict = {
        "question_id": q["question_id"],
        "category": q["category"],
        "budget": budget,
        "tau": tau,
        "returned_spans": retrieved.returned_spans,
        "used_chars": retrieved.used_chars,
        "abstained": retrieved.is_empty(),
    }
    if q["unanswerable"]:
        return result

    sufficient, required = _groups(q, "sufficient"), _groups(q, "required")
    #: The protocol defines evidence recall over ``sufficient`` groups. A
    #: multi-hop question has no sufficient group — every hop is ``required`` —
    #: so recall would be undefined for the whole cross-reference category,
    #: which is the one category whose partial credit is worth seeing. Recall is
    #: therefore the share of *all* gold groups hit, and ``all_groups`` stays the
    #: strict all-or-nothing multi-hop metric beside it.
    scored = sufficient + required
    if scored:
        result["evidence_recall"] = sum(_group_hit(retrieved, g, tau) for g in scored) / len(scored)
    if required:
        result["all_groups"] = float(all(_group_hit(retrieved, g, tau) for g in required))

    gold_docs = {loc["document_id"] for g in scored for loc in g["locations"]}
    result["document_recall"] = (
        sum(any(loc["document_id"] in retrieved.documents for loc in g["locations"]) for g in scored)
        / len(scored)) if scored else None
    result["gold_documents"] = sorted(gold_docs)

    if retrieved.used_chars:
        inside = 0
        for _rank, did, start, end in retrieved.pieces:
            gold_here = sorted((loc["char_start"], loc["char_end"])
                               for g in scored for loc in g["locations"] if loc["document_id"] == did)
            inside += _overlap(gold_here, start, end)
        result["evidence_precision"] = inside / retrieved.used_chars

    if with_rank:
        ranks = [_group_rank(spans, g, tau, budget) for g in scored]
        result["mrr"] = (sum(1 / r if r else 0.0 for r in ranks) / len(ranks)) if ranks else None
    return result


# --- Answers --------------------------------------------------------------------------


def _citation_keys(q: dict) -> list[set[tuple[str, str]]]:
    """For each ``must_cite`` entry, every citation that is the same provision.

    A system that cites the same Act stored under a different handle has cited
    the right provision (``BENCHMARK_DESIGN.md`` §4). The right Act with the
    wrong section is still wrong: the key must match.
    """
    acceptable: list[set[tuple[str, str]]] = []
    for entry in q["must_cite"]:
        want = (entry["document_id"], entry["key"])
        same = {want}
        for group in q["gold_evidence"]:
            keys = {(loc["document_id"], loc["key"]) for loc in group["locations"]}
            if want in keys:
                same |= keys
        acceptable.append(same)
    return acceptable


def quotes(text: str) -> list[str]:
    return [a or b for a, b in _QUOTE.findall(text or "")]


def score_answer(q: dict, answer: dict, corpus: Optional[Corpus] = None,
                 extra_documents: Iterable[str] = ()) -> dict:
    """Fact coverage, citation precision/recall, quote grounding, abstention.

    Reported separately and never collapsed into one number: a system can be
    right about the facts and wrong about where they come from, and the point of
    the benchmark is to see that.
    """
    text = (answer or {}).get("text", "") or ""
    abstained = bool((answer or {}).get("abstained")) or not text.strip()
    result: dict = {"question_id": q["question_id"], "category": q["category"],
                    "abstained": abstained}
    if q["unanswerable"]:
        return result

    facts = q["required_facts"]
    if facts:
        result["fact_coverage"] = sum(
            any(terms.appears_in(variant, text) for variant in group) for group in facts) / len(facts)

    predicted = [(c["document_id"], c.get("key") or _key_of(c)) for c in (answer or {}).get("citations", [])]
    acceptable = _citation_keys(q)
    if acceptable:
        result["citation_recall"] = sum(any(p in ok for p in predicted) for ok in acceptable) / len(acceptable)
    if predicted:
        result["citation_precision"] = sum(any(p in ok for ok in acceptable) for p in predicted) / len(predicted)

    quoted = quotes(text)
    if quoted and corpus is not None:
        documents = {loc["document_id"] for g in q["gold_evidence"] for loc in g["locations"]}
        documents |= {did for did, _ in predicted} | set(extra_documents)
        haystack = " ".join(terms.normalise(corpus.text(d)) for d in sorted(documents) if corpus.meta(d))
        result["quote_grounding"] = sum(terms.normalise(s) in haystack for s in quoted) / len(quoted)
    result["quotes"] = len(quoted)
    return result


def _key_of(citation: dict) -> str:
    """The provision key a ``unit_path`` names, for systems that cite paths."""
    return "/".join(f"{u['unit_type']}:{u['number']}" for u in citation.get("unit_path", []))


# --- A whole run ------------------------------------------------------------------------


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _collect(rows: list[dict], key: str) -> Optional[float]:
    return _mean([r[key] for r in rows if r.get(key) is not None])


def corpus_fingerprint(corpus_dir: Path) -> dict:
    """What a reported result must name to be comparable (``EVALUATION_PROTOCOL.md`` §5)."""
    checksums = Path(corpus_dir) / config.CHECKSUMS_FILENAME
    digest = hashlib.sha256(checksums.read_bytes()).hexdigest() if checksums.exists() else None
    documents = sum(1 for _ in open(Path(corpus_dir) / config.DOCUMENTS_FILENAME, encoding="utf-8"))
    return {"corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
            "checksums_sha256": digest, "documents": documents}


def run_system(system: RetrievalSystem, questions: Sequence[dict],
               budgets: Sequence[int] = DEFAULT_BUDGETS,
               progress: Optional[object] = None) -> dict[str, dict[int, list[Span]]]:
    """Ask the system for spans, once per question per budget. No scoring here."""
    run: dict[str, dict[int, list[Span]]] = {}
    for n, q in enumerate(questions, start=1):
        run[q["question_id"]] = {b: [tuple(s) for s in system.retrieve(q["question"], b)] for b in budgets}
        if progress and n % 25 == 0:
            print(f"  {n}/{len(questions)} questions", flush=True)
    return run


def score_run(questions: Sequence[dict], run: dict[str, dict[int, list[Span]]], *,
              system: str, corpus_dir: Path, corpus: Optional[Corpus] = None,
              answers: Optional[dict[str, dict]] = None,
              budgets: Sequence[int] = DEFAULT_BUDGETS, tau: float = DEFAULT_TAU,
              split: str = "all", tuned_on_test: bool = False) -> dict:
    """The report. Per category, per budget — with no number that averages categories."""
    answerable = [q for q in questions if not q["unanswerable"]]
    unanswerable = [q for q in questions if q["unanswerable"]]
    categories = sorted({q["category"] for q in questions})

    retrieval: dict[str, dict[str, dict]] = {}
    stricter: dict[str, dict[str, dict]] = {}
    for category in categories:
        in_category = [q for q in answerable if q["category"] == category]
        if not in_category:
            continue
        retrieval[category] = {}
        stricter[category] = {}
        for budget in budgets:
            rows = [score_question(q, run.get(q["question_id"], {}).get(budget, []), budget, tau=tau)
                    for q in in_category]
            retrieval[category][str(budget)] = {
                "questions": len(rows),
                "evidence_recall": _collect(rows, "evidence_recall"),
                "all_groups": _collect(rows, "all_groups"),
                "evidence_precision": _collect(rows, "evidence_precision"),
                "document_recall": _collect(rows, "document_recall"),
                "mrr": _collect(rows, "mrr"),
                "mean_chars_used": _mean([float(r["used_chars"]) for r in rows]),
                "empty_results": sum(r["abstained"] for r in rows),
            }
            stricter[category][str(budget)] = {
                str(t): _collect([score_question(q, run.get(q["question_id"], {}).get(budget, []),
                                                 budget, tau=t, with_rank=False)
                                  for q in in_category], "evidence_recall")
                for t in STRICTER_TAUS}

    headline = budgets[len(budgets) // 2] if budgets else DEFAULT_BUDGETS[1]
    abstained_unanswerable = [bool(within_budget(run.get(q["question_id"], {}).get(headline, []), headline).is_empty())
                              for q in unanswerable]
    abstained_answerable = [bool(within_budget(run.get(q["question_id"], {}).get(headline, []), headline).is_empty())
                            for q in answerable]
    report = {
        "scorer_version": SCORER_VERSION,
        "system": system,
        "corpus": corpus_fingerprint(corpus_dir),
        "split": split,
        "tuned_on_test": tuned_on_test,
        "budgets": list(budgets),
        "tau": tau,
        "questions": {"total": len(questions), "answerable": len(answerable),
                      "unanswerable": len(unanswerable),
                      "by_category": {c: sum(q["category"] == c for q in questions) for c in categories}},
        "retrieval": {"by_category": retrieval, "stricter_tau": stricter},
        "abstention": {
            "budget": headline,
            "abstention_rate": _mean([float(x) for x in abstained_unanswerable]),
            "false_abstention": _mean([float(x) for x in abstained_answerable]),
            "unanswerable_questions": len(unanswerable),
            "answerable_questions": len(answerable),
        },
        "answers": None,
    }

    if answers is not None:
        by_category: dict[str, dict] = {}
        for category in categories:
            rows = [score_answer(q, answers.get(q["question_id"], {}), corpus)
                    for q in questions if q["category"] == category]
            answerable_rows = [r for r in rows if not next(
                q for q in questions if q["question_id"] == r["question_id"])["unanswerable"]]
            by_category[category] = {
                "questions": len(rows),
                "fact_coverage": _collect(answerable_rows, "fact_coverage"),
                "citation_precision": _collect(answerable_rows, "citation_precision"),
                "citation_recall": _collect(answerable_rows, "citation_recall"),
                "quote_grounding": _collect(answerable_rows, "quote_grounding"),
                "abstained": sum(r["abstained"] for r in rows),
            }
        report["answers"] = {
            "by_category": by_category,
            "abstention_accuracy": {
                "abstention_rate": _mean([float(score_answer(q, answers.get(q["question_id"], {}))["abstained"])
                                          for q in unanswerable]),
                "false_abstention": _mean([float(score_answer(q, answers.get(q["question_id"], {}))["abstained"])
                                           for q in answerable]),
            },
        }
    return report


def format_report(report: dict) -> str:
    """The report as a person reads it: one table per category, recall beside precision."""
    lines = [f"system: {report['system']}   split: {report['split']}   "
             f"tau: {report['tau']}   corpus: {report['corpus']['checksums_sha256']}",
             f"questions: {report['questions']['total']} "
             f"({report['questions']['answerable']} answerable, "
             f"{report['questions']['unanswerable']} unanswerable)"]
    if report["tuned_on_test"]:
        lines.append("WARNING: tuned on test")
    header = f"{'category':>18} {'B':>6} {'recall':>8} {'prec':>8} {'all-grp':>8} {'doc':>8} {'mrr':>8}"
    lines.append(header)
    for category, budgets in report["retrieval"]["by_category"].items():
        for budget, m in budgets.items():
            def show(key: str) -> str:
                return "     -  " if m[key] is None else f"{m[key]:8.3f}"
            lines.append(f"{category:>18} {budget:>6} {show('evidence_recall')} "
                         f"{show('evidence_precision')} {show('all_groups')} "
                         f"{show('document_recall')} {show('mrr')}")
    a = report["abstention"]
    lines.append(f"abstention (B={a['budget']}): unanswerable {a['abstention_rate']} "
                 f"/ false {a['false_abstention']}")
    return "\n".join(lines)


# --- Command line ------------------------------------------------------------------------


def add_arguments(parser) -> None:
    """The ``score`` options, shared by the repository CLI and the copied-out kit."""
    parser.add_argument("--system", help="A shipped baseline: oracle | bm25-windows | bm25-two-stage.")
    parser.add_argument("--predictions", type=Path,
                        help="A predictions JSON written by your own system (see benchmark/predictions.py).")
    parser.add_argument("--corpus", type=Path, default=None,
                        help="The published corpus directory (default: data/corpus).")
    parser.add_argument("--questions", type=Path, default=None,
                        help="Question directory (default: the one inside this package).")
    parser.add_argument("--split", default="test", help="test | dev | all")
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--index", type=Path, default=None, help="BM25 index, for the BM25 baselines.")
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--report", type=Path, default=None, help="Write the JSON report here.")
    parser.add_argument("--tuned-on-test", action="store_true",
                        help="Record that this system was tuned on the test split.")


def run(args) -> int:
    """Score a baseline or a predictions file. Returns a process exit code."""
    from . import splits
    from .jsonio import dumps, write_atomic
    from .questions import QUESTIONS_DIR, Corpus, load_questions

    if bool(args.system) == bool(args.predictions):
        print("give exactly one of --system (a shipped baseline) or --predictions (your own run)")
        return 2

    corpus_dir = Path(args.corpus) if args.corpus else Path("data") / config.CORPUS_SUBDIR
    if not (corpus_dir / config.DOCUMENTS_FILENAME).exists():
        print(f"no corpus at {corpus_dir} — pass --corpus")
        return 2

    verified = load_questions(args.questions or QUESTIONS_DIR, verified_only=True)
    split_file = args.split_file
    if split_file is None and args.split != "all":
        found = sorted(splits.SPLITS_DIR.glob("split-s*.json"))
        if len(found) != 1:
            print(f"specify --split-file: {len(found)} split files in {splits.SPLITS_DIR}")
            return 2
        split_file = found[0]
    questions = splits.select(verified, split_file, args.split)
    budgets = tuple(args.budgets) if args.budgets else DEFAULT_BUDGETS
    corpus = Corpus(corpus_dir)
    answers = None

    if args.predictions:
        from . import predictions as predictions_module
        loaded = predictions_module.load(args.predictions, questions, budgets)
        name, run_spans, answers = loaded["system"], loaded["run"], loaded["answers"]
        if loaded["split"] and loaded["split"] != args.split:
            print(f"WARNING: the file says split {loaded['split']!r}, scoring {args.split!r}")
        if loaded["missing"]:
            print(f"WARNING: {len(loaded['missing'])} question(s) have no prediction and are "
                  f"scored as abstentions, starting with {loaded['missing'][0]}")
    else:
        from . import baselines
        system = baselines.load(args.system, corpus_dir=corpus_dir, questions=questions,
                                index_path=args.index)
        name = system.name
        print(f"running {name} over {len(questions)} questions at budgets {budgets}")
        run_spans = run_system(system, questions, budgets, progress=True)

    report = score_run(questions, run_spans, system=name, corpus_dir=corpus_dir, corpus=corpus,
                       answers=answers, budgets=budgets, tau=args.tau or DEFAULT_TAU,
                       split=args.split, tuned_on_test=args.tuned_on_test)
    print(format_report(report))
    if args.report:
        write_atomic(args.report, dumps(report))
        print(f"wrote {args.report}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Standalone entry point: ``python -m benchmark score …`` in a copied-out kit."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="python -m benchmark score",
        description="Score a RAG system against the IndiaStatRAG question set.")
    add_arguments(parser)
    return run(parser.parse_args(argv))
