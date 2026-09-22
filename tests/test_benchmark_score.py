"""The scorer: the five properties ``EVALUATION_PROTOCOL.md`` says must hold.

Every test here is named after the thing that goes wrong if the scorer is
written carelessly, and each builds a real miniature corpus rather than mocking
one, so a change to the corpus format is caught here too.
"""

from __future__ import annotations

import json

import pytest

from benchmark import score, splits
from benchmark.baselines.oracle import Oracle
from benchmark.corpus import build_corpus
from benchmark.duplicates import write_duplicates
from benchmark.questions import Corpus, location
from tests.processedbuild import make_page, write_processed

LICENSING = """THE LICENSING AUTHORITY ACT, 2001
1. Short title.—This Act may be called the Licensing Authority Act, 2001, and it shall apply to every district of the territory from the day it is notified.
2. Definitions.—In this Act, unless the context otherwise requires, "authority" means the licensing authority appointed under section 3 of this Act for any district.
3. Authority.—The Government shall appoint a licensing authority for each district, who shall decide every application within thirty days of its receipt."""

MARKETS = """THE MARKETS ACT, 2003
1. Short title.—This Act may be called the Markets Act, 2003, and it extends to the whole of the State of Bihar from the appointed day.
2. Market committee.—The State Government shall constitute a market committee for every market area declared under this Act by notification.
3. Fees.—A market committee may levy a fee not exceeding two rupees on every hundred rupees of the value of the produce sold in the market area."""


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    write_processed(data, "licensing-act__handle-1", [make_page(1, LICENSING)],
                    title="The Licensing Authority Act, 2001", category="central_acts",
                    document_type="central_act")
    write_processed(data, "licensing-act-copy__handle-2", [make_page(1, LICENSING)],
                    title="The Licensing Authority Act, 2001", category="state_acts",
                    document_type="state_act")
    write_processed(data, "markets-act__handle-3", [make_page(1, MARKETS)],
                    title="The Markets Act, 2003", category="state_acts",
                    document_type="state_act", jurisdiction="Bihar")
    corpus_dir = data / "corpus"
    build_corpus(data, corpus_dir)
    write_duplicates(corpus_dir)
    return {"dir": corpus_dir, "corpus": Corpus(corpus_dir)}


def question(env, qid, category, *, groups, facts=(), must_cite=(), unanswerable=False):
    corpus = env["corpus"]
    gold = [{"group_id": f"g{i}", "requirement": requirement,
             "locations": [location(corpus, did, key, "sampled") for did, key in locations]}
            for i, (requirement, locations) in enumerate(groups, start=1)]
    return {
        "question_id": qid, "question": f"question {qid}", "category": category,
        "answer_type": "abstractive", "jurisdiction_hint": None, "gold_evidence": gold,
        "proposed_alternatives": [], "gold_answer": "an answer",
        "required_facts": [list(f) for f in facts],
        "must_cite": [{"document_id": d, "key": k} for d, k in must_cite],
        "unanswerable": unanswerable, "difficulty": {},
        "provenance": {"status": "verified"},
    }


def one(env, qid="IN-STAT-0001", category="situational", **kwargs):
    return question(env, qid, category,
                    groups=[("sufficient", [("licensing-act__handle-1", "section:3")])], **kwargs)


def span_of(env, did, key):
    p = env["corpus"].provision(did, key)
    return (did, p["char_start"], p["char_end"])


# --- The five checks the protocol names ------------------------------------------------


def test_the_oracle_scores_perfect_recall_at_every_budget(env):
    """If this fails the scorer is wrong, not the system. The project's most important test."""
    questions = [
        one(env),
        question(env, "IN-STAT-0002", "cross_reference", groups=[
            ("required", [("licensing-act__handle-1", "section:2")]),
            ("required", [("licensing-act__handle-1", "section:3")])]),
        question(env, "IN-STAT-0003", "jurisdictional", groups=[
            ("sufficient", [("markets-act__handle-3", "section:3")])]),
    ]
    run = score.run_system(Oracle(questions), questions, score.DEFAULT_BUDGETS)
    report = score.score_run(questions, run, system="oracle", corpus_dir=env["dir"])
    for category, budgets in report["retrieval"]["by_category"].items():
        for budget, metrics in budgets.items():
            assert metrics["evidence_recall"] == 1.0, (category, budget)
            assert metrics["document_recall"] == 1.0
            if metrics["all_groups"] is not None:
                assert metrics["all_groups"] == 1.0
            assert metrics["evidence_precision"] == pytest.approx(1.0)


def test_the_oracle_cannot_half_cover_a_provision_longer_than_twice_the_budget(env):
    """The B=2,000 ceiling is arithmetic, not a fault — and must never be "fixed"
    by lowering tau or measuring coverage against the budget instead of the gold."""
    q = one(env)
    did, start, end = span_of(env, "licensing-act__handle-1", "section:3")
    length = end - start
    oracle = Oracle([q])
    tight = length // 2 - 5
    assert score.score_question(q, oracle.retrieve(q["question"], tight), tight)["evidence_recall"] == 0.0
    assert score.score_question(q, oracle.retrieve(q["question"], length), length)["evidence_recall"] == 1.0


def test_the_oracle_is_perfect_at_every_stricter_tau(env):
    questions = [one(env)]
    run = score.run_system(Oracle(questions), questions, score.DEFAULT_BUDGETS)
    report = score.score_run(questions, run, system="oracle", corpus_dir=env["dir"])
    for budgets in report["retrieval"]["stricter_tau"].values():
        for by_tau in budgets.values():
            assert set(by_tau.values()) == {1.0}


def test_overlapping_spans_are_counted_once_against_the_budget(env):
    """Padding the ranked list with near-duplicates must buy nothing."""
    did, start, end = span_of(env, "licensing-act__handle-1", "section:3")
    once = score.within_budget([(did, start, end)], 10_000)
    padded = score.within_budget(
        [(did, start, end), (did, start, end), (did, start + 10, end - 10)], 10_000)
    assert padded.used_chars == once.used_chars == end - start


def test_returning_a_whole_document_scores_high_recall_and_near_zero_precision(env):
    """The anti-gaming property: "return the Act" must not be a winning strategy."""
    q = one(env)
    whole = [("licensing-act__handle-1", 0, len(env["corpus"].text("licensing-act__handle-1")))]
    result = score.score_question(q, whole, 32_000)
    assert result["evidence_recall"] == 1.0
    assert result["evidence_precision"] < 0.5


def test_right_act_with_wrong_section_scores_zero_on_citation(env):
    q = one(env, must_cite=[("licensing-act__handle-1", "section:3")])
    wrong = {"text": "The authority decides within thirty days.",
             "citations": [{"document_id": "licensing-act__handle-1", "key": "section:2"}]}
    result = score.score_answer(q, wrong, env["corpus"])
    assert result["citation_recall"] == 0.0
    assert result["citation_precision"] == 0.0


def test_the_report_never_averages_categories_together(env):
    questions = [one(env), question(env, "IN-STAT-0003", "jurisdictional", groups=[
        ("sufficient", [("markets-act__handle-3", "section:3")])])]
    run = score.run_system(Oracle(questions), questions, (8000,))
    report = score.score_run(questions, run, system="oracle", corpus_dir=env["dir"], budgets=(8000,))
    assert set(report["retrieval"]["by_category"]) == {"situational", "jurisdictional"}
    flat = json.dumps(report)
    for forbidden in ('"overall"', '"mean"', '"macro"', '"average"'):
        assert forbidden not in flat


# --- Budget mechanics --------------------------------------------------------------------


def test_a_span_larger_than_the_budget_is_truncated_not_dropped(env):
    did = "licensing-act__handle-1"
    retrieved = score.within_budget([(did, 0, 5000)], 100)
    assert retrieved.used_chars == 100
    assert retrieved.pieces == [(1, did, 0, 100)]


def test_the_budget_stops_the_ranked_list(env):
    did = "licensing-act__handle-1"
    retrieved = score.within_budget([(did, 0, 60), (did, 100, 160), (did, 200, 260)], 100)
    assert retrieved.used_chars == 100
    assert [p[0] for p in retrieved.pieces] == [1, 2]


def test_a_provision_split_across_two_chunks_still_counts(env):
    """Coverage is computed over the union, or small chunks are punished for being small."""
    did, start, end = span_of(env, "licensing-act__handle-1", "section:3")
    middle = (start + end) // 2
    q = one(env)
    halves = [(did, start, middle), (did, middle, end)]
    assert score.score_question(q, halves, 8000)["evidence_recall"] == 1.0
    assert score.score_question(q, halves[:1], 8000, tau=0.8)["evidence_recall"] == 0.0


def test_half_a_provision_hits_at_the_default_tau_but_not_at_one(env):
    did, start, end = span_of(env, "licensing-act__handle-1", "section:3")
    half = [(did, start, start + (end - start) // 2 + 5)]
    q = one(env)
    assert score.score_question(q, half, 8000, tau=0.5)["evidence_recall"] == 1.0
    assert score.score_question(q, half, 8000, tau=1.0)["evidence_recall"] == 0.0


def test_a_hit_in_a_duplicate_copy_of_the_act_counts(env):
    """Two handles, one Act: retrieving either is retrieving the provision."""
    q = question(env, "IN-STAT-0004", "situational", groups=[
        ("sufficient", [("licensing-act__handle-1", "section:3"),
                        ("licensing-act-copy__handle-2", "section:3")])])
    copy_span = span_of(env, "licensing-act-copy__handle-2", "section:3")
    assert score.score_question(q, [copy_span], 8000)["evidence_recall"] == 1.0


def test_the_wrong_act_scores_zero_on_recall_and_on_documents(env):
    q = one(env)
    result = score.score_question(q, [span_of(env, "markets-act__handle-3", "section:3")], 8000)
    assert result["evidence_recall"] == 0.0
    assert result["document_recall"] == 0.0
    assert result["evidence_precision"] == 0.0


def test_mrr_rewards_ranking_the_gold_span_first(env):
    q = one(env)
    gold = span_of(env, "licensing-act__handle-1", "section:3")
    noise = span_of(env, "markets-act__handle-3", "section:2")
    assert score.score_question(q, [gold, noise], 8000)["mrr"] == 1.0
    assert score.score_question(q, [noise, gold], 8000)["mrr"] == 0.5
    assert score.score_question(q, [noise], 8000)["mrr"] == 0.0


# --- Abstention and answers ----------------------------------------------------------------


def test_abstention_is_reported_as_a_pair(env):
    """A system that abstains on everything scores 1.0 and 1.0 — visibly useless."""
    answerable = one(env)
    unanswerable = question(env, "IN-STAT-0009", "unanswerable", groups=[], unanswerable=True)
    questions = [answerable, unanswerable]

    class Silent:
        name = "silent"

        def retrieve(self, question, budget_chars):
            return []

    run = score.run_system(Silent(), questions, (8000,))
    report = score.score_run(questions, run, system="silent", corpus_dir=env["dir"], budgets=(8000,))
    assert report["abstention"]["abstention_rate"] == 1.0
    assert report["abstention"]["false_abstention"] == 1.0
    assert "unanswerable" not in report["retrieval"]["by_category"]


def test_an_unanswerable_question_is_never_scored_on_retrieval(env):
    q = question(env, "IN-STAT-0009", "unanswerable", groups=[], unanswerable=True)
    result = score.score_question(q, [span_of(env, "markets-act__handle-3", "section:3")], 8000)
    assert "evidence_recall" not in result and result["abstained"] is False


def test_fact_coverage_counts_a_variant_as_the_fact(env):
    q = one(env, facts=[("thirty days", "30 days"), ("licensing authority",)])
    result = score.score_answer(q, {"text": "It must be decided within 30 days."}, env["corpus"])
    assert result["fact_coverage"] == 0.5


def test_citing_another_copy_of_the_same_act_is_correct(env):
    """The same Act under a second handle is the same provision (design §4)."""
    q = question(env, "IN-STAT-0005", "situational",
                 groups=[("sufficient", [("licensing-act__handle-1", "section:3"),
                                         ("licensing-act-copy__handle-2", "section:3")])],
                 must_cite=[("licensing-act__handle-1", "section:3")])
    answer = {"text": "Thirty days.",
              "citations": [{"document_id": "licensing-act-copy__handle-2", "key": "section:3"}]}
    result = score.score_answer(q, answer, env["corpus"])
    assert result["citation_recall"] == 1.0 and result["citation_precision"] == 1.0


def test_an_invented_quotation_is_not_grounded(env):
    q = one(env)
    real = '"shall decide every application within thirty days"'
    fake = '"shall decide every application within three working days"'
    assert score.score_answer(q, {"text": real}, env["corpus"])["quote_grounding"] == 1.0
    assert score.score_answer(q, {"text": fake}, env["corpus"])["quote_grounding"] == 0.0


def test_a_report_states_what_it_was_measured_against(env):
    q = one(env)
    run = score.run_system(Oracle([q]), [q], (8000,))
    report = score.score_run([q], run, system="oracle", corpus_dir=env["dir"], budgets=(8000,),
                             split="test", tuned_on_test=True)
    assert report["corpus"]["checksums_sha256"] and report["corpus"]["documents"] == 3
    assert report["split"] == "test" and report["tuned_on_test"] is True
    assert report["tau"] == score.DEFAULT_TAU and report["budgets"] == [8000]


# --- Splits -------------------------------------------------------------------------------


def test_the_split_is_thirty_seventy_and_deterministic(env):
    questions = [one(env, qid=f"IN-STAT-{i:04d}", category=c)
                 for i, c in enumerate(["situational"] * 20 + ["definitional"] * 10, start=1)]
    first = splits.assign(questions, seed=7, corpus=env["corpus"])
    again = splits.assign(questions, seed=7, corpus=env["corpus"])
    assert first == again
    assert sum(v == "dev" for v in first.values()) == 9
    other = splits.assign(questions, seed=8, corpus=env["corpus"])
    assert other != first


def test_every_category_appears_in_both_halves(env):
    questions = [one(env, qid=f"IN-STAT-{i:04d}", category=c)
                 for i, c in enumerate(["situational"] * 20 + ["definitional"] * 10, start=1)]
    assignment = splits.assign(questions, seed=7, corpus=env["corpus"])
    for category in ("situational", "definitional"):
        halves = {assignment[q["question_id"]] for q in questions if q["category"] == category}
        assert halves == {"dev", "test"}


def test_scoring_a_split_refuses_a_question_it_has_never_seen(env, tmp_path):
    questions = [one(env, qid=f"IN-STAT-{i:04d}") for i in range(1, 11)]
    path = tmp_path / "split.json"
    splits.write_split(questions[:5], path, seed=1, corpus=env["corpus"])
    with pytest.raises(KeyError):
        splits.select(questions, path, "test")
    assert len(splits.select(questions, None, "all")) == 10
