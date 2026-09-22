"""Questions: the draft workflow, and every rule a question must pass."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmark import authoring
from benchmark.corpus import build_corpus
from benchmark.duplicates import write_duplicates
from benchmark.evidence import build_pool
from benchmark.questions import CHECKLIST, Corpus, validate, validate_set
from tests.processedbuild import make_act, make_page, write_processed

LICENSING = """THE LICENSING AUTHORITY ACT, 2001
1. Short title.—This Act may be called the Licensing Authority Act, 2001, and it shall apply to every district of the territory from the day it is notified.
2. Definitions.—In this Act, unless the context otherwise requires, "authority" means the licensing authority appointed under section 3 of this Act for any district.
3. Authority.—The Government shall appoint a licensing authority for each district, who shall decide every application within thirty days of its receipt."""

LICENSING_TITLE = "The Licensing Authority Act, 2001"


@pytest.fixture
def env(tmp_path):
    data = tmp_path / "data"
    docs = {
        "licensing-act__handle-1": ([make_page(1, LICENSING)], LICENSING_TITLE, "central_acts", "central_act"),
        "licensing-act-copy__handle-2": ([make_page(1, LICENSING)], LICENSING_TITLE, "state_acts", "state_act"),
        "licensing-act-2004__handle-5": ([make_page(1, LICENSING.replace("of its receipt.", "of its receipt, recording its reasons in writing."))],
                                        LICENSING_TITLE, "state_acts", "state_act"),
        "licensing-act-2010__handle-6": ([make_page(1, LICENSING.replace("thirty days", "sixty days"))],
                                        LICENSING_TITLE, "state_acts", "state_act"),
        "bihar-markets__handle-3": (make_act("The Bihar Markets Act, 2003", long=True, topic="markets"),
                                    "The Bihar Markets Act, 2003", "state_acts", "state_act"),
        "assam-markets__handle-4": (make_act("The Assam Markets Act, 2003", long=True, topic="markets"),
                                    "The Assam Markets Act, 2003", "state_acts", "state_act"),
    }
    for did, (pages, title, category, doctype) in docs.items():
        write_processed(data, did, pages, title=title, category=category, document_type=doctype,
                        jurisdiction={"bihar-markets__handle-3": "Bihar",
                                      "assam-markets__handle-4": "Assam"}.get(did, "India"))
    corpus = data / "corpus"
    build_corpus(data, corpus)
    write_duplicates(corpus)
    rows, _ = build_pool(corpus)
    sample = tmp_path / "samples" / "sample-s1-n9.json"
    sample.parent.mkdir()
    sample.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    index = {(r["document_id"], r["key"]): i for i, r in enumerate(rows)}
    return {"corpus": corpus, "sample": sample, "index": index, "qdir": tmp_path / "questions"}


def draft(env, did, key, category):
    q = authoring.new_draft(env["corpus"], env["sample"], env["index"][(did, key)], category,
                            directory=env["qdir"])
    q["provenance"]["drafting_method"] = "human"
    return q


def written(env) -> dict:
    """A correct situational question on section 3 of the Licensing Authority Act."""
    q = draft(env, "licensing-act__handle-1", "section:3", "situational")
    q.update(question="How long does an official deciding permits have to respond once someone applies?",
             gold_answer="The licensing authority must decide an application within thirty days of receiving it.",
             required_facts=[["thirty days", "30 days"]])
    return q


def problems(env, q):
    return validate(q, Corpus(env["corpus"]))


def has(found, fragment):
    return any(fragment in p for p in found), found


# --- Drafts ------------------------------------------------------------------------


def test_a_draft_resolves_its_evidence_and_adds_same_instrument_copies(env):
    q = draft(env, "licensing-act__handle-1", "section:3", "situational")
    locs = q["gold_evidence"][0]["locations"]
    assert [(l["document_id"], l["source"]) for l in locs] == [
        ("licensing-act__handle-1", "sampled"), ("licensing-act-copy__handle-2", "same_instrument_equivalent")]
    assert locs[0]["citation"] == "The Licensing Authority Act, 2001, s. 3"
    # The amended versions of the Act are proposed, not added: their section 3 differs.
    assert {(l["document_id"], l["source"]) for l in q["proposed_alternatives"]} == {
        ("licensing-act-2004__handle-5", "same_instrument_version"),
        ("licensing-act-2010__handle-6", "same_instrument_version")}


def test_another_states_identical_section_is_proposed_not_added(env):
    q = draft(env, "bihar-markets__handle-3", "section:5", "situational")
    assert [l["document_id"] for l in q["gold_evidence"][0]["locations"]] == ["bihar-markets__handle-3"]
    assert [(l["document_id"], l["source"]) for l in q["proposed_alternatives"]] == [
        ("assam-markets__handle-4", "cross_instrument_equivalent")]


def test_a_cross_reference_draft_adds_the_cited_provision_as_a_second_required_group(env):
    q = draft(env, "licensing-act__handle-1", "section:2", "cross_reference")
    assert [g["requirement"] for g in q["gold_evidence"]] == ["required", "required"]
    assert q["gold_evidence"][1]["locations"][0]["key"] == "section:3"


def test_a_blank_draft_is_not_valid(env):
    found = problems(env, draft(env, "licensing-act__handle-1", "section:3", "situational"))
    assert has(found, "question text is empty")[0] and has(found, "required_facts is empty")[0]


def test_ids_count_up(env):
    authoring.save_question(written(env), env["qdir"])
    assert authoring.next_id(env["qdir"]) == "IN-STAT-0002"


# --- The rules -----------------------------------------------------------------------


def test_a_well_written_question_passes(env):
    assert problems(env, written(env)) == []


def test_giving_the_provision_number_is_refused(env):
    q = written(env)
    q["question"] = "Under section 3, how long does an official have to decide a permit application?"
    assert has(problems(env, q), "gives the provision number")[0]


def test_naming_the_act_is_refused(env):
    q = written(env)
    q["question"] = "Under the licensing authority law, how long may a permit decision take?"
    assert has(problems(env, q), "names the Act")[0]


def test_a_provision_lookup_may_name_the_act_and_section(env):
    q = draft(env, "licensing-act__handle-1", "section:3", "provision_lookup")
    q.update(question="What does section 3 of the Licensing Authority Act, 2001 require?",
             gold_answer="A licensing authority for each district, deciding applications within thirty days.",
             required_facts=[["thirty days"]])
    assert problems(env, q) == []


def test_a_question_made_of_the_provisions_own_words_is_refused(env):
    q = written(env)
    q["question"] = "Who shall decide every application within thirty days of its receipt for each district?"
    assert has(problems(env, q), "content words are the provision's own")[0]


def test_a_fact_the_evidence_does_not_contain_is_refused(env):
    q = written(env)
    q["required_facts"] = [["ninety days"]]
    assert has(problems(env, q), "no variant appears in the evidence")[0]


def test_the_provisions_own_number_is_never_a_required_fact(env):
    q = written(env)
    q["required_facts"] = [["thirty days"], ["section 3"]]
    assert has(problems(env, q), "provision's own number")[0]


def test_a_numeric_question_needs_a_quantity_fact(env):
    q = written(env)
    q["category"] = "numeric_threshold"
    q["required_facts"] = [["licensing authority"]]
    assert has(problems(env, q), "quantity with a unit")[0]
    q["required_facts"] = [["thirty days"]]
    assert problems(env, q) == []


def test_a_span_that_no_longer_matches_the_corpus_is_refused(env):
    q = written(env)
    q["gold_evidence"][0]["locations"][0]["char_start"] += 1
    assert has(problems(env, q), "the corpus says")[0]


def test_evidence_that_is_not_in_the_corpus_is_refused_without_crashing(env):
    q = written(env)
    q["gold_evidence"][0]["locations"][0]["key"] = "section:99"
    assert has(problems(env, q), "not a provision in the corpus")[0]


def test_a_group_with_no_gold_eligible_anchor_is_refused(env):
    q = written(env)
    # Keep only the same-instrument copy, and pretend it is in a title conflict.
    q["gold_evidence"][0]["locations"] = q["gold_evidence"][0]["locations"][1:]
    corpus = Corpus(env["corpus"])
    corpus._conflicted = {"licensing-act-copy__handle-2"}
    assert has(validate(q, corpus), "no location is gold-eligible")[0]


def test_a_jurisdictional_question_may_name_its_state_and_must_match_it(env):
    q = draft(env, "bihar-markets__handle-3", "section:5", "jurisdictional")
    q.update(question="In Bihar, how quickly must officials take up a market matter after it arrives?",
             gold_answer="Within fifteen days of receipt.", required_facts=[["15 days"]],
             proposed_alternatives=[])
    assert q["jurisdiction_hint"] == "India" or q["jurisdiction_hint"] == "Bihar"
    q["jurisdiction_hint"] = "Bihar"
    assert problems(env, q) == []
    q["jurisdiction_hint"] = "Assam"
    assert has(problems(env, q), "jurisdiction the question names")[0]


def test_an_unanswerable_question_has_no_evidence(env):
    q = written(env)
    q.update(category="unanswerable", unanswerable=True, gold_evidence=[], must_cite=[], required_facts=[],
             question="What fee does the licensing authority charge to renew a permit in Goa?",
             gold_answer="The corpus does not say.")
    assert problems(env, q) == []
    q["must_cite"] = [{"document_id": "x", "key": "y"}]
    assert has(problems(env, q), "no evidence, citations")[0]


def test_a_rejected_question_needs_only_a_reason(env):
    q = draft(env, "licensing-act__handle-1", "section:3", "situational")
    q["provenance"]["status"] = "rejected"
    assert has(problems(env, q), "must say why")[0]
    q["provenance"]["rejection_reason"] = "section too thin to ask a fair question"
    assert problems(env, q) == []


def test_two_questions_on_one_sampled_provision_are_refused(env):
    a, b = written(env), written(env)
    b["question_id"] = "IN-STAT-0002"
    found = validate_set([a, b], Corpus(env["corpus"]))
    assert has(found["IN-STAT-0002"], "already the sampled evidence")[0]


# --- Verification --------------------------------------------------------------------


def test_verified_without_a_full_checklist_is_refused(env):
    q = written(env)
    q["provenance"].update(status="verified", verified_by="A. Reviewer", verified_at="2026-09-23",
                           verification_method="official_pdf",
                           verification={item: True for item in CHECKLIST[:-1]})
    assert has(problems(env, q), "checklist incomplete")[0]


def test_proposed_alternatives_must_be_decided_before_verifying(env):
    q = draft(env, "bihar-markets__handle-3", "section:5", "situational")
    q.update(question="How quickly must officials take up a market matter after it arrives?",
             gold_answer="Within fifteen days of receipt.", required_facts=[["15 days"]])
    q["provenance"].update(status="verified", verified_by="A. Reviewer", verified_at="2026-09-23",
                           verification_method="official_pdf",
                           verification={item: True for item in CHECKLIST})
    assert has(problems(env, q), "proposed_alternatives must be decided")[0]


def test_exported_verdicts_are_applied_only_when_complete(env, tmp_path):
    good, partial, reject = written(env), written(env), written(env)
    partial["question_id"], reject["question_id"] = "IN-STAT-0002", "IN-STAT-0003"
    for q in (good, partial, reject):
        q["proposed_alternatives"] = []          # decided: the amended versions are left out
        authoring.save_question(q, env["qdir"])
    full = {item: True for item in CHECKLIST}
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(json.dumps({"verifier": "A. Reviewer", "verdicts": {
        "IN-STAT-0001": {"decision": "verified", "checklist": full, "date": "2026-09-23"},
        "IN-STAT-0002": {"decision": "verified", "checklist": {**full, "text_undamaged": False}},
        "IN-STAT-0003": {"decision": "rejected", "checklist": {}, "reason": ""},
    }}), encoding="utf-8")
    assert authoring.apply_verdicts(verdicts, env["qdir"]) == {"verified": 1, "rejected": 0, "skipped": 2}
    stored = {q["question_id"]: q for q in authoring.load_questions(env["qdir"])}
    assert stored["IN-STAT-0001"]["provenance"]["verified_by"] == "A. Reviewer"
    assert problems(env, stored["IN-STAT-0001"]) == []
    assert stored["IN-STAT-0002"]["provenance"]["status"] == "draft"


def test_verdicts_without_a_verifier_are_refused(env, tmp_path):
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(json.dumps({"verifier": " ", "verdicts": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        authoring.apply_verdicts(verdicts, env["qdir"])


def test_the_review_page_links_the_official_pdf_and_escapes_text(env, tmp_path):
    q = written(env)
    q["question"] = "Is <script>alert(1)</script> escaped when a permit decision is late?"
    authoring.save_question(q, env["qdir"])
    page = tmp_path / "review.html"
    assert authoring.build_review_page(env["corpus"], page, env["qdir"]) == 1
    text = page.read_text(encoding="utf-8")
    assert "#page=1" in text and "open official PDF page" in text
    assert "<script>alert(1)" not in text and "&lt;script&gt;" in text
    assert "thirty days of its receipt" in text


# --- Batches -------------------------------------------------------------------------


def batch(env, tmp_path, items, name="batch.json"):
    import shutil
    from benchmark import config
    target = config.SAMPLES_DIR
    spec = tmp_path / name
    spec.write_text(json.dumps({"sample": env["sample"].name, "questions": items}), encoding="utf-8")
    return spec


@pytest.fixture
def samples_dir(env, monkeypatch):
    from benchmark import config
    monkeypatch.setattr(config, "SAMPLES_DIR", env["sample"].parent)
    return env["sample"].parent


def test_a_batch_keeps_a_version_that_states_every_fact_and_drops_one_that_does_not(env, tmp_path, samples_dir):
    spec = batch(env, tmp_path, [{
        "index": env["index"][("licensing-act__handle-1", "section:3")], "category": "situational",
        "question": "How long does an official deciding permits have to respond once someone applies?",
        "gold_answer": "Within thirty days of receiving the application.",
        "required_facts": [["thirty days"]], "answer_type": "numeric"}])
    result = authoring.apply_batch(env["corpus"], spec, drafting_method="language_model", directory=env["qdir"])
    [(qid, found)] = result.items()
    assert found == []
    q = authoring.load_questions(env["qdir"])[0]
    docs = {l["document_id"] for l in q["gold_evidence"][0]["locations"]}
    assert "licensing-act-2004__handle-5" in docs and "licensing-act-2010__handle-6" not in docs
    reasons = {d["document_id"]: d["reason"] for d in q["provenance"]["alternatives_decided"]}
    assert reasons["licensing-act-2010__handle-6"] == "does not state every required fact"
    assert q["provenance"]["drafting_method"] == "language_model"


def test_re_applying_a_batch_updates_rather_than_duplicates(env, tmp_path, samples_dir):
    item = {"index": env["index"][("licensing-act__handle-1", "section:3")], "category": "situational",
            "question": "How long does an official deciding permits have to respond once someone applies?",
            "gold_answer": "Within thirty days.", "required_facts": [["thirty days"]]}
    authoring.apply_batch(env["corpus"], batch(env, tmp_path, [item]), drafting_method="human", directory=env["qdir"])
    item["gold_answer"] = "Within thirty days of receipt."
    authoring.apply_batch(env["corpus"], batch(env, tmp_path, [item]), drafting_method="human", directory=env["qdir"])
    stored = authoring.load_questions(env["qdir"])
    assert len(stored) == 1 and stored[0]["gold_answer"] == "Within thirty days of receipt."


def test_a_jurisdictional_batch_never_accepts_another_states_copy(env, tmp_path, samples_dir):
    spec = batch(env, tmp_path, [{
        "index": env["index"][("bihar-markets__handle-3", "section:5")], "category": "jurisdictional",
        "question": "In Bihar, how quickly must officials take up a market matter after it arrives?",
        "gold_answer": "Within fifteen days of receipt.", "required_facts": [["15 days"]],
        "jurisdiction_hint": "Bihar"}])
    authoring.apply_batch(env["corpus"], spec, drafting_method="human", directory=env["qdir"])
    q = authoring.load_questions(env["qdir"])[0]
    assert {l["document_id"] for l in q["gold_evidence"][0]["locations"]} == {"bihar-markets__handle-3"}


def test_an_unanswerable_batch_item_needs_no_sample_row(env, tmp_path, samples_dir):
    spec = batch(env, tmp_path, [{"ref": "absent-renewal-fee", "category": "unanswerable",
                                  "question": "What fee does a licensing authority charge to renew a permit in Goa?",
                                  "gold_answer": "The corpus does not say."}])
    result = authoring.apply_batch(env["corpus"], spec, drafting_method="human", directory=env["qdir"])
    assert list(result.values()) == [[]]


def test_a_model_verification_records_its_method(env, tmp_path, samples_dir):
    q = written(env)
    q["proposed_alternatives"] = []
    authoring.save_question(q, env["qdir"])
    verdicts = tmp_path / "verdicts.json"
    verdicts.write_text(json.dumps({"verifier": "language-model", "method": "text_review", "verdicts": {
        q["question_id"]: {"decision": "verified", "checklist": {i: True for i in CHECKLIST}}}}), encoding="utf-8")
    authoring.apply_verdicts(verdicts, env["qdir"])
    stored = authoring.load_questions(env["qdir"])[0]
    assert stored["provenance"]["verification_method"] == "text_review"
    assert problems(env, stored) == []


def test_a_short_cited_provision_is_valid_second_hop_evidence(env):
    """Section 1 of the fixture is under 150 characters' worth of question, but a
    cited provision needs no length floor — only the sampled one does."""
    q = draft(env, "licensing-act__handle-1", "section:2", "cross_reference")
    q.update(question="Which official decides applications, and how long do they have?",
             gold_answer="The licensing authority appointed for the district; thirty days.",
             required_facts=[["licensing authority"], ["thirty days"]], proposed_alternatives=[])
    from benchmark import evidence
    real = evidence.exclusion
    try:
        evidence_calls = []
        import benchmark.questions as bq
        bq.exclusion = lambda p, m, c, t: ("too_short" if p["key"] == "section:3" else real(p, m, c, t))
        assert problems(env, q) == []
        bq.exclusion = lambda p, m, c, t: ("too_long" if p["key"] == "section:3" else real(p, m, c, t))
        assert has(problems(env, q), "group g2: no location is gold-eligible")[0]
    finally:
        bq.exclusion = real


def test_re_categorising_a_batch_item_rebuilds_its_evidence_and_keeps_its_id(env, tmp_path, samples_dir):
    idx = env["index"][("licensing-act__handle-1", "section:2")]
    base = {"index": idx, "question": "Which official decides applications, and how long do they have?",
            "gold_answer": "The licensing authority; thirty days.",
            "required_facts": [["licensing authority"], ["thirty days"]]}
    authoring.apply_batch(env["corpus"], batch(env, tmp_path, [{**base, "category": "cross_reference"}]),
                          drafting_method="human", directory=env["qdir"])
    authoring.apply_batch(env["corpus"], batch(env, tmp_path, [{
        **base, "category": "situational", "required_facts": [["licensing authority"]],
        "question": "Which official is meant by the licensing body in this law?"}]),
        drafting_method="human", directory=env["qdir"])
    [q] = authoring.load_questions(env["qdir"])
    assert q["question_id"] == "IN-STAT-0001" and q["category"] == "situational"
    assert len(q["gold_evidence"]) == 1 and q["gold_evidence"][0]["requirement"] == "sufficient"


def test_an_undeclared_drafting_method_is_refused(env):
    q = written(env)
    q["provenance"]["drafting_method"] = None
    assert has(problems(env, q), "drafting_method")[0]
