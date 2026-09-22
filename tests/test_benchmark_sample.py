"""The gold evidence pool and the sampler drawn from it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark import config
from benchmark.corpus import build_corpus
from benchmark.duplicates import write_duplicates
from benchmark.evidence import build_pool, exclusion
from benchmark.sample import draw, load_pool, verify_sample, write_pool, write_sample
from tests.processedbuild import make_act, make_page, rewrite_document, write_processed

GOOD_META = {"document_id": "a", "quality": {"ocr_text_source": "born_digital",
                                             "structure_confidence": "structured"}}
BODY = "x" * 400


def provision(**kw):
    p = {"evidence_confidence": "high", "key_ambiguous": False, "char_start": 0, "char_end": 400}
    p.update(kw)
    return p


# --- The eligibility rules, one at a time ------------------------------------------


def test_a_clean_provision_is_eligible():
    assert exclusion(provision(), GOOD_META, set(), BODY) is None


@pytest.mark.parametrize("change, reason", [
    ({"evidence_confidence": "medium"}, "tier"),
    ({"key_ambiguous": True}, "ambiguous"),
    ({"char_end": 100}, "too_short"),
    ({"char_end": 7000}, "too_long"),
])
def test_each_provision_rule_excludes_with_its_reason(change, reason):
    assert exclusion(provision(**change), GOOD_META, set(), BODY) == reason


def test_a_document_in_a_title_conflict_supplies_no_gold():
    assert exclusion(provision(), GOOD_META, {"a"}, BODY) == "title_conflict"


def test_a_scanned_pdf_with_its_own_text_layer_supplies_no_gold():
    meta = {**GOOD_META, "quality": {**GOOD_META["quality"], "ocr_text_source": "scan_with_good_ocr"}}
    assert exclusion(provision(), meta, set(), BODY) == "text_source"


def test_a_partially_structured_document_supplies_no_gold():
    meta = {**GOOD_META, "quality": {**GOOD_META["quality"], "structure_confidence": "partial"}}
    assert exclusion(provision(), meta, set(), BODY) == "structure"


def test_an_editorial_omission_is_a_stub_even_with_a_long_heading():
    body = ("57. Act not to prevent use of Aadhaar number for other purposes under law.—"
            "Omitted by the Aadhaar and Other Laws (Amendment) Act 2019 (Act 14 of 2019), "
            "s. 25 (w.e.f. 25-07-2019).")
    assert exclusion(provision(char_end=len(body)), GOOD_META, set(), body) == "stub"


def test_a_real_repeal_section_is_law_not_a_stub():
    body = ("45. Repeal and savings.—(1) The Abhilashi University Ordinance, 2014 is hereby "
            "repealed. (2) Notwithstanding such repeal, anything done or any action taken under "
            "the said Ordinance shall be deemed to have been done or taken under this Act.")
    assert exclusion(provision(char_end=len(body)), GOOD_META, set(), body) is None


def test_an_operative_amendment_that_omits_words_is_law_not_a_stub():
    body = ("2. Amendment of section 4.—In section 4 of the principal Act, the words "
            "\"or any other officer\" shall be omitted, and after clause (b) the following "
            "clause shall be inserted, namely: (c) the Collector of the district.")
    assert exclusion(provision(char_end=len(body)), GOOD_META, set(), body) is None


# --- The pool over a real miniature corpus ----------------------------------------


DEFS = """THE DEFINED ACT, 2001
1. Short title.—This Act may be called the Defined Act, 2001, and it shall apply to every district of the territory from the day it is notified.
2. Definitions.—In this Act, unless the context otherwise requires, "authority" means the licensing authority appointed under section 3 of this Act for any district.
3. Authority.—The Government shall appoint a licensing authority for each district, who shall decide every application within thirty days of its receipt."""


def make_data(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    docs = {
        # A Central Act and a state-collection copy of it: one instrument.
        "central-act__handle-1": ([*make_act("The Central Act, 1999", long=True)], "The Central Act, 1999", "central_acts"),
        "central-act-copy__handle-2": ([*make_act("The Central Act, 1999", long=True)], "The Central Act, 1999", "state_acts"),
        "defined-act__handle-3": ([make_page(1, DEFS)], "The Defined Act, 2001", "state_acts"),
        "scanned-act__handle-4": ([*make_act("The Scanned Act, 2002", long=True, topic="cargo")],
                                  "The Scanned Act, 2002", "state_acts"),
    }
    # Six unrelated sets of rules: distinct subjects and titles, so none is a
    # copy of another and each is its own instrument.
    for n, topic in zip(range(5, 11), ("bridges", "ferries", "markets", "wells", "forests", "canals")):
        docs[f"{topic}-rules__rule-{n}"] = ([*make_act(f"The {topic.title()} Rules, 2005", sections=4,
                                                       long=True, topic=topic)],
                                            f"The {topic.title()} Rules, 2005", "rules")
    for did, (pages, title, category) in docs.items():
        write_processed(data, did, pages, title=title, category=category,
                        document_type={"central_acts": "central_act", "state_acts": "state_act",
                                       "rules": "rule"}[category])
    rewrite_document(data, "scanned-act__handle-4",
                     lambda d: d["ocr_decision"].update(text_source="scan_with_good_ocr"))
    corpus = data / "corpus"
    build_corpus(data, corpus)
    write_duplicates(corpus)
    return data, corpus


def test_the_pool_counts_every_exclusion_and_tags_what_it_keeps(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, report = build_pool(corpus)
    assert report["excluded"]["text_source"] == 20
    assert not any(r["document_id"] == "scanned-act__handle-4" for r in rows)
    assert report["pool"] + sum(report["excluded"].values()) == report["provisions"]
    by_key = {(r["document_id"], r["key"]): r for r in rows}
    assert "definitional" in by_key[("defined-act__handle-3", "section:2")]["tags"]
    assert by_key[("defined-act__handle-3", "section:2")]["references"] == ["section:3"]
    assert "numeric" in by_key[("central-act__handle-1", "section:1")]["tags"]


def test_difficulty_signals_count_what_they_claim(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    row = next(r for r in rows if r["document_id"] == "central-act__handle-1" and r["key"] == "section:1")
    assert row["signals"]["cluster_size"] == 2
    assert row["signals"]["equivalents"] == 1
    assert row["signals"]["number_ambiguity"] >= 9      # every miniature Act has a section 1


# --- The draw ---------------------------------------------------------------------


def test_the_same_seed_draws_the_same_sample_and_another_seed_does_not(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    first, _ = draw(rows, seed=1, size=8)
    assert draw(rows, seed=1, size=8)[0] == first
    assert [r["key"] for r in draw(rows, seed=2, size=8)[0]] != [r["key"] for r in first] or len(first) < 2


def test_an_instrument_and_its_copies_are_drawn_at_most_once_per_round(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    drawn, report = draw(rows, seed=7, size=6,
                         allocation={"central_acts": 0.5, "state_acts": 0.5, "rules": 0.0, "regulations": 0.0})
    # The copy lives in the central stratum only: the cluster has one home.
    assert not any(r["document_id"] == "central-act-copy__handle-2" and r["stratum"] == "state_acts"
                   for r in drawn)
    assert report["state_acts"]["instruments_available"] == 1


def test_word_for_word_identical_provisions_are_never_both_drawn(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    drawn, _ = draw(rows, seed=3, size=40,
                    allocation={"central_acts": 1.0, "state_acts": 0.0, "rules": 0.0, "regulations": 0.0})
    classes = [r["equivalence_class"] for r in drawn if r["equivalence_class"]]
    assert len(classes) == len(set(classes))


def test_targets_always_sum_to_the_requested_size(tmp_path):
    from benchmark.sample import _targets
    for size in (1, 7, 10, 999, 1000):
        assert sum(_targets(size, config.DEFAULT_ALLOCATION).values()) == size


def test_an_exhausted_category_says_so_rather_than_padding(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    drawn, report = draw(rows, seed=5, size=100,
                         allocation={"central_acts": 0.0, "state_acts": 0.0, "rules": 1.0, "regulations": 0.0})
    assert report["rules"]["drawn"] == len(drawn) < 100
    assert report["rules"]["rounds"] > 1


def test_an_allocation_that_does_not_sum_to_one_is_refused(tmp_path):
    with pytest.raises(ValueError):
        draw([], seed=1, size=10, allocation={"central_acts": 0.5})


def test_a_tagged_top_up_draws_only_that_tag_and_never_overlaps_the_base(tmp_path):
    _, corpus = make_data(tmp_path)
    rows, _ = build_pool(corpus)
    base, _ = draw(rows, seed=1, size=3,
                   allocation={"central_acts": 0.0, "state_acts": 0.0, "rules": 1.0, "regulations": 0.0})
    top, report = draw(rows, seed=2, size=50, tag="numeric", exclude=base, allocation="proportional")
    assert top and all("numeric" in r["tags"] for r in top)
    base_instruments = {r["cluster_id"] or r["document_id"] for r in base}
    assert not base_instruments & {r["cluster_id"] or r["document_id"] for r in top}
    assert sum(v["target"] for v in report.values()) == 50


# --- Written samples ---------------------------------------------------------------


def test_a_written_sample_is_deterministic_and_verifies(tmp_path):
    data, corpus = make_data(tmp_path)
    build = data / "benchmark_build"
    write_pool(corpus, build)
    path = write_sample(corpus, build, seed=11, size=6, out_dir=tmp_path / "samples")
    first = path.read_bytes()
    assert write_sample(corpus, build, seed=11, size=6, out_dir=tmp_path / "samples").read_bytes() == first
    sample = json.loads(first)
    assert sample["seed"] == 11 and sample["rows"]
    assert all(r["tags"] is not None and r["sample_index"] == i for i, r in enumerate(sample["rows"]))
    assert verify_sample(path, corpus) == []


def test_verify_catches_a_sample_read_against_a_changed_corpus(tmp_path):
    data, corpus = make_data(tmp_path)
    build = data / "benchmark_build"
    write_pool(corpus, build)
    path = write_sample(corpus, build, seed=11, size=6, out_dir=tmp_path / "samples")
    row = json.loads(path.read_text(encoding="utf-8"))["rows"][0]
    text_path = corpus / "text" / f"{row['document_id']}.txt"
    text = text_path.read_text(encoding="utf-8")
    i = row["char_start"] + text[row["char_start"]:row["char_end"]].index("authority")
    text_path.write_bytes((text[:i] + "committee" + text[i + len("authority"):]).encode("utf-8"))
    problems = verify_sample(path, corpus)
    assert any("span no longer reproduces" in p for p in problems)
    assert any("document text changed" in p for p in problems)


def test_a_stale_pool_is_refused(tmp_path):
    data, corpus = make_data(tmp_path)
    build = data / "benchmark_build"
    write_pool(corpus, build)
    (corpus / config.CHECKSUMS_FILENAME).write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pool(build, corpus)
