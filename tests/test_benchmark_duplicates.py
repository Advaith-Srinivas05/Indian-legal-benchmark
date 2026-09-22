"""Duplicate detection: copies cluster, parallel laws do not, and provisions match exactly."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark import config
from benchmark.corpus import build_corpus, verify_corpus
from benchmark.duplicates import (classify, find_duplicates, normalise_provision,
                                  title_similarity, write_duplicates)
from tests.processedbuild import make_act, make_page, write_processed

TITLE = "The Sample Act, 1999"


def corpus(tmp_path: Path, docs: dict) -> Path:
    """Build a corpus from ``{document_id: (pages, title, pdf_sha256)}``."""
    data = tmp_path / "data"
    for did, (pages, title, pdf) in docs.items():
        write_processed(data, did, pages, title=title, pdf_sha256=pdf)
    build_corpus(data, data / "corpus")
    return data / "corpus"


def edge(payload: dict, a: str, b: str):
    a, b = sorted((a, b))
    for e in [e for c in payload["clusters"] for e in c["edges"]] + payload["parallel"] + payload["contains"]:
        if (e["a"], e["b"]) == (a, b):
            return e
    return None


def cluster_of(payload: dict, did: str):
    return next((c["cluster_id"] for c in payload["clusters"]
                 if did in {d["document_id"] for d in c["documents"]}), None)


def test_a_rehosted_copy_is_the_same_instrument(tmp_path):
    out = corpus(tmp_path, {"central__handle-1": (make_act(TITLE), TITLE, None),
                            "ut-copy__handle-2": (make_act(TITLE), TITLE, None)})
    payload, _ = find_duplicates(out)
    assert edge(payload, "central__handle-1", "ut-copy__handle-2")["relation"] == "identical"
    assert cluster_of(payload, "central__handle-1") == cluster_of(payload, "ut-copy__handle-2")


def test_an_amended_version_with_the_same_title_joins_the_cluster(tmp_path):
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": (make_act(TITLE, changed={3, 7, 11, 15}), TITLE, None)})
    payload, _ = find_duplicates(out)
    e = edge(payload, "a__handle-1", "b__handle-2")
    assert e["relation"] == "same_instrument" and 0.5 <= e["jaccard"] < 0.95


def test_a_parallel_law_from_another_state_is_never_clustered(tmp_path):
    """Bihar's GST Act and Uttar Pradesh's share most wording and are different law."""
    out = corpus(tmp_path, {
        "bihar__handle-1": (make_act("The Bihar Goods and Services Tax Act, 2017"),
                            "The Bihar Goods and Services Tax Act, 2017", None),
        "up__handle-2": (make_act("The Uttar Pradesh Goods and Services Tax Act, 2017", changed={2, 9, 14}),
                         "The Uttar Pradesh Goods and Services Tax Act, 2017", None)})
    payload, _ = find_duplicates(out)
    assert edge(payload, "bihar__handle-1", "up__handle-2")["relation"] == "parallel"
    assert cluster_of(payload, "bihar__handle-1") is None


def test_near_identical_text_clusters_even_under_a_garbled_title(tmp_path):
    """India Code's titles are noisy: 'MP IRRGATION RULES 1974'."""
    out = corpus(tmp_path, {"a__rule-1": (make_act(TITLE, sections=40), "M.P. Irrigation Rules-1974", None),
                            "b__rule-2": (make_act(TITLE, sections=40, changed={5}), "MP IRRGATION RULES 1974", None)})
    payload, _ = find_duplicates(out)
    e = edge(payload, "a__rule-1", "b__rule-2")
    assert e["relation"] == "same_instrument" and e["jaccard"] >= 0.95 and e["title_similarity"] < 0.5


def test_the_same_pdf_bytes_are_identical_even_when_the_text_differs(tmp_path):
    """One PDF processed before and after OCR existed gives two slightly different texts."""
    out = corpus(tmp_path, {"early__handle-1": (make_act(TITLE), TITLE, "f" * 64),
                            "late__handle-2": (make_act(TITLE, changed={4}), TITLE, "f" * 64)})
    payload, _ = find_duplicates(out)
    assert edge(payload, "early__handle-1", "late__handle-2")["relation"] == "identical"


def test_empty_documents_are_not_copies_of_each_other(tmp_path):
    out = corpus(tmp_path, {"empty-a__handle-1": ([make_page(1, "")], TITLE, None),
                            "empty-b__handle-2": ([make_page(1, "")], TITLE, None)})
    payload, _ = find_duplicates(out)
    assert payload["clusters"] == []


def test_unrelated_laws_have_no_edge(tmp_path):
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": ([make_page(1, "THE OTHER ACT\n1. Other.—Nothing in common at all here today.")],
                                            "The Other Act", None)})
    payload, _ = find_duplicates(out)
    assert edge(payload, "a__handle-1", "b__handle-2") is None


def test_an_unchanged_section_in_an_amended_version_is_an_equivalent_provision(tmp_path):
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": (make_act(TITLE, changed={3}), TITLE, None)})
    _, classes = find_duplicates(out)
    members = {frozenset((m["document_id"], m["key"]) for m in c["members"]) for c in classes}
    assert frozenset({("a__handle-1", "section:1"), ("b__handle-2", "section:1")}) in members
    changed = [c for c in classes if any(m["key"] == "section:3" for m in c["members"])]
    assert changed == [], "a section whose text differs must not be called equivalent"
    assert all(c["same_instrument"] for c in classes)


def test_an_identical_provision_in_a_parallel_law_is_marked_cross_instrument(tmp_path):
    out = corpus(tmp_path, {
        "bihar__handle-1": (make_act("The Bihar Tax Act, 2017"), "The Bihar Tax Act, 2017", None),
        "up__handle-2": (make_act("The Uttar Pradesh Tax Act, 2017", changed={2, 9, 14}),
                         "The Uttar Pradesh Tax Act, 2017", None)})
    _, classes = find_duplicates(out)
    assert classes and not any(c["same_instrument"] for c in classes)


def test_normalisation_ignores_wrapping_but_not_numbers():
    assert normalise_provision("within 30\n  days") == normalise_provision("within 30 days")
    assert normalise_provision("within 30 days") != normalise_provision("within 60 days")


def test_title_similarity_ignores_case_punctuation_and_years():
    assert title_similarity("ESSENTIAL COMMODITIES ACT, 1955", "Essential commodities act 1955") == 1.0
    assert title_similarity("The Bihar GST Act", "The Uttar Pradesh GST Act") < 0.8


def test_classification_boundaries():
    assert classify(0.2, 0.0, 0.2, same_file=True) == "identical"
    assert classify(1.0, 0.0, 1.0, same_text=True) == "identical"
    assert classify(0.95, 0.0, 0.95) == "same_instrument"
    assert classify(0.6, 0.9, 0.6) == "same_instrument"
    assert classify(0.6, 0.5, 0.6) == "parallel"
    assert classify(0.3, 0.1, 0.95) == "contains"
    assert classify(0.3, 0.1, 0.3) == "unrelated"


def test_different_places_block_every_merge_except_the_same_file():
    assert classify(0.96, 0.75, 0.96, different_places=True) == "parallel"
    assert classify(1.0, 0.1, 1.0, same_text=True, different_places=True) == "parallel"
    assert classify(1.0, 0.1, 1.0, same_file=True, different_places=True) == "identical"


def test_two_territories_street_vendor_schemes_are_two_instruments(tmp_path):
    """Chandigarh's and Daman and Diu's schemes share 96 % of their lines."""
    out = corpus(tmp_path, {
        "chd__regulation-1": (make_act(TITLE, sections=40),
                              "Union Territory of Chandigarh Street Vendors Scheme", None),
        "dd__regulation-2": (make_act(TITLE, sections=40, changed={5}),
                             "Union Territory of Daman and Diu Street Vendors Scheme", None)})
    payload, _ = find_duplicates(out)
    e = edge(payload, "chd__regulation-1", "dd__regulation-2")
    assert e["jaccard"] >= 0.95 and e["places_differ"] and e["relation"] == "parallel"
    assert payload["clusters"] == []


def test_a_law_adopted_verbatim_by_another_state_is_parallel_but_its_provisions_stay_linked(tmp_path):
    """Chhattisgarh adopted Madhya Pradesh's Acts word for word in 2000."""
    out = corpus(tmp_path, {
        "cg__handle-1": (make_act(TITLE), "Chhattisgarh Panchayats (Recovery) Act, 1976", None),
        "mp__handle-2": (make_act(TITLE), "The Madhaya Pradesh Panchayats (Recovery) Act, 1976", None)})
    payload, classes = find_duplicates(out)
    e = edge(payload, "cg__handle-1", "mp__handle-2")
    assert e["same_text"] and e["relation"] == "parallel"
    assert classes and not any(c["same_instrument"] for c in classes)


def test_a_typo_in_a_title_does_not_split_one_act_in_two(tmp_path):
    out = corpus(tmp_path, {
        "a__handle-1": (make_act(TITLE), "PROTECTION OF HUMAN RIGHTS ACT, 1993", None),
        "b__handle-2": (make_act(TITLE, changed={2, 6, 9, 13}), "PROTECTION OF HUMEN RIGHTS ACT, 1993", None)})
    payload, _ = find_duplicates(out)
    assert edge(payload, "a__handle-1", "b__handle-2")["relation"] == "same_instrument"


def test_identical_text_under_disagreeing_titles_is_listed_for_review(tmp_path):
    """Rajasthan's "registration act, 1908" holds the Unlawful Activities Act."""
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), "registration act, 1908", None),
                            "b__handle-2": (make_act(TITLE), "unlawful activities (prevention) act, 1967", None)})
    payload, _ = find_duplicates(out)
    assert [(e["a"], e["b"]) for e in payload["title_conflicts"]] == [("a__handle-1", "b__handle-2")]
    assert payload["summary"]["title_conflicts"] == 1


def test_places_are_found_by_name_old_name_and_abbreviation():
    from benchmark.duplicates import places_differ, places_in
    assert places_in("M.P. Irrigation Rules-1974") == {"madhya pradesh"}
    assert places_in("The Orissa Excise Act") == {"odisha"}
    assert places_in("Goa, Daman & Diu Agricultural Tenancy Act") == {"goa", "daman diu"}
    assert places_in("Explosive Substances Act, 1908") == set()
    assert not places_differ("Goa, Daman and Diu Tenancy Act", "Daman and Diu Tenancy Act")
    assert not places_differ("Explosive Substances Act", "Chandigarh Explosive Substances Act")
    assert places_differ("Bihar Tax Act", "Uttar Pradesh Tax Act")


def test_duplicate_output_is_deterministic_and_checksummed(tmp_path):
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": (make_act(TITLE), TITLE, None),
                            "c__handle-3": (make_act(TITLE, changed={1, 2}), TITLE, None)})
    write_duplicates(out)
    first = [(out / n).read_bytes() for n in (config.DUPLICATES_FILENAME, config.PROVISION_EQUIVALENTS_FILENAME)]
    write_duplicates(out)
    assert first == [(out / n).read_bytes() for n in (config.DUPLICATES_FILENAME, config.PROVISION_EQUIVALENTS_FILENAME)]
    listed = (out / config.CHECKSUMS_FILENAME).read_text(encoding="utf-8")
    assert config.DUPLICATES_FILENAME in listed and config.PROVISION_EQUIVALENTS_FILENAME in listed
    assert verify_corpus(out)["problems"] == []


def test_the_review_sample_is_stratified_blank_and_spreadsheet_safe(tmp_path):
    import csv
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": (make_act(TITLE), "=HYPERLINK(evil)", None),
                            "c__handle-3": (make_act(TITLE, changed={1, 2}), TITLE, None)})
    write_duplicates(out)
    from benchmark.duplicates import write_review_sample
    sheet = tmp_path / "review.csv"
    assert write_review_sample(out, sheet, per_band=5) > 0
    assert sheet.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(sheet.open(encoding="utf-8-sig")))
    assert {r["verdict"] for r in rows} == {""}
    assert all(not r["b_title"].startswith("=") for r in rows)
    first = sheet.read_bytes()
    write_review_sample(out, sheet, per_band=5)
    assert sheet.read_bytes() == first


def test_a_rebuild_removes_stale_duplicate_findings(tmp_path):
    out = corpus(tmp_path, {"a__handle-1": (make_act(TITLE), TITLE, None),
                            "b__handle-2": (make_act(TITLE), TITLE, None)})
    write_duplicates(out)
    build_corpus(out.parent, out, force=True)
    assert not (out / config.DUPLICATES_FILENAME).exists()
    assert config.DUPLICATES_FILENAME not in (out / config.CHECKSUMS_FILENAME).read_text(encoding="utf-8")
