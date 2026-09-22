"""The shipped baselines, on a real FTS5 index built over a miniature corpus."""

from __future__ import annotations

import pytest

from benchmark import baselines, score
from benchmark.baselines.bm25 import (Bm25Index, Bm25TwoStage, Bm25Windows, _query_string,
                                      build_index, windows_of)
from benchmark.corpus import build_corpus
from tests.processedbuild import make_page, write_processed

LICENSING = """THE LICENSING AUTHORITY ACT, 2001
1. Short title.—This Act may be called the Licensing Authority Act, 2001.
2. Definitions.—In this Act "authority" means the licensing authority appointed under section 3.
3. Authority.—The Government shall appoint a licensing authority for each district, who shall decide every application within thirty days of its receipt."""

MARKETS = """THE MARKETS ACT, 2003
1. Short title.—This Act may be called the Markets Act, 2003.
2. Market committee.—The State Government shall constitute a market committee for every market area.
3. Fees.—A market committee may levy a fee on the value of the produce sold in the market area."""


@pytest.fixture
def indexed(tmp_path):
    data = tmp_path / "data"
    write_processed(data, "licensing-act__handle-1", [make_page(1, LICENSING)],
                    title="The Licensing Authority Act, 2001")
    write_processed(data, "markets-act__handle-3", [make_page(1, MARKETS)],
                    title="The Markets Act, 2003", category="state_acts",
                    document_type="state_act", jurisdiction="Bihar")
    corpus_dir = data / "corpus"
    build_corpus(data, corpus_dir)
    path = tmp_path / "bm25.sqlite"
    report = build_index(corpus_dir, path, window=120, stride=90)
    return {"dir": corpus_dir, "path": path, "report": report}


def test_windows_cover_the_whole_text_with_overlap():
    text = "x" * 1000
    pieces = list(windows_of(text, window=400, stride=300))
    assert pieces[0] == (0, 400) and pieces[-1][1] == 1000
    assert all(b - a <= 400 for a, b in pieces)
    assert {c for _, c in pieces} and pieces[1][0] < pieces[0][1]        # they overlap


def test_the_query_keeps_content_words_in_order_without_repeats():
    assert _query_string("The authority of the authority shall decide") == '"authority" OR "decide"'
    assert _query_string("of the and") == ""


def test_the_index_records_its_own_geometry(indexed):
    assert indexed["report"]["documents"] == 2
    assert indexed["report"]["windows"] > 2
    assert indexed["report"]["window_chars"] == 120 and indexed["report"]["stride_chars"] == 90
    index = Bm25Index(indexed["path"])
    assert index.meta["window_chars"] == 120
    index.close()


def test_bm25_finds_the_act_the_question_is_about(indexed):
    system = Bm25Windows(Bm25Index(indexed["path"]))
    spans = system.retrieve("Who appoints the licensing authority for a district?", 2000)
    assert spans and spans[0][0] == "licensing-act__handle-1"
    system.index.close()


def test_two_stage_searches_only_the_documents_it_resolved(indexed):
    system = Bm25TwoStage(Bm25Index(indexed["path"]), documents=1)
    spans = system.retrieve("What fee may a market committee levy on produce?", 2000)
    assert spans and {s[0] for s in spans} == {"markets-act__handle-3"}
    system.index.close()


def test_a_baseline_stops_once_the_budget_is_covered(indexed):
    system = Bm25Windows(Bm25Index(indexed["path"]))
    small = system.retrieve("licensing authority district application", 200)
    large = system.retrieve("licensing authority district application", 2000)
    assert sum(e - s for _, s, e in small) < sum(e - s for _, s, e in large)
    assert score.within_budget(small, 200).used_chars <= 200
    system.index.close()


def test_a_question_of_only_stopwords_returns_nothing(indexed):
    system = Bm25Windows(Bm25Index(indexed["path"]))
    assert system.retrieve("of the and", 2000) == []
    system.index.close()


def test_loading_a_baseline_by_name(indexed):
    system = baselines.load("bm25-two-stage", corpus_dir=indexed["dir"], index_path=indexed["path"])
    assert system.name == "bm25-two-stage"
    system.index.close()
    with pytest.raises(ValueError):
        baselines.load("dense", corpus_dir=indexed["dir"])
    with pytest.raises(ValueError):
        baselines.load("bm25-windows", corpus_dir=indexed["dir"])       # no index


def test_a_missing_index_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m benchmark index"):
        Bm25Index(tmp_path / "absent.sqlite")
