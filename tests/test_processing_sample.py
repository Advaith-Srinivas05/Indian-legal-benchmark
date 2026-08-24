"""Tests for deterministic, representative benchmark sampling.

Two properties are load-bearing and both are tested directly: the same corpus
must always yield the same sample (otherwise no two benchmark runs are
comparable), and the sample must actually spread across categories, eras, sizes
and jurisdictions (otherwise it measures one corner of the corpus and calls it
the whole).
"""

from __future__ import annotations

import pytest

from processing import sample
from processing.errors import SamplingError
from processing.models import CorpusDocument


def document(document_id: str, *, category="state_acts", year=1999,
             size=200_000, jurisdiction="Assam") -> CorpusDocument:
    return CorpusDocument(
        document_id=document_id,
        category=category,
        document_type=category.rstrip("s"),
        title=document_id,
        pdf_path=None,          # sampling never opens a PDF
        pdf_relpath=f"raw/{document_id}.pdf",
        sha256="0" * 64,
        bytes=size,
        jurisdiction=jurisdiction,
        year=year,
    )


def big_corpus() -> list[CorpusDocument]:
    """A corpus spanning every category, era, size band and several states."""
    categories = ["central_acts", "state_acts", "rules", "regulations"]
    years = [1850, 1920, 1975, 2005, 2020, None]
    sizes = [50_000, 300_000, 1_000_000, 4_000_000, 20_000_000]
    states = ["Assam", "Kerala", "Goa", "Punjab", "Bihar", "India"]
    documents = []
    for category in categories:
        for year in years:
            for size in sizes:
                for state in states:
                    documents.append(document(
                        f"{category}-{year}-{size}-{state}",
                        category=category, year=year, size=size,
                        jurisdiction=state,
                    ))
    return documents


class TestBands:
    def test_era_bands(self):
        assert sample.era_band(1850) == "pre_1900"
        assert sample.era_band(1920) == "1900_1949"
        assert sample.era_band(1975) == "1950_1990"
        assert sample.era_band(2005) == "1991_2010"
        assert sample.era_band(2020) == "2011_plus"
        assert sample.era_band(None) == "year_unknown"

    def test_size_bands(self):
        assert sample.size_band(1_000) == "tiny"
        assert sample.size_band(200_000) == "small"
        assert sample.size_band(1_000_000) == "medium"
        assert sample.size_band(4_000_000) == "large"
        assert sample.size_band(30_000_000) == "huge"

    def test_band_edges_land_in_the_higher_band(self):
        assert sample.size_band(100 * 1024) == "small"
        assert sample.size_band(8 * 1024 ** 2) == "huge"


class TestDeterminism:
    def test_the_same_corpus_yields_the_same_sample(self):
        corpus = big_corpus()
        first = [r["document_id"] for r in sample.select(corpus, target=100)]
        second = [r["document_id"] for r in sample.select(corpus, target=100)]
        assert first == second

    def test_the_sample_does_not_depend_on_corpus_ordering(self):
        corpus = big_corpus()
        forward = [r["document_id"] for r in sample.select(corpus, target=100)]
        backward = [r["document_id"] for r in sample.select(
            list(reversed(corpus)), target=100)]
        assert forward == backward

    def test_selection_key_is_recorded_for_audit(self):
        records = sample.select(big_corpus(), target=20)
        for record in records:
            assert record["selection_key"] == \
                sample.document_rank(record["document_id"])[:16]
            assert record["selection_reason"]


class TestRepresentativeness:
    @pytest.fixture()
    def records(self):
        return sample.select(big_corpus(), target=100)

    def test_the_target_size_is_reached(self, records):
        assert len(records) == 100

    def test_category_quotas_are_honoured(self, records):
        counts = sample.describe(records)["by_category"]
        assert counts == {"central_acts": 25, "state_acts": 30,
                          "rules": 25, "regulations": 20}

    def test_every_era_band_present_in_the_corpus_is_sampled(self, records):
        eras = set(sample.describe(records)["by_era"])
        assert {"pre_1900", "1900_1949", "1950_1990", "1991_2010",
                "2011_plus"} <= eras

    def test_every_size_band_is_sampled(self, records):
        bands = set(sample.describe(records)["by_size_band"])
        assert bands == {"tiny", "small", "medium", "large", "huge"}

    def test_the_largest_size_band_is_reached(self, records):
        # The band where scanned gazette reproductions live. Without it the
        # benchmark would never meet a scanned document.
        assert sample.describe(records)["by_size_band"]["huge"] > 0

    def test_many_jurisdictions_are_covered(self, records):
        assert sample.describe(records)["distinct_jurisdictions"] >= 5

    def test_no_document_is_selected_twice(self, records):
        ids = [r["document_id"] for r in records]
        assert len(ids) == len(set(ids))


class TestQuotaScaling:
    def test_a_smaller_sample_keeps_the_same_shape(self):
        scaled = sample.scale_quotas(dict(central_acts=25, state_acts=30,
                                          rules=25, regulations=20), 20)
        assert sum(scaled.values()) == 20
        assert scaled["state_acts"] >= scaled["regulations"]

    def test_every_category_keeps_at_least_one_document(self):
        scaled = sample.scale_quotas(dict(a=25, b=30, c=25, d=20), 5)
        assert min(scaled.values()) >= 1
        assert sum(scaled.values()) == 5

    def test_the_full_size_is_unchanged(self):
        quotas = dict(central_acts=25, state_acts=30, rules=25, regulations=20)
        assert sample.scale_quotas(quotas, 100) == quotas


class TestSmallAndAwkwardCorpora:
    def test_a_corpus_smaller_than_the_target_returns_everything(self):
        corpus = [document(f"d{i}") for i in range(7)]
        records = sample.select(corpus, target=100)
        assert len(records) == 7

    def test_a_short_category_is_topped_up_from_the_rest(self):
        corpus = (
            [document(f"c{i}", category="central_acts") for i in range(2)]
            + [document(f"s{i}", category="state_acts") for i in range(40)]
        )
        records = sample.select(corpus, target=20)
        assert len(records) == 20
        assert any(r["selection_reason"] == "quota_top_up" for r in records)

    def test_an_empty_corpus_is_an_error(self):
        with pytest.raises(SamplingError):
            sample.select([], target=10)

    def test_documents_with_unknown_years_are_still_sampled(self):
        corpus = [document(f"d{i}", year=None) for i in range(40)]
        records = sample.select(corpus, target=10)
        assert all(r["era_band"] == "year_unknown" for r in records)


class TestDescribe:
    def test_summary_reports_ranges(self):
        records = sample.select(big_corpus(), target=100)
        summary = sample.describe(records)
        assert summary["size"] == 100
        assert summary["year_range"][0] <= summary["year_range"][1]
        assert summary["bytes_range"][0] <= summary["bytes_range"][1]
