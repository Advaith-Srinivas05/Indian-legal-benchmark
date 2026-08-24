"""Tests for reading the corpus through the ingestion manifest.

The project spec requires successful documents to be discovered through the manifest
rather than by walking ``data/raw/``. These tests hold that line: the manifest
decides what the corpus is, the inventory only enriches it, and a disagreement
between the manifest and the disk is surfaced rather than absorbed.
"""

from __future__ import annotations

import json

import pytest

from ingestion import config as ingestion_config
from processing.corpus import Corpus
from processing.errors import CorpusError
from tests import corpusbuild


@pytest.fixture()
def data_dir(tmp_path):
    entries = [
        corpusbuild.add_document(tmp_path, "sample-act-1999__handle-1",
                                 category="central_acts", year=1999),
        corpusbuild.add_document(tmp_path, "assam-land-act-1971__handle-2",
                                 category="state_acts", year=1971,
                                 jurisdiction="Assam"),
        corpusbuild.add_document(tmp_path, "sample-rules-2001__rule-3",
                                 category="rules", year=2001,
                                 jurisdiction="Kerala"),
    ]
    corpusbuild.write_corpus(tmp_path, entries)
    return tmp_path


class TestLoading:
    def test_every_manifest_entry_becomes_a_document(self, data_dir):
        corpus = Corpus.load(data_dir)
        assert len(corpus) == 3
        assert {d.document_id for d in corpus} == {
            "sample-act-1999__handle-1",
            "assam-land-act-1971__handle-2",
            "sample-rules-2001__rule-3",
        }

    def test_pdf_paths_resolve_against_the_data_directory(self, data_dir):
        for document in Corpus.load(data_dir):
            assert document.pdf_path.exists()
            assert document.pdf_path.is_absolute() or document.pdf_path.parts

    def test_identity_fields_come_from_the_manifest(self, data_dir):
        document = Corpus.load(data_dir).get("sample-act-1999__handle-1")
        assert document.category == "central_acts"
        assert document.document_type == "central_act"
        assert document.language == "en"
        assert len(document.sha256) == 64
        assert document.bytes > 0

    def test_year_and_jurisdiction_come_from_the_inventory(self, data_dir):
        document = Corpus.load(data_dir).get("assam-land-act-1971__handle-2")
        assert document.jurisdiction == "Assam"
        assert document.year == 1971

    def test_documents_are_returned_in_a_stable_order(self, data_dir):
        first = [d.document_id for d in Corpus.load(data_dir)]
        second = [d.document_id for d in Corpus.load(data_dir)]
        assert first == second == sorted(first)

    def test_by_category(self, data_dir):
        corpus = Corpus.load(data_dir)
        assert len(corpus.by_category("state_acts")) == 1
        assert corpus.by_category("regulations") == []


class TestMissingArtefacts:
    def test_a_missing_manifest_is_an_error(self, tmp_path):
        with pytest.raises(CorpusError, match="No manifest"):
            Corpus.load(tmp_path)

    def test_a_missing_inventory_is_tolerated_by_default(self, tmp_path):
        entries = [corpusbuild.add_document(tmp_path, "a__handle-1")]
        corpusbuild.write_corpus(tmp_path, entries, inventory=False)
        corpus = Corpus.load(tmp_path)
        assert corpus.get("a__handle-1").year is None

    def test_a_missing_inventory_is_an_error_when_required(self, tmp_path):
        entries = [corpusbuild.add_document(tmp_path, "a__handle-1")]
        corpusbuild.write_corpus(tmp_path, entries, inventory=False)
        with pytest.raises(CorpusError, match="inventory"):
            Corpus.load(tmp_path, require_inventory=True)

    def test_a_manifest_entry_without_a_pdf_path_is_an_error(self, tmp_path):
        (tmp_path / ingestion_config.MANIFEST_FILENAME).write_text(
            json.dumps({"documents": {"x": {"document_id": "x"}}}), encoding="utf-8")
        with pytest.raises(CorpusError, match="pdf_relpath"):
            Corpus.load(tmp_path)


class TestDiskAgreement:
    def test_present_and_missing_partition_the_corpus(self, data_dir):
        corpus = Corpus.load(data_dir)
        assert len(corpus.present()) == 3
        assert corpus.missing() == []

    def test_a_manifest_entry_whose_pdf_is_gone_is_reported_not_ignored(self, data_dir):
        corpus = Corpus.load(data_dir)
        target = corpus.get("sample-rules-2001__rule-3")
        target.pdf_path.unlink()

        reloaded = Corpus.load(data_dir)
        assert [d.document_id for d in reloaded.missing()] == ["sample-rules-2001__rule-3"]
        assert len(reloaded.present()) == 2
