"""Tests for per-document processing and the files it writes.

The invariants under test are the ones a later phase will depend on and cannot
re-derive: page boundaries survive, page text is stored verbatim, provenance
back to the exact raw PDF is recorded, a failure is reported rather than raised,
and nothing under ``data/raw/`` is written to.
"""

from __future__ import annotations

import json

import pytest

from processing import config
from processing.corpus import Corpus
from processing.process import output_dir, process_document
from tests import corpusbuild


@pytest.fixture()
def data_dir(tmp_path):
    entries = [
        corpusbuild.add_document(tmp_path, "sample-act-1999__handle-1",
                                 category="central_acts", title="Sample Act, 1999"),
        corpusbuild.add_document(tmp_path, "broken__handle-2",
                                 category="state_acts", kind="corrupt"),
        corpusbuild.add_document(tmp_path, "scan__handle-3",
                                 category="rules", kind="scanned"),
    ]
    corpusbuild.write_corpus(tmp_path, entries)
    return tmp_path


def process(data_dir, document_id, **kwargs):
    document = Corpus.load(data_dir).get(document_id)
    return document, process_document(document, data_dir, **kwargs)


class TestSuccessfulProcessing:
    def test_returns_extraction_and_structure(self, data_dir):
        _, result = process(data_dir, "sample-act-1999__handle-1")
        assert result.ok
        assert result.extraction.page_count == 1
        assert result.structure.counts["section"] == 3

    def test_writes_both_files_in_the_mandated_location(self, data_dir):
        _, result = process(data_dir, "sample-act-1999__handle-1")
        directory = output_dir(data_dir, "sample-act-1999__handle-1")
        assert result.output_dir == directory
        assert (directory / config.DOCUMENT_FILENAME).exists()
        assert (directory / config.PAGES_FILENAME).exists()
        assert directory.parent == data_dir / "processed" / "indiacode"


class TestPagesJson:
    @pytest.fixture()
    def payload(self, data_dir):
        process(data_dir, "sample-act-1999__handle-1")
        path = output_dir(data_dir, "sample-act-1999__handle-1") / config.PAGES_FILENAME
        return json.loads(path.read_text(encoding="utf-8"))

    def test_page_boundaries_are_preserved(self, payload):
        assert payload["page_count"] == len(payload["pages"])
        assert [p["page_number"] for p in payload["pages"]] == [1]

    def test_minimum_page_representation_is_present(self, payload):
        page = payload["pages"][0]
        assert {"page_number", "text", "char_count"} <= set(page)
        assert page["char_count"] == len(page["text"])

    def test_page_text_is_stored_verbatim(self, data_dir, payload):
        # The stored text must equal what the backend produced, so an extraction
        # bug found later can be diagnosed from the file alone.
        _, result = process(data_dir, "sample-act-1999__handle-1")
        assert payload["pages"][0]["text"] == result.extraction.pages[0].text

    def test_debugging_evidence_is_preserved(self, payload):
        page = payload["pages"][0]
        for key in ("alpha_char_count", "image_area_ratio", "is_image_backed",
                    "has_text", "quality_signals", "furniture", "tables"):
            assert key in page


class TestDocumentJson:
    @pytest.fixture()
    def payload(self, data_dir):
        process(data_dir, "sample-act-1999__handle-1")
        path = output_dir(data_dir, "sample-act-1999__handle-1") / config.DOCUMENT_FILENAME
        return json.loads(path.read_text(encoding="utf-8"))

    def test_provenance_points_back_at_the_exact_raw_pdf(self, payload):
        assert payload["artifacts"]["raw_pdf"] == payload["source"]["pdf_relpath"]
        assert len(payload["source"]["sha256"]) == 64

    def test_ingestion_metadata_is_carried_forward_not_re_derived(self, payload):
        source = payload["source"]
        assert source["title"] == "Sample Act, 1999"
        assert source["document_type"] == "central_act"
        assert source["language"] == "en"

    def test_classification_is_recorded_with_its_evidence(self, payload):
        extraction = payload["extraction"]
        assert extraction["pdf_type"] in config.PDF_TYPES
        assert extraction["text_extraction_status"] in config.TEXT_EXTRACTION_STATUSES
        assert extraction["classification_evidence"]["thresholds"]

    def test_structure_units_carry_page_provenance(self, payload):
        units = payload["structure"]["units"]
        assert units
        for unit in units:
            assert unit["page_start"] >= 1
            assert unit["page_end"] >= unit["page_start"]
            assert unit["detected_by"]

    def test_schema_version_is_recorded(self, payload):
        assert payload["schema_version"] == config.PROCESSING_SCHEMA_VERSION


class TestFailureHandling:
    def test_an_unreadable_pdf_is_reported_not_raised(self, data_dir):
        _, result = process(data_dir, "broken__handle-2")
        assert not result.ok
        assert result.error_type == "PDFOpenError"
        assert result.error_message

    def test_a_failed_document_writes_nothing(self, data_dir):
        process(data_dir, "broken__handle-2")
        assert not output_dir(data_dir, "broken__handle-2").exists()

    def test_a_manifest_entry_with_no_file_on_disk_is_reported(self, data_dir):
        document = Corpus.load(data_dir).get("scan__handle-3")
        document.pdf_path.unlink()
        result = process_document(document, data_dir)
        assert not result.ok
        assert "not on disk" in result.error_message


class TestRawCorpusIsUntouched:
    def test_processing_does_not_modify_the_source_pdf(self, data_dir):
        document = Corpus.load(data_dir).get("sample-act-1999__handle-1")
        before = document.pdf_path.read_bytes()
        process_document(document, data_dir)
        assert document.pdf_path.read_bytes() == before

    def test_output_never_lands_under_data_raw(self, data_dir):
        process(data_dir, "scan__handle-3")
        raw_files = {p.name for p in (data_dir / "raw").rglob("*")}
        assert config.DOCUMENT_FILENAME not in raw_files
        assert config.PAGES_FILENAME not in raw_files


class TestWriteToggle:
    def test_write_false_measures_without_producing_files(self, data_dir):
        _, result = process(data_dir, "sample-act-1999__handle-1", write=False)
        assert result.ok
        assert result.output_dir is None
        assert not output_dir(data_dir, "sample-act-1999__handle-1").exists()
