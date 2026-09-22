"""The reader: every silent trap in the processed data must fail loudly or read right."""

from __future__ import annotations

import pytest

from benchmark.errors import LineStreamMismatchError, NotEligibleError, SourceError
from benchmark.source import (assessed_pages, line_stream, load_document, load_pages,
                              page_text, require_eligible)
from tests.processedbuild import (ACT_PAGE_ONE, ACT_PAGE_TWO, make_page, rewrite_document,
                                  write_processed)

DOC = "sample-act-1999__handle-1"


def test_an_ocr_page_is_read_from_its_ocr_text_not_the_empty_backend_text(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE),
                                    make_page(2, "", ocr_text=ACT_PAGE_TWO)])
    pages, _ = load_pages(tmp_path, DOC)
    assert pages[1].text == ""
    assert pages[1].selected_text == ACT_PAGE_TWO
    stream = line_stream(load_document(tmp_path, DOC), pages)
    assert any("Power to make rules" in line.text for line in stream)


def test_page_text_resolves_the_reading_from_a_raw_dictionary():
    assert page_text({"text": "", "text_source": "ocr", "ocr": {"text": "engine"}}) == "engine"
    assert page_text({"text": "backend", "text_source": "backend"}) == "backend"
    assert page_text({"text": "backend"}) == "backend"


def test_missing_page_fields_are_reported_not_hidden(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE)],
                    omit_page_fields=("text_source", "ocr", "quality"))
    pages, missing = load_pages(tmp_path, DOC)
    assert missing == ["ocr", "quality", "text_source"]
    assert pages[0].text_source == "backend"


def test_the_stream_is_built_over_indexable_pages_only(tmp_path):
    hindi = make_page(2, "यह पृष्ठ हिंदी में है", indexable=False)
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE), hindi, make_page(3, ACT_PAGE_TWO)])
    pages, _ = load_pages(tmp_path, DOC)
    assert [p.page_number for p in assessed_pages(pages)] == [1, 3]
    stream = line_stream(load_document(tmp_path, DOC), pages)
    assert {line.page_number for line in stream} == {1, 3}


def test_a_line_count_disagreement_is_refused(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE)])
    rewrite_document(tmp_path, DOC, lambda d: d["structure"]["counts"].update(line_count=999))
    pages, _ = load_pages(tmp_path, DOC)
    with pytest.raises(LineStreamMismatchError):
        line_stream(load_document(tmp_path, DOC), pages)


def test_a_missing_line_count_is_refused_rather_than_trusted(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE)])
    rewrite_document(tmp_path, DOC, lambda d: d["structure"]["counts"].pop("line_count"))
    pages, _ = load_pages(tmp_path, DOC)
    with pytest.raises(LineStreamMismatchError):
        line_stream(load_document(tmp_path, DOC), pages)


def test_an_unknown_schema_version_is_refused(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE)], schema_version=99)
    with pytest.raises(SourceError):
        load_document(tmp_path, DOC)


def test_a_quarantined_document_is_refused(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE)], eligible=False)
    with pytest.raises(NotEligibleError):
        require_eligible(load_document(tmp_path, DOC))


def test_a_missing_document_is_a_source_error(tmp_path):
    with pytest.raises(SourceError):
        load_document(tmp_path, "no-such-document")
