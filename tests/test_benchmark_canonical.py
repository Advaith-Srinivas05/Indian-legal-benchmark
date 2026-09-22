"""Canonical text: the coordinate system every gold span depends on."""

from __future__ import annotations

import pytest

from benchmark.canonical import build_canonical
from benchmark.errors import SpanMismatchError
from benchmark.source import load_document, load_pages
from tests.processedbuild import (ACT_PAGE_ONE, ACT_PAGE_TWO, PLAIN_RULES, SCHEDULE_PAGE,
                                  make_page, rewrite_document, walk_units, write_processed)

DOC = "sample-act-1999__handle-1"


def build(tmp_path, pages, **kwargs):
    write_processed(tmp_path, DOC, pages, **kwargs)
    document = load_document(tmp_path, DOC)
    loaded, _ = load_pages(tmp_path, DOC)
    return document, build_canonical(document, loaded)


def test_every_provision_span_reproduces_its_text(tmp_path):
    document, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    sections = [u for u in walk_units(document["structure"]["units"]) if u["unit_type"] == "section"]
    assert [p["number"] for p in canonical.provisions] == ["1", "2", "3", "4"]
    for unit, provision in zip(sections, canonical.provisions):
        assert canonical.text[provision["char_start"]:provision["char_end"]] == unit["text"]


def test_the_page_map_tiles_the_whole_text_without_gaps(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    cursor = 0
    for page in canonical.page_map:
        assert page["char_start"] == cursor
        cursor = page["char_end"]
    assert cursor == len(canonical.text)
    assert [p["page_number"] for p in canonical.page_map] == [1, 2]


def test_a_provision_lies_inside_the_pages_the_page_map_gives_it(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    by_page = {p["page_number"]: p for p in canonical.page_map}
    for provision in canonical.provisions:
        assert provision["char_start"] >= by_page[provision["page_start"]]["char_start"]
        assert provision["char_end"] <= by_page[provision["page_end"]]["char_end"]


def test_a_line_break_character_inside_a_line_does_not_shift_offsets(tmp_path):
    page = ACT_PAGE_ONE.replace("It extends to", "It extends to").replace("Sample Act, 1999.", "Sample Act,\r1999.")
    _, canonical = build(tmp_path, [make_page(1, page), make_page(2, ACT_PAGE_TWO)])
    assert canonical.replaced_line_boundaries == 2
    assert len(canonical.text.splitlines()) == canonical.line_count
    first = canonical.provisions[0]
    assert "It extends to the whole of India" in canonical.text[first["char_start"]:first["char_end"]]


def test_an_ocr_page_elsewhere_does_not_demote_a_provision(tmp_path):
    """The Right to Information Act regression: one OCR-read schedule page must
    not push every born-digital section out of the high tier."""
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO),
                                    make_page(3, "", ocr_text=SCHEDULE_PAGE)])
    assert canonical.ocr_pages == [3]
    assert {p["evidence_confidence"] for p in canonical.provisions} == {"high"}
    assert all(p["ocr_in_document"] and not p["ocr_in_span"] for p in canonical.provisions)


def test_a_provision_on_an_ocr_page_is_not_high(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE),
                                    make_page(2, "", ocr_text=ACT_PAGE_TWO)])
    tiers = {p["number"]: p["evidence_confidence"] for p in canonical.provisions}
    assert tiers["1"] == "high" and tiers["3"] == "medium" and tiers["4"] == "medium"


def test_a_weak_detector_provision_is_never_high(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, PLAIN_RULES)], title="The Plain Rules, 2001")
    assert canonical.provisions, "the plain numbered headings should be found"
    assert {p["detected_by"] for p in canonical.provisions} == {"numbered_heading_line"}
    assert {p["evidence_confidence"] for p in canonical.provisions} == {"medium"}


def test_a_chapter_does_not_claim_its_span_as_its_text(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    chapter = next(u for u in canonical.structure if u["unit_type"] == "chapter")
    assert chapter["text_matches_span"] is False
    assert chapter["char_start"] is not None


def test_provision_keys_carry_the_full_path(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    keys = [p["key"] for p in canonical.provisions]
    assert keys[0] == "chapter:I/section:1"
    assert len(set(keys)) == len(keys)


def test_an_act_printed_twice_gets_unique_keys(tmp_path):
    pages = [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO),
             make_page(3, ACT_PAGE_ONE), make_page(4, ACT_PAGE_TWO)]
    _, canonical = build(tmp_path, pages)
    keys = [p["key"] for p in canonical.provisions]
    assert len(set(keys)) == len(keys)
    assert canonical.duplicate_provision_keys > 0
    assert any(k.endswith("#2") for k in keys)


def test_the_first_of_a_repeated_citation_is_marked_ambiguous_too(tmp_path):
    pages = [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO),
             make_page(3, ACT_PAGE_ONE), make_page(4, ACT_PAGE_TWO)]
    _, canonical = build(tmp_path, pages)
    first = next(p for p in canonical.provisions if p["key"] == "chapter:I/section:1")
    assert first["key_ambiguous"] is True
    assert canonical.ambiguous_provisions == 2 * canonical.duplicate_provision_keys


def test_a_unique_citation_is_not_marked_ambiguous(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
    assert not any(p["key_ambiguous"] for p in canonical.provisions)


def test_a_provision_whose_text_disagrees_with_its_span_is_refused(tmp_path):
    write_processed(tmp_path, DOC, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])

    def corrupt(document):
        section = next(u for u in walk_units(document["structure"]["units"])
                       if u["unit_type"] == "section")
        section["text"] = section["text"].replace("Sample", "Simple")

    rewrite_document(tmp_path, DOC, corrupt)
    document = load_document(tmp_path, DOC)
    pages, _ = load_pages(tmp_path, DOC)
    with pytest.raises(SpanMismatchError):
        build_canonical(document, pages)


def test_the_canonical_text_never_contains_a_carriage_return(tmp_path):
    _, canonical = build(tmp_path, [make_page(1, ACT_PAGE_ONE.replace("\n", "\r\n"))])
    assert "\r" not in canonical.text
