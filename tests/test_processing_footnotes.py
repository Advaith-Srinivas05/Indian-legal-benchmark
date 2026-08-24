"""Tests for footnote detection and separation.

The regression: India Code prints amendment history as numbered footnotes at
the foot of the page, and extracted linearly they landed inside the provision
above them. In the first benchmark, the Code of Civil Procedure's section 1
ended with "1. This Act has been amended in its application to Assam by Assam
Acts 2 of 1941…" as though it were law.

Footnotes are separated, never deleted, so two things are tested together
throughout: that the block is found, and that its text is still in the page.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from processing import footnotes
from processing.backends import LineMetric
from processing.extract import extract_document
from processing.structure import parse_structure
from tests import pdfbuild

#: ``{n}`` is replaced with the page number by the builder, so no two pages are
#: identical — identical pages would be running-head material, which is a
#: different rule entirely.
BODY = (
    "{n}. Short title and extent.-(1) This Act may be called the Sample Act, "
    "199{n}.\n(2) It extends to the whole of India and applies also to citizens "
    "of India who are outside India, save as otherwise expressly provided in "
    "this section {n} or in any rule made under this Act."
)

AMENDMENT_NOTE = (
    "1. Subs. by Act 3{n} of 1965, s. 3, for the former sub-section "
    "(w.e.f. 1-10-1967).\n"
    "2. Ins. by Act {n} of 1941, s. 2 and the Sch. (w.e.f. 1-1-1942)."
)


def _page(text, metrics, *, height=842.0, number=1, furniture=()):
    return SimpleNamespace(
        text=text, height=height, page_number=number,
        line_metrics=metrics, furniture=list(furniture),
    )


def _metrics(sizes_and_positions):
    return [
        LineMetric(line_index=i, y0=y, y1=y + 10, size=size)
        for i, (y, size) in enumerate(sizes_and_positions)
    ]


class TestDetectionOnSyntheticPages:
    def test_a_smaller_block_at_the_foot_is_found(self):
        text = "Body line one\nBody line two\n1. Ins. by Act 30 of 1965, s. 3"
        metrics = _metrics([(100, 11.0), (120, 11.0), (700, 9.0)])
        page = _page(text, metrics)
        assert footnotes.detect([page])[1][0].line_start == 2

    def test_body_sized_text_at_the_foot_is_not_a_footnote(self):
        text = "Body line one\nBody line two\n1. Ins. by Act 30 of 1965, s. 3"
        metrics = _metrics([(100, 11.0), (120, 11.0), (700, 11.0)])
        assert footnotes.detect([_page(text, metrics)]) == {}

    def test_small_text_in_the_middle_of_a_page_is_not_a_footnote(self):
        text = "Body one\n1. Ins. by Act 30 of 1965, s. 3\nBody two at the foot"
        metrics = _metrics([(100, 11.0), (300, 9.0), (700, 11.0)])
        assert footnotes.detect([_page(text, metrics)]) == {}

    def test_a_block_with_no_marker_is_not_a_footnote(self):
        text = "Body line\ncontinued small print without any marker at all"
        metrics = _metrics([(100, 11.0), (700, 9.0)])
        assert footnotes.detect([_page(text, metrics)]) == {}

    def test_a_numbered_list_at_the_foot_in_body_type_is_left_alone(self):
        # The thing that must not be eaten: a genuine numbered list that
        # happens to end the page.
        text = ("2. Definitions.—In this Act,—\n"
                "1. the first defined expression;\n"
                "2. the second defined expression;")
        metrics = _metrics([(100, 11.0), (680, 11.0), (700, 11.0)])
        assert footnotes.detect([_page(text, metrics)]) == {}

    def test_a_page_that_is_entirely_small_type_is_not_swallowed(self):
        # A continuation page of an overflowing note has no body-sized line
        # above the block, so there is nothing to separate it from.
        text = "1. Ins. by Act 30 of 1965\ncontinued\nstill continued"
        metrics = _metrics([(100, 9.0), (300, 9.0), (700, 9.0)])
        assert footnotes.detect([_page(text, metrics)]) == {}

    def test_pages_without_aligned_geometry_contribute_nothing(self):
        assert footnotes.detect([_page("1. Ins. by Act 30 of 1965", [])]) == {}

    def test_furniture_lines_are_skipped(self):
        from processing.models import FurnitureLine
        text = "Body\n1. Ins. by Act 30 of 1965, s. 3\n42"
        metrics = _metrics([(100, 11.0), (700, 9.0), (780, 9.0)])
        furniture = [FurnitureLine(line_index=2, text="42", kind="page_number",
                                   reason="x")]
        block = footnotes.detect([_page(text, metrics, furniture=furniture)])[1][0]
        assert "42" not in block.text


class TestDocumentBodySize:
    def test_body_size_is_taken_across_the_whole_document(self):
        # A page that is mostly footnote must not define the footnote size as
        # the body size — the failure that stopped the CPC's longest notes from
        # being separated.
        mostly_notes = _page(
            "Body\n" + "\n".join(["note"] * 20),
            _metrics([(80, 11.0)] + [(200 + i * 20, 9.0) for i in range(20)]),
            number=1,
        )
        normal = _page(
            "\n".join(["body"] * 20),
            _metrics([(80 + i * 20, 11.0) for i in range(20)]),
            number=2,
        )
        assert footnotes.document_body_size([mostly_notes, normal]) == 11.0

    def test_no_geometry_gives_no_body_size(self):
        assert footnotes.document_body_size([_page("x", [])]) is None


class TestOnRealPdfLayout:
    @pytest.fixture()
    def extracted(self, tmp_path):
        path = pdfbuild.footnoted_pdf(
            tmp_path / "act.pdf", BODY, AMENDMENT_NOTE, page_count=3)
        return extract_document(path, "act", detect_tables=False)

    def test_footnote_blocks_are_found_on_every_page(self, extracted):
        assert extracted.footnote_count == 3
        assert all(page.footnotes for page in extracted.pages)

    def test_the_block_carries_page_provenance(self, extracted):
        block = extracted.pages[1].footnotes[0]
        assert block.page_number == 2
        assert block.line_start >= 0
        assert block.line_end >= block.line_start
        assert block.detected_by

    def test_the_block_serialises_as_a_footnote_unit(self, extracted):
        payload = extracted.pages[0].footnotes[0].to_dict()
        assert payload["type"] == "footnote"
        assert payload["page_start"] == payload["page_end"] == 1
        assert "Subs. by Act 31 of 1965" in payload["text"]

    def test_the_text_is_still_present_in_the_page(self, extracted):
        # Separated, not deleted.
        assert "Subs. by Act 31 of 1965" in extracted.pages[0].text

    def test_the_provision_no_longer_absorbs_the_note(self, extracted):
        structure = parse_structure(extracted.pages, metadata_title="Sample Act")
        sections = [u for u in structure.all_units() if u.unit_type == "section"]
        assert sections
        assert "Subs. by Act 31 of 1965" not in sections[0].text
        assert "This Act may be called" in sections[0].text

    def test_the_structure_keeps_the_footnotes(self, extracted):
        structure = parse_structure(extracted.pages)
        assert len(structure.footnotes) == 3
        assert structure.counts["footnote"] == 3
        assert structure.to_dict()["footnotes"][0]["type"] == "footnote"

    def test_a_document_without_footnotes_reports_none(self, tmp_path):
        path = pdfbuild.text_pdf(tmp_path / "plain.pdf", [BODY, BODY])
        extracted = extract_document(path, "plain", detect_tables=False)
        assert extracted.footnote_count == 0
