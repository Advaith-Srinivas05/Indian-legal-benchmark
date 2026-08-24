"""Tests for PDF extraction and page/document classification.

Two kinds of test here, deliberately separated:

* Against **real generated PDFs** (``tests/pdfbuild.py``) — page boundaries,
  image coverage, ruled tables, unreadable files. These are things only an
  actual PDF can exercise.
* Against a **fake backend** — the classification thresholds themselves. Driving
  those through a generated PDF would mean tuning fixture greyscale until a
  ratio crossed a line; feeding :class:`~processing.backends.RawPage` objects
  directly states the case under test outright.
"""

from __future__ import annotations

import pytest

from processing import config
from processing.backends import RawDocument, RawPage, RawTable
from processing.errors import PDFOpenError
from processing.extract import extract_document
from tests import pdfbuild

A4_W, A4_H = pdfbuild.A4

# Enough real prose that a page clears the "has usable text" threshold.
PROSE = (
    "1. Short title and extent. This Act may be called the Sample Act, 1999 and "
    "it extends to the whole of India, save as otherwise provided in this "
    "section, and shall come into force on such date as the Central Government "
    "may, by notification in the Official Gazette, appoint in this behalf."
)


class FakeBackend:
    """A backend that returns exactly the pages a test describes."""

    name = "fake"

    def __init__(self, pages: list[RawPage], **document_kwargs):
        self.pages = pages
        self.document_kwargs = document_kwargs

    def read(self, path, *, detect_tables: bool = True) -> RawDocument:
        return RawDocument(
            page_count=len(self.pages), pages=self.pages, backend=self.name,
            **self.document_kwargs,
        )


def raw_page(number: int, *, text: str = "", images=(), tables=()) -> RawPage:
    return RawPage(
        page_number=number, text=text, width=A4_W, height=A4_H,
        image_bboxes=list(images), tables=list(tables),
    )


def full_page_image():
    return (0.0, 0.0, A4_W, A4_H)


# --- Real PDFs ------------------------------------------------------------------


class TestTextPdf:
    @pytest.fixture()
    def extracted(self, tmp_path):
        path = pdfbuild.text_pdf(
            tmp_path / "act.pdf",
            [f"Page {n} body.\n{PROSE}" for n in range(1, 4)],
        )
        return extract_document(path, "act")

    def test_page_boundaries_are_preserved(self, extracted):
        assert extracted.page_count == 3
        assert [p.page_number for p in extracted.pages] == [1, 2, 3]

    def test_each_page_reports_its_own_text_and_char_count(self, extracted):
        for number, page in enumerate(extracted.pages, start=1):
            assert f"Page {number} body." in page.text
            assert page.char_count == len(page.text)
            assert page.has_text

    def test_classified_as_text_based_and_ok(self, extracted):
        assert extracted.pdf_type == "text_based"
        assert extracted.text_extraction_status == "ok"
        assert extracted.classification_evidence["image_backed_pages"] == 0

    def test_classification_evidence_records_the_thresholds_used(self, extracted):
        thresholds = extracted.classification_evidence["thresholds"]
        assert thresholds["min_page_alpha_chars"] == config.MIN_PAGE_ALPHA_CHARS
        assert thresholds["image_backed_area_ratio"] == config.IMAGE_BACKED_AREA_RATIO


class TestScannedPdf:
    def test_image_only_pages_require_ocr(self, tmp_path):
        path = pdfbuild.scanned_pdf(tmp_path / "scan.pdf", page_count=3)
        extracted = extract_document(path, "scan")
        assert extracted.pdf_type == "scanned"
        assert extracted.text_extraction_status == "requires_ocr"
        assert all(p.is_image_backed for p in extracted.pages)
        assert any("requires OCR" in w for w in extracted.warnings)

    def test_a_scan_with_an_ocr_layer_is_still_scanned(self, tmp_path):
        # The case that matters most in this corpus: text comes out, so nothing
        # looks broken, but the pages are photographs of a gazette.
        path = pdfbuild.scanned_pdf(
            tmp_path / "ocr.pdf", page_count=3, ocr_text=[PROSE, PROSE, PROSE],
        )
        extracted = extract_document(path, "ocr")
        assert extracted.pdf_type == "scanned"
        assert extracted.text_extraction_status == "ok"
        assert all(p.is_image_backed and p.has_text for p in extracted.pages)


class TestMixedPdf:
    def test_text_pages_plus_scanned_pages(self, tmp_path):
        path = pdfbuild.mixed_pdf(
            tmp_path / "mixed.pdf",
            text_pages=[PROSE, PROSE, PROSE],
            image_pages=2,
        )
        extracted = extract_document(path, "mixed")
        assert extracted.pdf_type == "mixed"
        assert extracted.text_extraction_status == "partial"
        evidence = extracted.classification_evidence
        assert evidence["image_backed_pages"] == 2
        assert evidence["pages_with_text"] == 3


class TestBlankAndBrokenPdfs:
    def test_blank_pages_are_counted_not_hidden(self, tmp_path):
        path = pdfbuild.blank_pdf(tmp_path / "blank.pdf", page_count=3)
        extracted = extract_document(path, "blank")
        assert extracted.empty_page_count == 3
        assert extracted.text_extraction_status == "requires_ocr"
        assert any("extracted as empty" in w for w in extracted.warnings)

    def test_a_page_with_neither_text_nor_image_says_so(self, tmp_path):
        path = pdfbuild.blank_pdf(tmp_path / "blank.pdf", page_count=3)
        extracted = extract_document(path, "blank")
        assert any("no usable text and no page-covering image" in w
                   for w in extracted.pages[0].warnings)

    def test_an_unreadable_file_raises(self, tmp_path):
        path = pdfbuild.corrupt_pdf(tmp_path / "broken.pdf")
        with pytest.raises(PDFOpenError):
            extract_document(path, "broken")

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(PDFOpenError):
            extract_document(tmp_path / "absent.pdf", "absent")


class TestTables:
    def test_a_ruled_table_is_detected_and_retained(self, tmp_path):
        rows = [
            ["Serial", "Name of officer", "Area"],
            ["1", "Regional Passport Officer", "Delhi"],
            ["2", "District Magistrate", "Assam"],
            ["3", "Collector", "Goa"],
        ]
        path = pdfbuild.table_pdf(tmp_path / "table.pdf", rows, heading="SCHEDULE I")
        extracted = extract_document(path, "table")
        retained = extracted.pages[0].retained_tables
        assert retained, "a fully ruled 4x3 grid should be detected"
        assert retained[0].row_count >= 3
        assert retained[0].col_count >= 2
        flat = " ".join(
            str(cell) for row in retained[0].rows for cell in row if cell)
        assert "Regional Passport Officer" in flat

    def test_table_detection_can_be_switched_off(self, tmp_path):
        rows = [["a", "b"], ["c", "d"], ["e", "f"]]
        path = pdfbuild.table_pdf(tmp_path / "table.pdf", rows)
        extracted = extract_document(path, "table", detect_tables=False)
        assert extracted.table_candidate_count == 0


# --- Classification thresholds, through the fake backend ------------------------


class TestPageClassification:
    def test_a_page_covered_by_one_image_is_image_backed(self):
        backend = FakeBackend([raw_page(1, images=[full_page_image()])])
        extracted = extract_document("x.pdf", "x", backend=backend)
        assert extracted.pages[0].image_area_ratio == pytest.approx(1.0)
        assert extracted.pages[0].is_image_backed

    def test_overlapping_images_do_not_exceed_full_coverage(self):
        backend = FakeBackend([
            raw_page(1, images=[full_page_image(), full_page_image()])
        ])
        extracted = extract_document("x.pdf", "x", backend=backend)
        assert extracted.pages[0].image_area_ratio <= 1.0

    def test_a_small_illustration_does_not_make_a_page_a_scan(self):
        backend = FakeBackend([
            raw_page(1, text=PROSE, images=[(50.0, 50.0, 200.0, 200.0)])
        ])
        extracted = extract_document("x.pdf", "x", backend=backend)
        assert not extracted.pages[0].is_image_backed
        assert extracted.pdf_type == "text_based"

    def test_a_page_with_too_few_letters_has_no_usable_text(self):
        backend = FakeBackend([raw_page(1, text="12")])
        extracted = extract_document("x.pdf", "x", backend=backend)
        assert not extracted.pages[0].has_text

    def test_an_ocr_noise_page_is_flagged_but_kept(self):
        noise = " ".join(["~t1diO", "<Sox.ette", "~fM~", "\\Iq'", "lIlT"] * 12)
        backend = FakeBackend([raw_page(1, text=noise + " " + PROSE)])
        extracted = extract_document("x.pdf", "x", backend=backend)
        page = extracted.pages[0]
        assert page.text_quality_suspect
        # Flagged, never discarded.
        assert noise[:20] in page.text


class TestDocumentClassification:
    def test_boundary_between_mixed_and_scanned(self):
        # 9 of 10 image-backed pages is exactly the scanned threshold.
        image_pages = [raw_page(n, images=[full_page_image()]) for n in range(1, 10)]
        text_pages = [raw_page(10, text=PROSE)]
        backend = FakeBackend(image_pages + text_pages)
        assert extract_document("x.pdf", "x", backend=backend).pdf_type == "scanned"

    def test_a_single_scanned_page_among_many_is_mixed(self):
        pages = [raw_page(n, text=PROSE) for n in range(1, 5)]
        pages.append(raw_page(5, images=[full_page_image()]))
        backend = FakeBackend(pages)
        assert extract_document("x.pdf", "x", backend=backend).pdf_type == "mixed"

    def test_a_document_with_no_pages_is_an_error(self):
        backend = FakeBackend([])
        with pytest.raises(PDFOpenError):
            extract_document("x.pdf", "x", backend=backend)

    def test_furniture_is_attached_to_pages_during_extraction(self):
        pages = [
            raw_page(n, text=f"THE GAZETTE OF INDIA\n{n}\n{PROSE} {n}")
            for n in range(1, 5)
        ]
        extracted = extract_document("x.pdf", "x", backend=FakeBackend(pages))
        kinds = {f.kind for page in extracted.pages for f in page.furniture}
        assert kinds == {"header", "page_number"}
        # Labelled, not removed.
        assert extracted.pages[0].text.startswith("THE GAZETTE OF INDIA")


class TestTableQualityFilter:
    """PyMuPDF reports a grid wherever ruled lines or column alignment appear,
    which on legal prose means indented sub-clauses come back as "tables"."""

    def _extract_one(self, rows):
        table = RawTable(bbox=(0.0, 0.0, 100.0, 100.0), rows=rows)
        backend = FakeBackend([raw_page(1, text=PROSE, tables=[table])])
        return extract_document("x.pdf", "x", backend=backend).pages[0].tables[0]

    def test_a_dense_grid_is_retained(self):
        table = self._extract_one([["a", "b"], ["c", "d"], ["e", "f"]])
        assert table.retained
        assert table.rejected_reason is None

    def test_a_serial_number_column_does_not_make_it_a_contents_listing(self):
        # A column of bare numbers is the hallmark of a real schedule, not of an
        # arrangement-of-sections page.
        table = self._extract_one([
            ["1", "Regional Passport Officer", "Delhi"],
            ["2", "District Magistrate", "Assam"],
            ["3", "Collector", "Goa"],
        ])
        assert table.retained

    def test_a_contents_listing_is_rejected(self):
        table = self._extract_one([
            ["17A. Prohibition of picking of specified plant.", "x"],
            ["17B. Grants of permit for special purposes.", "y"],
            ["17C. Cultivation of specified plants prohibited.", "z"],
        ])
        assert not table.retained
        assert "arrangement-of-sections listing" in table.rejected_reason

    def test_a_footnote_block_is_rejected(self):
        table = self._extract_one([
            ["1. Ins. by Act 30 of 1965, s. 3", "w.e.f. 1-10-1967"],
            ["2. Subs. by Act 4 of 1941, s. 2", "w.e.f. 1-1-1942"],
            ["3. Omitted by Act 34 of 2019, s. 95", "w.e.f. 31-10-2019"],
        ])
        assert not table.retained
        assert "footnote block" in table.rejected_reason

    def test_a_column_that_is_nearly_empty_is_rejected(self):
        table = self._extract_one([
            ["Serial", "Name", ""],
            ["1", "Officer", ""],
            ["2", "Magistrate", ""],
            ["3", "Collector", "x"],
        ])
        assert not table.retained
        assert "populated" in table.rejected_reason

    def test_a_mostly_empty_grid_is_rejected_with_a_reason(self):
        table = self._extract_one(
            [["", "text", None], ["more", None, ""], ["", "", "x"]])
        assert not table.retained
        assert "cells contain text" in table.rejected_reason

    def test_a_two_row_candidate_is_rejected(self):
        table = self._extract_one([["a", "b"], ["c", "d"]])
        assert not table.retained
        assert "row(s)" in table.rejected_reason

    def test_a_single_column_grid_is_rejected(self):
        table = self._extract_one([["a"], ["b"], ["c"]])
        assert not table.retained
        assert "column" in table.rejected_reason

    def test_rejected_candidates_are_still_recorded(self):
        table = self._extract_one([["a"], ["b"], ["c"]])
        assert table.rows == [["a"], ["b"], ["c"]]
        assert table.detector == "pymupdf.find_tables"
