"""Tests for vector-outlined PDFs and the OCR decision pipeline.

``vector_outlined`` is a fourth PDF type added after the first benchmark, where
the UP Stamp Rules, 1942 was classified ``text_based`` with two thirds of its
pages reported as empty. Its glyphs had been converted to vector outlines: no
text layer, no image, about a thousand drawing paths per page. It is worse than
a scan, because plain OCR has nothing to read until the page is rendered.

The OCR pipeline tests hold the distinction the first benchmark collapsed — a
scan with a *good* OCR layer must not be re-OCR'd, a scan with a *bad* one must
be, and a scan with none is a third thing again.
"""

from __future__ import annotations

import pytest

from processing import config, ocr, quality
from processing.extract import extract_document
from processing.models import ExtractedDocument, PageText
from tests import pdfbuild
from tests.test_processing_extract import PROSE, FakeBackend, full_page_image, raw_page


class TestVectorOutlinedDetection:
    def test_a_wholly_outlined_document_is_classified_as_such(self, tmp_path):
        path = pdfbuild.vector_outlined_pdf(tmp_path / "outlined.pdf", page_count=4)
        extracted = extract_document(path, "outlined")
        assert extracted.pdf_type == "vector_outlined"
        assert extracted.vector_outlined_page_count == 4
        assert all(p.is_vector_outlined for p in extracted.pages)

    def test_it_is_not_called_text_based_just_because_there_is_no_image(self, tmp_path):
        # The exact regression: image coverage is zero on these pages, and the
        # old rule read zero image coverage as "born digital".
        path = pdfbuild.vector_outlined_pdf(tmp_path / "outlined.pdf", page_count=4)
        extracted = extract_document(path, "outlined")
        assert extracted.classification_evidence["image_backed_pages"] == 0
        assert extracted.pdf_type != "text_based"

    def test_the_pages_say_why(self, tmp_path):
        path = pdfbuild.vector_outlined_pdf(tmp_path / "outlined.pdf", page_count=3)
        extracted = extract_document(path, "outlined")
        assert any("vector outlines" in w for w in extracted.pages[0].warnings)
        assert any("rasterised" in w for w in extracted.warnings)
        assert extracted.pages[0].drawing_path_count >= config.VECTOR_OUTLINE_MIN_PATHS

    def test_a_mostly_outlined_document_still_counts(self, tmp_path):
        # The UP Stamp Rules shape: 97 outlined pages of 148, the rest typeset.
        path = pdfbuild.vector_outlined_pdf(
            tmp_path / "mixed.pdf", page_count=6, text_pages=[PROSE, PROSE, PROSE])
        extracted = extract_document(path, "mixed")
        assert extracted.pdf_type == "vector_outlined"
        assert extracted.vector_outlined_page_count == 6

    def test_a_blank_page_is_not_vector_outlined(self, tmp_path):
        path = pdfbuild.blank_pdf(tmp_path / "blank.pdf", page_count=3)
        extracted = extract_document(path, "blank")
        assert extracted.pdf_type != "vector_outlined"
        assert not any(p.is_vector_outlined for p in extracted.pages)

    def test_a_ruled_form_with_text_is_not_vector_outlined(self, tmp_path):
        # Drawing paths alone are not the signal; the absence of text is.
        rows = [["Serial", "Name", "Area"], ["1", "Officer", "Delhi"],
                ["2", "Magistrate", "Assam"], ["3", "Collector", "Goa"]]
        path = pdfbuild.table_pdf(tmp_path / "form.pdf", rows, heading=PROSE[:200])
        extracted = extract_document(path, "form")
        assert not any(p.is_vector_outlined for p in extracted.pages)

    def test_a_scanned_page_is_not_vector_outlined(self, tmp_path):
        path = pdfbuild.scanned_pdf(tmp_path / "scan.pdf", page_count=3)
        extracted = extract_document(path, "scan")
        assert extracted.pdf_type == "scanned"
        assert extracted.vector_outlined_page_count == 0


def _document(pages, pdf_type, status, **evidence) -> ExtractedDocument:
    return ExtractedDocument(
        document_id="x",
        page_count=len(pages),
        pages=pages,
        pdf_type=pdf_type,
        text_extraction_status=status,
        classification_evidence={"pages_with_text": len(pages), **evidence},
    )


def _page(number: int, text: str = "", **kwargs) -> PageText:
    return PageText(page_number=number, text=text, char_count=len(text), **kwargs)


class TestTextSourceStates:
    """The three states hiding inside "scanned"."""

    def test_scan_with_good_ocr(self):
        pages = [_page(1, PROSE * 6)]
        document = _document(pages, "scanned", "ok")
        assessment = quality.assess(pages)
        assert assessment.classification == "good"
        assert ocr.text_source(document, assessment) == "scan_with_good_ocr"

    def test_scan_with_bad_ocr(self):
        from tests.test_processing_quality import NFSA_OCR
        pages = [_page(1, NFSA_OCR)]
        document = _document(pages, "scanned", "ok")
        assessment = quality.assess(pages)
        assert ocr.text_source(document, assessment) == "scan_with_bad_ocr"

    def test_scan_without_ocr(self):
        pages = [_page(1, "")]
        document = _document(pages, "scanned", "requires_ocr", pages_with_text=0)
        assert ocr.text_source(document, quality.assess(pages)) == "scan_without_ocr"

    def test_born_digital(self):
        pages = [_page(1, PROSE * 6)]
        document = _document(pages, "text_based", "ok")
        assert ocr.text_source(document, quality.assess(pages)) == "born_digital"

    def test_vector_outlined(self):
        pages = [_page(1, "", is_vector_outlined=True)]
        document = _document(pages, "vector_outlined", "requires_ocr", pages_with_text=0)
        assert ocr.text_source(document, quality.assess(pages)) == "vector_outlined"


class TestOcrDecisions:
    def test_good_text_is_left_alone(self):
        pages = [_page(1, PROSE * 6)]
        document = _document(pages, "text_based", "ok")
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "use_extracted_text"
        assert decision.estimated_pages == 0

    def test_a_good_ocr_layer_is_not_re_ocred(self):
        # The point of separating pdf_type from text quality: re-OCRing a sound
        # layer costs time and would probably make it worse.
        pages = [_page(1, PROSE * 6)]
        document = _document(pages, "scanned", "ok")
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "use_extracted_text"
        assert decision.text_source == "scan_with_good_ocr"

    def test_a_bad_ocr_layer_is_queued(self):
        from tests.test_processing_quality import NFSA_OCR
        pages = [_page(1, NFSA_OCR)]
        document = _document(pages, "scanned", "ok")
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "ocr_recommended"
        assert decision.estimated_pages == 1

    def test_no_text_layer_requires_ocr(self):
        pages = [_page(n, "") for n in range(1, 4)]
        document = _document(pages, "scanned", "requires_ocr", pages_with_text=0)
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "ocr_required"
        assert decision.estimated_pages == 3

    def test_vector_outlined_must_be_rasterised_first(self):
        pages = [_page(n, "", is_vector_outlined=True) for n in range(1, 5)]
        document = _document(pages, "vector_outlined", "requires_ocr",
                             pages_with_text=0)
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "rasterize_then_ocr"
        assert "rendered" in decision.reason

    def test_partial_extraction_queues_only_the_missing_pages(self):
        pages = [_page(1, PROSE * 6), _page(2, ""), _page(3, "")]
        document = _document(pages, "mixed", "partial", pages_with_text=1)
        decision = ocr.decide(document, quality.assess(pages))
        assert decision.action == "ocr_recommended"
        assert decision.estimated_pages == 2

    def test_a_non_english_document_is_quarantined_not_queued(self):
        from processing.language import LanguageAssessment
        from tests.test_processing_quality import SHATTERED
        pages = [_page(1, SHATTERED)]
        document = _document(pages, "scanned", "ok")
        decision = ocr.decide(
            document, quality.assess(pages),
            language=LanguageAssessment("non_en", False),
        )
        assert decision.action == "review_text"
        assert decision.estimated_pages == 0
        assert "not English" in decision.reason

    def test_every_action_is_from_the_documented_vocabulary(self):
        pages = [_page(1, PROSE * 6)]
        for pdf_type in config.PDF_TYPES:
            for status in config.TEXT_EXTRACTION_STATUSES:
                document = _document(pages, pdf_type, status)
                decision = ocr.decide(document, quality.assess(pages))
                assert decision.action in ocr.ACTIONS
                assert decision.text_source in ocr.TEXT_SOURCES
                assert decision.reason


class TestOcrSummary:
    def test_counts_documents_and_pages_needing_ocr(self):
        decisions = [
            ocr.OcrDecision("use_extracted_text", "born_digital", "", 0, 0),
            ocr.OcrDecision("ocr_required", "scan_without_ocr", "", 4, 27),
            ocr.OcrDecision("ocr_recommended", "scan_with_bad_ocr", "", 3, 18),
            ocr.OcrDecision("rasterize_then_ocr", "vector_outlined", "", 5, 97),
            ocr.OcrDecision("review_text", "born_digital", "", 1, 0),
        ]
        summary = ocr.summarise(decisions)
        assert summary["documents_needing_ocr"] == 3
        assert summary["pages_needing_ocr"] == 142
        assert summary["by_action"]["ocr_required"] == 1

    def test_an_empty_run_summarises_to_zero(self):
        summary = ocr.summarise([])
        assert summary["documents_needing_ocr"] == 0
        assert summary["pages_needing_ocr"] == 0
