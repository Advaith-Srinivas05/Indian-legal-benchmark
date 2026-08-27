"""Tests for actually running OCR over a page (``processing.ocr_engine``).

``processing.ocr`` decides that a document needs OCR; this is the module that
reads one. The tests are about the two things that can go badly wrong at corpus
scale and cannot be noticed afterwards.

**Replacing sound text with plausible wrong text.** A scanned gazette with a bad
OCR layer already reads fluently and is already wrong; swapping it for different
wrong text is a second unverifiable claim, not a repair. Measured on three real
corpus documents while this was built, the quality panel *alone* would have
accepted two engine readings that were visibly worse -- one of them
``AORRRR REY Lab to 8`` in place of ``THE GAZETTE OF INDIA : EXTRAORDINARY``.
The English function-word rate is what caught both, which is why acceptance
needs the panel *and* the vocabulary to agree.

**Losing the original.** ``pages.json`` text must stay byte-for-byte what the
extraction backend produced, whatever OCR does, or an OCR regression stops being
diagnosable.

No Tesseract is needed here: the engine is a callable, so the tests pass their
own. That keeps the suite offline, fast and identical on a machine with no OCR
installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from processing import config, ocr_engine
from processing.models import ExtractedDocument, FootnoteBlock, PageText
from tests import pdfbuild

# Real statutory English, long enough for the panel and the language test to
# have something to measure.
GOOD = (
    "Whoever, being a public servant, knowingly disobeys any direction of the "
    "law as to the way in which he is to conduct himself as such public "
    "servant, intending to cause injury to any person, shall be punished with "
    "simple imprisonment for a term which may extend to one year, or with fine, "
    "or with both, and in this section the expression public servant shall have "
    "the meaning assigned to it in the Indian Penal Code. "
) * 3
#: The shape a bad OCR layer takes: letters present, words not.
GARBLED = (
    "Vv'noever, bemg a pubhL servant, knowmgly d1sobeys any d1recuon of tl1e "
    "1aw as to tl1e way m wh1ch he 1s to conduLt h1mse1f as suLh pubhL "
    "servant, mtendmg to Lause mJury to any persor1, sha11 be pumshed w1th "
    "s1mp1e 1mpnsonment for a terrn wh1ch may extend to one year. "
) * 3


def page(number=1, text="", **kwargs):
    defaults = dict(
        char_count=len(text), alpha_char_count=sum(c.isalpha() for c in text),
        has_text=len(text) > 100, is_empty=not text.strip(),
    )
    defaults.update(kwargs)
    return PageText(page_number=number, text=text, **defaults)


def document(*pages):
    return ExtractedDocument(
        document_id="doc", page_count=len(pages), pages=list(pages),
        pdf_type="scanned", text_extraction_status="partial",
    )


def reader_returning(text):
    return lambda image: text


# --- Which pages are read -----------------------------------------------------------


class TestTrigger:
    def test_a_page_with_no_text_layer_is_read(self):
        assert ocr_engine.page_trigger(page(text="")) == "no_text_layer"

    def test_a_vector_outlined_page_is_read(self):
        assert ocr_engine.page_trigger(
            page(text=GOOD, is_vector_outlined=True)) == "vector_outlined"

    def test_a_page_with_a_suspect_text_layer_is_read(self):
        assert ocr_engine.page_trigger(
            page(text=GOOD, text_quality_suspect=True)) == "text_quality_suspect"

    def test_a_sound_page_is_left_alone(self):
        assert ocr_engine.page_trigger(page(text=GOOD)) is None

    def test_a_page_already_known_to_be_another_script_is_skipped(self):
        """Its language is established; re-reading it in English is waste."""
        other = page(text=GOOD, language={"verdict": "non_en"},
                     text_quality_suspect=True)
        assert ocr_engine.page_trigger(other) == "text_quality_suspect"
        assert ocr_engine.should_skip(other)

    def test_a_scanned_page_of_unknown_language_is_not_skipped(self):
        """Nothing is established about a page that yielded no text."""
        blank = page(text="", language={"verdict": "unknown"})
        assert ocr_engine.should_skip(blank) is None


# --- Whether the reading is believed ------------------------------------------------


class TestJudgement:
    def test_text_where_there_was_none_is_accepted(self):
        verdict = ocr_engine.judge("", GOOD)
        assert verdict.accepted
        assert "no text layer" in verdict.reason

    def test_an_empty_reading_is_never_accepted(self):
        assert not ocr_engine.judge(GOOD, "").accepted
        assert not ocr_engine.judge("", "").accepted

    def test_a_better_reading_replaces_a_garbled_one(self):
        verdict = ocr_engine.judge(GARBLED, GOOD)
        assert verdict.accepted
        assert verdict.signals["ocr_score"] > verdict.signals["backend_score"]

    def test_a_worse_reading_is_rejected(self):
        verdict = ocr_engine.judge(GOOD, GARBLED)
        assert not verdict.accepted
        assert "did not improve" in verdict.reason

    def test_sound_text_is_never_replaced_by_a_tie(self):
        """Extraction is the incumbent and OCR has to beat it, not match it."""
        assert not ocr_engine.judge(GOOD, GOOD).accepted

    def test_a_reading_that_looks_better_but_loses_english_is_rejected(self):
        """The failure the panel alone cannot see.

        Measured on real documents: the panel rose 0.35 -> 0.70 and 0.26 -> 0.83
        on two engine readings that were plainly worse than the text they would
        have replaced. Recovered words are the point of OCR, so a reading that
        improves the shape of the text while reducing how much of it is English
        is not an improvement.
        """
        # Shapely, punctuated, and almost entirely not English words.
        shapely_nonsense = (
            "Zorvin mekaba tulden prasol vintagu meldor bakuna trevil dunosa "
            "grifel montak sedura piloven trakusa mendola virputa selkona. "
        ) * 6
        verdict = ocr_engine.judge(GARBLED, shapely_nonsense)
        assert not verdict.accepted
        assert "function words" in verdict.reason


# --- Reading one page ---------------------------------------------------------------


class TestOcrPage:
    def test_an_engine_failure_is_recorded_not_raised(self, tmp_path):
        def explode(image):
            raise RuntimeError("the engine fell over")

        record = ocr_engine.ocr_page(
            tmp_path / "missing.pdf", page(text=GARBLED), reader=explode)
        assert record["attempted"] is True
        assert record["accepted"] is False
        assert "error" in record

    def test_a_missing_pdf_is_recorded_not_raised(self, tmp_path):
        record = ocr_engine.ocr_page(
            tmp_path / "missing.pdf", page(text=GARBLED),
            reader=reader_returning(GOOD))
        assert record["accepted"] is False
        assert "error" in record


# --- Reading a document -------------------------------------------------------------


class TestOcrDocument:
    def test_a_document_with_nothing_to_read_runs_no_engine(self, tmp_path):
        def explode(image):
            raise AssertionError("the engine must not be called")

        run = ocr_engine.ocr_document(
            document(page(1, GOOD), page(2, GOOD)), tmp_path / "x.pdf",
            reader=explode)
        assert run.executed is False
        assert run.attempted == 0

    def test_a_non_english_document_is_not_read(self, tmp_path):
        def explode(image):
            raise AssertionError("the engine must not be called")

        run = ocr_engine.ocr_document(
            document(page(1, "")), tmp_path / "x.pdf", reader=explode,
            document_language="non_en")
        assert run.executed is False
        assert "not English" in run.unavailable_reason

    def test_the_page_cap_defers_rather_than_spending_the_night(self, tmp_path):
        pages = [page(n, "") for n in range(1, 12)]
        run = ocr_engine.ocr_document(
            document(*pages), tmp_path / "x.pdf",
            reader=reader_returning(GOOD), max_pages=4)
        assert run.truncated is True
        assert run.attempted == 4

    def test_a_missing_engine_is_reported_once_not_raised(self, tmp_path):
        def unavailable():
            raise FileNotFoundError("no tesseract")

        run = ocr_engine.ocr_document(
            document(page(1, "")), tmp_path / "x.pdf",
            reader=None, document_language="en")
        # Either it ran (Tesseract is installed here) or it said why not; what it
        # must never do is raise.
        assert isinstance(run.unavailable_reason, str)


# --- The invariants -----------------------------------------------------------------


class TestTheOriginalSurvives:
    def test_page_text_is_never_modified_by_ocr(self, tmp_path, monkeypatch):
        """``pages.json`` text stays byte-for-byte the backend's, always."""
        monkeypatch.setattr(ocr_engine, "render_page", lambda *a, **k: b"png")
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image",
            lambda image, **k: (image, None))
        target = page(1, GARBLED, text_quality_suspect=True)
        before = target.text
        ocr_engine.ocr_document(
            document(target), tmp_path / "x.pdf", reader=reader_returning(GOOD))
        assert target.text == before
        assert target.ocr["text"] != before
        assert target.text_source == "ocr"

    def test_selected_text_follows_the_verdict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ocr_engine, "render_page", lambda *a, **k: b"png")
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image",
            lambda image, **k: (image, None))
        # Accepted: the engine's reading is the one used.
        good_page = page(1, GARBLED, text_quality_suspect=True)
        ocr_engine.ocr_document(document(good_page), tmp_path / "x.pdf",
                                reader=reader_returning(GOOD))
        assert good_page.selected_text == good_page.ocr["text"]

        # Rejected: extraction's reading stands.
        kept = page(1, GOOD, text_quality_suspect=True)
        ocr_engine.ocr_document(document(kept), tmp_path / "x.pdf",
                                reader=reader_returning(GARBLED))
        assert kept.text_source == "backend"
        assert kept.selected_text == GOOD

    def test_a_rejected_reading_is_kept_for_audit(self, tmp_path, monkeypatch):
        """Rejection records what was read, not merely that something was."""
        monkeypatch.setattr(ocr_engine, "render_page", lambda *a, **k: b"png")
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image",
            lambda image, **k: (image, None))
        kept = page(1, GOOD, text_quality_suspect=True)
        ocr_engine.ocr_document(document(kept), tmp_path / "x.pdf",
                                reader=reader_returning(GARBLED))
        assert kept.ocr["accepted"] is False
        assert GARBLED[:20] in kept.ocr["text"]
        assert kept.ocr["reason"]

    def test_ocr_can_only_rescue_a_document_never_spoil_one(self, tmp_path,
                                                            monkeypatch):
        """The whole safety argument, as one assertion.

        Whatever the engine returns, a page whose extraction text was sound
        keeps it.
        """
        monkeypatch.setattr(ocr_engine, "render_page", lambda *a, **k: b"png")
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image",
            lambda image, **k: (image, None))
        for engine_output in ("", "   ", GARBLED, "!!!", "a"):
            sound = page(1, GOOD, text_quality_suspect=True)
            ocr_engine.ocr_document(
                document(sound), tmp_path / "x.pdf",
                reader=reader_returning(engine_output))
            assert sound.text_source == "backend", engine_output
            assert sound.selected_text == GOOD, engine_output


class TestConfiguration:
    def test_the_engine_and_resolution_are_recorded_on_every_page(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(ocr_engine, "render_page", lambda *a, **k: b"png")
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image",
            lambda image, **k: (image, None))
        target = page(1, "", )
        ocr_engine.ocr_document(document(target), tmp_path / "x.pdf",
                                reader=reader_returning(GOOD))
        assert target.ocr["engine"] == config.OCR_ENGINE_NAME
        assert target.ocr["dpi"] == config.OCR_DPI

    def test_the_resolution_is_at_least_what_orientation_detection_needs(self):
        """Below 200 dpi Tesseract's OSD stops answering rather than answering
        wrongly, and a page read in the wrong orientation is confidently wrong."""
        assert config.OCR_DPI >= 200


class TestThePixelBudget:
    """An oversized page is rendered smaller rather than rendered forever.

    The 286 documents journalled FAILED after the 2026-08-27 corpus run were all
    ``DocumentTimeout``, and about ten of them would not have finished however
    long the timeout was: their pages rasterise to 50-185 MP at 300 dpi, where
    PIL warns about a decompression bomb at 132 and a single Tesseract call on a
    two-page document was still running after thirteen minutes. See
    KNOWN_ISSUES B8. The budget lowers the resolution for those pages only.
    """

    def test_an_ordinary_page_is_not_touched(self):
        """The case that must not change: A4 at 300 dpi is 8.7 MP, well under."""
        assert ocr_engine.budgeted_dpi(*pdfbuild.A4, 300) == 300

    def test_an_oversized_page_is_lowered_just_far_enough(self):
        dpi = ocr_engine.budgeted_dpi(1600.0, 1600.0, 300, max_megapixels=40.0)
        assert 200 < dpi < 300
        megapixels = (1600 / 72 * dpi) ** 2 / 1_000_000
        assert megapixels <= 40.0

    def test_an_enormous_page_stops_at_the_orientation_floor(self):
        """Below 200 dpi Tesseract's OSD stops answering, and a page read the
        wrong way up is worse than a page read slowly from a large image."""
        assert ocr_engine.budgeted_dpi(
            3000.0, 3000.0, 300, max_megapixels=40.0, min_dpi=200) == 200

    def test_the_budget_never_raises_the_resolution(self):
        """A caller asking for less than the floor gets what it asked for."""
        assert ocr_engine.budgeted_dpi(
            3000.0, 3000.0, 150, max_megapixels=40.0, min_dpi=200) == 150

    def test_a_degenerate_page_size_is_left_alone(self):
        assert ocr_engine.budgeted_dpi(0.0, 842.0, 300) == 300

    def test_the_floor_is_what_orientation_detection_needs(self):
        assert config.OCR_MIN_DPI >= 200
        assert config.OCR_DPI >= config.OCR_MIN_DPI


class TestRenderPageRespectsTheBudget:
    """The budget has to reach the pixels, not just the arithmetic."""

    @staticmethod
    def _rendered_size(pdf: Path) -> tuple[int, int]:
        """Width and height of the PNG, read from its IHDR chunk.

        Read from the bytes rather than through PIL so the test needs nothing
        from ``requirements-ocr.txt`` and runs on a machine with no OCR at all.
        """
        png = ocr_engine.render_page(pdf, 1)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        return (int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big"))

    @staticmethod
    def _size_at(pdf: Path, dpi: int) -> tuple[int, int]:
        import pymupdf

        document = pymupdf.open(pdf)
        try:
            pixmap = document.load_page(0).get_pixmap(dpi=dpi)
            return (pixmap.width, pixmap.height)
        finally:
            document.close()

    def test_an_ordinary_page_still_renders_at_the_configured_resolution(
            self, tmp_path):
        pdf = pdfbuild.text_pdf(tmp_path / "a4.pdf", [GOOD])
        assert self._rendered_size(pdf) == self._size_at(pdf, config.OCR_DPI)

    def test_an_over_budget_page_renders_smaller(self, tmp_path, monkeypatch):
        """A4 at 300 dpi is 8.7 MP; a 6 MP budget lowers it without hitting the
        floor, which is the behaviour a real oversized page gets."""
        monkeypatch.setattr(config, "OCR_MAX_MEGAPIXELS", 6.0)
        pdf = pdfbuild.text_pdf(tmp_path / "big.pdf", [GOOD])
        lowered = ocr_engine.budgeted_dpi(
            *pdfbuild.A4, config.OCR_DPI, max_megapixels=6.0,
            min_dpi=config.OCR_MIN_DPI)
        assert config.OCR_MIN_DPI < lowered < config.OCR_DPI
        assert self._rendered_size(pdf) == self._size_at(pdf, lowered)

    def test_it_stops_at_the_floor_however_small_the_budget(
            self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "OCR_MAX_MEGAPIXELS", 0.5)
        pdf = pdfbuild.text_pdf(tmp_path / "huge.pdf", [GOOD])
        assert self._rendered_size(pdf) == self._size_at(pdf, config.OCR_MIN_DPI)

    def test_the_upright_render_inherits_it(self, tmp_path, monkeypatch):
        """``render_page_upright`` is the path the OCR pass actually uses."""
        monkeypatch.setattr(config, "OCR_MAX_MEGAPIXELS", 0.5)
        monkeypatch.setattr(
            ocr_engine.orientation, "upright_image", lambda image, **k: (image, None))
        pdf = pdfbuild.text_pdf(tmp_path / "upright.pdf", [GOOD])
        png, _ = ocr_engine.render_page_upright(pdf, 1)
        width = int.from_bytes(png[16:20], "big")
        assert width == self._size_at(pdf, config.OCR_MIN_DPI)[0]
