"""Page orientation: sideways pages, reversed writing direction, and rotation.

Every case here is drawn from something the 100-document benchmark actually
contains. The two that matter most:

* the Rajasthan Legislative Assembly Secretariat recruitment rules, page 19 —
  ``/Rotate 0`` and half the document's lines set at 90°, whose OCR layer
  extracts with the characters reversed ("Jo siseq oy3 uQ" for "On the basis
  of"); and
* the Madhya Pradesh Goods and Services Tax Act, 2017 — 64 of 309 pages in
  Devanagari behind a broken character map, extracting as reversed Latin
  nonsense while the document's pooled language check still says English.

Both were invisible before this module existed. These tests exist so they cannot
become invisible again.

Offline: the orientation logic works on writing directions and rotations, not on
pixels, so nothing here needs a PDF or an OCR engine.
"""

from __future__ import annotations

import pytest

from processing import config, extract, orientation
from processing.backends import RawDocument, RawPage

HORIZONTAL = (1.0, 0.0)
VERTICAL_DOWN = (0.0, 1.0)
VERTICAL_UP = (0.0, -1.0)
REVERSED = (-1.0, 0.0)


def directions(direction, count):
    return [direction] * count


class TestSidewaysDetection:
    """The axis test: is this page's text running vertically *on screen*?"""

    def test_an_ordinary_page_is_upright(self):
        result = orientation.assess_page(1, 0, directions(HORIZONTAL, 40))
        assert result.orientation == "upright"
        assert result.needs_rotation is False
        assert result.sideways_line_ratio == 0.0

    def test_vertical_lines_on_an_unrotated_page_are_sideways(self):
        """The Rajasthan page: /Rotate 0, lines at 90°."""
        result = orientation.assess_page(19, 0, directions(VERTICAL_DOWN, 89))
        assert result.orientation == "sideways"
        assert result.needs_rotation is True
        assert result.sideways_line_ratio == 1.0

    @pytest.mark.parametrize(
        "rotation,direction",
        [
            (90, VERTICAL_UP),      # skins-grading-and-marking-rules-1937
            (270, VERTICAL_DOWN),   # jserc-terms-conditions-for-distribution-tariff
        ],
    )
    def test_rotation_that_the_viewer_applies_makes_a_page_upright(
        self, rotation, direction
    ):
        """Vertical lines on a 90/270-rotated page display horizontally.

        Both of these documents render perfectly upright and extract perfectly
        readable text. Reading the writing direction without accounting for
        ``/Rotate`` would condemn them, and re-OCRing a rotated copy of a page
        that was already right would make them worse.
        """
        result = orientation.assess_page(1, rotation, directions(direction, 30))
        assert result.orientation == "upright"
        assert result.needs_rotation is False

    def test_a_180_degree_rotation_does_not_change_the_axis(self):
        """The Gujarat regulations declare /Rotate 180 and are upright."""
        result = orientation.assess_page(1, 180, directions(HORIZONTAL, 30))
        assert result.orientation == "upright"

    def test_a_rotated_page_with_horizontal_file_lines_is_sideways(self):
        result = orientation.assess_page(1, 90, directions(HORIZONTAL, 30))
        assert result.orientation == "sideways"

    def test_a_page_with_too_few_lines_is_not_judged(self):
        result = orientation.assess_page(1, 0, directions(VERTICAL_DOWN, 3))
        assert result.orientation == "unknown"
        assert result.needs_rotation is False
        assert result.source == "insufficient_lines"

    def test_a_page_with_no_text_layer_is_not_judged(self):
        """A scan's orientation is a question for OCR, not for line geometry."""
        result = orientation.assess_page(1, 0, [])
        assert result.orientation == "unknown"
        assert result.measured_lines == 0

    def test_a_half_sideways_page_is_neither(self):
        """A rotated table beside upright prose. Rotating it would ruin the prose."""
        mixture = directions(HORIZONTAL, 20) + directions(VERTICAL_DOWN, 20)
        result = orientation.assess_page(1, 0, mixture)
        assert result.orientation == "unknown"
        assert result.needs_rotation is False

    def test_the_declared_rotation_is_recorded_whatever_the_verdict(self):
        result = orientation.assess_page(1, 270, directions(VERTICAL_DOWN, 30))
        assert result.declared_rotation == 270
        assert result.to_dict()["declared_rotation"] == 270


class TestReversedWritingDirection:
    """Pages that run backwards relative to their own document."""

    def test_a_consistent_document_flags_nothing(self):
        pages = [orientation.assess_page(n, 0, directions(HORIZONTAL, 30))
                 for n in range(1, 6)]
        orientation.flag_direction_inconsistency(pages)
        assert not any(p.direction_inconsistent for p in pages)

    def test_a_document_consistently_reversed_flags_nothing(self):
        """No page disagrees with the others, so there is nothing to report.

        The sign of a writing direction is not trustworthy on its own — it does
        not survive ``/Rotate`` in a way this code can rely on — so only
        disagreement *within* a document is evidence.
        """
        pages = [orientation.assess_page(n, 0, directions(REVERSED, 30))
                 for n in range(1, 6)]
        orientation.flag_direction_inconsistency(pages)
        assert not any(p.direction_inconsistent for p in pages)

    def test_the_minority_direction_is_the_one_flagged(self):
        """The Madhya Pradesh GST Act: 2 of 6 pages behind a broken font."""
        pages = [orientation.assess_page(n, 0, directions(HORIZONTAL, 30))
                 for n in range(1, 5)]
        pages += [orientation.assess_page(n, 0, directions(REVERSED, 30))
                  for n in (5, 6)]
        orientation.flag_direction_inconsistency(pages)
        flagged = [p.page_number for p in pages if p.direction_inconsistent]
        assert flagged == [5, 6]

    def test_the_summary_counts_both_faults_separately(self):
        pages = [orientation.assess_page(1, 0, directions(VERTICAL_DOWN, 30))]
        pages += [orientation.assess_page(n, 0, directions(HORIZONTAL, 30))
                  for n in (2, 3)]
        pages += [orientation.assess_page(4, 0, directions(REVERSED, 30))]
        orientation.flag_direction_inconsistency(pages)
        summary = orientation.summarise(pages, page_count=4)
        assert summary["sideways_pages"] == 1
        assert summary["sideways_page_numbers"] == [1]
        assert summary["direction_inconsistent_pages"] == 1
        assert summary["direction_inconsistent_page_numbers"] == [4]
        assert summary["orientation_suspect"] is True

    def test_a_clean_document_is_not_suspect(self):
        pages = [orientation.assess_page(n, 0, directions(HORIZONTAL, 30))
                 for n in range(1, 21)]
        orientation.flag_direction_inconsistency(pages)
        summary = orientation.summarise(pages, page_count=20)
        assert summary["orientation_suspect"] is False
        assert summary["sideways_pages"] == 0


class TestExtractionIntegration:
    """Orientation reaches the extracted document, its pages and its warnings."""

    @staticmethod
    def _document(page_specs):
        pages = [
            RawPage(page_number=number, text="line\n" * 30, width=595.0,
                    height=842.0, rotation=rotation,
                    line_directions=directions(direction, 30))
            for number, (rotation, direction) in enumerate(page_specs, start=1)
        ]
        return RawDocument(page_count=len(pages), pages=pages, backend="stub")

    def _extract(self, monkeypatch, page_specs):
        raw = self._document(page_specs)

        class StubBackend:
            name = "stub"

            def read(self, path, *, detect_tables=True):
                return raw

        return extract.extract_document("ignored.pdf", "doc", backend=StubBackend())

    def test_sideways_pages_are_counted_and_warned_about(self, monkeypatch):
        document = self._extract(
            monkeypatch,
            [(0, VERTICAL_DOWN), (0, VERTICAL_DOWN), (0, HORIZONTAL)],
        )
        assert document.sideways_page_count == 2
        assert document.orientation["sideways_pages"] == 2
        assert any("sideways" in w for w in document.warnings)
        assert any("sideways" in w for w in document.pages[0].warnings)

    def test_an_upright_document_reports_no_orientation_problem(self, monkeypatch):
        document = self._extract(monkeypatch, [(0, HORIZONTAL)] * 4)
        assert document.sideways_page_count == 0
        assert document.direction_inconsistent_page_count == 0
        assert not any("sideways" in w for w in document.warnings)

    def test_reversed_pages_are_counted_and_warned_about(self, monkeypatch):
        document = self._extract(
            monkeypatch,
            [(0, HORIZONTAL)] * 4 + [(0, REVERSED)],
        )
        assert document.direction_inconsistent_page_count == 1
        assert any("writing direction" in w for w in document.warnings)
        assert any("character map" in w for w in document.pages[4].warnings)
        assert not any("character map" in w for w in document.pages[0].warnings)

    def test_orientation_is_serialised_with_the_page(self, monkeypatch):
        document = self._extract(monkeypatch, [(90, VERTICAL_UP)] * 3)
        page = document.pages[0].to_dict()
        assert page["orientation"]["declared_rotation"] == 90
        assert page["orientation"]["orientation"] == "upright"

    def test_page_text_is_never_rewritten(self, monkeypatch):
        """Detection labels; it does not touch the extracted text."""
        document = self._extract(monkeypatch, [(0, VERTICAL_DOWN)] * 3)
        assert document.pages[0].text == "line\n" * 30


class TestOcrRouting:
    """A sideways document is routed to rotate-then-OCR before anything else."""

    def test_sideways_pages_demand_rotation_first(self, monkeypatch):
        raw = TestExtractionIntegration._document([(0, VERTICAL_DOWN)] * 3)

        class StubBackend:
            name = "stub"

            def read(self, path, *, detect_tables=True):
                return raw

        from processing import ocr, quality

        document = extract.extract_document("x.pdf", "doc", backend=StubBackend())
        decision = ocr.decide(document, quality.assess(document.pages))
        assert decision.action == "rasterize_then_ocr"
        assert decision.preprocessing == ["rotate_upright"]
        assert decision.estimated_pages == 3


class TestImageOrientation:
    """The pixel-level half, which needs an OCR engine and says so when absent."""

    def test_an_unavailable_detector_is_reported_not_assumed(self, monkeypatch):
        """"No rotation needed" and "nobody looked" must not be the same value."""
        monkeypatch.setattr(orientation, "_configured_pytesseract", lambda: None)
        result = orientation.detect_image_rotation(b"not-an-image")
        assert result.known is False
        assert result.rotate_degrees == 0
        assert "orientation detection" in result.detail

    def test_an_image_is_left_alone_when_nothing_could_be_detected(self, monkeypatch):
        monkeypatch.setattr(orientation, "_configured_pytesseract", lambda: None)
        image = b"png-bytes"
        result, detected = orientation.upright_image(image)
        assert result is image
        assert detected.known is False

    def test_a_low_confidence_detection_is_not_acted_on(self, monkeypatch):
        monkeypatch.setattr(
            orientation, "detect_image_rotation",
            lambda image: orientation.ImageOrientation(
                rotate_degrees=90, confidence=0.2, source="tesseract_osd"),
        )
        image = b"png-bytes"
        result, detected = orientation.upright_image(image)
        assert result is image
        assert "below" in detected.detail

    def test_a_confident_detection_rotates_the_image(self, monkeypatch):
        monkeypatch.setattr(
            orientation, "detect_image_rotation",
            lambda image: orientation.ImageOrientation(
                rotate_degrees=90, confidence=12.0, source="tesseract_osd"),
        )
        monkeypatch.setattr(orientation, "rotate_image",
                            lambda image, degrees: b"rotated")
        result, detected = orientation.upright_image(b"png-bytes")
        assert result == b"rotated"
        assert detected.rotate_degrees == 90

    def test_zero_degrees_is_not_a_rotation(self, monkeypatch):
        monkeypatch.setattr(
            orientation, "detect_image_rotation",
            lambda image: orientation.ImageOrientation(
                rotate_degrees=0, confidence=12.0, source="tesseract_osd"),
        )
        image = b"png-bytes"
        result, _ = orientation.upright_image(image)
        assert result is image

    def test_rotating_by_zero_returns_the_same_bytes(self):
        image = b"png-bytes"
        assert orientation.rotate_image(image, 0) is image


class TestThresholds:
    """The thresholds are part of the contract; a silent change is a regression."""

    def test_thresholds_are_recorded_in_the_summary(self):
        summary = orientation.summarise([], page_count=0)
        assert summary["thresholds"] == {
            "sideways_line_ratio": config.ORIENTATION_SIDEWAYS_LINE_RATIO,
            "suspect_page_ratio": config.ORIENTATION_SUSPECT_PAGE_RATIO,
            "min_lines": config.ORIENTATION_MIN_LINES,
        }


class TestOnlyPagesWithTextVote:
    """A page with no horizontal text has no opinion about which way is forward.

    Without this, a scanned gazette of two hundred image-only pages would outvote
    its three text pages and flag them as inconsistent on no evidence at all.
    """

    def test_pages_with_no_text_do_not_outvote_pages_with_text(self):
        blanks = [orientation.assess_page(n, 0, []) for n in range(1, 21)]
        text = [orientation.assess_page(n, 0, directions(REVERSED, 30))
                for n in (21, 22, 23)]
        orientation.flag_direction_inconsistency(blanks + text)
        assert not any(p.direction_inconsistent for p in blanks + text)

    def test_sideways_pages_do_not_vote_either(self):
        sideways = [orientation.assess_page(n, 0, directions(VERTICAL_DOWN, 30))
                    for n in range(1, 11)]
        text = [orientation.assess_page(n, 0, directions(REVERSED, 30))
                for n in (11, 12)]
        orientation.flag_direction_inconsistency(sideways + text)
        assert not any(p.direction_inconsistent for p in sideways + text)

    def test_a_page_records_how_many_lines_it_voted_with(self):
        page = orientation.assess_page(1, 0, directions(HORIZONTAL, 30))
        assert page.horizontal_lines == 30
        assert page.to_dict()["horizontal_lines"] == 30

    def test_a_sideways_page_contributes_no_votes(self):
        page = orientation.assess_page(1, 0, directions(VERTICAL_DOWN, 30))
        assert page.horizontal_lines == 0

    def test_the_minority_is_still_flagged_among_pages_that_do_vote(self):
        blanks = [orientation.assess_page(n, 0, []) for n in range(1, 6)]
        forward = [orientation.assess_page(n, 0, directions(HORIZONTAL, 30))
                   for n in (6, 7, 8, 9)]
        reverse = [orientation.assess_page(n, 0, directions(REVERSED, 30))
                   for n in (10, 11)]
        orientation.flag_direction_inconsistency(blanks + forward + reverse)
        flagged = [p.page_number
                   for p in blanks + forward + reverse if p.direction_inconsistent]
        assert flagged == [10, 11]


class TestProportionateRouting:
    """A document with a sideways *page* is not a sideways *document*.

    The Code of Civil Procedure, 1908 has two landscape schedules among 347
    pages. Routing a sound born-digital Central Act through OCR because of them
    would be both expensive and a downgrade in text quality.
    """

    @staticmethod
    def _decide(page_specs):
        from processing import ocr, quality

        raw = TestExtractionIntegration._document(page_specs)

        class StubBackend:
            name = "stub"

            def read(self, path, *, detect_tables=True):
                return raw

        document = extract.extract_document("x.pdf", "doc", backend=StubBackend())
        return document, ocr.decide(document, quality.assess(document.pages))

    def test_a_few_sideways_pages_do_not_reroute_the_document(self):
        specs = [(0, HORIZONTAL)] * 38 + [(0, VERTICAL_DOWN)] * 2
        document, decision = self._decide(specs)
        assert document.sideways_page_count == 2
        assert document.orientation["orientation_suspect"] is False
        assert decision.action != "rasterize_then_ocr"
        assert decision.preprocessing == []

    def test_but_they_are_still_named_in_the_decision(self):
        """Proportionate is not silent: the pages still have to be findable."""
        specs = [(0, HORIZONTAL)] * 38 + [(0, VERTICAL_DOWN)] * 2
        _, decision = self._decide(specs)
        assert "2 of 40 pages are set sideways" in decision.reason

    def test_a_substantially_sideways_document_is_rerouted(self):
        specs = [(0, HORIZONTAL)] * 20 + [(0, VERTICAL_DOWN)] * 20
        document, decision = self._decide(specs)
        assert document.orientation["orientation_suspect"] is True
        assert decision.action == "rasterize_then_ocr"
        assert decision.preprocessing == ["rotate_upright"]
        assert decision.estimated_pages == 20

    def test_an_upright_document_says_nothing_about_orientation(self):
        _, decision = self._decide([(0, HORIZONTAL)] * 40)
        assert "sideways" not in decision.reason
        assert decision.preprocessing == []


class TestEligibilityGate:
    """Orientation is a third, independent gate on indexing.

    Not redundant with extraction quality, although on the benchmark sample it
    happens to be: quality is measured over the whole document, so a 300-page act
    with a 40-page sideways section keeps a respectable average while a seventh
    of it is reversed characters.
    """

    @staticmethod
    def _result(orientation_summary):
        from processing.models import ExtractedDocument, ProcessedDocument

        extraction = ExtractedDocument(
            document_id="doc", page_count=10, pages=[], pdf_type="text_based",
            text_extraction_status="ok", orientation=orientation_summary,
        )
        return ProcessedDocument(
            document=None,                              # not read by the property
            extraction=extraction,
            language=type("L", (), {"eligible_for_indexing": True})(),
            quality=type("Q", (), {"classification": "good"})(),
            ok=True,
        )

    def test_good_english_upright_text_is_eligible(self):
        assert self._result({"orientation_suspect": False}).eligible_for_indexing

    def test_a_substantially_sideways_document_is_quarantined(self):
        """Even with English content and text the quality panel calls good."""
        assert not self._result({"orientation_suspect": True}).eligible_for_indexing

    def test_a_document_with_no_orientation_data_is_not_penalised(self):
        assert self._result({}).eligible_for_indexing
