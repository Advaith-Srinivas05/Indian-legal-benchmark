"""Tests for the pure text helpers underneath extraction and parsing."""

from __future__ import annotations

from processing.textutils import (
    alpha_count,
    is_blank,
    is_page_number_line,
    normalise_line,
    quality_signals,
    split_lines,
    union_area,
)


class TestNormalisation:
    def test_case_and_whitespace_are_folded(self):
        assert normalise_line("  THE   Gazette of India ") == "the gazette of india"

    def test_digit_runs_fold_so_page_numbers_compare_equal(self):
        # This is what lets one repetition rule catch running heads *and* page
        # numbers: "Page 12" and "Page 13" must normalise to the same thing.
        assert normalise_line("Page 12") == normalise_line("Page 13")
        assert normalise_line("12") == normalise_line("7")

    def test_unicode_is_folded_before_comparison(self):
        assert normalise_line("ﬁrst") == normalise_line("first")

    def test_blank_line_normalises_to_empty(self):
        assert normalise_line("   \t ") == ""


class TestPageNumberLines:
    def test_recognises_bare_and_decorated_numbers(self):
        for line in ("7", " 12 ", "Page 3", "3 of 40", "iv", "12.", "5/40"):
            assert is_page_number_line(line), line

    def test_does_not_treat_a_numbered_heading_as_a_page_number(self):
        assert not is_page_number_line("12. Punishment for murder")
        assert not is_page_number_line("(1) This Act may be called")
        assert not is_page_number_line("")


class TestCounting:
    def test_alpha_count_ignores_punctuation_and_digits(self):
        assert alpha_count("...123... abc") == 3

    def test_is_blank(self):
        assert is_blank("   \n\t ")
        assert not is_blank(" x ")

    def test_split_lines_keeps_empty_lines(self):
        assert split_lines("a\n\nb") == ["a", "", "b"]


class TestQualitySignals:
    def test_clean_legal_english_scores_low(self):
        text = (
            "Whoever commits murder shall be punished with death or imprisonment "
            "for life, and shall also be liable to fine under this section."
        )
        signals = quality_signals(text)
        assert signals["vowelless_ratio"] == 0.0
        assert signals["symbol_ratio"] < 0.01

    def test_ocr_noise_scores_high(self):
        # Verbatim from the OCR layer of a real India Code scanned gazette
        # (The Passport Rules, 1967), which extracts "successfully" and is
        # nevertheless unusable as legal text.
        text = (
            "UG,UTmuID ~-QPiT ~he <Sox.ette of ~t1diO EXTRAORDINARY "
            "tQ ltA ti f~ 'l~~~fM~ '3fT~i\\'~ f~ f~ lf~ ~~;r ..r \\Iq' i~ lIlT ri I"
        )
        signals = quality_signals(text)
        assert signals["symbol_ratio"] > 0.08

    def test_empty_text_does_not_divide_by_zero(self):
        signals = quality_signals("")
        assert signals["vowelless_ratio"] == 0.0
        assert signals["symbol_ratio"] == 0.0


class TestUnionArea:
    def test_single_rectangle(self):
        assert union_area([(0, 0, 10, 10)], (0, 0, 100, 100)) == 100.0

    def test_overlapping_rectangles_are_counted_once(self):
        # Two identical full-page images must not report 200% coverage.
        rects = [(0, 0, 10, 10), (0, 0, 10, 10)]
        assert union_area(rects, (0, 0, 100, 100)) == 100.0

    def test_partial_overlap(self):
        rects = [(0, 0, 10, 10), (5, 5, 15, 15)]
        assert union_area(rects, (0, 0, 100, 100)) == 175.0

    def test_rectangles_are_clipped_to_the_page(self):
        assert union_area([(-50, -50, 50, 50)], (0, 0, 10, 10)) == 100.0

    def test_no_rectangles_or_empty_clip(self):
        assert union_area([], (0, 0, 10, 10)) == 0.0
        assert union_area([(0, 0, 10, 10)], (0, 0, 0, 0)) == 0.0
