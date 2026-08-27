"""Tests for the pure text helpers underneath extraction and parsing."""

from __future__ import annotations

import time

import pytest

from processing import config
from processing.textutils import (
    alpha_count,
    is_blank,
    is_page_number_line,
    normalise_line,
    quality_signals,
    split_lines,
    union_area,
)


def _exact_union_area(rects, clip):
    """The unbounded reference implementation, kept as the oracle.

    ``union_area`` only runs this below ``UNION_AREA_EXACT_MAX_RECTS``. Keeping
    a copy here is what lets the raster fallback be checked against the answer
    it approximates, on inputs small enough for the exact method to finish.
    """
    cx0, cy0, cx1, cy1 = clip
    clipped = []
    for x0, y0, x1, y1 in rects:
        nx0, ny0 = max(min(x0, x1), cx0), max(min(y0, y1), cy0)
        nx1, ny1 = min(max(x0, x1), cx1), min(max(y0, y1), cy1)
        if nx1 > nx0 and ny1 > ny0:
            clipped.append((nx0, ny0, nx1, ny1))
    if not clipped:
        return 0.0
    xs = sorted({v for r in clipped for v in (r[0], r[2])})
    ys = sorted({v for r in clipped for v in (r[1], r[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            for rx0, ry0, rx1, ry1 in clipped:
                if rx0 <= x0 and rx1 >= x1 and ry0 <= y0 and ry1 >= y1:
                    area += (x1 - x0) * (y1 - y0)
                    break
    return area


def _tiled_page(count, per_row=None, width=612.0, height=792.0):
    """The first *count* tiles of a *per_row* x *per_row* tiling of the page.

    This is how the scans that stalled the run are built: abutting tiles, not
    one image. Passing ``count == per_row ** 2`` covers the page completely.
    """
    per_row = per_row or int(count ** 0.5) + 1
    tw, th = width / per_row, height / per_row
    tiles = []
    for i in range(count):
        row, col = divmod(i, per_row)
        tiles.append((col * tw, row * th, (col + 1) * tw, (row + 1) * th))
    return tiles


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


class TestUnionAreaAtScale:
    """The tiled-scan case that stalled the first full-corpus run.

    A page in ``uttarakhand-public-library-act__handle-3439`` carries 16,376
    image placements. The exact method is O(n^4) in that count and does not
    return; seven of the run's eight workers ended up inside it, holding the
    GIL and starving the rest of the process. These tests exist so that never
    silently comes back.
    """

    CLIP = (0.0, 0.0, 612.0, 792.0)
    PAGE = 612.0 * 792.0

    def test_a_speckled_page_returns_promptly_and_is_not_image_backed(self):
        # The real shape that stalled the run: 16,376 placements 0.2-0.8 pt
        # across — scanner dust kept as individual images — over an ordinary
        # text page. They cover almost nothing, so the only hard part was ever
        # arriving at that answer.
        speckles = [
            (x * 0.37 % 600.0, x * 0.61 % 780.0,
             x * 0.37 % 600.0 + 0.4, x * 0.61 % 780.0 + 0.4)
            for x in range(16376)
        ]
        started = time.perf_counter()
        ratio = union_area(speckles, self.CLIP) / self.PAGE
        assert time.perf_counter() - started < 1.0
        assert ratio < config.IMAGE_BACKED_AREA_RATIO
        assert ratio < 0.05

    def test_a_tiled_scan_of_thousands_of_placements_returns_promptly(self):
        started = time.perf_counter()
        area = union_area(_tiled_page(16376), self.CLIP)
        elapsed = time.perf_counter() - started
        # The exact method needs ~1.8e13 operations for this input. A second is
        # a generous ceiling that still fails by orders of magnitude if the
        # fallback is ever removed.
        assert elapsed < 1.0
        assert area / self.PAGE > 0.99          # the tiles do cover the page

    def test_thousands_of_large_overlapping_placements_return_promptly(self):
        # The other shape: not tiles but many big overlapping rectangles, which
        # is what defeats a "sum the areas and cap at 1.0" shortcut.
        rects = [(0.0, 0.0, 612.0, 792.0)]
        rects += [(x * 0.05, x * 0.07, 612.0, 792.0) for x in range(5000)]
        started = time.perf_counter()
        area = union_area(rects, self.CLIP)
        assert time.perf_counter() - started < 1.0
        assert area == self.PAGE                 # one of them already covers it

    def test_repeated_placements_are_deduplicated_back_to_the_exact_path(self):
        # 5,000 copies of one full-page image is 1 distinct rectangle, so this
        # must come back exact rather than rasterised.
        rects = [(0.0, 0.0, 612.0, 792.0)] * 5000
        assert union_area(rects, self.CLIP) == self.PAGE

    def test_a_single_full_page_scan_is_still_exactly_the_page(self):
        # The case the threshold actually turns on must not be approximated.
        assert union_area([(0.0, 0.0, 612.0, 792.0)], self.CLIP) == self.PAGE

    def test_the_fallback_stays_close_to_the_exact_answer(self):
        # Centre sampling makes the error two-sided, so this bounds the size of
        # the error rather than its direction.
        for size in (30.0, 80.0, 160.0):
            rects = [
                (x * 7.0 % 400.0, x * 11.0 % 500.0,
                 x * 7.0 % 400.0 + size, x * 11.0 % 500.0 + size)
                for x in range(config.UNION_AREA_EXACT_MAX_RECTS + 1)
            ]
            exact = _exact_union_area(rects, self.CLIP)
            assert union_area(rects, self.CLIP) == pytest.approx(exact, rel=0.05)

    @pytest.mark.parametrize("count", [400, 4000, 16376])
    def test_a_tiled_page_reports_the_fraction_its_tiles_actually_cover(self, count):
        # The regression that matters. These tiles abut, so the covered fraction
        # is exactly tiles / grid slots, whatever the tiles' size relative to a
        # raster cell. A rule that required whole cells to be contained would
        # lose the cells straddling every tile boundary and report a fully
        # covered page as roughly half covered.
        per_row = int(count ** 0.5) + 1
        expected = count / (per_row * per_row)
        ratio = union_area(_tiled_page(count), self.CLIP) / self.PAGE
        assert ratio == pytest.approx(expected, abs=0.002)

    def test_a_fully_tiled_page_is_not_demoted_below_the_image_backed_threshold(self):
        # State the consequence directly: a scan laid down as tiles must still
        # classify as image-backed.
        for per_row in (7, 13, 32, 64, 128):
            tiles = _tiled_page(per_row * per_row, per_row=per_row)
            ratio = union_area(tiles, self.CLIP) / self.PAGE
            assert ratio >= config.IMAGE_BACKED_AREA_RATIO
            assert ratio == pytest.approx(1.0, abs=0.002)

    def test_the_cap_is_where_the_behaviour_changes(self):
        # At the cap the answer is exact; one rectangle past it, it is not
        # required to be — but it must still be close and still under.
        cap = config.UNION_AREA_EXACT_MAX_RECTS
        tiles = _tiled_page(cap * 4)[:cap]
        assert union_area(tiles, self.CLIP) == _exact_union_area(tiles, self.CLIP)
