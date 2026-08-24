"""Tests for running header / footer / page-number detection.

The project spec allows furniture to be *detected* but not blindly deleted, and
requires any removal rule to have tests. These are those tests. Two things must
hold: real furniture is found, and legal text is never labelled as furniture —
the second matters more, because a false positive silently removes law from the
parsed structure.

Pages here are built with a realistic number of lines. A three-line page would
put every line in both the top and the bottom zone, which is not the shape any
real page has and would test the detector against a situation it never meets.
"""

from __future__ import annotations

from processing.furniture import content_lines, detect
from processing.models import FurnitureLine

_SENTENCES = [
    "and whereas it is expedient to provide for the matters hereinafter",
    "appearing, the provisions of this Chapter shall have effect according",
    "to their tenor and be construed as one with the principal enactment",
    "for every purpose, including the making of rules thereunder and the",
    "issue of any notification, order or direction by the State Government",
    "in respect of any area to which this section has been extended by a",
    "notification published in the Official Gazette in that behalf, and",
    "no such notification shall be called in question in any court merely",
]


def _body(page_number: int) -> str:
    """Five lines of plausible statutory prose, different on every page."""
    start = (page_number * 3) % len(_SENTENCES)
    return "\n".join(_SENTENCES[(start + i) % len(_SENTENCES)] for i in range(5))


def _pages(header: str = "", footer: str = "", count: int = 4) -> list[str]:
    """*count* realistic pages, optionally wrapped in furniture."""
    pages = []
    for number in range(1, count + 1):
        parts = []
        if header:
            parts.append(header.format(n=number))
        parts.append(f"{number}. Provision number {number} of the Act.")
        parts.append(_body(number))
        if footer:
            parts.append(footer.format(n=number))
        pages.append("\n".join(parts))
    return pages


class TestRunningHeads:
    def test_repeated_header_is_detected(self):
        found = detect(_pages(header="THE GAZETTE OF INDIA"))
        assert set(found) == {1, 2, 3, 4}
        assert all(f.kind == "header" for lines in found.values() for f in lines)
        assert found[1][0].text == "THE GAZETTE OF INDIA"
        assert found[1][0].line_index == 0

    def test_repeated_footer_is_detected(self):
        found = detect(_pages(footer="Ministry of Law and Justice"))
        assert set(found) == {1, 2, 3, 4}
        assert all(f.kind == "footer" for lines in found.values() for f in lines)

    def test_reason_records_the_evidence(self):
        found = detect(_pages(header="THE GAZETTE OF INDIA", count=3))
        assert "3/3 pages" in found[1][0].reason


class TestPageNumbers:
    def test_changing_page_numbers_are_detected(self):
        # Digit folding is what makes this work: "1", "2", "3" all compare equal
        # once digits are folded, even though no two pages share the literal
        # line.
        found = detect(_pages(footer="{n}"))
        assert set(found) == {1, 2, 3, 4}
        assert found[2][0].kind == "page_number"
        assert found[2][0].text == "2"
        assert "digits folded" in found[2][0].reason

    def test_page_x_of_y_footers(self):
        found = detect(_pages(footer="Page {n} of 4"))
        assert found[1][0].kind == "page_number"

    def test_a_header_and_a_page_number_are_both_labelled(self):
        found = detect(_pages(header="THE PASSPORTS ACT, 1967", footer="{n}"))
        assert {f.kind for f in found[3]} == {"header", "page_number"}


class TestConservatism:
    def test_short_documents_are_left_alone(self):
        # With two pages, "recurs on most pages" is not evidence of anything.
        assert detect(_pages(header="THE GAZETTE OF INDIA", count=2)) == {}

    def test_legal_text_that_does_not_repeat_is_not_furniture(self):
        assert detect(_pages()) == {}

    def test_body_text_containing_numbers_is_not_furniture(self):
        # The danger of folding digits for every line: these differ only in a
        # number, but they are law, not a running head.
        pages = [
            f"In the year 196{n}, the Central Government did notify\n{_body(n)}"
            for n in range(4)
        ]
        assert detect(pages) == {}

    def test_a_line_repeated_on_a_minority_of_pages_is_not_furniture(self):
        pages = _pages(count=5)
        pages[0] = "THE GAZETTE OF INDIA\n" + pages[0]
        pages[1] = "THE GAZETTE OF INDIA\n" + pages[1]
        # Two pages out of five is below the 60% threshold.
        assert detect(pages) == {}

    def test_long_repeated_lines_are_left_alone(self):
        # A long recurring line is more likely repeated legal text (a form, a
        # recital) than a running head, so the rule refuses to touch it.
        assert detect(_pages(header="x" * 200)) == {}

    def test_a_repeated_line_in_the_middle_of_a_page_is_not_furniture(self):
        pages = [
            f"page top {n}\n{_body(n)}\nREPEATED MIDDLE LINE\n{_body(n + 1)}\n"
            f"page bottom {n}"
            for n in range(4)
        ]
        found = detect(pages)
        assert all(
            f.text.strip() != "REPEATED MIDDLE LINE"
            for lines in found.values() for f in lines
        )


class TestContentLines:
    def test_furniture_is_skipped_but_indices_are_preserved(self):
        text = "HEAD\n1. Short title.\nbody"
        furniture = [FurnitureLine(line_index=0, text="HEAD", kind="header", reason="x")]
        assert content_lines(text, furniture) == [(1, "1. Short title."), (2, "body")]

    def test_detection_never_modifies_the_input(self):
        pages = _pages(header="THE GAZETTE OF INDIA")
        before = list(pages)
        detect(pages)
        assert pages == before
