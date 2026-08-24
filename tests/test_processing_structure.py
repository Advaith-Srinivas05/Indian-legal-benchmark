"""Tests for legal-structure parsing.

The parser takes :class:`~processing.models.PageText`, not PDFs, so these tests
use real strings with the exact typography India Code uses — em dashes, bracketed
inserted sections, roman sub-clauses. That is deliberate: font support in a
generated fixture PDF would otherwise decide whether an em dash survives, and the
punctuation is the whole signal.

The bias under test is conservatism. It is worse for the parser to invent a
section than to miss one: a missed section leaves text that is still searchable,
while an invented one produces a citation to a provision that does not exist.
"""

from __future__ import annotations

import pytest

from processing.models import PageText
from processing.structure import build_line_stream, classify_label, detect_toc_pages, parse_structure


def page(number: int, text: str) -> PageText:
    return PageText(page_number=number, text=text, char_count=len(text))


def pages(*texts: str) -> list[PageText]:
    return [page(i, t) for i, t in enumerate(texts, start=1)]


def units_of(structure, unit_type):
    return [u for u in structure.all_units() if u.unit_type == unit_type]


# A Central Act in India Code's own house style, verbatim in shape.
ACT = """THE SAMPLE ACT, 1999
ACT NO. 7 OF 1999
[24th June, 1999.]
An Act to provide for the regulation of samples and for matters connected therewith.
BE it enacted by Parliament in the Fiftieth Year of the Republic of India as follows:—
CHAPTER I
PRELIMINARY
1. Short title and extent.—(1) This Act may be called the Sample Act, 1999.
(2) It extends to the whole of India.
2. Definitions.—In this Act, unless the context otherwise requires,—
(a) "appointed day" means the day on which this Act comes into force;
(b) "prescribed" means prescribed by rules made under this Act, and includes—
(i) a rule made by the Central Government;
(ii) a rule made by the State Government;
(c) "sample" has the meaning assigned to it in section 3.
CHAPTER II
REGULATION OF SAMPLES
3. Power to make rules.—The Central Government may, by notification, make rules.
Provided that no such rule shall be made without previous publication.
Explanation.—In this section, "rules" means rules made under this Act.
"""


class TestNormalActExtraction:
    @pytest.fixture()
    def structure(self):
        return parse_structure(pages(ACT), metadata_title="Sample Act, 1999")

    def test_chapters_and_sections_are_found(self, structure):
        assert [u.number for u in units_of(structure, "chapter")] == ["I", "II"]
        assert [u.number for u in units_of(structure, "section")] == ["1", "2", "3"]

    def test_chapter_caption_on_the_following_line_is_captured(self, structure):
        chapters = units_of(structure, "chapter")
        assert chapters[0].heading == "PRELIMINARY"
        assert chapters[0].detected_by == "chapter_heading_caption_below"

    def test_sections_are_nested_under_their_chapter(self, structure):
        chapters = units_of(structure, "chapter")
        assert [c.number for c in chapters[0].children] == ["1", "2"]
        assert [c.number for c in chapters[1].children] == ["3"]

    def test_section_heading_is_separated_from_its_body(self, structure):
        section = units_of(structure, "section")[0]
        assert section.heading == "Short title and extent"
        assert section.detected_by == "numbered_heading_dash"
        assert "This Act may be called" in section.text

    def test_preamble_is_identified_from_the_enacting_formula(self, structure):
        assert structure.preamble is not None
        assert "An Act to provide" in structure.preamble.text
        assert structure.preamble.detected_by == "enacting_formula"

    def test_title_comes_from_india_code_metadata(self, structure):
        assert structure.title == "Sample Act, 1999"
        assert structure.title_source == "india_code_metadata_confirmed_on_page"

    def test_confidence(self, structure):
        assert structure.confidence == "structured"


class TestSubsectionsAndClauses:
    @pytest.fixture()
    def structure(self):
        return parse_structure(pages(ACT))

    def test_inline_first_subsection_after_the_heading_dash_is_found(self, structure):
        section = units_of(structure, "section")[0]
        assert [c.number for c in section.children] == ["1", "2"]
        assert all(c.unit_type == "subsection" for c in section.children)
        assert section.children[0].detected_by == "numeric label"

    def test_clauses_are_found(self, structure):
        section = units_of(structure, "section")[1]
        assert [c.number for c in section.children] == ["a", "b", "c"]
        assert all(c.unit_type == "clause" for c in section.children)

    def test_roman_labels_inside_a_clause_are_subclauses(self, structure):
        clause_b = units_of(structure, "section")[1].children[1]
        assert [c.unit_type for c in clause_b.children] == ["subclause", "subclause"]
        assert [c.number for c in clause_b.children] == ["i", "ii"]

    def test_provisos_and_explanations_attach_to_their_section(self, structure):
        section = units_of(structure, "section")[2]
        kinds = [c.unit_type for c in section.children]
        assert "proviso" in kinds and "explanation" in kinds

    def test_clause_text_starts_at_its_label(self, structure):
        clause = units_of(structure, "clause")[0]
        assert clause.text.startswith('(a) "appointed day"')


class TestLabelDisambiguation:
    """``(i)`` is both the ninth clause letter and the first roman numeral."""

    def test_digits_are_always_subsections(self):
        assert classify_label("1", open_types=[], last_clause=None,
                              last_subclause=None)[0] == "subsection"
        assert classify_label("2A", open_types=[], last_clause=None,
                              last_subclause=None)[0] == "subsection"

    def test_i_after_h_continues_the_clause_letters(self):
        kind, reason = classify_label("i", open_types=[], last_clause="h",
                                      last_subclause=None)
        assert kind == "clause"
        assert "continues clause sequence" in reason

    def test_i_inside_an_open_clause_opens_a_roman_run(self):
        kind, reason = classify_label("i", open_types=["subsection", "clause"],
                                      last_clause="b", last_subclause=None)
        assert kind == "subclause"

    def test_roman_run_continues(self):
        kind, _ = classify_label("iii", open_types=["clause"], last_clause="b",
                                 last_subclause="ii")
        assert kind == "subclause"

    def test_plain_letters_are_clauses(self):
        assert classify_label("b", open_types=[], last_clause="a",
                              last_subclause=None)[0] == "clause"

    def test_every_decision_records_a_reason(self):
        for label in ("1", "a", "i", "iv", "B"):
            _, reason = classify_label(label, open_types=["clause"],
                                       last_clause="a", last_subclause=None)
            assert reason


class TestContentsListings:
    TOC = """THE SAMPLE ACT, 1999
_________
ARRANGEMENT OF SECTIONS
_________
SECTIONS
1. Short title and extent.
2. Definitions.
3. Power to make rules.
4. Penalty.
5. Repeal and savings.
"""

    def test_a_marked_contents_page_is_recognised(self):
        structure = parse_structure(pages(self.TOC, ACT))
        assert structure.toc_pages == [1]

    def test_contents_entries_are_not_reported_as_sections(self):
        structure = parse_structure(pages(self.TOC, ACT))
        numbers = [u.number for u in units_of(structure, "section")]
        # Only the three real sections on page 2 — not the five listed on page 1.
        assert numbers == ["1", "2", "3"]
        assert all(u.page_start == 2 for u in units_of(structure, "section"))

    def test_an_unmarked_dense_number_list_is_recognised(self):
        # A two-column contents page extracts as bare numbers with the captions
        # in a separate run; there is no marker line to key on.
        listing = "Section\n1.\n2.\n3.\n4.\n5.\n6.\nShort title.\nDefinitions.\n"
        assert detect_toc_pages(build_line_stream(pages(listing))) == {1}

    def test_contents_page_text_is_still_present_in_the_pages(self):
        # Excluding a page from the section *index* must not remove its text.
        page_objects = pages(self.TOC, ACT)
        parse_structure(page_objects)
        assert "ARRANGEMENT OF SECTIONS" in page_objects[0].text


class TestPageProvenance:
    ACROSS_PAGES_1 = """1. Short title.—This Act may be called the Sample Act.
2. Definitions.—In this Act, unless the context otherwise requires,—
(a) "appointed day" means the day on which this Act comes into force;
"""
    ACROSS_PAGES_2 = """(b) "prescribed" means prescribed by rules made under this Act;
(c) "sample" has the meaning assigned to it in section 3.
3. Penalty.—Whoever contravenes any provision shall be punished.
"""

    @pytest.fixture()
    def structure(self):
        return parse_structure(pages(self.ACROSS_PAGES_1, self.ACROSS_PAGES_2))

    def test_a_section_spanning_a_page_break_records_both_pages(self, structure):
        section_2 = units_of(structure, "section")[1]
        assert section_2.number == "2"
        assert (section_2.page_start, section_2.page_end) == (1, 2)

    def test_units_on_the_second_page_are_attributed_to_it(self, structure):
        clause_c = [u for u in units_of(structure, "clause") if u.number == "c"][0]
        assert clause_c.page_start == clause_c.page_end == 2

    def test_every_unit_carries_a_page_range_and_line_range(self, structure):
        for unit in structure.all_units():
            assert unit.page_start >= 1
            assert unit.page_end >= unit.page_start
            assert unit.line_start >= 0
            assert unit.line_end >= unit.line_start

    def test_a_chapter_page_range_covers_its_sections(self):
        structure = parse_structure(pages(
            "CHAPTER I\nPRELIMINARY\n1. Short title.—This Act may be called X.",
            "2. Definitions.—In this Act,—\n3. Penalty.—Whoever contravenes.",
        ))
        chapter = units_of(structure, "chapter")[0]
        assert (chapter.page_start, chapter.page_end) == (1, 2)


class TestSchedules:
    def test_a_schedule_is_a_top_level_unit(self):
        structure = parse_structure(pages(
            "1. Short title.—This Act may be called X.\n"
            "2. Definitions.—In this Act,—\n"
            "3. Penalty.—Whoever contravenes.\n"
            "THE FIRST SCHEDULE\n"
            "FORM OF APPLICATION\n"
            "1. Name of applicant\n"
        ))
        schedules = units_of(structure, "schedule")
        assert len(schedules) == 1
        assert schedules[0].heading == "FORM OF APPLICATION"


class TestWeakerStyleFallback:
    PLAIN = """THE STATE SAMPLE ACT, 1975
1. Short title and commencement
This Act may be called the State Sample Act, 1975.
2. Definitions
In this Act, unless the context otherwise requires, the following words shall
have the meanings assigned to them.
3. Power to make rules
The State Government may make rules for carrying out the purposes of this Act.
"""

    def test_line_style_sections_are_found_when_no_dashed_style_exists(self):
        structure = parse_structure(pages(self.PLAIN))
        assert [u.number for u in units_of(structure, "section")] == ["1", "2", "3"]
        assert all(u.detected_by == "numbered_heading_line"
                   for u in units_of(structure, "section"))

    def test_the_weaker_style_is_reported_as_such(self):
        structure = parse_structure(pages(self.PLAIN))
        assert any("weaker" in w for w in structure.warnings)

    def test_the_weak_pattern_is_not_mixed_into_a_dashed_document(self):
        structure = parse_structure(pages(ACT))
        assert all(u.detected_by == "numbered_heading_dash"
                   for u in units_of(structure, "section"))


class TestConservatism:
    def test_unstructured_text_produces_no_invented_hierarchy(self):
        text = (
            "This is an extract of correspondence between two departments of the "
            "Government concerning the transfer of certain records, and it carries "
            "no numbering, no headings and no divisions of any kind whatsoever."
        )
        structure = parse_structure(pages(text))
        assert structure.units == []
        assert structure.confidence == "unstructured"
        assert any("no legal structure" in w for w in structure.warnings)

    def test_a_sentence_beginning_with_chapter_is_not_a_chapter(self):
        text = (
            "1. Short title.—This Act may be called X.\n"
            "2. Application.—Chapter V of the principal Act shall apply to every "
            "person to whom this section extends, and shall be construed as one.\n"
            "3. Penalty.—Whoever contravenes.\n"
        )
        structure = parse_structure(pages(text))
        assert units_of(structure, "chapter") == []

    def test_out_of_sequence_candidates_are_excluded_and_reported(self):
        text = (
            "1. Short title.—This Act may be called X.\n"
            "2. Definitions.—In this Act,—\n"
            "99. Form No.—This is a numbered form caption inside section 2.\n"
            "3. Penalty.—Whoever contravenes.\n"
            "4. Repeal.—The earlier Act is repealed.\n"
            "5. Savings.—Nothing in this Act shall affect.\n"
        )
        structure = parse_structure(pages(text))
        assert [u.number for u in units_of(structure, "section")] == \
            ["1", "2", "3", "4", "5"]
        assert any("out-of-sequence" in w for w in structure.warnings)

    def test_wholly_disordered_numbering_keeps_everything_and_warns(self):
        # If most candidates break the order, the assumption is wrong, not the
        # document; dropping half the sections would be the worse error.
        text = "\n".join(
            f"{n}. Heading {n}.—Body of the provision." for n in (9, 2, 7, 1, 5, 3)
        )
        structure = parse_structure(pages(text))
        assert len(units_of(structure, "section")) == 6
        assert any("not in ascending order" in w for w in structure.warnings)

    def test_empty_input_is_handled(self):
        structure = parse_structure([])
        assert structure.units == []
        assert structure.confidence == "unstructured"

    def test_lettered_sections_are_preserved(self):
        text = (
            "10. Variation.—The passport authority may vary a passport.\n"
            "10A. Suspension of passports.—The authority may suspend.\n"
            "10B. Validation of intimations.—Any intimation is valid.\n"
            "11. Appeals.—An appeal shall lie.\n"
        )
        structure = parse_structure(pages(text))
        assert [u.number for u in units_of(structure, "section")] == \
            ["10", "10A", "10B", "11"]


class TestArticles:
    """Articles are a first-class unit, not Sections by another name.

    The Constitution and the Portuguese-derived civil codes still in force in
    Goa number their provisions as Articles. In the first benchmark the
    Portuguese Civil Code, 1867 reported 191 "sections" — which were the
    numbered list items *inside* its articles — while its 4,900 actual Articles
    went unrecognised. Citing an Article as a section would be a citation to a
    provision that does not exist.
    """

    GOA = """PORTUGUESE CIVIL CODE, 1867
Article 14 – Conflict of rights – Whoever in exercise of his own right, seeks
advantages should in case of conflict concede in favour of whoever intends to
avoid losses.
Article 15 – Plain concurrence of rights – In case of concurrence of rights
which are equal the interested parties shall reciprocally relinquish that which
is necessary.
Article 18 – Acquisition of Portuguese citizenship – The following are
Portuguese citizens:
1. Those who are born in Portuguese territory, of Portuguese father;
2. Those who are born abroad of a Portuguese father;
3. Those who obtain naturalization under the law;
Article 19 – Loss of citizenship – Portuguese citizenship is lost as follows.
"""

    CONSTITUTION = """PART III
FUNDAMENTAL RIGHTS
Article 19. Protection of certain rights regarding freedom of speech, etc.
Article 20. Protection in respect of conviction for offences.
Article 21. Protection of life and personal liberty.
"""

    def test_articles_are_identified_as_articles(self):
        structure = parse_structure(pages(self.GOA))
        assert [u.number for u in units_of(structure, "article")] == \
            ["14", "15", "18", "19"]
        assert units_of(structure, "section") == []

    def test_the_document_records_which_vocabulary_it_uses(self):
        assert parse_structure(pages(self.GOA)).unit_vocabulary == "article"
        assert parse_structure(pages(ACT)).unit_vocabulary == "section"

    def test_the_dashed_form_separates_heading_from_body(self):
        article = units_of(parse_structure(pages(self.GOA)), "article")[0]
        assert article.heading == "Conflict of rights"
        assert article.detected_by == "article_heading_dash"
        assert "Whoever in exercise" in article.text

    def test_numbered_items_inside_an_article_are_not_sections(self):
        # The exact failure: "1." and "2." inside Article 18 were reported as
        # sections of the Code.
        structure = parse_structure(pages(self.GOA))
        assert units_of(structure, "section") == []
        assert any("numbers its provisions as Articles" in w
                   for w in structure.warnings)

    def test_a_sentence_mentioning_an_article_is_not_an_article(self):
        text = (
            "1. Short title.—This Act may be called X.\n"
            "2. Application.—Article 348 of the Constitution provides for an "
            "English translation of legislation, and this Act is published "
            "accordingly by the Government.\n"
            "3. Penalty.—Whoever contravenes.\n"
        )
        structure = parse_structure(pages(text))
        assert units_of(structure, "article") == []
        assert len(units_of(structure, "section")) == 3

    def test_constitution_style_articles_are_found(self):
        structure = parse_structure(pages(self.CONSTITUTION))
        assert [u.number for u in units_of(structure, "article")] == \
            ["19", "20", "21"]
        assert units_of(structure, "part")[0].number == "III"

    def test_articles_carry_page_provenance(self):
        structure = parse_structure(pages(self.GOA, self.GOA.replace("1", "2")))
        for article in units_of(structure, "article"):
            assert article.page_start >= 1
            assert article.page_end >= article.page_start
            assert article.detected_by

    def test_articles_count_towards_structure_confidence(self):
        assert parse_structure(pages(self.GOA)).confidence == "structured"

    def test_sections_and_articles_keep_separate_number_sequences(self):
        # Article 5 following section 200 is not a break in the numbering.
        text = (
            "1. Short title.—This Act may be called X.\n"
            "2. Definitions.—In this Act,—\n"
            "200. Repeal.—The earlier Act is repealed.\n"
            "Article 1 – Scope – This Article applies to the Schedule.\n"
            "Article 2 – Effect – It has effect from the appointed day.\n"
            "Article 3 – Savings – Nothing herein affects vested rights.\n"
        )
        structure = parse_structure(pages(text))
        assert [u.number for u in units_of(structure, "article")] == ["1", "2", "3"]
        assert [u.number for u in units_of(structure, "section")] == \
            ["1", "2", "200"]
        assert not any("not in ascending order" in w for w in structure.warnings)


class TestTitle:
    def test_metadata_title_is_used_when_the_page_does_not_show_it(self):
        structure = parse_structure(
            pages("1. Short title.—This Act may be called X."),
            metadata_title="Some Act, 1999",
        )
        assert structure.title == "Some Act, 1999"
        assert structure.title_source == "india_code_metadata"

    def test_a_page_heading_is_only_used_when_there_is_no_metadata_title(self):
        structure = parse_structure(pages("THE SAMPLE ACT, 1999\nbody text here"))
        assert structure.title == "THE SAMPLE ACT, 1999"
        assert structure.title_source == "pdf_first_page_heading"
