"""Tests for content-level language validation.

The regression these exist for: India Code served a Devanagari rules
notification and a mirrored Gujarati scan under ``Files(Eng)``, and the
extraction benchmark accepted both because the ingestion phase — correctly,
under its own rules — had recorded the publisher's label as the evidence.

The fixtures below are shaped after the real documents, including the awkward
part: both OCR'd into **Latin** glyphs and contain no Devanagari at all, so a
"contains Devanagari" rule would clear them while rejecting the perfectly good
English acts that quote a Hindi title.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from processing import config, language
from processing.structure import parse_structure

ENGLISH = """
    Whoever, being a public servant, knowingly disobeys any direction of the law
    as to the way in which he is to conduct himself as such public servant,
    intending to cause injury to any person, shall be punished with simple
    imprisonment for a term which may extend to one year, or with fine, or with
    both. In this section, the expression "public servant" shall have the meaning
    assigned to it in section 21 of the Indian Penal Code, and nothing in this
    section shall apply to any act done in good faith by an officer of the
    Central Government or of a State Government in the discharge of his official
    duty under any rule made under this Act or under any other law for the time
    being in force in the territories to which this Act extends.
""" * 4

#: Shaped after sadasy-penison-niyam-1977: a Devanagari page whose OCR produced
#: Latin glyph noise. Verbatim fragments from that document's extracted text.
TRANSLITERATED_NOISE = """
    rf.W. Faum. MIT 'H 4t'4 41TR' 17PT, 1977 Sia/T (10191'(3711tiMin 14770 (1)
    4-26-77-t.g.-77/2/21-3T, itFTW 31 Pik 1977 A. crm-ifta. 31ffrfl1Wr fa.FTW 4
    th-Cdt 1980 MIT aiftriTFT W. 1385, itFTW 3 aleil& 1996 C.51i1 if 3:0-414aVr
    Fdt11T 1977 ch6Meicii. ir T r-01- # 3r rzrr 3Trtrea. S #31thrg- 37t444at
    ittu-07311 iceq adj, 31Td1 MIT ei'M. 3i1trdzixf, 1972 sheiich 7W 1973
    Niqq 3ftiela t facwii CIF r 1"11. 3arera. t1Mar faTTF Rail 31# ail ata* 311FT
""" * 8

#: Shaped after national-food-security-act-2013: English under heavy OCR damage.
#: This must still come out ``en`` — corrupted English is a *quality* problem,
#: and calling it another language would send it down the wrong path entirely.
DAMAGED_ENGLISH = """
    Any publiL' SerVi!ll1 or authorilY found guilty, by the Slate COlllmis~i()n at
    lhl: time uf deeiding any eomplaint or ;lppeal. uf f"illllg to pruvick the
    relief recommcndeJ by the Di~lrid G,.iev~ll".:e !(cd,css;t! Ofri\\:cr, withoul
    reavlIlabk causc. or II'ilfully Ignoring .-;ud1 Iccolllillendatiull. shall be
    l1able to penalty Ilot exceeding five thuusand rupees under this Act and the
    rules made thereunder by the Central Government in that behalf.
""" * 8


def pages(*texts: str):
    return [SimpleNamespace(text=t, page_number=i, furniture=[], footnotes=[])
            for i, t in enumerate(texts, start=1)]


#: A Devanagari sentence long enough for a page of it to be measurable, set out
#: as numbered dash-headed provisions -- the shape ``processing.patterns`` reads
#: as a section, and the shape a Hindi translation of a rule actually takes.
_HINDI_SENTENCE = (
    "\u0907\u0928 \u0928\u093f\u092f\u092e\u094b\u0902 \u0915\u093e "
    "\u0938\u0902\u0915\u094d\u0937\u093f\u092a\u094d\u0924 \u0928\u093e\u092e "
    "\u0936\u093f\u0915\u094d\u0937\u0941\u0924\u093e \u0928\u093f\u092f\u092e "
    "\u0939\u0948\u0964"
)
HINDI_SECTION_PAGE = "".join(
    f"{n}. {_HINDI_SENTENCE}\u2014{_HINDI_SENTENCE} {_HINDI_SENTENCE}\n"
    for n in range(1, 9)
)
ENGLISH_SECTION_PAGE = (
    "1. Short title and commencement.\u2014These rules may be called the Model "
    "Rules, 2019, and they shall come into force at once.\n"
    "2. Definitions.\u2014In these rules, unless the context otherwise requires, "
    "the expressions used shall have the meaning assigned to them.\n"
) + ENGLISH


class TestEnglishContent:
    def test_english_legal_text_is_accepted(self):
        result = language.assess_text(ENGLISH)
        assert result.content_language == "en"
        assert result.signals["function_word_rate"] >= config.LANGUAGE_ENGLISH_RATE

    def test_metadata_english_plus_english_content_is_eligible(self):
        result = language.assess_pages(pages(ENGLISH), metadata_language="en")
        assert result.content_language == "en"
        assert result.eligible_for_indexing

    def test_a_few_devanagari_characters_do_not_reject_a_document(self):
        # The Passports Act's own metadata carries its Hindi title. An English
        # act that quotes one is still an English act.
        text = ENGLISH + "\nपासपोर्ट अधिनियम, 1967\n"
        result = language.assess_text(text)
        assert result.content_language == "en"
        assert result.signals["non_latin_letter_ratio"] < \
            config.LANGUAGE_NON_LATIN_LETTER_RATIO

    def test_latin_legal_abbreviations_and_proper_nouns_are_tolerated(self):
        text = ENGLISH + (
            "\nThe doctrine of ultra vires, res judicata and audi alteram partem "
            "was considered in Kesavananda Bharati v. State of Kerala, and in "
            "Maneka Gandhi v. Union of India, by Chandrachud, Bhagwati and Krishna "
            "Iyer JJ. See also Halsbury's Laws of England, 4th edn.\n"
        )
        assert language.assess_text(text).content_language == "en"


class TestNonEnglishContent:
    def test_devanagari_transliterated_into_latin_noise_is_rejected(self):
        # The sadasy-penison-niyam-1977 regression. No Devanagari characters
        # survive OCR, so only the readable-word evidence catches this.
        result = language.assess_text(TRANSLITERATED_NOISE)
        assert result.content_language == "non_en"
        assert not result.eligible_for_indexing
        assert result.signals["non_latin_letter_ratio"] == 0.0
        assert any("not English" in reason for reason in result.reasons)

    def test_a_document_actually_set_in_devanagari_is_rejected_on_script(self):
        text = (
            "यह अधिनियम भारत के संसद द्वारा पारित किया गया है और यह सम्पूर्ण भारत "
            "पर लागू होता है। इस अधिनियम की धारा के अंतर्गत सभी नियम बनाए जाएंगे। "
        ) * 12
        result = language.assess_text(text)
        assert result.content_language == "non_en"
        assert "devanagari" in " ".join(result.reasons).lower()

    def test_english_metadata_cannot_clear_non_english_content(self):
        result = language.assess_pages(
            pages(TRANSLITERATED_NOISE), metadata_language="en")
        assert result.content_language == "non_en"
        assert not result.eligible_for_indexing

    def test_non_english_metadata_blocks_eligibility_whatever_the_content(self):
        result = language.assess_pages(pages(ENGLISH), metadata_language="hi")
        assert result.content_language == "en"
        assert not result.eligible_for_indexing
        assert any("not English" in r for r in result.reasons)

    def test_missing_metadata_language_blocks_eligibility(self):
        result = language.assess_pages(pages(ENGLISH), metadata_language=None)
        assert not result.eligible_for_indexing


class TestNoisyButEnglish:
    def test_heavily_damaged_english_is_still_english(self):
        # The National Food Security Act regression. Its text is unusable, but
        # that is a quality verdict; the language is not in doubt.
        result = language.assess_text(DAMAGED_ENGLISH)
        assert result.content_language == "en"

    def test_damaged_english_scores_far_above_transliterated_noise(self):
        english = language.assess_text(DAMAGED_ENGLISH)
        noise = language.assess_text(TRANSLITERATED_NOISE)
        assert english.signals["function_word_rate"] > \
            2 * noise.signals["function_word_rate"]


class TestShortAndAmbiguousDocuments:
    def test_a_short_but_clearly_english_document_is_accepted(self):
        # A one-page repeal act is short by nature, not doubtful.
        text = (
            "THE RAJASTHAN SPECIAL COURTS (REPEAL) ACT, 2020\n"
            "An Act to repeal the Rajasthan Special Courts Act, 2012.\n"
            "Be it enacted by the Rajasthan State Legislature in the Seventy-first "
            "Year of the Republic of India as follows:—\n"
            "1. Short title and commencement.—(1) This Act may be called the "
            "Rajasthan Special Courts (Repeal) Act, 2020.\n"
            "(2) It shall come into force at once.\n"
            "2. Repeal and savings.—The Rajasthan Special Courts Act, 2012 is "
            "hereby repealed, but such repeal shall not affect anything duly done "
            "or suffered under the said Act before the commencement of this Act.\n"
        )
        result = language.assess_text(text)
        assert result.content_language == "en"
        assert result.signals["clean_word_count"] < config.LANGUAGE_MIN_CLEAN_WORDS
        assert any("short document" in r for r in result.reasons)

    def test_a_document_with_almost_no_text_is_uncertain_not_rejected(self):
        result = language.assess_text("Government of West Bengal\n1948\n")
        assert result.content_language == "uncertain"
        assert not result.eligible_for_indexing
        assert any("cannot be established" in r for r in result.reasons)

    def test_uncertain_is_reached_between_the_two_bounds(self):
        # Enough words to judge, but the readable ones are only borderline
        # English: neither confirmed nor ruled out.
        text = ("Ram Shyam Gopal Krishna Radha Sita Lakshmi Ganga Yamuna Kaveri "
                "Narmada Godavari Tapti Mahanadi Chenab Ravi Beas Sutlej of the ") * 20
        result = language.assess_text(text)
        assert result.content_language in ("uncertain", "non_en")
        assert not result.eligible_for_indexing


class TestEvidence:
    def test_every_verdict_carries_its_signals_and_reasons(self):
        for text in (ENGLISH, TRANSLITERATED_NOISE, "short"):
            result = language.assess_text(text)
            assert result.reasons
            assert "function_word_rate" in result.signals
            assert "thresholds" in result.signals

    def test_script_profile_counts_letters_only(self):
        # Digits and punctuation are shared between scripts and would dilute the
        # ratio. Combining vowel signs are not letters either, which is why the
        # Devanagari count is 2 rather than 3.
        profile = language.script_profile("abc 123 ... देव")
        assert profile["letters"] == 5
        assert profile["by_script"]["LATIN"] == 3
        assert profile["by_script"]["DEVANAGARI"] == 2

    def test_clean_words_exclude_ocr_wreckage(self):
        words = language.clean_words("the publiL' SerVi!ll1 Court of India t1diO")
        assert "the" in words and "court" in words and "india" in words
        assert "publil" not in words and "servi" not in words

    def test_clean_words_keep_hyphenated_compounds(self):
        # "White-browed" is two correctly-cased words, not a mis-cased one.
        assert language.clean_words("White-browed Fulvetta") == \
            ["white", "browed", "fulvetta"]


class TestBilingualDocumentsAndLegalStructure:
    """What page routing is ultimately protecting: the section index.

    Language routing would be an accounting change if it stopped at a verdict.
    It does not -- ``process_document`` parses structure over the indexable
    pages, and that is what keeps Devanagari out of the legal hierarchy.
    """

    def test_no_legal_unit_is_emitted_from_a_non_english_page(self):
        """Devanagari must never be emitted as a provision of English law.

        Observed on a real corpus document before this was fixed
        (``apprenticeship-amendment-rules-2019``): 8 of its 9 detected sections
        were Devanagari, each one a ``LegalUnit`` with Hindi body text, indexed
        as Indian law in English.
        """
        page_list = pages(ENGLISH_SECTION_PAGE, ENGLISH,
                          HINDI_SECTION_PAGE, HINDI_SECTION_PAGE)
        assessment = language.assess_pages(page_list, metadata_language="en")
        assert assessment.content_language == "bilingual_en"

        keep = language.indexable_pages(page_list, assessment)
        structure = parse_structure(keep, metadata_title="Model Rules, 2019")
        excluded = {3, 4}
        for unit in structure.all_units():
            assert unit.page_start not in excluded, (
                f"a {unit.unit_type} was emitted from Devanagari page "
                f"{unit.page_start}: {(unit.heading or unit.text or '')[:40]!r}")

    def test_the_translation_had_been_crowding_out_the_real_sections(self):
        """Not merely extra units -- the English ones were being displaced.

        Only one section style is used per document, so eight numbered Hindi
        provisions per page outvoted the two real English ones and took the
        index for themselves. Parsed over all four pages this document's entire
        section index is Devanagari; over the English pages it is the two
        sections the act actually has.
        """
        page_list = pages(ENGLISH_SECTION_PAGE, ENGLISH,
                          HINDI_SECTION_PAGE, HINDI_SECTION_PAGE)
        over_everything = parse_structure(page_list, metadata_title="Model Rules")
        sections = [u for u in over_everything.all_units()
                    if u.unit_type == "section"]
        assert sections, "fixture no longer reproduces the defect"
        assert all(u.page_start in (3, 4) for u in sections)

        assessment = language.assess_pages(page_list, metadata_language="en")
        keep = language.indexable_pages(page_list, assessment)
        recovered = [u for u in parse_structure(
            keep, metadata_title="Model Rules").all_units()
            if u.unit_type == "section"]
        assert [u.number for u in recovered] == ["1", "2"]
        assert all(u.page_start == 1 for u in recovered)

    def test_page_provenance_stays_absolute_after_filtering(self):
        """A citation names a page of the PDF, not a page of the filtered subset.

        ``structure.py`` works from ``line.page_number`` rather than from a
        position in the list it was handed, which is what makes filtering safe.
        If that ever changes, a bilingual document's sections start citing the
        wrong pages -- silently, and in a way only this test would notice.
        """
        page_list = pages(HINDI_SECTION_PAGE, HINDI_SECTION_PAGE,
                          ENGLISH_SECTION_PAGE, ENGLISH)
        assessment = language.assess_pages(page_list, metadata_language="en")
        keep = language.indexable_pages(page_list, assessment)
        assert [p.page_number for p in keep] == [3, 4]

        structure = parse_structure(keep, metadata_title="Model Rules, 2019")
        sections = [u for u in structure.all_units() if u.unit_type == "section"]
        assert sections
        # Page 3 of the PDF, not page 1 of the two pages that survived.
        assert all(u.page_start >= 3 for u in sections)


class TestLineLevelRouting:
    """A page is not the right unit when both languages share one.

    Page-level routing was measured on the pilot and lost 4.2% of all English in
    bilingual documents. The loss was not spread evenly: a gazette commonly ends
    its Hindi text partway down a page and starts the English notification
    below, so the page excluded whole was the one carrying the rule's number,
    its date and the provision it was made under -- the part a citation needs
    most. Line-level routing recovers 66% of that.
    """

    def test_a_line_is_only_excluded_on_positive_evidence(self):
        assert language.classify_line(
            "These rules may be called the Model Rules, 2019.") == "en"
        assert language.classify_line(_HINDI_SENTENCE) == "non_en"
        # Too short to carry evidence: kept, because discarding text requires
        # evidence and "no Latin letters here" is not evidence.
        assert language.classify_line("(2)") == "neutral"
        assert language.classify_line("21") == "neutral"
        assert language.classify_line("") == "neutral"
        assert language.classify_line("   ") == "neutral"

    def test_a_transition_page_keeps_its_english_half(self):
        """The G.S.R. 495(E) case, reduced to its shape.

        Real document: the Hindi text ends mid-page and the English begins with
        NOTIFICATION / New Delhi / G.S.R.495(E). Excluding the page lost all of
        it.
        """
        transition = (
            f"{_HINDI_SENTENCE}\n{_HINDI_SENTENCE}\n{_HINDI_SENTENCE}\n"
            "\n"
            "NOTIFICATION\n"
            "New Delhi, the 23rd May, 2017\n"
            "1. Short title.\u2014These rules may be called the Prevention of "
            "Cruelty to Animals Rules, 2017, and shall come into force at once.\n"
        ) + ENGLISH
        page_list = pages(ENGLISH_SECTION_PAGE, HINDI_SECTION_PAGE, transition)
        assessment = language.assess_pages(page_list, metadata_language="en")
        assert assessment.content_language == "bilingual_en"

        keep = language.indexable_pages(page_list, assessment)
        assert 3 in [pg.page_number for pg in keep], (
            "the transition page was excluded whole; its English is lost")

    def test_the_other_script_lines_are_labelled_not_removed(self):
        """``pages.json`` text stays byte for byte what the backend produced."""
        transition = f"{_HINDI_SENTENCE}\nThese rules may be called the Model "\
                     f"Rules, 2019 and come into force at once.\n"
        page = pages(transition)[0]
        marked = language.non_english_lines(page)
        assert [m["line_index"] for m in marked] == [0]
        assert "another script" in marked[0]["reason"]
        # The text is untouched; only the label knows.
        assert page.text == transition
        assert _HINDI_SENTENCE in page.text

    def test_english_line_text_drops_only_the_marked_lines(self):
        transition = f"{_HINDI_SENTENCE}\nThese rules may be called the Model "\
                     f"Rules, 2019 and come into force at once.\n(2)\n"
        page = pages(transition)[0]
        english = language.english_line_text(page)
        assert _HINDI_SENTENCE not in english
        assert "Model Rules, 2019" in english
        assert "(2)" in english          # neutral lines survive

    def test_a_provision_on_a_transition_page_reaches_the_structure_parser(self):
        """The whole point, asserted end to end.

        A section printed below the end of the Hindi text must be found, and
        must be cited at its real page number.
        """
        transition = (
            f"{_HINDI_SENTENCE}\n{_HINDI_SENTENCE}\n"
            "1. Short title and commencement.\u2014These rules may be called "
            "the Prevention of Cruelty to Animals Rules, 2017.\n"
            "2. Definitions.\u2014In these rules, unless the context otherwise "
            "requires, the expressions used shall have the meaning assigned.\n"
        ) + ENGLISH
        page_list = pages(ENGLISH, HINDI_SECTION_PAGE, transition)
        assessment = language.assess_pages(page_list, metadata_language="en")
        keep = language.indexable_pages(page_list, assessment)
        for pg in keep:
            pg.non_english_lines = language.non_english_lines(pg)

        structure = parse_structure(keep, metadata_title="PCA Rules, 2017")
        sections = [u for u in structure.all_units() if u.unit_type == "section"]
        assert [u.number for u in sections] == ["1", "2"]
        assert all(u.page_start == 3 for u in sections)
        for unit in structure.all_units():
            assert _HINDI_SENTENCE not in unit.text

    def test_a_stray_other_script_line_is_excluded_from_a_plain_english_act(self):
        """Marked whatever the document is, not only in bilingual ones.

        The old page-ratio rule ignored anything under 15% of pages, so a single
        Devanagari line inside an English act went into the index unremarked.
        """
        page = pages(ENGLISH + "\n" + _HINDI_SENTENCE + "\n")[0]
        assert [m["line_index"] for m in language.non_english_lines(page)] == [
            len(page.text.split("\n")) - 2]


class TestControlledVocabulary:
    def test_only_the_three_documented_values_are_produced(self):
        for text in (ENGLISH, TRANSLITERATED_NOISE, DAMAGED_ENGLISH, "", "x y z"):
            assert language.assess_text(text).content_language in \
                config.CONTENT_LANGUAGES

    @pytest.mark.parametrize("value", ["non_en", "uncertain"])
    def test_non_english_and_uncertain_are_never_eligible(self, value):
        texts = {"non_en": TRANSLITERATED_NOISE, "uncertain": "too short"}
        result = language.assess_pages(pages(texts[value]), metadata_language="en")
        assert result.content_language == value
        assert not result.eligible_for_indexing


#: A page of Devanagari, as a bilingual gazette prints it. Real script, not a
#: transliteration — which is the case the page-level check exists for.
DEVANAGARI_PAGE = """
    अध्याय ३ कर का उद्ग्रहण और संग्रहण प्रदाय की परिधि संयुक्त और मिश्रित प्रदायों पर
    कर का दायित्व उद्ग्रहण और संग्रहण प्रदाय का समय और मूल्य माल का प्रदाय का समय
    सेवाओं के प्रदाय का समय माल या सेवाओं के प्रदाय के संबंध में कर की दर में परिवर्तन
    कराधेय प्रदाय का मूल्य इनपुट कर प्रत्यय लेने के लिए पात्रता और शर्तें प्रत्यय की
    विशेष परिस्थितियों में प्रत्यय की उपलब्धता जाब वर्क के लिए किये गये इनपुट की भेजे
    गए पूंजी माल के संबंध में इनपुट कर प्रत्यय का लिया जाना इनपुट सेवा वितरक द्वारा
    प्रत्यय के वितरण की रीति आंशिक में वितरित प्रत्यय की वसूली की रीति रजिस्ट्रीकरण
""" * 4

#: A schedule of protected species, from the Wild Life (Protection) Act, 1972.
#: Two columns of names, no sentences, an English function-word rate of zero —
#: and unambiguously part of an English Central Act.
SPECIES_SCHEDULE = """
    36. Caracal Caracal caracal
    37. Cheetah Acinonyx jubatus
    38. Clouded Leopard Neofelis nebulosa
    39. Desert Cat Felis silvestris
    40. Eurasian Lynx Lynx lynx
    41. Fishing Cat Prionailurus viverrinus
    43. Leopard Panthera pardus
    44. Leopard Cat Prionailurus bengalensis
    45. Marbled Cat Pardofelis marmorata
    47. Rusty Spotted Cat Prionailurus rubiginosus
    48. Snow Leopard Panthera uncia
    63. Bearded Vulture Gypaetus barbatus
    65. Black Baza Aviceda leuphotes
    66. Black Eagle Ictinaetus malaiensis
    69. Brahminy Kite Haliastur indus
    72. Cinereous Vulture Aegypius monachus
""" * 4


class TestMixedLanguageDocuments:
    """A document that is part English and part written in another script.

    Such a document is routed **page by page**: the English pages are indexed
    and the others are not. It is neither accepted whole nor rejected whole,
    because both were wrong in practice and in opposite directions.

    Rejecting whole threw away real law. On the 1,000-document proportional
    pilot, 69 of the 95 documents called ``non_en`` scored 0.28-0.56 on English
    function words -- inside the healthy English band -- and were rejected only
    for carrying 25-36% Devanagari. They are gazette notifications printing the
    same rule in both languages on separate pages. Extrapolated: ~1,370 corpus
    documents.

    Accepting whole was worse and quieter. The old rule only fired above 15%
    non-English pages, so one Devanagari page in twenty passed unremarked **and
    was indexed as English law**. Page-level routing closes that.

    Two regressions this must not re-introduce, both tested below: judging pages
    on vocabulary rather than script quarantined the Wild Life (Protection) Act,
    1972 for its own schedules of species; and a document transliterated into
    Latin glyphs has pages that *read* as English on script, so script must
    never be trusted to clear a document vocabulary has not already cleared.
    """

    def test_a_wholly_english_document_is_not_called_mixed(self):
        result = language.assess_pages(
            pages(*[ENGLISH] * 10), metadata_language="en")
        assert result.content_language == "en"
        assert result.eligible_for_indexing
        assert result.signals["page_language"]["mixed_language"] is False
        assert result.signals["page_language"]["non_english_pages"] == 0

    def test_a_block_of_devanagari_pages_is_routed_page_by_page(self):
        """Previously this quarantined the whole document, English half included.

        The Devanagari pages must not be indexed -- that part has not changed
        and is asserted below. What changed is that the eight English pages are
        no longer thrown away with them.
        """
        page_list = pages(*([ENGLISH] * 8 + [DEVANAGARI_PAGE] * 2))
        result = language.assess_pages(page_list, metadata_language="en")
        profile = result.signals["page_language"]
        assert profile["non_english_pages"] == 2
        assert profile["non_english_page_numbers"] == [9, 10]
        assert profile["english_pages"] == 8
        assert result.content_language == "bilingual_en"
        assert result.eligible_for_indexing

        keep = language.indexable_pages(page_list, result)
        assert [p.page_number for p in keep] == [1, 2, 3, 4, 5, 6, 7, 8]
        assert 9 not in [p.page_number for p in keep]
        assert 10 not in [p.page_number for p in keep]

    def test_the_devanagari_pages_are_never_indexable(self):
        """The invariant the old whole-document quarantine was protecting.

        It survives the change, and now holds at any ratio rather than only
        above 15%.
        """
        for english, hindi in ((8, 2), (19, 1), (2, 8), (1, 1)):
            page_list = pages(*([ENGLISH] * english + [DEVANAGARI_PAGE] * hindi))
            result = language.assess_pages(page_list, metadata_language="en")
            keep = language.indexable_pages(page_list, result)
            for page in keep:
                assert language.classify_page(page)["verdict"] != "non_en", (
                    f"a Devanagari page was indexable at {english}/{hindi}")

    def test_the_pooled_check_alone_would_have_cleared_it(self):
        """The reason the page-level pass exists, stated as a test.

        If this starts failing because the pooled check catches the document
        too, the page-level pass has stopped being what protects against this.
        """
        pooled = language.assess_text(
            "\n".join([ENGLISH] * 8 + [DEVANAGARI_PAGE] * 2))
        assert pooled.content_language == "en"

    def test_species_schedules_are_not_evidence_of_another_language(self):
        """The Wild Life (Protection) Act, 1972 must stay eligible.

        Its schedules run to a third of the act and score zero on English
        function words. Judging pages on script rather than on vocabulary is
        what keeps them out of the count.
        """
        result = language.assess_pages(
            pages(*([ENGLISH] * 6 + [SPECIES_SCHEDULE] * 6)),
            metadata_language="en",
        )
        assert result.signals["page_language"]["non_english_pages"] == 0
        assert result.content_language == "en"
        assert result.eligible_for_indexing

    def test_one_devanagari_page_is_excluded_rather_than_indexed(self):
        """The hole the old ratio rule left open.

        One Devanagari page in twenty is 5% -- under the old 15% limit -- so the
        document was cleared as English *and the Devanagari page went into the
        index as English law*. Nothing objected, because nothing looked.

        The document is still usable; it is the page that is excluded.
        """
        page_list = pages(*([ENGLISH] * 19 + [DEVANAGARI_PAGE]))
        result = language.assess_pages(page_list, metadata_language="en")
        assert result.signals["page_language"]["mixed_language"] is False
        assert result.content_language == "bilingual_en"
        assert result.eligible_for_indexing
        keep = language.indexable_pages(page_list, result)
        assert len(keep) == 19
        assert 20 not in [p.page_number for p in keep]

    def test_short_pages_are_not_counted_either_way(self):
        """Covers and part-title pages establish nothing about language."""
        result = language.assess_pages(
            pages(*([ENGLISH] * 6 + ["CHAPTER IV", "SCHEDULE II", ""])),
            metadata_language="en",
        )
        assert result.signals["page_language"]["measurable_pages"] == 6
        assert result.content_language == "en"

    def test_a_quoted_hindi_title_does_not_make_a_page_non_english(self):
        page = ENGLISH + "\nपासपोर्ट अधिनियम, 1967\n"
        profile = language.page_language_profile(pages(*[page] * 5))
        assert profile["non_english_pages"] == 0

    def test_script_never_clears_what_vocabulary_rejected(self):
        """Why the two tests are ordered, and why neither is enough alone.

        A transliterated Devanagari document extracts as *Latin* glyphs, so
        every one of its pages reads as English on script. If page-level script
        were allowed to pick indexable pages inside a document the vocabulary
        test had rejected, the whole thing would be indexed as English law.

        Vocabulary rejects the document; script then only chooses among the
        pages of documents vocabulary has already cleared.
        """
        page_list = pages(*[TRANSLITERATED_NOISE] * 6)
        result = language.assess_pages(page_list, metadata_language="en")
        assert result.content_language == "non_en"
        assert language.indexable_pages(page_list, result) == []

    def test_an_ordinary_english_document_keeps_every_page(self):
        page_list = pages(*[ENGLISH] * 10)
        result = language.assess_pages(page_list, metadata_language="en")
        assert result.content_language == "en"
        assert len(language.indexable_pages(page_list, result)) == 10

    def test_the_indexable_list_is_not_truncated_at_fifty(self):
        """``english_page_numbers`` is capped at 50 for display.

        Building the indexable list from it would silently drop every English
        page after the fiftieth in a long bilingual act.
        """
        page_list = pages(*([ENGLISH] * 120 + [DEVANAGARI_PAGE] * 10))
        result = language.assess_pages(page_list, metadata_language="en")
        assert result.content_language == "bilingual_en"
        assert len(language.indexable_pages(page_list, result)) == 120

    def test_short_pages_stay_in_rather_than_leaving_a_hole(self):
        """A cover carries no language and dropping it would break the run."""
        page_list = pages(*(["CHAPTER IV"] + [ENGLISH] * 6 + [DEVANAGARI_PAGE] * 2))
        result = language.assess_pages(page_list, metadata_language="en")
        assert result.content_language == "bilingual_en"
        kept = [p.page_number for p in language.indexable_pages(page_list, result)]
        assert 1 in kept                      # the short cover page
        assert kept == [1, 2, 3, 4, 5, 6, 7]

    def test_a_wholly_non_english_document_stays_non_en(self):
        """``uncertain`` would be a weaker and less accurate answer than
        ``non_en``, so the mixed-language rule must not reach a document the
        pooled check has already settled.
        """
        result = language.assess_pages(
            pages(*[TRANSLITERATED_NOISE] * 6), metadata_language="en")
        assert result.content_language == "non_en"
        assert not result.eligible_for_indexing

    def test_damaged_english_pages_are_not_mistaken_for_another_language(self):
        """Corrupted English is a quality problem, not a language one.

        If OCR damage counted as evidence of another language, every scanned
        gazette in the corpus would be quarantined for the wrong reason and the
        real fix — re-OCR — would never be scheduled.
        """
        result = language.assess_pages(
            pages(*([ENGLISH] * 6 + [DAMAGED_ENGLISH] * 4)),
            metadata_language="en",
        )
        assert result.signals["page_language"]["non_english_pages"] == 0
        assert result.content_language == "en"

    def test_the_profile_records_its_basis_and_thresholds(self):
        profile = language.page_language_profile(pages(ENGLISH))
        assert profile["basis"] == "non_latin_script"
        assert profile["thresholds"] == {
            "page_min_letters": config.LANGUAGE_PAGE_MIN_LETTERS,
            "non_latin_letter_ratio": config.LANGUAGE_NON_LATIN_LETTER_RATIO,
            "mixed_non_en_page_ratio": config.LANGUAGE_MIXED_NON_EN_PAGE_RATIO,
        }

    def test_a_document_with_no_measurable_pages_is_not_mixed(self):
        profile = language.page_language_profile(pages("", "CHAPTER I"))
        assert profile["measurable_pages"] == 0
        assert profile["mixed_language"] is False
