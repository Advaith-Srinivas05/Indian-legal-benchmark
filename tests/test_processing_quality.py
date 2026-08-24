"""Tests for the multi-signal extraction-quality assessment.

These exist because the first benchmark's single heuristic caught about two of
the eight documents a stronger check found. The fixtures are therefore built
from the real failures, and the load-bearing test is
:meth:`TestKnownBadDocuments.test_national_food_security_act_is_flagged` — that
document extracted "successfully" and is not quotable.

The other half of the job is not flagging sound documents, so the healthy
fixtures include the two shapes that fooled an earlier draft: schedules of
hyphenated species names, and documents that are simply short.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from processing import config, quality

CLEAN_LEGAL_TEXT = """
Whoever, being a public servant, knowingly disobeys any direction of the law as
to the way in which he is to conduct himself as such public servant, intending
to cause injury to any person, shall be punished with simple imprisonment for a
term which may extend to one year, or with fine, or with both. Provided that
nothing in this section shall apply to any act done in good faith by an officer
of the Central Government in the discharge of his official duty under any rule
made under this Act. Explanation.—In this section, the expression "public
servant" has the meaning assigned to it in section 21 of the Indian Penal Code.
""" * 4

#: Verbatim in shape from national-food-security-act-2013's OCR layer.
NFSA_OCR = """
33. Any publiL' SerVi!ll1 or authorilY found guilty, by the Slate COlllmis~i()n at
lhl: time uf deeiding any eomplaint or ;lppeal. uf f"illllg to pruvick the relief
recommcndeJ by the Di~lrid G,.iev~ll".:e !(cd,css;t! Ofri\\:cr, withoul reavlIlabk
causc. or II'ilfully Ignoring .-;ud1 Iccolllillendatiull. shall be l1able to penalty
Ilot exceeding five thuusand rupees: M lSCELLAI'EOUS CJ-I,\\JYrER XTTI Ac"1 I" ha\\'c
o\\'crr;dillg etl~cl P,."~r In ",nellll SLh~',htle, Oll"" \\'clf;lre <chctll,·s, 1\\)lI'cr 11'
udcg:J1C by C~llIl:oI C;l>\\ cr"ml'n! 'lliU Slale C;""crnml'nl, I'"wer "r ('c",,~1
(io"",nmenl tll IP"C Ulfcclion, I'ower to aJj\\lUi~'''ll', Slcp, to further ",Jva'K~'
""" * 6

#: Verbatim in shape from the-gujarat-civil-services-tribunal-regulations-1977,
#: a mirrored Gujarati scan: shattered into single characters.
SHATTERED = """
-~~re U11~1.;: 1:"F.i1 f,-,,~J ~1.,J 13ft l l'ld lltiD !Ja11t1}-ll<l.AS1 n u~m lbM<t
lJ,1itlbkfL 11,1 lt '1,r:: 16":·cs~ lrht ·1~1-m1L ~11,-' ·1,1:2 u,1;,c J'!t1b~ U 11<JNJ
1}-U{IHc 1bra1<t 1,11_: 1~, ll c }hlLi:1 I.!-'?: 11;J i:1~ ll:1>'11,lr!:'. t hl,1:JbJ-1?~ E
·1~~fl:'! lbl-)rn~ tctrnh. lit1H:1~1,, l}Gl't: u·1:: 1~1"t ~~~ lbH l-)S11~ 11_2fl- 'k3l:-
""" * 8


def pages(*texts: str):
    return [SimpleNamespace(text=t, page_number=i)
            for i, t in enumerate(texts, start=1)]


class TestHealthyText:
    def test_clean_legal_prose_is_good(self):
        result = quality.assess(pages(CLEAN_LEGAL_TEXT))
        assert result.classification == "good"
        assert result.score >= config.QUALITY_GOOD_SCORE
        assert result.failed_checks == []
        assert result.is_usable

    def test_a_schedule_of_hyphenated_species_names_is_not_damage(self):
        # The Wildlife Protection Act regression: "White-browed Fulvetta" is two
        # correctly-cased words, and an earlier draft read every one of them as
        # a mis-cased word and flagged a sound Central Act.
        birds = (
            "White-browed Fulvetta, Long-tailed Shrike, Black-throated Thrush, "
            "Jerdon's Baza, Hume's Wheatear, White-cheeked Barbet, Blyth's Pipit, "
            "Grey-headed Fish Eagle, Slender-billed Vulture, Red-necked Falcon. "
        ) * 10
        result = quality.assess(pages(CLEAN_LEGAL_TEXT + birds))
        assert result.classification == "good"
        assert "mixed_case_rate" not in result.failed_checks

    def test_a_short_document_is_still_measured(self):
        short = (
            "1. Short title and commencement.—(1) This Act may be called the "
            "Rajasthan Special Courts (Repeal) Act, 2020, and it shall come into "
            "force at once in the whole of the State of Rajasthan.\n"
            "2. Repeal and savings.—The Rajasthan Special Courts Act, 2012 is "
            "hereby repealed, but such repeal shall not affect anything duly done "
            "or suffered under the said Act before this commencement.\n"
        )
        result = quality.assess(pages(short))
        assert not result.insufficient_text
        assert result.classification == "good"

    def test_text_below_the_measurable_floor_is_questionable_not_bad(self):
        result = quality.assess(pages("Government of West Bengal 1948"))
        assert result.insufficient_text
        assert result.classification == "questionable"
        assert result.failed_checks == ["insufficient_text"]
        assert any("not the same as the text being bad" in r for r in result.reasons)


class TestKnownBadDocuments:
    def test_national_food_security_act_is_flagged(self):
        # The explicit requirement. This document extracted "successfully":
        # pages, characters and section numbers all came out.
        result = quality.assess(pages(NFSA_OCR))
        assert result.classification == "bad"
        assert not result.is_usable
        assert "mean_word_length" in result.failed_checks
        assert "single_char_rate" in result.failed_checks

    def test_shattered_ocr_is_flagged(self):
        result = quality.assess(pages(SHATTERED))
        assert result.classification == "bad"
        assert result.score < config.QUALITY_QUESTIONABLE_SCORE

    def test_the_old_single_signal_would_have_missed_the_food_security_act(self):
        # Documents the regression: the retired heuristic looked only at
        # vowel-free words and stray symbols, and this text clears both.
        signals = quality.text_signals(NFSA_OCR)
        assert signals["vowelless_ratio"] < 0.20
        assert signals["symbol_ratio"] < 0.08
        # …and the panel still catches it.
        assert quality.assess(pages(NFSA_OCR)).classification == "bad"


class TestSignalPanel:
    def test_intra_word_capitals_are_measured(self):
        signals = quality.text_signals("publiL authorilY SerVi normal words here")
        assert signals["mixed_case_rate"] > 0

    def test_digits_inside_words_are_measured(self):
        signals = quality.text_signals("t1diO SerVi1l1 plain words")
        assert signals["alnum_mix_rate"] > 0

    def test_replacement_characters_are_counted(self):
        assert quality.text_signals("a � b")["replacement_char_count"] == 1

    def test_every_failed_check_produces_a_reason(self):
        result = quality.assess(pages(NFSA_OCR))
        assert len(result.reasons) == len(result.failed_checks)
        for reason in result.reasons:
            assert any(character.isdigit() for character in reason)

    def test_score_is_declared_not_to_be_a_probability(self):
        note = quality.assess(pages(CLEAN_LEGAL_TEXT)).to_dict()["note"]
        assert "not a calibrated probability" in note

    def test_classification_is_from_the_controlled_vocabulary(self):
        for text in (CLEAN_LEGAL_TEXT, NFSA_OCR, SHATTERED, "tiny"):
            assert quality.assess(pages(text)).classification in config.QUALITY_LEVELS


class TestPageConsistency:
    def test_a_document_with_a_few_ruined_pages_is_distinguished(self):
        good_pages = pages(*([CLEAN_LEGAL_TEXT] * 9))
        mixed = good_pages + pages(NFSA_OCR)
        result = quality.assess(mixed)
        assert result.signals["suspect_page_count"] >= 1
        # One bad page in ten does not condemn the document.
        assert result.classification == "good"

    def test_a_uniformly_bad_document_fails_consistency(self):
        result = quality.assess(pages(*([NFSA_OCR] * 5)))
        assert "page_consistency" in result.failed_checks

    def test_worst_pages_are_reported_for_review(self):
        mixed = pages(CLEAN_LEGAL_TEXT, NFSA_OCR, CLEAN_LEGAL_TEXT)
        assert quality.worst_pages(mixed, limit=1) == [2]


class TestSectionNumberPlausibility:
    def _structure(self, numbers):
        units = [
            SimpleNamespace(unit_type="section", number=str(n)) for n in numbers
        ]
        return SimpleNamespace(all_units=lambda: iter(units))

    def test_ascending_numbering_passes(self):
        result = quality.section_number_plausibility(
            self._structure([1, 2, 3, 4, 5, 6]))
        assert result == 1.0

    def test_scrambled_numbering_scores_low(self):
        result = quality.section_number_plausibility(
            self._structure([9, 2, 77, 1, 5, 3]))
        assert result < 0.5

    def test_too_few_sections_is_not_evidence(self):
        assert quality.section_number_plausibility(self._structure([1, 2])) is None

    def test_no_structure_is_not_evidence(self):
        assert quality.section_number_plausibility(None) is None

    def test_the_signal_feeds_the_panel(self):
        scrambled = quality.assess(
            pages(CLEAN_LEGAL_TEXT),
            structure=self._structure([9, 2, 77, 1, 5, 3]),
        )
        assert "section_number_plausibility" in scrambled.failed_checks
        assert "section_number_ascending_ratio" in scrambled.signals


class TestPageLevelSuspicion:
    def test_a_clean_page_is_not_suspect(self):
        assert not quality.page_is_suspect(CLEAN_LEGAL_TEXT)

    def test_a_damaged_page_is_suspect(self):
        assert quality.page_is_suspect(NFSA_OCR)

    def test_a_page_with_too_little_text_is_not_judged(self):
        assert not quality.page_is_suspect("12")


class TestPageLevelIndexability:
    """``indexable_pages`` — kept as a diagnostic, NOT wired to eligibility.

    Built to let a quarantined document contribute its sound pages, and reverted
    when the pilot showed the pages it admits are corrupted in ways the
    page-level checks do not see. It stays because the per-page verdict is worth
    recording and because a future word-validity signal would slot in here — but
    ``process_document`` does not use it to decide indexability, and
    ``ProcessedDocument.eligible_for_indexing`` still asks for a good document.
    """

    def _assessment(self, classification="good", insufficient=False):
        return SimpleNamespace(
            classification=classification, insufficient_text=insufficient)

    def test_a_clean_page_in_a_bad_document_survives(self):
        page_list = pages(CLEAN_LEGAL_TEXT, NFSA_OCR, NFSA_OCR)
        kept = quality.indexable_pages(page_list, self._assessment("bad"))
        assert [p.page_number for p in kept] == [1]

    def test_a_damaged_page_in_a_good_document_is_dropped(self):
        page_list = pages(CLEAN_LEGAL_TEXT, NFSA_OCR)
        kept = quality.indexable_pages(page_list, self._assessment("good"))
        assert [p.page_number for p in kept] == [1]

    def test_a_document_with_no_sound_page_keeps_none(self):
        kept = quality.indexable_pages(
            pages(NFSA_OCR, NFSA_OCR), self._assessment("bad"))
        assert kept == []

    def test_an_unjudgeable_page_follows_a_good_document(self):
        # A cover or a part-title carries no evidence of its own.
        kept = quality.indexable_pages(
            pages(CLEAN_LEGAL_TEXT, "SCHEDULE II"), self._assessment("good"))
        assert [p.page_number for p in kept] == [1, 2]

    def test_an_unjudgeable_page_does_not_follow_a_bad_one(self):
        kept = quality.indexable_pages(
            pages(NFSA_OCR, "SCHEDULE II"), self._assessment("bad"))
        assert kept == []

    def test_a_document_whose_quality_was_never_established_keeps_nothing(self):
        # insufficient_text is not a quality verdict, and must not read as one.
        kept = quality.indexable_pages(
            pages(CLEAN_LEGAL_TEXT), self._assessment("questionable", insufficient=True))
        assert kept == []

    def test_the_text_to_judge_can_be_overridden(self):
        # process.py judges the English lines, not the raw page.
        page_list = pages(NFSA_OCR)
        kept = quality.indexable_pages(
            page_list, self._assessment("good"),
            page_text=lambda p: CLEAN_LEGAL_TEXT)
        assert len(kept) == 1

    def test_the_gate_applies_the_same_thresholds_as_the_panel(self):
        # Not a relaxation: a page the panel calls suspect is the page this
        # drops, on the identical checks.
        for text in (CLEAN_LEGAL_TEXT, NFSA_OCR):
            page_list = pages(text)
            kept = quality.indexable_pages(page_list, self._assessment("good"))
            assert bool(kept) is not quality.page_is_suspect(text)


class TestPageVerdict:
    def test_a_clean_page_reads_good(self):
        assert quality.page_verdict(CLEAN_LEGAL_TEXT)["verdict"] == "good"

    def test_a_damaged_page_reads_suspect_and_says_why(self):
        verdict = quality.page_verdict(NFSA_OCR)
        assert verdict["verdict"] == "suspect"
        assert verdict["reason"]

    def test_a_short_page_is_unjudged_not_bad(self):
        verdict = quality.page_verdict("SCHEDULE II")
        assert verdict["verdict"] == "unjudged"
        assert "not the same as bad text" in verdict["reason"]
