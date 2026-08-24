"""Repairing OCR word spacing without inventing words.

The engine comparison measured RapidOCR at a 10.5% space ratio against 15.5%
for the PDFs' own text layers, and 6.3% of its tokens at 12+ characters against
2.6% for the embedded layer. The text is *read* correctly and the word
boundaries are gone::

    the power to revise the valuationandassessmentconferred
    bysection65of theMunicipalActand thepowertoamend

Both halves of this module's contract are tested here, and the second matters
more than the first:

1. it puts back spaces that were dropped; and
2. it never puts back a space that was not dropped.

Failing (2) would put words into a legal corpus that were never on the page,
which is worse than leaving a token glued — so most of what follows is (2).
"""

from __future__ import annotations

import pytest

from processing import respace
from processing.respace import glued_token_rate, respace as repair


class TestCaseBoundaries:
    """``theMunicipalAct`` -> ``the Municipal Act``."""

    def test_a_lowercase_uppercase_boundary_is_a_lost_space(self):
        assert repair("theMunicipalActand").text == "the Municipal Actand"

    def test_a_single_letter_fragment_abandons_the_split(self):
        """``McGregor``-shaped tokens and stray initials are left alone."""
        assert repair("aBcdefghij").text == "aBcdefghij"

    def test_capitals_alone_are_not_a_boundary(self):
        assert repair("COMMISSIONER").text == "COMMISSIONER"


class TestDigitBoundaries:
    """``section65of`` -> ``section 65 of``, without breaking section numbers."""

    def test_a_word_glued_to_a_number_is_split(self):
        assert repair("bysection65of").text == "by section 65 of"

    @pytest.mark.parametrize(
        "token,intact",
        [("undersection65Aofthe", "65A"),
         ("inarticle21Aandthe", "21A"),
         ("after1stand2ndof", "1st")],
    )
    def test_section_numbering_and_ordinals_survive(self, token, intact):
        """``65A``, ``21A`` and ``1st`` must come through whole.

        A letter suffix on a section number is part of the number — ``s. 65A``
        is not ``s. 65`` — so splitting it would change a citation. These tokens
        are the hard case: the letters after the number are long *only* because
        more words are glued behind them, so a rule that measured the run's
        length would cut every one of them.
        """
        assert intact in repair(token).text

    def test_a_bare_number_is_untouched(self):
        assert repair("121,122,123,124").text == "121,122,123,124"

    def test_a_word_running_into_a_number_is_still_split(self):
        assert repair("Act30of1965").text == "Act 30 of 1965"


class TestLexiconSegmentation:
    """The part that needs a word list, and the part that most needs guarding."""

    def test_a_run_of_known_words_is_split(self):
        assert repair("thepowertoamend").text == "the power to amend"

    def test_known_words_may_adjoin_an_unrecognised_run(self):
        assert repair("shallbedeemedtohavebeen").text == "shallbedeemed to have been"

    def test_an_unrecognised_run_may_lead(self):
        assert repair("appointedbythe").text == "appointed by the"

    @pytest.mark.parametrize(
        "word",
        ["information", "therefore", "whosoever", "understanding",
         "notwithstanding", "permission", "consisting", "commencement"],
    )
    def test_a_real_word_beginning_with_a_function_word_is_never_split(self, word):
        """``information`` must not become ``in formation``.

        This is what the "at least two known words" rule buys, and it is the
        failure mode a small lexicon produces most readily.
        """
        assert repair(word).text == word

    def test_a_word_is_not_carved_up_from_the_middle(self):
        """``withholdwritten`` covers as ``withholdwr`` + ``it`` + ``ten``.

        Every piece is defensible on its own and the word is destroyed. The
        known-coverage floor is what rejects it.
        """
        assert repair("withholdwrittenpermission").text == "withholdwrittenpermission"

    def test_a_token_with_no_lexicon_support_is_left_alone(self):
        assert (repair("valuationandassessmentconferred").text
                == "valuationandassessmentconferred")

    def test_hyphenated_tokens_are_segmented_run_by_run(self):
        """The hyphen stays exactly where it was; the runs either side are split."""
        assert repair("shallbemadebythe-committee").text == \
            "shall be made by the-committee"

    def test_a_very_long_run_is_not_searched(self):
        """Beyond the cap this is OCR debris, not lost spacing."""
        debris = "the" * 40
        assert len(debris) > respace.MAX_SEGMENT_LENGTH
        assert repair(debris).text == debris


class TestDoesNotDamageGoodText:
    """The safety property. Correctly-spaced legal prose must come through
    byte-for-byte."""

    SPECIMEN = (
        "Notwithstanding anything contained in section 65A of the principal "
        "Act, the Commissioner shall, before the commencement of proceedings, "
        "communicate in writing to the person concerned the particulars of the "
        "contravention alleged against him and afford a reasonable opportunity "
        "of representation.\n"
        "(2) Every such notification shall be published in the Official "
        "Gazette and shall come into force on the 1st day of April, 2017.\n"
        "1. Substituted by Act 30 of 1965, s. 3, for sub-section (1) (w.e.f. "
        "15-10-1965).\n"
    )

    def test_well_spaced_prose_is_unchanged(self):
        result = repair(self.SPECIMEN)
        assert result.text == self.SPECIMEN
        assert result.spaces_inserted == 0
        assert result.tokens_changed == 0
        assert result.tokens_examined > 0        # it did look

    def test_line_structure_is_preserved(self):
        text = "theMunicipalAct\nsecond line\n\nfourth"
        assert repair(text).text.count("\n") == text.count("\n")

    def test_no_character_is_ever_removed_or_changed(self):
        text = "bysection65of theMunicipalActand thepowertoamend (2)(a) 1[State]"
        repaired = repair(text).text
        assert repaired.replace(" ", "") == text.replace(" ", "")

    def test_empty_input(self):
        assert repair("").text == ""


class TestReporting:
    """The repair has to be auditable, not merely applied."""

    def test_changed_tokens_are_recorded_with_their_originals(self):
        result = repair("thepowertoamend and unremarkable prose")
        assert result.tokens_changed == 1
        assert ("thepowertoamend", "the power to amend") in result.examples
        assert result.spaces_inserted == 3
        assert result.changed is True

    def test_the_summary_serialises(self):
        payload = repair("thepowertoamend").to_dict()
        assert payload["spaces_inserted"] == 3
        assert payload["examples"][0] == {
            "before": "thepowertoamend", "after": "the power to amend"}

    def test_examples_are_capped(self):
        result = repair(" ".join(["thepowertoamend"] * 30))
        assert len(result.to_dict(example_limit=5)["examples"]) == 5


class TestGluedTokenRate:
    """The metric the engine comparison reports."""

    def test_ordinary_prose_scores_low(self):
        """Real statutory English does contain long words ("Notwithstanding").

        The measure is a rate, not a rule: what makes it useful is the gap
        between well-spaced prose and glued output, not any single reading.
        """
        assert glued_token_rate(TestDoesNotDamageGoodText.SPECIMEN) < 0.10

    def test_glued_output_scores_high(self):
        text = "bysection65of theMunicipalActand thepowertoamendandrepealthesame"
        assert glued_token_rate(text) > 0.3

    def test_glued_output_and_prose_are_far_apart(self):
        glued = "bysection65of theMunicipalActand thepowertoamendandrepealthesame"
        assert (glued_token_rate(glued)
                > 3 * glued_token_rate(TestDoesNotDamageGoodText.SPECIMEN))

    def test_repair_lowers_the_rate(self):
        text = ("bysection65of theMunicipalActand thepowertoamend "
                "conditionsas maybeimposed byanyrulesmadebythe")
        assert glued_token_rate(repair(text).text) < glued_token_rate(text)

    def test_empty_text_has_no_rate(self):
        assert glued_token_rate("") == 0.0
