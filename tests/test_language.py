"""Unit tests for the English-only language gate.

The property under test is asymmetric and deliberately so:

* any credible Hindi signal rejects a file, and
* English is only ever accepted on positive evidence — never because a file
  merely failed to look Hindi.
"""

from __future__ import annotations

import pytest

from ingestion import language
from ingestion.errors import LanguageError
from ingestion.models import BitstreamRef, ParsedItem

ENGLISH_TITLES = ["The Passports Act, 1967", "Passports Act, 1967"]
HINDI_TITLES = ["पासपोर्ट अधिनियम, 1967"]


def classify(**kwargs):
    kwargs.setdefault("filename", "file.pdf")
    kwargs.setdefault("english_titles", ENGLISH_TITLES)
    kwargs.setdefault("hindi_titles", HINDI_TITLES)
    return language.classify_bitstream(**kwargs)


class TestHindiIsRejected:
    def test_explicit_hindi_label(self):
        verdict = classify(filename="196715.pdf", explicit_label="Hindi")
        assert verdict.language == "hi"
        assert verdict.source == "indiacode_bitstream_label"

    def test_devanagari_label(self):
        assert classify(explicit_label="हिंदी").language == "hi"

    def test_link_text_matches_hindi_title_field(self):
        verdict = classify(filename="H1967-15.pdf", link_text="पासपोर्ट अधिनियम, 1967")
        assert verdict.language == "hi"
        assert verdict.source == "indiacode_hindi_title"

    def test_devanagari_link_text_without_a_matching_title(self):
        verdict = classify(filename="x.pdf", link_text="कोई अन्य अधिनियम, 1971")
        assert verdict.language == "hi"
        assert verdict.source == "indiacode_devanagari_script"

    def test_hindi_filename_convention(self):
        verdict = classify(filename="H1967-15.pdf", link_text="")
        assert verdict.language == "hi"
        assert verdict.source == "indiacode_filename_pattern"

    def test_an_english_title_quoting_a_hindi_phrase_is_not_hindi(self):
        # Real case: "The Viksit Bharat—Guarantee for Rozgar And Ajeevika
        # Mission (Gramin): VB—G Ram G (विकसित भारत—जी राम जी) Act, 2025" is an
        # English act whose title quotes Hindi. Containing Devanagari is not
        # the same as being written in Devanagari.
        title = (
            "The Viksit Bharat—Guarantee for Rozgar And Ajeevika Mission "
            "(Gramin): VB—G Ram G (विकसित "
            "भारत—जी राम "
            "जी)Act, 2025"
        )
        verdict = classify(
            filename="a2025-36.pdf", link_text=title,
            english_titles=[title], hindi_titles=[],
        )
        assert verdict.language == "en"
        assert verdict.source == "indiacode_metadata_title"

    def test_a_wholly_devanagari_title_is_still_hindi(self):
        assert classify(filename="x.pdf", link_text="पासपोर्ट अधिनियम, 1967").language == "hi"

    def test_hindi_evidence_beats_the_citation_pdf_rule(self):
        # Even the item's own citation PDF is rejected when it is Hindi.
        verdict = classify(
            filename="H1975-40.pdf", link_text="हिंदी अधिनियम, 1975", is_citation_pdf=True
        )
        assert verdict.language == "hi"


class TestEnglishMustBeProven:
    def test_explicit_english_label(self):
        verdict = classify(filename="196715.pdf", explicit_label="English")
        assert verdict.language == "en"
        assert verdict.source == "indiacode_bitstream_label"

    def test_english_label_as_a_short_phrase(self):
        assert classify(explicit_label="English version").language == "en"

    def test_link_text_matches_india_code_english_title(self):
        verdict = classify(filename="196715.pdf", link_text="The Passports Act, 1967")
        assert verdict.language == "en"
        assert verdict.source == "indiacode_metadata_title"

    def test_leading_article_and_punctuation_are_ignored_when_matching(self):
        # "Short Title" and "DC.title" differ only by "The" / commas.
        assert classify(link_text="Passports Act 1967").language == "en"

    def test_citation_pdf_of_a_latin_titled_item(self):
        verdict = classify(filename="196715.pdf", link_text="", is_citation_pdf=True)
        assert verdict.language == "en"
        assert verdict.source == "indiacode_citation_pdf_url"

    def test_absence_of_hindi_signals_is_not_english(self):
        # The core rule: an ordinary-looking filename with no evidence at all
        # stays undetermined instead of defaulting to English.
        verdict = classify(filename="document123.pdf", link_text="Download PDF")
        assert verdict.is_undetermined
        assert verdict.language is None

    def test_a_title_containing_the_word_english_is_not_a_language_label(self):
        verdict = classify(
            filename="a1968-33.pdf",
            link_text="The English Language Education Act, 1968",
            english_titles=[],
            hindi_titles=[],
        )
        assert verdict.is_undetermined


class TestIsDevanagariText:
    @pytest.mark.parametrize("text,expected", [
        ("पासपोर्ट अधिनियम, 1967", True),
        ("हिंदी", True),
        ("The Passports Act, 1967", False),
        ("The Viksit Bharat Act (विकसित भारत) 2025", False),  # mostly Latin
        ("", False),
        (None, False),
    ])
    def test_script_dominance(self, text, expected):
        assert language.is_devanagari_text(text) is expected

    def test_contains_is_weaker_than_is(self):
        mixed = "The Viksit Bharat Act (विकसित भारत) 2025"
        assert language.contains_devanagari(mixed) is True
        assert language.is_devanagari_text(mixed) is False


class TestLanguageLabelOf:
    @pytest.mark.parametrize("text,expected", [
        ("English", "en"),
        ("  english  ", "en"),
        ("English version", "en"),
        ("Hindi", "hi"),
        ("हिंदी", "hi"),
        ("PDF in Hindi", "hi"),
        ("The Passports Act, 1967", None),
        ("254 KB", None),
        ("", None),
        (None, None),
    ])
    def test_labels(self, text, expected):
        assert language.language_label_of(text) == expected

    def test_phrase_matching_is_opt_out(self):
        # With allow_phrase=False the whole string must *be* a language name,
        # which is how a file's link text (normally a title) is treated.
        assert language.language_label_of("PDF in Hindi", allow_phrase=False) is None
        assert language.language_label_of("Hindi", allow_phrase=False) == "hi"


def _item(*bitstreams, requested=None):
    return ParsedItem(
        source_url="https://www.indiacode.nic.in/handle/123456789/1372",
        handle="123456789/1372",
        handle_id="1372",
        title="Passports Act, 1967",
        bitstreams=list(bitstreams),
        requested_bitstream_url=requested,
    )


def _ref(filename, lang, *, primary=False, url=None):
    return BitstreamRef(
        url=url or f"https://www.indiacode.nic.in/bitstream/123456789/1372/1/{filename}",
        filename=filename,
        is_primary=primary,
        language=lang,
        language_source="test",
    )


class TestSelectEnglishBitstream:
    def test_picks_the_english_file_when_hindi_is_also_offered(self):
        english = _ref("196715.pdf", "en", primary=True)
        hindi = _ref("H1967-15.pdf", "hi")
        assert language.select_english_bitstream(_item(english, hindi)) is english

    def test_single_english_file(self):
        english = _ref("196715.pdf", "en", primary=True)
        assert language.select_english_bitstream(_item(english)) is english

    def test_hindi_only_item_is_refused(self):
        with pytest.raises(LanguageError, match="No English version"):
            language.select_english_bitstream(_item(_ref("H1967-15.pdf", "hi", primary=True)))

    def test_undetermined_language_is_refused(self):
        with pytest.raises(LanguageError, match="could not be determined"):
            language.select_english_bitstream(_item(_ref("doc.pdf", None, primary=True)))

    def test_two_english_files_resolve_to_the_primary_one(self):
        primary = _ref("196715.pdf", "en", primary=True)
        extra = _ref("196715-schedule.pdf", "en")
        assert language.select_english_bitstream(_item(primary, extra)) is primary

    def test_two_english_files_without_a_primary_is_refused(self):
        first = _ref("a.pdf", "en")
        second = _ref("b.pdf", "en")
        with pytest.raises(LanguageError, match="refusing to guess"):
            language.select_english_bitstream(_item(first, second))

    def test_item_without_files_is_refused(self):
        with pytest.raises(LanguageError):
            language.select_english_bitstream(_item())

    def test_requested_bitstream_confines_the_choice(self):
        english = _ref("196715.pdf", "en", primary=True)
        hindi = _ref("H1967-15.pdf", "hi")
        # Asking for the Hindi file directly must not silently yield the English
        # one — and must not download the Hindi one either.
        with pytest.raises(LanguageError, match="No English version"):
            language.select_english_bitstream(_item(english, hindi, requested=hindi.url))
        item = _item(english, hindi, requested=english.url)
        assert language.select_english_bitstream(item) is english

    def test_requested_bitstream_missing_from_the_page_is_refused(self):
        english = _ref("196715.pdf", "en", primary=True)
        item = _item(english, requested="https://www.indiacode.nic.in/bitstream/1/2/3/x.pdf")
        with pytest.raises(LanguageError, match="was not found"):
            language.select_english_bitstream(item)
