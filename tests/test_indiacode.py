"""Tests for India Code URL handling and landing-page parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion import indiacode, language
from ingestion.errors import InvalidURLError, LanguageError

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "passports_act_1372.html"
BASE_URL = "https://www.indiacode.nic.in/handle/123456789/1372"


def parse_fixture(name: str, handle_id: str = "1372"):
    url = f"https://www.indiacode.nic.in/handle/123456789/{handle_id}"
    html = (FIXTURES / name).read_text(encoding="utf-8")
    return indiacode.parse_item_page(html, base_url=url, source_url=url)


@pytest.fixture(scope="module")
def parsed():
    html = FIXTURE.read_text(encoding="utf-8")
    return indiacode.parse_item_page(html, base_url=BASE_URL, source_url=BASE_URL)


class TestNormaliseUrl:
    def test_handle_url(self):
        kind, canonical, handle, filename = indiacode.normalise_url(BASE_URL)
        assert kind == "handle"
        assert handle == "123456789/1372"
        assert canonical == BASE_URL
        assert filename is None

    def test_bitstream_url(self):
        url = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
        kind, canonical, handle, filename = indiacode.normalise_url(url)
        assert kind == "bitstream"
        assert handle == "123456789/1372"
        assert filename == "196715.pdf"

    def test_handle_net_is_rewritten(self):
        kind, canonical, handle, _ = indiacode.normalise_url(
            "https://hdl.handle.net/123456789/1372"
        )
        assert kind == "handle"
        assert handle == "123456789/1372"
        assert canonical.startswith("https://www.indiacode.nic.in/handle/")

    @pytest.mark.parametrize("bad", [
        "",
        "not a url",
        "ftp://www.indiacode.nic.in/handle/123456789/1372",
        "https://example.com/handle/123456789/1372",
        "https://www.indiacode.nic.in/browse?type=title",
    ])
    def test_rejects_invalid_urls(self, bad):
        with pytest.raises(InvalidURLError):
            indiacode.normalise_url(bad)


class TestParseItemPage:
    def test_title_and_handle(self, parsed):
        assert parsed.title == "Passports Act, 1967"
        assert parsed.handle == "123456789/1372"
        assert parsed.handle_id == "1372"

    def test_primary_pdf_url_from_citation(self, parsed):
        assert parsed.primary_pdf_url == \
            "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"

    def test_enumerates_all_bitstreams(self, parsed):
        urls = {b.url for b in parsed.bitstreams}
        assert "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf" in urls
        assert "https://www.indiacode.nic.in/bitstream/123456789/1372/2/H1967-15.pdf" in urls

    def test_exactly_one_primary(self, parsed):
        primaries = [b for b in parsed.bitstreams if b.is_primary]
        assert len(primaries) == 1
        assert primaries[0].filename == "196715.pdf"

    def test_language_is_read_from_the_pages_own_labels(self, parsed):
        # This fixture labels each file "English"/"Hindi" in a neighbouring cell.
        english = next(b for b in parsed.bitstreams if b.filename == "196715.pdf")
        hindi = next(b for b in parsed.bitstreams if b.filename == "H1967-15.pdf")
        assert (english.language, english.language_source) == (
            "en", "indiacode_bitstream_label",
        )
        assert (hindi.language, hindi.language_source) == ("hi", "indiacode_bitstream_label")
        assert hindi.is_primary is False

    def test_english_bitstream_is_the_one_selected(self, parsed):
        assert language.select_english_bitstream(parsed).filename == "196715.pdf"

    def test_curated_metadata_fields(self, parsed):
        md = parsed.metadata
        assert md["act_number"] == "15"
        assert md["act_year"] == 1967  # coerced to int
        assert md["enactment_date"] == "1967-06-24"
        assert md["enforcement_date"] == "24-06-1967"
        assert md["short_title"] == "The Passports Act, 1967"
        assert md["ministry"] == "Ministry of External Affairs"
        assert md["india_code_act_id"] == "196715"
        assert md["long_title"].startswith("An Act to provide for the issue of passports")

    def test_hindi_title_captured(self, parsed):
        assert "hindi_title" in parsed.metadata
        assert parsed.metadata["hindi_title"]  # non-empty Devanagari string

    def test_no_invented_keys(self, parsed):
        # A field India Code does not provide must simply be absent.
        assert "repeal_date" not in parsed.metadata

    def test_raw_dublin_core_preserved(self, parsed):
        assert parsed.dublin_core["citation_pdf_url"].endswith("196715.pdf")
        # DC.title repeats (English + Hindi) -> stored as a list.
        assert isinstance(parsed.dublin_core["DC.title"], list)


class TestRealIndiaCodePage:
    """Against a verbatim capture of the live page for handle 123456789/1372.

    The real page has *no* "English"/"Hindi" label anywhere: each file's link is
    simply labelled with the document's title in its own language. This is the
    case the corpus actually has to handle.
    """

    @pytest.fixture(scope="class")
    def real(self):
        return parse_fixture("real_passports_act_1372.html")

    def test_both_language_versions_are_present(self, real):
        assert {b.filename for b in real.bitstreams} == {"196715.pdf", "H1967-15.pdf"}

    def test_english_and_hindi_are_told_apart_from_metadata(self, real):
        english = next(b for b in real.bitstreams if b.filename == "196715.pdf")
        hindi = next(b for b in real.bitstreams if b.filename == "H1967-15.pdf")
        assert english.language == "en"
        assert english.language_source == "indiacode_metadata_title"
        assert hindi.language == "hi"
        assert hindi.language_source == "indiacode_hindi_title"

    def test_only_the_english_pdf_is_selected(self, real):
        selected = language.select_english_bitstream(real)
        assert selected.filename == "196715.pdf"
        assert selected.url.endswith("/1/196715.pdf")


class TestEnglishOnlyItem:
    def test_single_english_file_is_selected(self):
        item = parse_fixture("english_only_1699.html", "1699")
        assert len(item.bitstreams) == 1
        selected = language.select_english_bitstream(item)
        assert selected.filename == "AAA1956___25.pdf"
        assert selected.language_source == "indiacode_metadata_title"


class TestHindiOnlyItem:
    def test_hindi_only_item_yields_nothing(self):
        item = parse_fixture("hindi_only_9001.html", "9001")
        # The Hindi PDF is even this item's citation_pdf_url; it is still refused.
        assert item.bitstreams[0].language == "hi"
        with pytest.raises(LanguageError, match="No English version"):
            language.select_english_bitstream(item)


class TestAmbiguousItem:
    def test_files_without_language_evidence_are_undetermined(self):
        item = parse_fixture("ambiguous_language_9002.html", "9002")
        assert [b.language for b in item.bitstreams] == [None, None]
        assert [b.language_source for b in item.bitstreams] == [None, None]

    def test_ambiguous_item_is_refused_not_assumed_english(self):
        item = parse_fixture("ambiguous_language_9002.html", "9002")
        with pytest.raises(LanguageError, match="could not be determined"):
            language.select_english_bitstream(item)
