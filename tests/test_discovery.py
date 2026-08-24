"""Tests for India Code corpus discovery (parsing side).

All fixtures are saved pages — several captured verbatim from the live site —
so nothing here touches the network.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from ingestion import discovery, indiacode
from ingestion.discovery import ActListing, Collection, Outcome
from ingestion.errors import FetchError

FIXTURES = Path(__file__).parent / "fixtures"
ACT_URL = "https://www.indiacode.nic.in/handle/123456789/1372"
EDGE_URL = "https://www.indiacode.nic.in/handle/123456789/8100"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def listing_for(handle="123456789/1372", url=ACT_URL, **kwargs) -> ActListing:
    defaults = dict(
        handle=handle, url=url, short_title="The Passports Act, 1967",
        act_number="15", enactment_date="24-Jun-1967",
        collection_name="Central Acts", collection_handle="123456789/1362",
        document_type="central_act", jurisdiction="India",
        browse_url="https://www.indiacode.nic.in/handle/123456789/1362/browse",
    )
    defaults.update(kwargs)
    return ActListing(**defaults)


class TestParseCollections:
    @pytest.fixture(scope="class")
    def collections(self):
        return discovery.parse_collections(read("indiacode_nav.html"), ACT_URL)

    def test_finds_central_and_state_collections(self, collections):
        assert len(collections) > 30
        types = Counter(c.document_type for c in collections)
        assert types["central_act"] == 1
        assert types["state_act"] == len(collections) - 1

    def test_central_collection(self, collections):
        central = next(c for c in collections if c.document_type == "central_act")
        assert central.handle == "123456789/1362"
        assert central.jurisdiction == "India"
        assert central.url.endswith("/handle/123456789/1362")

    def test_state_jurisdiction_is_the_state_name(self, collections):
        ap = next(c for c in collections if c.name == "Andhra Pradesh")
        assert ap.jurisdiction == "Andhra Pradesh"
        assert ap.document_type == "state_act"
        assert ap.handle.startswith("123456789/")

    def test_no_duplicate_collections(self, collections):
        handles = [c.handle for c in collections]
        assert len(handles) == len(set(handles))

    def test_browse_links_are_not_mistaken_for_states(self, collections):
        # "Short Title", "Act Number", ... are browse indexes of the central
        # collection, not collections in their own right.
        names = {c.name for c in collections}
        assert not names & {"Short Title", "Act Number", "Act Year", "Ministry"}


class TestParseBrowsePage:
    @pytest.fixture(scope="class")
    def collection(self):
        return Collection("Central Acts", "123456789/1362", "central_act", "India", "u")

    @pytest.fixture(scope="class")
    def listings(self, collection):
        return discovery.parse_browse_page(
            read("browse_central_acts.html"), "https://browse.example", collection
        )

    def test_one_listing_per_result_row(self, listings):
        assert len(listings) == 5

    def test_listing_fields_come_from_the_table_columns(self, listings):
        first = listings[0]
        assert first.handle == "123456789/18935"
        assert first.short_title == "The Bengal Indigo Contracts Act, 1836"
        assert first.act_number == "10"
        assert first.enactment_date == "11-Apr-1836"
        assert first.url == "https://www.indiacode.nic.in/handle/123456789/18935"

    def test_collection_context_is_attached(self, listings):
        assert all(item.document_type == "central_act" for item in listings)
        assert all(item.jurisdiction == "India" for item in listings)
        assert all(item.collection_handle == "123456789/1362" for item in listings)
        assert all(item.browse_url == "https://browse.example" for item in listings)

    def test_empty_page_ends_pagination(self, collection):
        assert discovery.parse_browse_page("<html><body></body></html>", "u", collection) == []


class TestBrowseCollection:
    def test_pages_until_empty(self, monkeypatch):
        collection = Collection("Central Acts", "123456789/1362", "central_act", "India", "u")
        pages = {0: read("browse_central_acts.html"), 200: "<html></html>"}
        calls = []

        def fake_get_text(session, url):
            offset = int(url.rsplit("offset=", 1)[1])
            calls.append(offset)
            return pages.get(offset, "<html></html>")

        monkeypatch.setattr(discovery, "get_text", fake_get_text)
        listings = discovery.browse_collection(None, collection)
        assert len(listings) == 5
        assert calls == [0, 200]

    def test_repeated_rows_across_pages_are_not_duplicated(self, monkeypatch):
        collection = Collection("Central Acts", "123456789/1362", "central_act", "India", "u")
        page = read("browse_central_acts.html")
        seen = {"n": 0}

        def fake_get_text(session, url):
            seen["n"] += 1
            return page if seen["n"] <= 2 else "<html></html>"

        monkeypatch.setattr(discovery, "get_text", fake_get_text)
        listings = discovery.browse_collection(None, collection)
        assert len(listings) == 5  # the second page repeats the same handles


class TestParseSubordinateRows:
    """Against the verbatim Rules/Regulations markup of the live act page."""

    @pytest.fixture(scope="class")
    def rows(self):
        return discovery.parse_subordinate_rows(read("act_with_subordinate_1372.html"), ACT_URL)

    def test_rows_are_not_swallowed_by_unclosed_tr_tags(self, rows):
        # India Code never closes <tr>, so every row parses as nested inside the
        # previous one. Reading only direct <td> children keeps them separate.
        assert len(rows) == 5
        assert all(row.description for row in rows)
        assert len({row.english_url for row in rows}) == 5

    def test_rules_are_identified_by_their_button_label(self, rows):
        assert {row.document_type for row in rows} == {"rule"}
        assert {row.tab_label for row in rows} == {"Rules"}

    def test_file_columns_are_read_from_the_headers(self, rows):
        assert all(row.english_column_label == "Files(Eng)" for row in rows)
        assert all(row.hindi_column_label == "Files(Hindi)" for row in rows)

    def test_english_file_url_and_name(self, rows):
        first = rows[0]
        assert first.description == "The Passport Rules, 1967 (10.05.1967)"
        assert first.english_url.startswith(
            "https://www.indiacode.nic.in/ViewFileUploaded?path="
        )
        assert first.english_filename.endswith(".pdf")
        assert " " not in first.english_url  # trailing space in the href is stripped

    def test_out_of_scope_tabs_are_ignored(self):
        rows = discovery.parse_subordinate_rows(read("act_subordinate_edge_cases.html"), EDGE_URL)
        assert "notification" not in {row.document_type for row in rows}
        assert {row.document_type for row in rows} == {"rule", "regulation"}


class TestSubordinateLanguageGate:
    """The edge cases the live Passports page does not happen to contain."""

    @pytest.fixture(scope="class")
    def result(self):
        html = read("act_subordinate_edge_cases.html")
        listing = listing_for(handle="123456789/8100", url=EDGE_URL, short_title=None)
        item = indiacode.parse_item_page(html, base_url=EDGE_URL, source_url=EDGE_URL)
        result = discovery._act_entry(item, listing)
        result.entries.extend(discovery._subordinate_entries(html, item, listing, result))
        return result

    def test_english_file_is_taken_when_both_languages_exist(self, result):
        rules = [e for e in result.entries if e.document_type == "rule"]
        assert len(rules) == 1
        assert rules[0].english_pdf_url.endswith("edge_case_rules_1991.pdf")
        assert rules[0].language == "en"
        assert rules[0].language_source == "indiacode_bitstream_label"
        assert "Files(Eng)" in rules[0].language_evidence

    def test_no_hindi_file_is_ever_listed(self, result):
        urls = " ".join(e.english_pdf_url for e in result.entries)
        assert "ruleshindifile" not in urls
        assert "H_edge_case" not in urls

    def test_hindi_only_row_is_rejected(self, result):
        reasons = Counter(r.reason for r in result.rejections)
        assert reasons[Outcome.HINDI_REJECTED] == 2

    def test_row_with_no_file_is_reported_as_no_pdf(self, result):
        no_pdf = [r for r in result.rejections if r.reason == Outcome.NO_PDF]
        assert len(no_pdf) == 1
        assert "Second Amendment" in no_pdf[0].title

    def test_devanagari_description_beats_the_english_column(self, result):
        # Conflicting evidence must reject, never accept.
        hindi = [r for r in result.rejections
                 if r.reason == Outcome.HINDI_REJECTED and r.detail.count("devanagari_row")]
        assert len(hindi) == 1

    def test_unlabelled_file_column_is_ambiguous_not_english(self, result):
        ambiguous = [r for r in result.rejections if r.reason == Outcome.AMBIGUOUS]
        assert len(ambiguous) == 1
        assert "edge_case_regs_1995.pdf" in ambiguous[0].detail
        assert not [e for e in result.entries if e.document_type == "regulation"]


class TestInspectAct:
    def test_act_and_its_rules_are_discovered(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "get_text",
            lambda session, url: read("act_with_subordinate_1372.html"),
        )
        result = discovery.inspect_act(None, listing_for())

        assert result.outcome == Outcome.ENGLISH_CONFIRMED
        types = Counter(e.document_type for e in result.entries)
        assert types["central_act"] == 1
        assert types["rule"] == 5

        act = next(e for e in result.entries if e.document_type == "central_act")
        assert act.document_id == "passports-act-1967__handle-1372"
        assert act.english_pdf_url.endswith("/1/196715.pdf")
        assert act.pdf_url_kind == "bitstream"
        assert act.language == "en"
        assert act.language_source == "indiacode_metadata_title"
        assert act.year == 1967
        assert act.act_number == "15"
        assert act.ministry == "Ministry of External Affairs"
        assert act.india_code_act_id == "196715"
        assert act.handle == "123456789/1372"

    def test_no_pdf_is_ever_fetched(self, monkeypatch):
        """Discovery must read pages only — never a PDF."""
        fetched = []

        def fake_get_text(session, url):
            fetched.append(url)
            return read("act_with_subordinate_1372.html")

        monkeypatch.setattr(discovery, "get_text", fake_get_text)
        discovery.inspect_act(None, listing_for())
        assert fetched == [ACT_URL]
        assert not [u for u in fetched if u.lower().endswith(".pdf")]

    def test_rules_are_traceable_to_their_parent_act(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "get_text",
            lambda session, url: read("act_with_subordinate_1372.html"),
        )
        result = discovery.inspect_act(None, listing_for())
        rule = next(e for e in result.entries if e.document_type == "rule")
        assert rule.parent_document_id == "passports-act-1967__handle-1372"
        assert rule.parent_handle == "123456789/1372"
        assert rule.india_code_url == ACT_URL
        assert rule.india_code_path.startswith("AC_CEN_")
        assert rule.sources[0]["via"] == "act_page_rules_table"
        assert rule.sources[0]["collection_name"] == "Central Acts"

    def test_hindi_only_act_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "get_text", lambda session, url: read("hindi_only_9001.html")
        )
        result = discovery.inspect_act(
            None, listing_for(handle="123456789/9001", url="https://x/handle/123456789/9001")
        )
        assert result.outcome == Outcome.HINDI_REJECTED
        assert result.entries == []
        assert result.rejections[0].reason == Outcome.HINDI_REJECTED

    def test_ambiguous_act_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "get_text",
            lambda session, url: read("ambiguous_language_9002.html"),
        )
        result = discovery.inspect_act(
            None, listing_for(handle="123456789/9002", url="https://x/handle/123456789/9002")
        )
        assert result.outcome == Outcome.AMBIGUOUS
        assert result.entries == []

    def test_act_without_files_is_reported_as_no_pdf(self, monkeypatch):
        monkeypatch.setattr(
            discovery, "get_text",
            lambda session, url: "<html><head><title>x</title></head><body></body></html>",
        )
        result = discovery.inspect_act(None, listing_for())
        assert result.outcome in {Outcome.NO_PDF, Outcome.ERROR}
        assert result.entries == []

    def test_fetch_failure_becomes_an_error_result(self, monkeypatch):
        def boom(session, url):
            raise FetchError("503 Service Unavailable")

        monkeypatch.setattr(discovery, "get_text", boom)
        result = discovery.inspect_act(None, listing_for())
        assert result.outcome == Outcome.ERROR
        assert "503" in result.error
        assert result.entries == []


class TestHelpers:
    @pytest.mark.parametrize("url,expected", [
        ("https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf", "bitstream"),
        ("https://www.indiacode.nic.in/ViewFileUploaded?path=X/&file=y.pdf", "viewfileuploaded"),
        ("https://example.com/somewhere/y.pdf", "other"),
    ])
    def test_pdf_url_kind(self, url, expected):
        assert discovery.pdf_url_kind(url) == expected

    def test_subordinate_document_id_is_stable_and_readable(self):
        row = discovery.SubordinateRow(
            document_type="rule", tab_label="Rules",
            description="The Passport Rules, 1967 (10.05.1967)",
            english_url="https://www.indiacode.nic.in/ViewFileUploaded?path=A/&file=b.pdf",
        )
        first = discovery._subordinate_document_id(row)
        assert first == discovery._subordinate_document_id(row)
        assert first.startswith("the-passport-rules-1967")
        assert "__rule-" in first

    def test_subordinate_document_id_differs_per_file(self):
        def make(url):
            return discovery._subordinate_document_id(
                discovery.SubordinateRow(
                    document_type="rule", tab_label="Rules", description="Same Name",
                    english_url=url,
                )
            )

        assert make("https://x/a.pdf") != make("https://x/b.pdf")

    @pytest.mark.parametrize("value,expected", [
        (1967, 1967), ("1967", 1967), ("24-Jun-1967", 1967),
        ("10-05-1967", 1967), (None, None), ("n/a", None), (99, None),
    ])
    def test_year_extraction(self, value, expected):
        assert discovery._as_year(value) == expected
