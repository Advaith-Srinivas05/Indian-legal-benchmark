"""Regression tests for the five defects found in the corpus download's failures.

The full 19,859-document run left 177 failures. Investigating them turned up
four faults on *our* side and one recoverable upstream fault:

1. ``requests`` re-decodes a redirect target as UTF-8, so India Code's raw
   cp1252 bytes in a ``Location`` header crashed the request before the caller
   ever saw the response (:class:`ingestion.http_client.IndiaCodeSession`).
2. ``get_text`` forced UTF-8 on any page ``requests`` reported as latin-1,
   which raises on a page carrying stray cp1252 bytes
   (:func:`ingestion.http_client.decode_response`).
3. A requested bitstream was matched against the page's links as raw text, so
   ``A2000-21 (1).pdf`` did not match the same file written
   ``A2000-21%20%281%29.pdf`` and a listed English file was reported missing
   (:func:`ingestion.utils.same_url`).
4. When India Code lists one file on several rows of an act's Rules table, only
   the first was consulted — making the language verdict depend on HTML order
   (:func:`ingestion.indiacode._classify_subordinate`).
5. India Code's ``/ViewFileUploaded`` lookup cannot resolve a file whose name
   contains a non-ASCII character: it answers ``302`` with no ``Location`` at
   all. The address it *would* have redirected to is derivable
   (:func:`ingestion.indiacode.showfile_url`) and serves the file.

Nothing here relaxes a validation rule: a non-PDF response is still rejected,
a redirect off India Code is still refused, and English is still proven rather
than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from ingestion import http_client, indiacode, language
from ingestion.errors import FetchError, HostNotAllowedError, LanguageError, NotPDFError
from ingestion.models import BitstreamRef, ParsedItem, SubordinateRow
from ingestion.utils import canonical_url, same_url

FIXTURES = Path(__file__).parent / "fixtures"

VIEW_URL = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_2_33_00044_193804_1523351752525/regulationindividualfile/"
    "&file=irdai_%28protection_of_policyholders%E2%80%99_interests%29_regulations%2C_2017.pdf"
)
SHOWFILE = (
    "https://upload.indiacode.nic.in/showfile"
    "?actid=AC_CEN_2_33_00044_193804_1523351752525&type=regulation"
    "&filename=irdai_(protection_of_policyholders%E2%80%99_interests)_regulations%2C_2017.pdf"
)


def pdf_bytes(body: bytes = b"body") -> bytes:
    padding = b"%" + b"x" * 128 + b"\n"
    return b"%PDF-1.4\n" + padding + body + b"\n%%EOF\n"


# --- 1. an undecodable Location header -------------------------------------------


class FakeHeaders(dict):
    """Case-insensitive enough for what ``get_redirect_target`` reads."""

    def get(self, key, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default

    def __getitem__(self, key):
        value = self.get(key, _MISSING)
        if value is _MISSING:
            raise KeyError(key)
        return value

    def __contains__(self, key):
        return self.get(key, _MISSING) is not _MISSING


_MISSING = object()


class FakeRedirect:
    """A 302 whose Location holds a byte that is not valid UTF-8.

    India Code writes file names into the header as raw bytes; ``http.client``
    decodes the header as latin-1, so the character arrives as ``\\xa0``.
    """

    is_redirect = True
    status_code = 302
    url = VIEW_URL

    def __init__(self, location):
        self.headers = FakeHeaders({"Location": location})


class TestUndecodableLocationHeader:
    LOCATION = "http://upload.indiacode.nic.in/showfile?filename=rules,2006\xa0.pdf"

    def test_plain_requests_would_raise(self):
        """Pins down the upstream behaviour this fix exists for."""
        with pytest.raises(UnicodeDecodeError):
            requests.Session().get_redirect_target(FakeRedirect(self.LOCATION))

    def test_our_session_keeps_the_latin1_reading_instead_of_crashing(self):
        session = http_client.IndiaCodeSession()
        assert session.get_redirect_target(FakeRedirect(self.LOCATION)) == self.LOCATION

    def test_a_valid_utf8_location_is_still_read_the_normal_way(self):
        location = "http://upload.indiacode.nic.in/showfile?filename=plain.pdf"
        session = http_client.IndiaCodeSession()
        assert session.get_redirect_target(FakeRedirect(location)) == location

    def test_a_non_redirect_still_yields_no_target(self):
        class NotARedirect:
            is_redirect = False
            headers = FakeHeaders()

        assert http_client.IndiaCodeSession().get_redirect_target(NotARedirect()) is None

    def test_build_session_uses_the_tolerant_session(self):
        assert isinstance(http_client.build_session(), http_client.IndiaCodeSession)


# --- 2. decoding a page the server did not label ---------------------------------


class TextResponse:
    def __init__(self, content: bytes, content_type: str):
        self.content = content
        self.headers = {"Content-Type": content_type}
        self.url = "https://www.indiacode.nic.in/handle/123456789/1"
        self.encoding = "iso-8859-1"   # what requests assumes when unlabelled

    @property
    def text(self):
        return self.content.decode(self.encoding)


#: A page carrying a cp1252 byte (0xAC, "¬") that is not valid UTF-8 — exactly
#: what "The Goa Ferries (Regu¬lation of Issue of Tickets) Rules" is served as.
CP1252_PAGE = "<html><body>The Goa Ferries (Regu¬lation) Rules</body></html>".encode("cp1252")
UTF8_PAGE = "<html><body>पासपोर्ट नियम</body></html>".encode("utf-8")


class TestRobustDecoding:
    def test_a_declared_charset_is_obeyed(self):
        response = TextResponse(UTF8_PAGE, "text/html;charset=UTF-8")
        assert "पासपोर्ट" in http_client.decode_response(response)

    def test_an_undeclared_page_is_read_as_utf8_not_latin1(self):
        """requests reports latin-1 when the server said nothing; UTF-8 is tried first."""
        response = TextResponse(UTF8_PAGE, "text/html")
        assert "पासपोर्ट" in http_client.decode_response(response)

    def test_a_cp1252_page_decodes_instead_of_raising(self):
        response = TextResponse(CP1252_PAGE, "text/html")
        assert "Regu¬lation" in http_client.decode_response(response)

    def test_nothing_is_replaced_by_a_substitution_character(self):
        """A lossy errors='replace' decode would silently corrupt the text."""
        response = TextResponse(CP1252_PAGE, "text/html")
        assert "�" not in http_client.decode_response(response)

    def test_a_declared_charset_wins_even_when_utf8_would_also_work(self):
        response = TextResponse("caf\xe9".encode("cp1252"), "text/html; charset=windows-1252")
        assert http_client.decode_response(response) == "café"

    def test_a_charset_with_quotes_and_spacing_is_parsed(self):
        response = TextResponse(UTF8_PAGE, 'text/html; charset="utf-8"')
        assert "पासपोर्ट" in http_client.decode_response(response)


# --- 3. comparing URLs that mean the same file -----------------------------------


class TestUrlComparison:
    @pytest.mark.parametrize(
        "left,right",
        [
            ("https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21 (1).pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21%20%281%29.pdf"),
            ("https://www.indiacode.nic.in/bitstream/123456789/2305/1/AAAA1938___)5.pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/2305/1/AAAA1938___%295.pdf"),
            ("https://WWW.IndiaCode.nic.in/bitstream/123456789/1/1/a.pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/1/1/a.pdf"),
            ("https://www.indiacode.nic.in/x?a=1&b=2", "https://www.indiacode.nic.in/x?b=2&a=1"),
        ],
    )
    def test_equivalent_urls_compare_equal(self, left, right):
        assert same_url(left, right)
        assert canonical_url(left) == canonical_url(right)

    @pytest.mark.parametrize(
        "left,right",
        [
            ("https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21.pdf",
             "https://www.indiacode.nic.in/bitstream/123456789/1999/2/H2000-21.pdf"),
            ("https://www.indiacode.nic.in/bitstream/123456789/1999/1/a.pdf",
             "https://evil.example.com/bitstream/123456789/1999/1/a.pdf"),
            ("https://www.indiacode.nic.in/x?a=1", "https://www.indiacode.nic.in/x?a=2"),
        ],
    )
    def test_different_urls_stay_different(self, left, right):
        assert not same_url(left, right)

    def test_none_is_handled(self):
        assert not same_url(None, "https://www.indiacode.nic.in/x")
        assert same_url(None, None)


def _item_with_encoded_bitstream(requested: str) -> ParsedItem:
    """An item whose page lists the file percent-encoded, as India Code does."""
    return ParsedItem(
        source_url="https://www.indiacode.nic.in/handle/123456789/1999",
        handle="123456789/1999",
        handle_id="1999",
        title="Information Technology Act, 2000",
        bitstreams=[
            BitstreamRef(
                url="https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21%20%281%29.pdf",
                filename="A2000-21 (1).pdf",
                is_primary=True,
                language="en",
                language_source="indiacode_metadata_title",
            ),
            BitstreamRef(
                url="https://www.indiacode.nic.in/bitstream/123456789/1999/2/H2000-21.pdf",
                filename="H2000-21.pdf",
                language="hi",
                language_source="indiacode_hindi_title",
            ),
        ],
        requested_bitstream_url=requested,
    )


class TestRequestedBitstreamIsFoundDespiteEncoding:
    RAW = "https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21 (1).pdf"

    def test_an_unencoded_request_matches_the_encoded_link(self):
        selected = language.select_english_bitstream(_item_with_encoded_bitstream(self.RAW))
        assert selected.filename == "A2000-21 (1).pdf"
        assert selected.language == "en"

    def test_the_url_actually_downloaded_is_the_one_the_page_publishes(self):
        selected = language.select_english_bitstream(_item_with_encoded_bitstream(self.RAW))
        assert selected.url.endswith("A2000-21%20%281%29.pdf")

    def test_a_genuinely_absent_file_is_still_refused(self):
        item = _item_with_encoded_bitstream(
            "https://www.indiacode.nic.in/bitstream/123456789/1999/9/absent.pdf"
        )
        with pytest.raises(LanguageError, match="not found among the bitstreams"):
            language.select_english_bitstream(item)

    def test_the_hindi_file_is_still_never_selected(self):
        item = _item_with_encoded_bitstream(
            "https://www.indiacode.nic.in/bitstream/123456789/1999/2/H2000-21.pdf"
        )
        with pytest.raises(LanguageError):
            language.select_english_bitstream(item)


# --- 4. one file listed on several rows ------------------------------------------


def row(description, hindi_description=None, *, column="Files(Eng)", filename="81.pdf"):
    return SubordinateRow(
        document_type="rule",
        tab_label="Rules",
        year="25-06-2024",
        description=description,
        hindi_description=hindi_description,
        english_url="https://www.indiacode.nic.in/ViewFileUploaded"
                    "?path=AC_KA_71_596_00014_2_1551958162885/rulesindividualfile/&file=81.pdf",
        english_filename=filename,
        english_column_label=column,
    )


KANNADA = "ಕರ್ನಾಟಕ ಅನುಸೂಚಿತ ಜಾತಿಗಳ ನಿಯಮಗಳು, 2024"
ENGLISH = "The Karnataka Scheduled Castes and Scheduled Tribes (Amendment) Rules, 2024."
DEVANAGARI = "कर्नाटक अनुसूचित जाति नियम, 2024"


class TestSeveralRowsForOneFile:
    def test_an_english_row_is_found_whichever_order_the_rows_appear_in(self):
        """India Code duplicates a row with the description copied into both
        the Description and Hindi Description cells; the English row decides."""
        forwards = indiacode._classify_subordinate([row(KANNADA, KANNADA), row(ENGLISH)])
        backwards = indiacode._classify_subordinate([row(ENGLISH), row(KANNADA, KANNADA)])
        assert forwards[1].language == "en"
        assert backwards[1].language == "en"
        assert forwards[0].description == ENGLISH

    def test_a_hindi_row_still_rejects_the_file(self):
        """The gate stays asymmetric: any credible Hindi signal wins."""
        _row, verdict = indiacode._classify_subordinate(
            [row(ENGLISH), row(DEVANAGARI, DEVANAGARI)]
        )
        assert verdict.language == "hi"

    def test_a_duplicated_description_is_not_treated_as_hindi_evidence(self):
        """``hindi_description`` repeating ``description`` verbatim is India
        Code copying a cell, not a statement that the file is Hindi."""
        assert indiacode._classify_row(row(KANNADA, KANNADA)).language == "en"

    def test_a_distinct_hindi_description_is_still_evidence(self):
        assert indiacode._classify_row(row(DEVANAGARI, DEVANAGARI)).language == "hi"
        assert indiacode._classify_row(row(ENGLISH, DEVANAGARI)).language == "en"

    def test_a_single_row_behaves_exactly_as_before(self):
        selected, verdict = indiacode._classify_subordinate([row(ENGLISH)])
        assert selected.description == ENGLISH
        assert verdict.language == "en"

    def test_undetermined_rows_stay_undetermined(self):
        _row, verdict = indiacode._classify_subordinate(
            [row("A rule", column="Files"), row("A rule", column="Files")]
        )
        assert verdict.is_undetermined


# --- 5. deriving the address /ViewFileUploaded redirects to -----------------------


class TestShowfileUrl:
    def test_a_rule_url_is_mapped_to_india_codes_own_redirect_target(self):
        url = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile/"
            "&file=The+Passport+Rules%2C+1967.pdf"
        )
        assert indiacode.showfile_url(url) == (
            "https://upload.indiacode.nic.in/showfile"
            "?actid=AC_CEN_10_10_00008_196715_1517807321481&type=rule"
            "&filename=The%20Passport%20Rules%2C%201967.pdf"
        )

    def test_a_regulation_url_is_typed_as_a_regulation(self):
        assert "&type=regulation" in indiacode.showfile_url(VIEW_URL)

    def test_the_derived_url_stays_on_an_allowed_download_host(self):
        derived = indiacode.showfile_url(VIEW_URL)
        # Must not raise: the same check the downloader applies to every hop.
        http_client._check_download_host(VIEW_URL, derived)

    def test_a_non_ascii_filename_is_percent_encoded_as_utf8(self):
        """India Code's file server matches on the UTF-8 encoding of the name."""
        assert "%E2%80%99" in indiacode.showfile_url(VIEW_URL)

    def test_an_ampersand_in_the_filename_is_escaped(self):
        url = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_23_31_00011_193701_1535099362507/rulesindividualfile/"
            "&file=lac_g%26m_rules_1950.pdf"
        )
        assert indiacode.showfile_url(url).endswith("&filename=lac_g%26m_rules_1950.pdf")

    def test_an_actid_containing_a_space_is_encoded(self):
        url = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_KA_71_153_00001_KARNATAKA ACT 2 OF 1957_1542625067080"
            "/rulesindividualfile/&file=a.pdf"
        )
        assert "actid=AC_KA_71_153_00001_KARNATAKA%20ACT%202%20OF%201957_1542625067080" \
            in indiacode.showfile_url(url)

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf",
            "https://www.indiacode.nic.in/handle/123456789/1372",
            # notifications/orders live in directories the file server types
            # differently; nothing is invented for them.
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_1/notificationindividualfile/&file=a.pdf",
            # no directory at all -> no type can be established
            "https://www.indiacode.nic.in/ViewFileUploaded?path=AC_CEN_1&file=a.pdf",
            "https://www.indiacode.nic.in/ViewFileUploaded?path=AC_CEN_1/rulesindividualfile/",
        ],
    )
    def test_nothing_is_derived_for_a_url_the_mapping_does_not_apply_to(self, url):
        assert indiacode.showfile_url(url) is None


# --- the fallback in the download path --------------------------------------------


class Response:
    def __init__(self, *, status=200, headers=None, chunks=(), url=""):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            error = requests.HTTPError(f"{self.status_code} Client Error")
            error.response = self
            raise error

    def iter_content(self, size):
        return iter(self._chunks)


class RoutingSession:
    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        try:
            return self.routes[url]
        except KeyError:
            raise AssertionError(f"unexpected request for {url}")


def location_less_302(url):
    """India Code's answer for a file name it cannot put in a header."""
    return Response(status=302, headers={}, url=url)


def pdf_response(url, body=b"v1"):
    return Response(
        headers={"Content-Type": "application/pdf"}, chunks=[pdf_bytes(body)], url=url
    )


def html_error(url, status=200):
    return Response(
        status=status,
        headers={"Content-Type": "text/html;charset=ISO-8859-1"},
        chunks=[b"<!DOCTYPE html><html><body>File not found</body></html>"],
        url=url,
    )


class TestDownloadFallback:
    def test_a_location_less_redirect_falls_back_to_the_derived_url(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            VIEW_URL: location_less_302(VIEW_URL),
            SHOWFILE: pdf_response(SHOWFILE),
        })
        fetched = http_client.download_stream(
            session, VIEW_URL, part, fallback_url=SHOWFILE
        )
        assert part.read_bytes() == pdf_bytes(b"v1")
        assert fetched.fallback_url == SHOWFILE
        assert fetched.final_url == SHOWFILE
        assert session.requested == [VIEW_URL, SHOWFILE]

    def test_a_404_from_the_lookup_servlet_falls_back(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            VIEW_URL: html_error(VIEW_URL, status=404),
            SHOWFILE: pdf_response(SHOWFILE),
        })
        fetched = http_client.download_stream(
            session, VIEW_URL, part, fallback_url=SHOWFILE
        )
        assert fetched.fallback_url == SHOWFILE
        assert part.read_bytes().startswith(b"%PDF-")

    def test_without_a_fallback_the_failure_is_unchanged(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({VIEW_URL: location_less_302(VIEW_URL)})
        with pytest.raises(FetchError, match="without a Location header"):
            http_client.download_stream(session, VIEW_URL, part)
        assert not part.exists()

    def test_the_fallback_is_not_used_when_the_file_is_simply_missing(self, tmp_path):
        """HTTP 200 with an HTML body means the redirect worked and the file is
        gone; asking a second way would fetch the same error page."""
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({VIEW_URL: html_error(VIEW_URL)})
        with pytest.raises(NotPDFError):
            http_client.download_stream(session, VIEW_URL, part, fallback_url=SHOWFILE)
        assert session.requested == [VIEW_URL]
        assert not part.exists()

    def test_the_fallback_response_must_still_be_a_pdf(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            VIEW_URL: location_less_302(VIEW_URL),
            SHOWFILE: html_error(SHOWFILE),
        })
        with pytest.raises(NotPDFError):
            http_client.download_stream(session, VIEW_URL, part, fallback_url=SHOWFILE)
        assert not part.exists()

    def test_a_redirect_off_india_code_is_never_worked_around(self, tmp_path):
        """A host violation is a security stop, not a transport failure."""
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            VIEW_URL: Response(
                status=302, headers={"Location": "https://evil.example.com/x.pdf"},
                url=VIEW_URL,
            ),
        })
        with pytest.raises(HostNotAllowedError):
            http_client.download_stream(session, VIEW_URL, part, fallback_url=SHOWFILE)
        assert SHOWFILE not in session.requested

    def test_the_normal_path_records_no_fallback(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        target = "https://upload.indiacode.nic.in/showfile?actid=X&type=rule&filename=a.pdf"
        session = RoutingSession({
            VIEW_URL: Response(status=302, headers={"Location": target}, url=VIEW_URL),
            target: pdf_response(target),
        })
        fetched = http_client.download_stream(
            session, VIEW_URL, part, fallback_url=SHOWFILE
        )
        assert fetched.fallback_url is None
        assert SHOWFILE not in session.requested


# --- retry classification ---------------------------------------------------------


class TestPermanentFailuresAreNotRetried:
    def test_a_fetch_error_carries_the_http_status(self):
        error = FetchError("gone", status_code=404)
        assert error.status_code == 404

    def test_a_fetch_error_without_a_status_still_works(self):
        assert FetchError("connection reset").status_code is None

    @pytest.mark.parametrize("status,expected", [
        (None, True),      # no response at all - worth retrying
        (500, True), (502, True), (503, True),
        (429, True), (408, True),
        (404, False), (403, False), (400, False), (410, False),
    ])
    def test_only_transient_statuses_are_retried(self, status, expected):
        from ingestion.bulk import is_transient
        from ingestion.models import DownloadResult, Outcome

        result = DownloadResult(
            source_url="https://www.indiacode.nic.in/x", outcome=Outcome.FAILED,
            error_type="FetchError", http_status=status,
        )
        assert is_transient(result) is expected

    def test_a_permanent_error_type_is_never_retried(self):
        from ingestion.bulk import is_transient
        from ingestion.models import DownloadResult, Outcome

        for error_type in ("NotPDFError", "LanguageError", "HostNotAllowedError"):
            result = DownloadResult(
                source_url="https://www.indiacode.nic.in/x", outcome=Outcome.FAILED,
                error_type=error_type,
            )
            assert is_transient(result) is False


# --- a failed document leaves nothing behind --------------------------------------


class TestFailedDownloadsLeaveNoTrace:
    def test_the_document_directory_is_removed_when_the_download_fails(
        self, tmp_path, monkeypatch
    ):
        from ingestion import pipeline
        from ingestion.manifest import Manifest
        from ingestion.models import Outcome
        from ingestion.storage import DocumentStore

        store = DocumentStore(tmp_path)
        manifest = Manifest.load(store.manifest_path)

        item = ParsedItem(
            source_url="https://www.indiacode.nic.in/handle/123456789/1",
            handle="123456789/1", handle_id="1", title="An Act",
            bitstreams=[BitstreamRef(
                url="https://www.indiacode.nic.in/bitstream/123456789/1/1/a.pdf",
                filename="a.pdf", is_primary=True, language="en",
                language_source="indiacode_metadata_title",
            )],
        )
        monkeypatch.setattr(pipeline.indiacode, "resolve", lambda *a, **k: item)

        def fail(*args, **kwargs):
            raise NotPDFError("not a PDF; nothing was written.")

        monkeypatch.setattr(pipeline, "download_stream", fail)

        result = pipeline.ingest_document(
            object(), store, manifest,
            "https://www.indiacode.nic.in/bitstream/123456789/1/1/a.pdf", "central_acts",
        )
        assert result.outcome is Outcome.FAILED
        assert not [p for p in (tmp_path / "raw").rglob("*") if p.is_file()]
        assert not [p for p in (tmp_path / "raw").rglob("*") if p.name == result.document_id], (
            "a failed document must not leave an empty directory behind"
        )

    def test_a_directory_holding_a_previous_version_is_kept(self, tmp_path, monkeypatch):
        """Only an *empty* staging directory is removed — never one with content."""
        from ingestion import pipeline

        directory = tmp_path / "doc"
        directory.mkdir()
        (directory / "keep.pdf").write_bytes(b"%PDF-1.4\n")
        pipeline._remove_if_empty(directory)
        assert directory.exists()

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        from ingestion import pipeline

        pipeline._remove_if_empty(tmp_path / "never-created")
