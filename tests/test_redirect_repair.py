"""Regression tests for India Code's malformed ``Location`` header.

``/ViewFileUploaded`` 302s to ``…/showfile?actid=…&type=…&filename=<name>.pdf``
with the file name written into the query **unescaped**. When the name contains
an ampersand ("GOA-IDC (Transfer & Sub-Lease Regulations), 2018") the file
server sees a truncated name and answers HTTP 200 with a small HTML error page
instead of the PDF — measured at ~6% of all Rules/Regulations.

The repair lives in :func:`ingestion.utils.repair_location` and is used by both
the downloader and the header-only sizing pass, so there is one implementation
of it. These tests pin down what it may and may not touch, and that the download
path applies it *before* following the redirect while keeping every existing
host check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import http_client, indiacode, pipeline, sizing
from ingestion.errors import FetchError, NotPDFError
from ingestion.manifest import Manifest
from ingestion.models import Outcome
from ingestion.storage import DocumentStore
from ingestion.utils import repair_location

FIXTURES = Path(__file__).parent / "fixtures"
PARENT_URL = "https://www.indiacode.nic.in/handle/123456789/1372"
RULE_URL = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile/"
    "&file=The+Passport+Rules%2C+1967+%2810.05.1967%29.pdf"
)


def act_page() -> str:
    return (FIXTURES / "act_with_subordinate_1372.html").read_text(encoding="utf-8")


def pdf_bytes(body: bytes = b"body") -> bytes:
    padding = b"%" + b"x" * 128 + b"\n"
    return b"%PDF-1.4\n" + padding + body + b"\n%%EOF\n"


SHOWFILE = "http://upload.indiacode.nic.in/showfile"
# A real-shaped redirect for a file whose name contains an ampersand.
AMPERSAND_LOCATION = (
    f"{SHOWFILE}?actid=AC_GA_65_853_00005_00005_1709710299779&type=regulation"
    "&filename=goa-idc_(transfer_&_sub-lease_regulations),_2018.pdf"
)
AMPERSAND_REPAIRED = AMPERSAND_LOCATION.replace("_&_sub", "_%26_sub")
PLAIN_LOCATION = (
    f"{SHOWFILE}?actid=AC_CEN_10_10_00008_196715_1517807321481&type=rule"
    "&filename=The Passport Rules, 1967 (10.05.1967).pdf"
)


# --- the repair itself ----------------------------------------------------------


class TestRepairLocation:
    def test_an_ampersand_inside_the_filename_is_escaped(self):
        assert repair_location(AMPERSAND_LOCATION) == AMPERSAND_REPAIRED

    def test_several_ampersands_in_one_filename_are_all_escaped(self):
        raw = f"{SHOWFILE}?actid=X&type=rule&filename=a_&_b_&_c_&_d.pdf"
        assert repair_location(raw) == (
            f"{SHOWFILE}?actid=X&type=rule&filename=a_%26_b_%26_c_%26_d.pdf"
        )

    def test_a_well_formed_location_is_returned_unchanged(self):
        assert repair_location(PLAIN_LOCATION) == PLAIN_LOCATION

    def test_the_separators_between_real_parameters_are_never_touched(self):
        raw = f"{SHOWFILE}?actid=X&type=rule&filename=plain.pdf"
        assert repair_location(raw) == raw
        assert raw.count("&") == repair_location(raw).count("&")

    def test_a_legitimate_parameter_after_filename_is_preserved(self):
        raw = f"{SHOWFILE}?filename=a_&_b.pdf&type=regulation&version=2"
        assert repair_location(raw) == (
            f"{SHOWFILE}?filename=a_%26_b.pdf&type=regulation&version=2"
        )

    def test_a_parameter_after_filename_survives_even_without_an_extension(self):
        raw = f"{SHOWFILE}?filename=a_&_b&type=regulation"
        assert repair_location(raw) == f"{SHOWFILE}?filename=a_%26_b&type=regulation"

    def test_a_url_without_a_filename_parameter_is_left_alone(self):
        raw = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf?a=1&b=2"
        assert repair_location(raw) == raw

    def test_the_downloader_and_the_sizing_pass_share_one_implementation(self):
        assert http_client.repair_location is repair_location
        assert sizing.repair_location is repair_location


# --- the download path -----------------------------------------------------------


class Response:
    """A canned streaming response; redirects carry no body."""

    def __init__(self, *, status=200, headers=None, chunks=(), url=""):
        self.status_code = status
        self.headers = headers or {}
        self._chunks = list(chunks)
        self.url = url
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            raise requests.HTTPError(f"{self.status_code} Server Error")

    def iter_content(self, size):
        return iter(self._chunks)


class RoutingSession:
    """Serves responses by URL and records exactly what was requested."""

    def __init__(self, routes):
        self.routes = routes
        self.requested: list[str] = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        assert kwargs.get("allow_redirects") is False, (
            "redirects must be followed by hand so the Location can be repaired"
        )
        try:
            return self.routes[url]
        except KeyError:  # pragma: no cover - a mis-built URL shows up here
            raise AssertionError(f"unexpected request for {url}")


def redirect_to(location):
    return Response(status=302, headers={"Location": location}, url=RULE_URL)


def pdf_response(url, body=b"v1"):
    return Response(
        headers={"Content-Type": "application/pdf"}, chunks=[pdf_bytes(body)], url=url
    )


def html_error(url):
    """What India Code's file server returns for a truncated file name."""
    return Response(
        headers={"Content-Type": "text/html;charset=ISO-8859-1"},
        chunks=[b"<!DOCTYPE html><html><head><title>India Code</title></head></html>"],
        url=url,
    )


class TestDownloadFollowsTheRepairedRedirect:
    def test_an_ampersand_filename_is_repaired_before_the_redirect_is_followed(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED),
        })
        fetched = http_client.download_stream(session, RULE_URL, part)

        assert part.read_bytes() == pdf_bytes(b"v1")
        assert fetched.final_url == AMPERSAND_REPAIRED
        assert fetched.redirects == [AMPERSAND_REPAIRED]
        # The raw, malformed Location was never requested.
        assert session.requested == [RULE_URL, AMPERSAND_REPAIRED]
        assert AMPERSAND_LOCATION not in session.requested

    def test_the_unrepaired_target_would_have_returned_html(self, tmp_path):
        """Without the repair the same download yields an HTML error page."""
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_LOCATION: html_error(AMPERSAND_LOCATION),
        })
        with pytest.raises(AssertionError, match="unexpected request"):
            # The repaired URL is the only one the downloader asks for, so this
            # route is never hit — which is precisely the fix.
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_normal_viewfileuploaded_redirect_still_works(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to(PLAIN_LOCATION),
            PLAIN_LOCATION: pdf_response(PLAIN_LOCATION),
        })
        fetched = http_client.download_stream(session, RULE_URL, part)
        assert part.exists()
        assert fetched.final_url == PLAIN_LOCATION
        assert session.requested[1] == PLAIN_LOCATION

    def test_a_bitstream_url_is_fetched_in_one_request_with_no_redirect(self, tmp_path):
        bitstream = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({bitstream: pdf_response(bitstream)})
        fetched = http_client.download_stream(session, bitstream, part)
        assert session.requested == [bitstream]
        assert fetched.redirects == []
        assert fetched.final_url == bitstream
        assert part.read_bytes() == pdf_bytes(b"v1")

    def test_an_html_error_page_is_still_rejected_after_a_redirect(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to(PLAIN_LOCATION),
            PLAIN_LOCATION: html_error(PLAIN_LOCATION),
        })
        with pytest.raises(NotPDFError, match="not a PDF"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_redirect_off_india_code_is_refused_before_it_is_requested(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({RULE_URL: redirect_to("https://evil.example/x.pdf")})
        with pytest.raises(FetchError, match="not an India Code host"):
            http_client.download_stream(session, RULE_URL, part)
        assert session.requested == [RULE_URL]  # the off-site URL was never fetched
        assert not part.exists()

    def test_a_redirect_without_a_location_is_a_clear_failure(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({RULE_URL: Response(status=302, url=RULE_URL)})
        with pytest.raises(FetchError, match="without a Location"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_redirect_chain_is_bounded(self, tmp_path):
        loop = "https://www.indiacode.nic.in/loop"
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to(loop),
            loop: Response(status=302, headers={"Location": loop}, url=loop),
        })
        with pytest.raises(FetchError, match="redirects"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_relative_location_is_resolved_against_the_request(self, tmp_path):
        absolute = "https://www.indiacode.nic.in/showfile?filename=x.pdf"
        part = tmp_path / "x.pdf.part"
        session = RoutingSession({
            RULE_URL: redirect_to("/showfile?filename=x.pdf"),
            absolute: pdf_response(absolute),
        })
        http_client.download_stream(session, RULE_URL, part)
        assert session.requested[1] == absolute


# --- through the whole pipeline ---------------------------------------------------


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """The real pipeline against the real download path; only HTTP is canned."""
    store = DocumentStore(tmp_path / "data")
    store.ensure_layout()
    manifest = Manifest.load(store.manifest_path)
    monkeypatch.setattr(indiacode, "get_text", lambda session, url: act_page())

    def run(routes, url=RULE_URL, category="rules", **kwargs):
        kwargs.setdefault("parent_url", PARENT_URL)
        session = RoutingSession(routes)
        result = pipeline.ingest_document(
            session=session, store=store, manifest=manifest,
            url=url, category=category, **kwargs,
        )
        return result, session

    return store, manifest, run


class TestAmpersandDocumentThroughThePipeline:
    def test_it_is_stored_as_a_pdf_with_the_original_url_preserved(self, harness):
        store, manifest, run = harness
        result, session = run({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED),
        })
        assert result.outcome is Outcome.NEW

        pdf = store.pdf_path("rules", result.document_id)
        assert pdf.read_bytes().startswith(b"%PDF-")

        meta = json.loads(
            store.metadata_path("rules", result.document_id).read_text(encoding="utf-8")
        )
        # Requirement: the India Code URL is preserved exactly, and the URL the
        # bytes actually came from is recorded separately.
        assert meta["primary_pdf_url"] == RULE_URL
        assert meta["pdf_url_kind"] == "viewfileuploaded"
        assert meta["download"]["final_url"] == AMPERSAND_REPAIRED
        assert meta["download"]["redirects"] == [AMPERSAND_REPAIRED]
        assert meta["download"]["content_type"] == "application/pdf"

        entry = manifest.get(result.document_id)
        assert entry["primary_pdf_url"] == RULE_URL
        assert entry["final_pdf_url"] == AMPERSAND_REPAIRED
        assert entry["language"] == "en"

    def test_the_language_gate_still_runs_before_anything_is_fetched(self, harness):
        """A file the parent page does not list as English is refused with no HTTP."""
        store, manifest, run = harness
        hindi = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_10_10_00008_196715_1517807321481/ruleshindifile/"
            "&file=H_The+Passport+Rules%2C+1967.pdf"
        )
        result, session = run({}, url=hindi)
        assert result.outcome is Outcome.FAILED
        assert session.requested == []  # not one byte was requested
        assert manifest.documents == {}

    def test_an_act_records_no_final_url_because_nothing_was_redirected(self, harness):
        store, manifest, run = harness
        bitstream = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
        result, session = run(
            {bitstream: pdf_response(bitstream)},
            url=PARENT_URL, category="central_acts", parent_url=None,
        )
        assert result.outcome is Outcome.NEW
        entry = manifest.get(result.document_id)
        assert entry["primary_pdf_url"] == bitstream
        assert "final_pdf_url" not in entry
        meta = json.loads(
            store.metadata_path("central_acts", result.document_id)
            .read_text(encoding="utf-8")
        )
        assert meta["download"]["redirects"] == []

    def test_re_downloading_the_repaired_document_is_unchanged(self, harness):
        store, manifest, run = harness
        routes = {
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED),
        }
        first, _ = run(routes)
        # Fresh responses: the canned ones are single-use iterators.
        second, _ = run({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED),
        })
        assert first.outcome is Outcome.NEW
        assert second.outcome is Outcome.UNCHANGED
        assert second.sha256 == first.sha256
        assert manifest.get(first.document_id)["version_count"] == 1

    def test_a_changed_ampersand_document_archives_the_old_version(self, harness):
        store, manifest, run = harness
        first, _ = run({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED),
        })
        second, _ = run({
            RULE_URL: redirect_to(AMPERSAND_LOCATION),
            AMPERSAND_REPAIRED: pdf_response(AMPERSAND_REPAIRED, b"amended"),
        })
        assert second.outcome is Outcome.UPDATED
        archived = list(
            (store.document_dir("rules", first.document_id) / "versions").glob("*.pdf")
        )
        assert len(archived) == 1
        assert archived[0].read_bytes() == pdf_bytes(b"v1")
