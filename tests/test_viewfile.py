"""Tests for downloading Rules/Regulations from ``/ViewFileUploaded`` URLs.

India Code does not publish subordinate legislation as DSpace bitstreams: it
lists it on the parent act's page and serves it from
``/ViewFileUploaded?path=…&file=…``. These files therefore carry no language
information of their own, and everything here turns on the parent page being
the thing that proves it.

Entirely offline: the parent page is a fixture captured from the live site and
every HTTP call is stubbed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import http_client, indiacode, language, pipeline
from ingestion.errors import FetchError, InvalidURLError, LanguageError, MetadataError, NotPDFError
from ingestion.manifest import Manifest
from ingestion.models import FetchResult, Outcome
from ingestion.storage import DocumentStore
from ingestion.utils import make_subordinate_document_id

FIXTURES = Path(__file__).parent / "fixtures"
PARENT_URL = "https://www.indiacode.nic.in/handle/123456789/1372"
RULE_URL = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile/"
    "&file=The+Passport+Rules%2C+1967+%2810.05.1967%29.pdf"
)
RULE_FILENAME = "The Passport Rules, 1967 (10.05.1967).pdf"
RULE_DESCRIPTION = "The Passport Rules, 1967 (10.05.1967)"


def act_page() -> str:
    return (FIXTURES / "act_with_subordinate_1372.html").read_text(encoding="utf-8")


def pdf_bytes(body: bytes = b"body") -> bytes:
    padding = b"%" + b"x" * 128 + b"\n"
    return b"%PDF-1.4\n" + padding + body + b"\n%%EOF\n"


class TestNormaliseViewFileUrl:
    def test_accepted_and_classified(self):
        kind, canonical, handle, filename = indiacode.normalise_url(RULE_URL)
        assert kind == "viewfile"
        assert handle is None                      # belongs to a parent act
        assert filename == RULE_FILENAME           # decoded for use, not for the URL

    def test_query_string_is_preserved_byte_for_byte(self):
        # Re-encoding path/file would produce a URL India Code does not serve.
        _kind, canonical, _handle, _filename = indiacode.normalise_url(RULE_URL)
        assert canonical == RULE_URL
        assert "%2C" in canonical and "+" in canonical

    def test_scheme_and_host_are_normalised(self):
        _kind, canonical, _handle, _filename = indiacode.normalise_url(
            RULE_URL.replace("https://www.indiacode.nic.in", "http://indiacode.nic.in")
        )
        assert canonical.startswith("https://www.indiacode.nic.in/ViewFileUploaded?")
        assert canonical.endswith("%2810.05.1967%29.pdf")

    def test_handle_and_bitstream_urls_still_work(self):
        assert indiacode.normalise_url(PARENT_URL)[0] == "handle"
        assert indiacode.normalise_url(
            "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
        )[0] == "bitstream"

    @pytest.mark.parametrize("bad", [
        "https://www.indiacode.nic.in/ViewFileUploaded",
        "https://www.indiacode.nic.in/ViewFileUploaded?path=AC_CEN_1/",
        "https://www.indiacode.nic.in/ViewFileUploaded?file=x.pdf",
        "https://www.indiacode.nic.in/ViewFileUploaded?path=&file=",
        "https://example.com/ViewFileUploaded?path=A/&file=x.pdf",
        "ftp://www.indiacode.nic.in/ViewFileUploaded?path=A/&file=x.pdf",
    ])
    def test_malformed_urls_are_rejected(self, bad):
        with pytest.raises(InvalidURLError):
            indiacode.normalise_url(bad)


class TestViewFileIdentity:
    def test_equivalent_encodings_are_the_same_file(self):
        plus = RULE_URL
        percent = RULE_URL.replace("+", "%20")
        assert indiacode.viewfile_identity(plus) == indiacode.viewfile_identity(percent)

    def test_identity_is_the_decoded_path_and_file(self):
        path, name = indiacode.viewfile_identity(RULE_URL)
        assert path == "AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile"
        assert name == RULE_FILENAME

    def test_different_files_differ(self):
        other = RULE_URL.replace("The+Passport+Rules", "The+Other+Rules")
        assert indiacode.viewfile_identity(other) != indiacode.viewfile_identity(RULE_URL)


class TestResolveViewFile:
    @pytest.fixture
    def parent(self, monkeypatch):
        fetched = []

        def fake_get_text(session, url):
            fetched.append(url)
            return act_page()

        monkeypatch.setattr(indiacode, "get_text", fake_get_text)
        return fetched

    def test_language_comes_from_the_files_eng_column(self, parent):
        item = indiacode.resolve(None, RULE_URL, parent_url=PARENT_URL)
        english = language.select_english_bitstream(item)
        assert english.language == "en"
        assert english.language_source == "indiacode_bitstream_label"
        assert "Files(Eng)" in english.language_evidence
        assert english.url == RULE_URL

    def test_parent_page_is_the_only_page_fetched(self, parent):
        indiacode.resolve(None, RULE_URL, parent_url=PARENT_URL)
        assert parent == [PARENT_URL]

    def test_subordinate_provenance_is_recorded(self, parent):
        item = indiacode.resolve(None, RULE_URL, parent_url=PARENT_URL)
        assert item.title == RULE_DESCRIPTION
        assert item.original_url == RULE_URL
        assert item.subordinate.document_type == "rule"
        assert item.subordinate.parent_handle == "123456789/1372"
        assert item.subordinate.parent_title == "Passports Act, 1967"
        assert item.subordinate.parent_url == PARENT_URL
        assert item.subordinate.india_code_path.startswith("AC_CEN_")

    def test_parent_act_fields_are_not_copied_onto_the_rule(self, parent):
        item = indiacode.resolve(None, RULE_URL, parent_url=PARENT_URL)
        # The parent's own identity must not masquerade as the rule's.
        assert "short_title" not in item.metadata
        assert "act_number" not in item.metadata
        # What genuinely still applies is kept.
        assert item.metadata["ministry"] == "Ministry of External Affairs"
        assert item.metadata["india_code_act_id"] == "196715"

    def test_alternative_encoding_still_matches_the_row(self, parent):
        item = indiacode.resolve(None, RULE_URL.replace("+", "%20"), parent_url=PARENT_URL)
        assert language.select_english_bitstream(item).language == "en"

    def test_without_a_parent_page_it_refuses(self, parent):
        with pytest.raises(MetadataError, match="parent act"):
            indiacode.resolve(None, RULE_URL, parent_url=None)
        assert parent == []          # nothing was even fetched

    def test_file_not_listed_on_the_parent_page_is_refused(self, parent):
        unlisted = RULE_URL.replace("The+Passport+Rules", "Some+Other+Document")
        with pytest.raises(MetadataError, match="not listed"):
            indiacode.resolve(None, unlisted, parent_url=PARENT_URL)


class TestViewFileLanguageGate:
    def test_a_hindi_only_row_is_refused(self, monkeypatch):
        html = (FIXTURES / "act_subordinate_edge_cases.html").read_text(encoding="utf-8")
        monkeypatch.setattr(indiacode, "get_text", lambda session, url: html)
        edge_parent = "https://www.indiacode.nic.in/handle/123456789/8100"
        hindi = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_TEST_8100/ruleshindifile/&file=H_edge_case_rules_1991.pdf"
        )
        # The Hindi file is not in the Files(Eng) column at all, so it is not a
        # verifiable English document and must be refused.
        with pytest.raises(MetadataError, match="not listed"):
            indiacode.resolve(None, hindi, parent_url=edge_parent)

    def test_an_unlabelled_file_column_is_refused(self, monkeypatch):
        html = (FIXTURES / "act_subordinate_edge_cases.html").read_text(encoding="utf-8")
        monkeypatch.setattr(indiacode, "get_text", lambda session, url: html)
        edge_parent = "https://www.indiacode.nic.in/handle/123456789/8100"
        regulation = (
            "https://www.indiacode.nic.in/ViewFileUploaded"
            "?path=AC_CEN_TEST_8100/regulationsindividualfile/&file=edge_case_regs_1995.pdf"
        )
        item = indiacode.resolve(None, regulation, parent_url=edge_parent)
        with pytest.raises(LanguageError, match="could not be determined"):
            language.select_english_bitstream(item)


# --- the download itself --------------------------------------------------------


class FakeResponse:
    def __init__(self, chunks, *, status=200, url=RULE_URL, content_type="application/pdf"):
        self._chunks = chunks
        self.status_code = status
        self.url = url
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} Server Error")

    def iter_content(self, size):
        return iter(self._chunks)


class FakeSession:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.requested = []

    def get(self, url, **kwargs):
        self.requested.append(url)
        if self._exc:
            raise self._exc
        return self._response


class TestDownloadStreamGuards:
    def test_a_pdf_is_written(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = FakeSession(FakeResponse([pdf_bytes(b"v1")]))
        fetched = http_client.download_stream(session, RULE_URL, part)
        assert fetched.content_type == "application/pdf"
        assert fetched.http_status == 200
        assert part.read_bytes() == pdf_bytes(b"v1")
        assert fetched.bytes == len(pdf_bytes(b"v1"))
        assert fetched.redirects == []

    def test_a_pdf_split_across_chunks_is_written(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        whole = pdf_bytes(b"chunked")
        session = FakeSession(FakeResponse([whole[:2], whole[2:9], whole[9:]]))
        http_client.download_stream(session, RULE_URL, part)
        assert part.read_bytes() == whole

    def test_a_non_pdf_response_is_rejected_before_anything_is_kept(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = FakeSession(
            FakeResponse([b"<html><body>Error</body></html>"], content_type="text/html")
        )
        with pytest.raises(NotPDFError, match="not a PDF"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_an_http_error_becomes_a_fetch_error(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = FakeSession(FakeResponse([], status=503))
        with pytest.raises(FetchError):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_connection_error_becomes_a_fetch_error(self, tmp_path):
        import requests
        part = tmp_path / "x.pdf.part"
        session = FakeSession(exc=requests.ConnectionError("connection reset"))
        with pytest.raises(FetchError, match="Failed to download"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_a_redirect_inside_india_code_is_allowed(self, tmp_path):
        # ViewFileUploaded legitimately 302s to India Code's file server.
        part = tmp_path / "x.pdf.part"
        session = FakeSession(FakeResponse(
            [pdf_bytes()],
            url="https://upload.indiacode.nic.in/showfile?actid=AC_CEN_1&type=rule",
        ))
        http_client.download_stream(session, RULE_URL, part)
        assert part.exists()

    def test_a_redirect_off_india_code_is_refused(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = FakeSession(FakeResponse([pdf_bytes()], url="https://evil.example/x.pdf"))
        with pytest.raises(FetchError, match="not an India Code host"):
            http_client.download_stream(session, RULE_URL, part)
        assert not part.exists()

    def test_oversize_download_is_aborted(self, tmp_path):
        part = tmp_path / "x.pdf.part"
        session = FakeSession(FakeResponse([pdf_bytes(b"x" * 500)]))
        with pytest.raises(FetchError, match="exceeded"):
            http_client.download_stream(session, RULE_URL, part, max_bytes=64)
        assert not part.exists()


# --- full pipeline --------------------------------------------------------------


@pytest.fixture
def rule_harness(tmp_path, monkeypatch):
    """The real pipeline, with only the two HTTP calls stubbed."""
    store = DocumentStore(tmp_path / "data")
    store.ensure_layout()
    manifest = Manifest.load(store.manifest_path)
    state = {"content": pdf_bytes(b"v1"), "downloads": 0, "urls": []}

    monkeypatch.setattr(indiacode, "get_text", lambda session, url: act_page())

    def fake_download(session, url, dest_part, *, max_bytes=None, expect_pdf=True, **kwargs):
        state["downloads"] += 1
        state["urls"].append(url)
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        dest_part.write_bytes(state["content"])
        return FetchResult(
            "application/pdf", len(state["content"]), 200,
            final_url=state.get("final_url", url),
            redirects=list(state.get("redirects", [])),
        )

    monkeypatch.setattr(pipeline, "download_stream", fake_download)

    def run(url=RULE_URL, category="rules", **kwargs):
        kwargs.setdefault("parent_url", PARENT_URL)
        return pipeline.ingest_document(
            session=None, store=store, manifest=manifest,
            url=url, category=category, **kwargs,
        )

    return store, manifest, state, run


EXPECTED_ID = make_subordinate_document_id("rule", RULE_DESCRIPTION, RULE_URL)


class TestViewFileThroughThePipeline:
    def test_a_rule_is_stored_like_any_other_document(self, rule_harness):
        store, manifest, state, run = rule_harness
        result = run()

        assert result.outcome is Outcome.NEW
        assert result.document_id == EXPECTED_ID
        assert state["urls"] == [RULE_URL]

        pdf = store.pdf_path("rules", EXPECTED_ID)
        assert pdf.exists() and pdf.read_bytes() == state["content"]
        assert store.metadata_path("rules", EXPECTED_ID).exists()

    def test_document_id_matches_the_inventory(self, rule_harness):
        # Discovery and the downloader must agree on what this document *is*.
        from ingestion import discovery
        row = discovery.SubordinateRow(
            document_type="rule", tab_label="Rules",
            description=RULE_DESCRIPTION, english_url=RULE_URL,
            english_filename=RULE_FILENAME,
        )
        assert discovery._subordinate_document_id(row) == EXPECTED_ID

    def test_metadata_records_the_url_kind_and_the_exact_url(self, rule_harness):
        store, manifest, state, run = rule_harness
        run()
        meta = json.loads(
            store.metadata_path("rules", EXPECTED_ID).read_text(encoding="utf-8")
        )
        assert meta["pdf_url_kind"] == "viewfileuploaded"
        assert meta["primary_pdf_url"] == RULE_URL      # preserved exactly
        assert meta["document_type"] == "rule"
        assert meta["language"] == "en"
        assert meta["language_source"] == "indiacode_bitstream_label"
        assert meta["parent_handle"] == "123456789/1372"
        assert meta["parent_title"] == "Passports Act, 1967"
        assert meta["parent_url"] == PARENT_URL
        assert meta["india_code_path"].startswith("AC_CEN_")
        assert meta["download"]["sha256"] == json.loads(
            store.manifest_path.read_text(encoding="utf-8")
        )["documents"][EXPECTED_ID]["sha256"]

    def test_manifest_records_the_url_kind(self, rule_harness):
        store, manifest, state, run = rule_harness
        result = run()
        entry = manifest.get(EXPECTED_ID)
        assert entry["pdf_url_kind"] == "viewfileuploaded"
        assert entry["primary_pdf_url"] == RULE_URL
        assert entry["document_type"] == "rule"
        assert entry["language"] == "en"
        assert entry["sha256"] == result.sha256
        assert entry["parent_handle"] == "123456789/1372"

    def test_sha256_is_computed_over_the_stored_file(self, rule_harness):
        from ingestion.utils import sha256_bytes
        store, manifest, state, run = rule_harness
        result = run()
        assert result.sha256 == sha256_bytes(state["content"])

    def test_unchanged_on_a_second_identical_run(self, rule_harness):
        store, manifest, state, run = rule_harness
        first = run()
        second = run()
        assert second.outcome is Outcome.UNCHANGED
        assert second.sha256 == first.sha256
        assert manifest.get(EXPECTED_ID)["version_count"] == 1
        assert not (store.document_dir("rules", EXPECTED_ID) / "versions").exists()

    def test_skip_existing_avoids_the_network(self, rule_harness):
        store, manifest, state, run = rule_harness
        run()
        before = state["downloads"]
        assert run(skip_existing=True).outcome is Outcome.UNCHANGED
        assert state["downloads"] == before

    def test_updated_content_archives_the_previous_version(self, rule_harness):
        store, manifest, state, run = rule_harness
        first = run()
        state["content"] = pdf_bytes(b"v2-amended")
        second = run()

        assert second.outcome is Outcome.UPDATED
        assert second.sha256 != first.sha256
        assert store.pdf_path("rules", EXPECTED_ID).read_bytes() == state["content"]

        archived = list((store.document_dir("rules", EXPECTED_ID) / "versions").glob("*.pdf"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == pdf_bytes(b"v1")

        entry = manifest.get(EXPECTED_ID)
        assert entry["version_count"] == 2
        assert entry["versions"][0]["archived"] is True
        assert entry["versions"][1]["archived"] is False

    def test_dry_run_downloads_nothing(self, rule_harness):
        store, manifest, state, run = rule_harness
        result = run(dry_run=True)
        assert result.outcome is Outcome.DRY_RUN
        assert state["downloads"] == 0
        assert not store.pdf_path("rules", EXPECTED_ID).exists()

    def test_missing_parent_url_fails_without_downloading(self, rule_harness):
        store, manifest, state, run = rule_harness
        result = run(parent_url=None)
        assert result.outcome is Outcome.FAILED
        assert "parent act" in result.message
        assert state["downloads"] == 0
        assert manifest.documents == {}

    def test_a_regulation_lands_in_the_regulations_category(self, rule_harness, monkeypatch):
        store, manifest, state, run = rule_harness
        # Re-point the fixture's Rules tab at Regulations for this one check.
        monkeypatch.setattr(
            indiacode, "get_text",
            lambda session, url: act_page().replace(">Rules<", ">Regulations<"),
        )
        result = run(category="regulations")
        assert result.outcome is Outcome.NEW
        assert result.document_id.endswith(result.document_id.split("__")[-1])
        assert "__regulation-" in result.document_id
        assert store.pdf_path("regulations", result.document_id).exists()


class TestActsAreUnaffected:
    def test_a_handle_url_still_resolves_and_downloads(self, tmp_path, monkeypatch):
        store = DocumentStore(tmp_path / "data")
        store.ensure_layout()
        manifest = Manifest.load(store.manifest_path)
        monkeypatch.setattr(indiacode, "get_text", lambda session, url: act_page())
        monkeypatch.setattr(
            pipeline, "download_stream",
            lambda session, url, dest_part, **kwargs: (
                dest_part.parent.mkdir(parents=True, exist_ok=True),
                dest_part.write_bytes(pdf_bytes()),
                FetchResult("application/pdf", len(pdf_bytes()), 200, final_url=url),
            )[-1],
        )
        result = pipeline.ingest_document(
            None, store, manifest, PARENT_URL, "central_acts"
        )
        assert result.outcome is Outcome.NEW
        assert result.document_id == "passports-act-1967__handle-1372"
        entry = manifest.get("passports-act-1967__handle-1372")
        assert entry["pdf_url_kind"] == "bitstream"
        assert entry["primary_pdf_url"].endswith("/1/196715.pdf")
        assert "parent_handle" not in entry
