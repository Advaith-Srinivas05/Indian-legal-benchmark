"""End-to-end tests of the ingest pipeline with the network stubbed out.

These exercise the core idempotency guarantees (NEW -> UNCHANGED on identical
content, UPDATED with the old PDF archived rather than destroyed) and the
English-only guarantee: a Hindi bitstream is never downloaded, stored, given a
``metadata.json`` or written into ``data/manifest.json``.
"""

from __future__ import annotations

import json

import pytest

from ingestion import pipeline
from ingestion.manifest import Manifest
from ingestion.models import BitstreamRef, FetchResult, Outcome, ParsedItem
from ingestion.storage import DocumentStore

DOC_ID = "passports-act-1967__handle-1372"
ENGLISH_URL = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
HINDI_URL = "https://www.indiacode.nic.in/bitstream/123456789/1372/2/H1967-15.pdf"


def _pdf_bytes(body: bytes = b"body") -> bytes:
    # Padded past MIN_PDF_BYTES so validation accepts it as a plausible PDF.
    padding = b"%" + b"x" * 128 + b"\n"
    return b"%PDF-1.4\n" + padding + body + b"\n%%EOF\n"


def _english_ref() -> BitstreamRef:
    return BitstreamRef(
        url=ENGLISH_URL, filename="196715.pdf", sequence=1, is_primary=True,
        link_text="The Passports Act, 1967",
        language="en", language_source="indiacode_metadata_title",
        language_evidence="link text matches India Code English title",
    )


def _hindi_ref() -> BitstreamRef:
    return BitstreamRef(
        url=HINDI_URL, filename="H1967-15.pdf", sequence=2,
        link_text="पासपोर्ट अधिनियम, 1967",
        language="hi", language_source="indiacode_hindi_title",
        language_evidence="link text matches India Code Hindi Title",
    )


def _item(*bitstreams: BitstreamRef) -> ParsedItem:
    """The Passports Act item, by default with both language versions attached."""
    refs = list(bitstreams) or [_english_ref(), _hindi_ref()]
    return ParsedItem(
        source_url="https://www.indiacode.nic.in/handle/123456789/1372",
        handle="123456789/1372",
        handle_id="1372",
        title="Passports Act, 1967",
        primary_pdf_url=ENGLISH_URL,
        bitstreams=refs,
        metadata={"act_number": "15", "act_year": 1967, "ministry": "Ministry of External Affairs"},
        dublin_core={"DC.title": "Passports Act, 1967"},
        metadata_table={"act number": "15"},
    )


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A store+manifest with resolve() and download_stream() stubbed.

    The download stub writes whatever ``state['content']`` currently holds and
    records every URL it was asked for, so tests can assert both that the
    network was/wasn't used and *which* file was fetched.
    """
    store = DocumentStore(tmp_path / "data")
    store.ensure_layout()
    manifest = Manifest.load(store.manifest_path)

    state = {"content": _pdf_bytes(b"v1"), "downloads": 0, "urls": [], "item": _item()}

    monkeypatch.setattr(
        pipeline.indiacode, "resolve",
        lambda session, url, **kwargs: state["item"],
    )

    def fake_download(session, url, dest_part, *, max_bytes=None, **kwargs):
        state["downloads"] += 1
        state["urls"].append(url)
        dest_part.parent.mkdir(parents=True, exist_ok=True)
        dest_part.write_bytes(state["content"])
        return FetchResult("application/pdf", len(state["content"]), 200, final_url=url)

    monkeypatch.setattr(pipeline, "download_stream", fake_download)

    def run(**kwargs):
        return pipeline.ingest_document(
            session=None, store=store, manifest=manifest,
            url="https://www.indiacode.nic.in/handle/123456789/1372",
            category="central_acts", **kwargs,
        )

    return store, manifest, state, run


class TestNewDownload:
    def test_new_document_is_stored_with_metadata(self, harness):
        store, manifest, state, run = harness
        result = run()

        assert result.outcome is Outcome.NEW
        doc_id = DOC_ID
        pdf = store.pdf_path("central_acts", doc_id)
        meta = store.metadata_path("central_acts", doc_id)
        assert pdf.exists() and pdf.read_bytes() == state["content"]
        assert meta.exists()

        metadata = json.loads(meta.read_text(encoding="utf-8"))
        assert metadata["document_id"] == doc_id
        assert metadata["document_type"] == "central_act"
        assert metadata["jurisdiction"] == "India"
        assert metadata["download"]["sha256"] == result.sha256
        # Manifest updated and persisted.
        assert manifest.get(doc_id)["sha256"] == result.sha256
        assert store.manifest_path.exists()


class TestEnglishOnly:
    """The item offers English *and* Hindi; only English may ever be taken."""

    def test_only_the_english_pdf_is_downloaded(self, harness):
        store, manifest, state, run = harness
        result = run()
        assert result.outcome is Outcome.NEW
        assert state["urls"] == [ENGLISH_URL]
        assert HINDI_URL not in state["urls"]

    def test_no_hindi_file_is_written_to_disk(self, harness):
        store, manifest, state, run = harness
        run()
        stored = sorted(p.name for p in store.document_dir("central_acts", DOC_ID).iterdir())
        assert stored == ["metadata.json", f"{DOC_ID}.pdf"]

    def test_manifest_records_english_and_never_mentions_hindi(self, harness):
        store, manifest, state, run = harness
        run()
        entry = manifest.get(DOC_ID)
        assert entry["language"] == "en"
        assert entry["language_source"] == "indiacode_metadata_title"
        assert entry["primary_pdf_url"] == ENGLISH_URL

        # The whole manifest file must be free of the Hindi bitstream.
        raw = store.manifest_path.read_text(encoding="utf-8")
        assert "H1967-15.pdf" not in raw
        assert HINDI_URL not in raw
        assert len(json.loads(raw)["documents"]) == 1

    def test_metadata_states_language_and_how_it_was_determined(self, harness):
        store, manifest, state, run = harness
        run()
        meta = json.loads(
            store.metadata_path("central_acts", DOC_ID).read_text(encoding="utf-8")
        )
        assert meta["language"] == "en"
        assert meta["language_source"] == "indiacode_metadata_title"
        assert meta["language_evidence"]
        assert meta["primary_pdf_url"] == ENGLISH_URL

    def test_hindi_bitstream_is_recorded_as_excluded_not_downloaded(self, harness):
        store, manifest, state, run = harness
        run()
        meta = json.loads(
            store.metadata_path("central_acts", DOC_ID).read_text(encoding="utf-8")
        )
        by_name = {b["filename"]: b for b in meta["available_bitstreams"]}
        assert by_name["196715.pdf"]["downloaded"] is True
        assert by_name["196715.pdf"]["role"] == "selected_english"
        hindi = by_name["H1967-15.pdf"]
        assert hindi["downloaded"] is False
        assert hindi["language"] == "hi"
        assert "English-only" in hindi["excluded_reason"]

    def test_all_bitstreams_still_refuses_hindi(self, harness):
        store, manifest, state, run = harness
        run(all_bitstreams=True)
        assert state["urls"] == [ENGLISH_URL]
        stored = sorted(p.name for p in store.document_dir("central_acts", DOC_ID).iterdir())
        assert stored == ["metadata.json", f"{DOC_ID}.pdf"]

    def test_dry_run_reports_the_english_pdf(self, harness):
        store, manifest, state, run = harness
        result = run(dry_run=True)
        assert result.outcome is Outcome.DRY_RUN
        assert ENGLISH_URL in result.message
        assert HINDI_URL not in result.message


class TestNoUsableEnglishVersion:
    """Nothing at all is stored when English cannot be identified."""

    def _assert_nothing_written(self, store, manifest, state):
        assert state["downloads"] == 0
        assert manifest.get(DOC_ID) is None
        assert manifest.documents == {}
        assert not store.document_dir("central_acts", DOC_ID).exists()

    def test_hindi_only_document_fails_without_downloading(self, harness):
        store, manifest, state, run = harness
        state["item"] = _item(_hindi_ref())
        result = run()
        assert result.outcome is Outcome.FAILED
        assert "No English version" in result.message
        self._assert_nothing_written(store, manifest, state)

    def test_hindi_only_document_fails_in_dry_run_too(self, harness):
        store, manifest, state, run = harness
        state["item"] = _item(_hindi_ref())
        assert run(dry_run=True).outcome is Outcome.FAILED

    def test_ambiguous_language_fails_without_downloading(self, harness):
        store, manifest, state, run = harness
        undetermined = BitstreamRef(
            url=ENGLISH_URL, filename="doc.pdf", sequence=1, is_primary=True,
            link_text="Download PDF",
        )
        state["item"] = _item(undetermined)
        result = run()
        assert result.outcome is Outcome.FAILED
        assert "could not be determined" in result.message
        assert "NOT assumed to be English" in result.message
        self._assert_nothing_written(store, manifest, state)


class TestIdempotency:
    def test_second_run_same_content_is_unchanged(self, harness):
        store, manifest, state, run = harness
        run()
        result = run()
        assert result.outcome is Outcome.UNCHANGED
        doc_id = "passports-act-1967__handle-1372"
        # No versions directory created; still a single version recorded.
        versions_dir = store.document_dir("central_acts", doc_id) / "versions"
        assert not versions_dir.exists()
        assert manifest.get(doc_id)["version_count"] == 1

    def test_skip_existing_avoids_network(self, harness):
        store, manifest, state, run = harness
        run()
        downloads_after_first = state["downloads"]
        result = run(skip_existing=True)
        assert result.outcome is Outcome.UNCHANGED
        assert state["downloads"] == downloads_after_first  # no new download


class TestUpdateDetection:
    def test_changed_content_archives_old_and_updates(self, harness):
        store, manifest, state, run = harness
        first = run()
        doc_id = "passports-act-1967__handle-1372"

        # Content changes upstream.
        state["content"] = _pdf_bytes(b"v2-amended")
        second = run()

        assert second.outcome is Outcome.UPDATED
        assert second.sha256 != first.sha256

        # Current PDF holds the new content.
        pdf = store.pdf_path("central_acts", doc_id)
        assert pdf.read_bytes() == state["content"]

        # Old version archived (never destroyed).
        versions_dir = store.document_dir("central_acts", doc_id) / "versions"
        archived = list(versions_dir.glob("*.pdf"))
        assert len(archived) == 1
        assert archived[0].read_bytes() == _pdf_bytes(b"v1")

        entry = manifest.get(doc_id)
        assert entry["version_count"] == 2
        assert entry["sha256"] == second.sha256
        assert entry["versions"][0]["archived"] is True
        assert entry["versions"][1]["archived"] is False


class TestDryRun:
    def test_dry_run_writes_nothing(self, harness):
        store, manifest, state, run = harness
        result = run(dry_run=True)
        assert result.outcome is Outcome.DRY_RUN
        assert state["downloads"] == 0
        doc_id = "passports-act-1967__handle-1372"
        assert not store.pdf_path("central_acts", doc_id).exists()
        assert manifest.get(doc_id) is None


class TestErrorHandling:
    def test_invalid_url_becomes_failed_result(self, harness, monkeypatch):
        store, manifest, state, run = harness
        from ingestion.errors import InvalidURLError

        def boom(session, url, **kwargs):
            raise InvalidURLError("bad url")

        monkeypatch.setattr(pipeline.indiacode, "resolve", boom)
        result = run()
        assert result.outcome is Outcome.FAILED
        assert "bad url" in result.message
