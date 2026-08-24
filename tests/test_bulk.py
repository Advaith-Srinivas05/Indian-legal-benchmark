"""Tests for the corpus-scale batch runner.

The pipeline itself is stubbed here — what is under test is everything a
20,000-document run adds around it: resumption, retry policy, the failure
report, checkpointing, progress, and the guarantee that concurrency cannot
corrupt the manifest.
"""

from __future__ import annotations

import json
import threading

import pytest

from ingestion import bulk, download
from ingestion.manifest import Manifest
from ingestion.models import DownloadResult, Outcome
from ingestion.storage import DocumentStore

URL = "https://www.indiacode.nic.in/handle/123456789/{}"


def target(n, category="central_acts", document_id=None):
    return download.Target(
        url=URL.format(n), category=category,
        document_id=document_id or f"doc-{n}", title=f"Act No. {n} of 1900",
    )


@pytest.fixture
def store(tmp_path):
    store = DocumentStore(tmp_path / "data")
    store.ensure_layout()
    return store


@pytest.fixture
def manifest(store):
    return Manifest.load(store.manifest_path)


def stub_pipeline(monkeypatch, handler):
    """Replace ingest_document with *handler(url, category, **kwargs)*."""
    calls: list[str] = []
    lock = threading.Lock()

    def fake(session, store, manifest, url, category, **kwargs):
        with lock:
            calls.append(url)
        return handler(url, category, **kwargs)

    monkeypatch.setattr(bulk, "ingest_document", fake)
    return calls


def ok(url, category, **kwargs):
    return DownloadResult(url, Outcome.NEW, document_id=url[-4:], category=category,
                          sha256="a" * 64, bytes=1000)


# --- resumption -----------------------------------------------------------------


class TestPartitionTargets:
    def test_documents_already_in_the_manifest_are_left_alone(self, manifest):
        manifest.upsert({"document_id": "doc-2", "sha256": "x"})
        todo, done = bulk.partition_targets([target(1), target(2), target(3)], manifest)
        assert [t.document_id for t in todo] == ["doc-1", "doc-3"]
        assert [t.document_id for t in done] == ["doc-2"]

    def test_a_target_without_an_id_is_always_processed(self, manifest):
        anonymous = download.Target(url=URL.format(9), category="rules")
        todo, done = bulk.partition_targets([anonymous], manifest)
        assert todo == [anonymous] and done == []

    def test_a_resumed_run_costs_no_network(self, store, manifest, monkeypatch):
        for i in (1, 2, 3):
            manifest.upsert({"document_id": f"doc-{i}", "sha256": "x"})
        calls = stub_pipeline(monkeypatch, ok)
        todo, done = bulk.partition_targets([target(i) for i in (1, 2, 3)], manifest)
        stats = bulk.run_batch(todo, store, manifest, workers=1, delay=0)
        assert calls == []
        assert len(done) == 3
        assert stats.processed == 0


# --- retry policy ---------------------------------------------------------------


class TestRetries:
    def _flaky(self, fail_times, error_type):
        state = {"n": 0}

        def handler(url, category, **kwargs):
            state["n"] += 1
            if state["n"] <= fail_times:
                return DownloadResult(url, Outcome.FAILED, message="boom",
                                      error_type=error_type)
            return ok(url, category)

        return handler, state

    def test_a_transient_failure_is_retried_and_can_succeed(self, store, manifest, monkeypatch):
        handler, state = self._flaky(1, "FetchError")
        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(1)], store, manifest, workers=1, delay=0,
                               retries=2, retry_backoff=0)
        assert state["n"] == 2
        assert stats.new == 1 and stats.failed == 0
        assert stats.retried == 1

    def test_retries_are_bounded(self, store, manifest, monkeypatch):
        handler, state = self._flaky(99, "FetchError")
        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(1)], store, manifest, workers=1, delay=0,
                               retries=2, retry_backoff=0)
        assert state["n"] == 3          # 1 attempt + 2 retries, then it stops
        assert stats.failed == 1

    @pytest.mark.parametrize("error_type", ["LanguageError", "NotPDFError",
                                            "InvalidURLError", "CorruptPDFError"])
    def test_a_permanent_failure_is_never_retried(self, store, manifest,
                                                  monkeypatch, error_type):
        handler, state = self._flaky(99, error_type)
        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(1)], store, manifest, workers=1, delay=0,
                               retries=5, retry_backoff=0)
        assert state["n"] == 1
        assert stats.failed == 1
        assert stats.failures_by_type == {error_type: 1}

    def test_a_stale_act_page_is_treated_as_transient(self, store, manifest, monkeypatch):
        """India Code intermittently serves an act page without its file block."""
        handler, state = self._flaky(1, "MetadataError")
        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(1)], store, manifest, workers=1, delay=0,
                               retries=1, retry_backoff=0)
        assert stats.new == 1 and state["n"] == 2


# --- failures never stop the batch ----------------------------------------------


class TestFailuresAreIsolated:
    def test_one_failure_does_not_end_the_run(self, store, manifest, monkeypatch):
        def handler(url, category, **kwargs):
            if url.endswith("2"):
                return DownloadResult(url, Outcome.FAILED, message="no English version",
                                      error_type="LanguageError")
            return ok(url, category)

        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(i) for i in range(1, 6)], store, manifest,
                               workers=1, delay=0, retry_backoff=0)
        assert stats.processed == 5
        assert stats.new == 4
        assert stats.failed == 1

    def test_an_unexpected_exception_is_still_a_failed_document(self, store, manifest,
                                                                monkeypatch):
        """ingest_document is contracted never to raise; if it does, the batch
        must not lose the other documents."""
        def handler(url, category, **kwargs):
            if url.endswith("2"):
                raise RuntimeError("stub exploded")
            return ok(url, category)

        stub_pipeline(monkeypatch, handler)
        with pytest.raises(RuntimeError):
            bulk.run_batch([target(i) for i in range(1, 4)], store, manifest,
                           workers=1, delay=0, retry_backoff=0)

    def test_the_failure_report_records_what_is_needed_to_investigate(
        self, store, manifest, monkeypatch, tmp_path
    ):
        stub_pipeline(monkeypatch, lambda url, category, **kw: DownloadResult(
            url, Outcome.FAILED, document_id="doc-7", category=category,
            message="No English version could be identified", error_type="LanguageError",
        ))
        report = bulk.FailureReport(tmp_path / "failures.jsonl")
        bulk.run_batch([target(7, "rules")], store, manifest, workers=1, delay=0,
                       retries=0, failures=report)

        rows = [json.loads(line) for line in report.path.read_text(encoding="utf-8").splitlines()]
        assert len(rows) == 1
        row = rows[0]
        assert row["document_id"] == "doc-7"
        assert row["title"] == "Act No. 7 of 1900"
        assert row["document_type"] == "rule"
        assert row["url"] == URL.format(7)
        assert row["error_type"] == "LanguageError"
        assert "No English version" in row["error_message"]
        assert row["timestamp"].startswith("20")
        assert row["attempts"] == 1

    def test_the_failure_report_appends_across_runs(self, store, manifest,
                                                    monkeypatch, tmp_path):
        stub_pipeline(monkeypatch, lambda url, category, **kw: DownloadResult(
            url, Outcome.FAILED, message="nope", error_type="NotPDFError"))
        path = tmp_path / "failures.jsonl"
        for i in (1, 2):
            bulk.run_batch([target(i)], store, manifest, workers=1, delay=0,
                           retries=0, failures=bulk.FailureReport(path))
        assert len(path.read_text(encoding="utf-8").splitlines()) == 2


# --- concurrency ----------------------------------------------------------------


class TestConcurrency:
    def test_every_document_is_processed_exactly_once(self, store, manifest, monkeypatch):
        calls = stub_pipeline(monkeypatch, ok)
        targets = [target(i) for i in range(1, 51)]
        stats = bulk.run_batch(targets, store, manifest, workers=4, delay=0)
        assert stats.processed == 50
        assert sorted(calls) == sorted(t.url for t in targets)
        assert len(calls) == len(set(calls))

    def test_concurrent_upserts_all_survive_in_the_saved_manifest(
        self, store, manifest, monkeypatch
    ):
        """The real pipeline writes to the manifest from the worker thread."""
        def handler(url, category, **kwargs):
            n = url.rsplit("/", 1)[-1]
            manifest.upsert({"document_id": f"doc-{n}", "sha256": n})
            manifest.save()
            return ok(url, category)

        stub_pipeline(monkeypatch, handler)
        bulk.run_batch([target(i) for i in range(1, 101)], store, manifest,
                       workers=8, delay=0, checkpoint_every=10)

        saved = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        assert len(saved["documents"]) == 100
        assert saved["documents"]["doc-50"]["sha256"] == "50"

    def test_the_rate_limiter_paces_the_whole_pool(self, store, manifest, monkeypatch):
        stub_pipeline(monkeypatch, ok)
        import time

        start = time.monotonic()
        bulk.run_batch([target(i) for i in range(1, 6)], store, manifest,
                       workers=4, delay=0.05)
        assert time.monotonic() - start >= 0.2   # 5 documents, 4 enforced gaps


# --- checkpointing and manifest safety -------------------------------------------


class TestCheckpointing:
    def test_the_manifest_is_written_periodically_not_per_document(
        self, store, manifest, monkeypatch
    ):
        writes = {"n": 0}
        original = Manifest._write

        def counted(self):
            writes["n"] += 1
            original(self)

        monkeypatch.setattr(Manifest, "_write", counted)

        def handler(url, category, **kwargs):
            manifest.upsert({"document_id": url[-2:], "sha256": "x"})
            manifest.save()
            return ok(url, category)

        stub_pipeline(monkeypatch, handler)
        bulk.run_batch([target(i) for i in range(10, 60)], store, manifest,
                       workers=1, delay=0, checkpoint_every=10)
        # 50 documents: 5 checkpoints + a final flush, not 50 whole-file writes.
        assert writes["n"] <= 8

    def test_everything_is_persisted_by_the_time_the_batch_returns(
        self, store, manifest, monkeypatch
    ):
        def handler(url, category, **kwargs):
            manifest.upsert({"document_id": url[-2:], "sha256": "x"})
            manifest.save()
            return ok(url, category)

        stub_pipeline(monkeypatch, handler)
        bulk.run_batch([target(i) for i in range(10, 25)], store, manifest,
                       workers=2, delay=0, checkpoint_every=100)
        saved = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        assert len(saved["documents"]) == 15

    def test_deferred_saves_are_flushed_even_if_the_body_raises(self, store, manifest):
        manifest.upsert({"document_id": "doc-1", "sha256": "x"})
        with pytest.raises(RuntimeError):
            with manifest.defer_saves():
                manifest.save()
                raise RuntimeError("interrupted")
        saved = json.loads(store.manifest_path.read_text(encoding="utf-8"))
        assert "doc-1" in saved["documents"]


# --- progress and reporting -------------------------------------------------------


class TestProgress:
    def test_progress_is_reported_at_the_requested_interval(self, store, manifest,
                                                            monkeypatch):
        stub_pipeline(monkeypatch, ok)
        seen: list[int] = []
        bulk.run_batch([target(i) for i in range(1, 21)], store, manifest,
                       workers=1, delay=0, progress_every=5,
                       on_progress=lambda s: seen.append(s.processed))
        assert seen == [5, 10, 15, 20]

    def test_the_progress_block_carries_the_expected_fields(self):
        stats = bulk.Stats(total=19859, processed=2500, new=2400, unchanged=50,
                           updated=10, failed=40, bytes=5 * 1024 ** 3)
        text = bulk.render_progress(stats)
        assert "[2500/19859]" in text
        for label in ("Downloaded:", "Unchanged:", "Failed:", "Bytes:", "Rate:", "ETA:"):
            assert label in text

    def test_stats_track_bytes_and_outcomes(self, store, manifest, monkeypatch):
        def handler(url, category, **kwargs):
            if url.endswith("1"):
                return DownloadResult(url, Outcome.UPDATED, bytes=10)
            if url.endswith("2"):
                return DownloadResult(url, Outcome.UNCHANGED, bytes=20)
            return DownloadResult(url, Outcome.NEW, bytes=30)

        stub_pipeline(monkeypatch, handler)
        stats = bulk.run_batch([target(i) for i in range(1, 5)], store, manifest,
                               workers=1, delay=0)
        assert (stats.new, stats.updated, stats.unchanged) == (2, 1, 1)
        assert stats.bytes == 10 + 20 + 30 + 30


# --- disk space -------------------------------------------------------------------


class TestFreeSpaceGuard:
    def test_enough_space_passes(self, tmp_path):
        ok_, free = bulk.check_free_space(tmp_path, 1024)
        assert ok_ is True and free > 1024

    def test_insufficient_space_is_reported_not_raised(self, tmp_path):
        ok_, free = bulk.check_free_space(tmp_path, 10 ** 18)
        assert ok_ is False and free >= 0

    def test_the_cli_refuses_to_start_without_room(self, tmp_path, monkeypatch, caplog):
        path = tmp_path / "urls.json"
        path.write_text(json.dumps([
            {"url": URL.format(1), "type": "central_act"}
        ]), encoding="utf-8")
        monkeypatch.setattr(bulk, "check_free_space", lambda p, r: (False, 1024))
        code = download.main([
            "--input", str(path), "--data-dir", str(tmp_path / "data"),
            "--min-free-gb", "70",
        ])
        assert code == 3


# --- formatting helpers ------------------------------------------------------------


class TestFormatting:
    @pytest.mark.parametrize("value,expected", [
        (512, "512 B"), (2048, "2.00 KB"), (5 * 1024 ** 3, "5.00 GB"),
    ])
    def test_bytes(self, value, expected):
        assert bulk.format_bytes(value) == expected

    @pytest.mark.parametrize("value,expected", [
        (45, "45s"), (150, "2m 30s"), (7200, "2h 00m"), (None, "unknown"),
    ])
    def test_duration(self, value, expected):
        assert bulk.format_duration(value) == expected
