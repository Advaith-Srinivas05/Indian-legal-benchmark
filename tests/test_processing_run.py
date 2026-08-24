"""Tests for the full-corpus processing runner (``python -m processing.run``).

The runner adds nothing to what one document does; what it adds is what 19,802
of them need. These tests are about exactly that: does an interrupted run resume
without redoing work, does a journal record claiming success get checked against
the files it names, does one unreadable PDF cost one record rather than the run,
and can a half-written document ever be mistaken for a finished one.

Everything is offline and runs against a miniature corpus built in a temporary
directory by :mod:`tests.corpusbuild`. Nothing here touches the real 39 GiB
corpus.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from processing import config, run as runner
from processing.corpus import Corpus
from processing.errors import ProcessingError, RunLockError
from processing.models import ProcessedDocument
from processing.process import output_dir
from tests import corpusbuild

# The tmp_path drive can be anywhere; the free-space guard is a production
# concern and is exercised on its own rather than in every run.
NO_SPACE_CHECK = {"min_free_bytes": 0, "min_headroom_bytes": 0}


# --- Fixtures -------------------------------------------------------------------


def build_corpus(data_dir: Path, specs) -> Path:
    entries = [corpusbuild.add_document(data_dir, document_id, **kwargs)
               for document_id, kwargs in specs]
    corpusbuild.write_corpus(data_dir, entries)
    return data_dir


@pytest.fixture()
def empty_corpus(tmp_path):
    corpusbuild.write_corpus(tmp_path, [])
    return tmp_path


@pytest.fixture()
def one_document(tmp_path):
    return build_corpus(tmp_path, [
        ("act-a__handle-1", {"category": "central_acts", "title": "Act A, 1999"}),
    ])


@pytest.fixture()
def mixed_corpus(tmp_path):
    """One clean act, one scanned document, one unreadable PDF, one state act."""
    return build_corpus(tmp_path, [
        ("act-a__handle-1", {"category": "central_acts", "title": "Act A, 1999"}),
        ("act-b__handle-2", {"category": "state_acts", "title": "Act B, 2001"}),
        ("scan-c__handle-3", {"category": "rules", "kind": "scanned"}),
        ("broken-d__handle-4", {"category": "regulations", "kind": "corrupt"}),
    ])


def journal_rows(data_dir: Path) -> list[dict]:
    return list(runner.StatusJournal(runner.journal_path(data_dir)).iter_records())


def statuses(data_dir: Path) -> dict[str, str]:
    return {row["document_id"]: row["status"]
            for row in runner.StatusJournal(runner.journal_path(data_dir)).load().values()}


# --- The empty corpus -----------------------------------------------------------


class TestEmptyCorpus:
    def test_a_corpus_with_no_documents_is_a_clean_no_op(self, empty_corpus):
        report = runner.run(empty_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["selected"] == 0
        assert report["this_run"]["attempted"] == 0
        assert report["corpus_totals"]["documents_in_journal"] == 0

    def test_it_still_writes_a_report(self, empty_corpus):
        runner.run(empty_corpus, workers=2, **NO_SPACE_CHECK)
        assert runner.report_path(empty_corpus).exists()

    def test_no_journal_is_created_for_nothing(self, empty_corpus):
        runner.run(empty_corpus, workers=2, **NO_SPACE_CHECK)
        assert not runner.journal_path(empty_corpus).exists()


# --- One document, and several ---------------------------------------------------


class TestOneDocument:
    def test_it_is_processed_and_journalled(self, one_document):
        report = runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 1
        assert report["this_run"]["successful"] == 1
        assert statuses(one_document) == {"act-a__handle-1": "SUCCESS"}

    def test_the_mandated_output_files_are_written(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        directory = output_dir(one_document, "act-a__handle-1")
        assert (directory / config.DOCUMENT_FILENAME).exists()
        assert (directory / config.PAGES_FILENAME).exists()

    def test_the_output_format_is_the_existing_one(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        payload = json.loads(
            (output_dir(one_document, "act-a__handle-1") / config.DOCUMENT_FILENAME)
            .read_text(encoding="utf-8"))
        assert payload["schema_version"] == config.PROCESSING_SCHEMA_VERSION
        assert payload["source"]["document_id"] == "act-a__handle-1"
        assert payload["artifacts"]["pages"] == config.PAGES_FILENAME


class TestSeveralDocuments:
    def test_every_document_gets_exactly_one_record(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        rows = journal_rows(mixed_corpus)
        assert len(rows) == 4
        assert len({row["document_id"] for row in rows}) == 4

    def test_outcomes_are_separated_by_status(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        by_id = statuses(mixed_corpus)
        assert by_id["act-a__handle-1"] == "SUCCESS"
        assert by_id["act-b__handle-2"] == "SUCCESS"
        assert by_id["scan-c__handle-3"] == "QUARANTINED"
        assert by_id["broken-d__handle-4"] == "FAILED"

    def test_every_status_is_from_the_controlled_vocabulary(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        for row in journal_rows(mixed_corpus):
            assert row["status"] in config.PROCESSING_STATUSES


# --- Provenance on the journal record --------------------------------------------


class TestJournalRecord:
    @pytest.fixture()
    def record(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        return journal_rows(one_document)[0]

    def test_it_carries_the_whole_provenance_chain(self, record):
        for key in ("document_id", "title", "document_type", "pdf_relpath",
                    "sha256", "source_url", "handle", "language"):
            assert key in record
        assert len(record["sha256"]) == 64

    def test_it_carries_the_four_verdicts(self, record):
        assert record["content_language"] in config.CONTENT_LANGUAGES
        assert record["extraction_quality"] in config.QUALITY_LEVELS
        assert record["orientation_suspect"] in (True, False)
        assert record["ocr_action"]

    def test_it_carries_extraction_and_output_facts(self, record):
        assert record["pages"] >= 1
        assert record["chars"] > 0
        assert record["output_relpath"].startswith("processed/indiacode/")
        assert record["output_bytes"]["document"] > 0
        assert record["output_bytes"]["pages"] > 0

    def test_it_is_versioned_and_timestamped(self, record):
        assert record["journal_schema_version"] == config.JOURNAL_SCHEMA_VERSION
        assert record["processing_schema_version"] == config.PROCESSING_SCHEMA_VERSION
        assert record["processed_at"].endswith("+00:00")
        assert record["attempt"] == 1


# --- OCR routing -----------------------------------------------------------------


class TestOcrRouting:
    def test_a_document_needing_ocr_is_quarantined_not_indexed(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "scan-c__handle-3"]
        assert record["ocr_action"] in config.OCR_ACTIONS_BLOCKING_INDEX
        assert record["eligible_for_indexing"] is False
        assert any(r.startswith("ocr_") for r in record["quarantine_reasons"])

    def test_the_existing_router_decides_and_no_engine_runs(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        # SUCCESS_OCR is reserved; nothing in this repository OCRs a document.
        assert report["this_run"]["successful_with_ocr"] == 0
        assert report["corpus_totals"]["pages_ocr"] == 0
        assert report["corpus_totals"]["documents_pending_ocr"] >= 1
        assert "must not be read as" in report["note"]

    def test_ocr_usage_is_recorded_in_the_report(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["documents_by_ocr_action"]
        assert report["documents_by_ocr_text_source"]


# --- Quarantine and the language gate --------------------------------------------


class TestQuarantine:
    def test_quarantine_records_an_explicit_reason(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "scan-c__handle-3"]
        assert record["status"] == "QUARANTINED"
        assert record["quarantine_reasons"]

    def test_quarantined_output_is_still_written(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        directory = output_dir(mixed_corpus, "scan-c__handle-3")
        assert (directory / config.DOCUMENT_FILENAME).exists()
        assert (directory / config.PAGES_FILENAME).exists()

    def test_the_language_gate_is_not_bypassed(self, monkeypatch, one_document):
        """A document the content-language gate rejects must be quarantined."""
        from processing import language as language_module

        real = language_module.assess_pages

        def uncertain(pages, **kwargs):
            assessment = real(pages, **kwargs)
            assessment.content_language = "uncertain"
            assessment.eligible_for_indexing = False
            return assessment

        monkeypatch.setattr("processing.process.language.assess_pages", uncertain)
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        record = journal_rows(one_document)[0]
        assert record["status"] == "QUARANTINED"
        assert "content_language_uncertain" in record["quarantine_reasons"]

    def test_the_report_counts_quarantine_by_reason(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["eligibility"]["quarantined_by_reason"]


# --- Failures --------------------------------------------------------------------


class TestProcessingFailure:
    def test_an_unreadable_pdf_is_recorded_not_raised(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "broken-d__handle-4"]
        assert record["status"] == "FAILED"
        assert record["error_type"] == "PDFOpenError"
        assert record["error_stage"] == "extract"

    def test_a_failure_record_carries_enough_to_debug_it(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "broken-d__handle-4"]
        for key in ("document_id", "title", "document_type", "pdf_relpath",
                    "error_type", "error_message", "processed_at", "attempt"):
            assert record.get(key) is not None

    def test_a_failed_document_writes_no_output(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert not output_dir(mixed_corpus, "broken-d__handle-4").exists()

    def test_one_bad_pdf_does_not_stop_the_others(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 4
        assert report["this_run"]["failed"] == 1
        assert report["this_run"]["successful"] == 2

    def test_a_failed_document_is_retried_on_the_next_run(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 1
        assert report["this_run"]["failed"] == 1

    def test_skip_failed_holds_it_back(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        report = runner.run(mixed_corpus, workers=2, retry_failed=False,
                            **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 0
        assert report["this_run"]["held_back_failed"] == 1

    def test_the_attempt_number_increases(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "broken-d__handle-4"]
        assert record["attempt"] == 2


class TestUnexpectedException:
    def test_an_exception_escaping_process_document_is_captured(
            self, monkeypatch, mixed_corpus):
        def explode(document, data_dir, **kwargs):
            if document.document_id == "act-b__handle-2":
                raise RuntimeError("something nobody predicted")
            return _real_process_document(document, data_dir, **kwargs)

        _real_process_document = runner.process_document
        monkeypatch.setattr(runner, "process_document", explode)

        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-b__handle-2"]
        assert record["status"] == "FAILED"
        assert record["error_type"] == "RuntimeError"
        assert record["error_stage"] == "unexpected"
        assert "something nobody predicted" in record["error_message"]
        # …and it is not silently swallowed: the traceback is kept.
        assert "RuntimeError" in record["traceback"]
        # …and the rest of the corpus still ran.
        assert report["this_run"]["attempted"] == 4
        assert report["this_run"]["successful"] == 1

    def test_a_write_failure_is_recorded_as_a_failure_not_a_success(
            self, monkeypatch, one_document):
        def refuse(result, data_dir):
            raise OSError("No space left on device")

        monkeypatch.setattr(runner, "write_document_output", refuse)
        report = runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        record = journal_rows(one_document)[0]
        assert record["status"] == "FAILED"
        assert record["error_stage"] == "write"
        assert report["this_run"]["successful"] == 0


class TestErrorGrouping:
    def test_errors_are_grouped_by_type_in_the_report(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["errors_by_type"]["PDFOpenError"] == 1
        assert report["this_run"]["errors_by_type"]["PDFOpenError"] == 1

    def test_failure_rows_are_carried_in_the_report(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert len(report["failures"]) == 1
        assert report["failures"][0]["document_id"] == "broken-d__handle-4"
        assert report["failures"][0]["pdf_relpath"]


# --- Documents the manifest names but disk does not have -------------------------


class TestUnavailableDocuments:
    def test_a_manifest_entry_with_no_pdf_is_skipped_not_failed(self, mixed_corpus):
        Corpus.load(mixed_corpus).get("act-b__handle-2").pdf_path.unlink()
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-b__handle-2"]
        assert record["status"] == "SKIPPED"
        assert record["skip_reason"] == "pdf_not_on_disk"
        assert report["this_run"]["skipped"] == 1

    def test_a_skipped_document_writes_no_output(self, mixed_corpus):
        Corpus.load(mixed_corpus).get("act-b__handle-2").pdf_path.unlink()
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert not output_dir(mixed_corpus, "act-b__handle-2").exists()

    def test_documents_outside_the_manifest_are_never_reached(self, mixed_corpus):
        """Enumeration is through the manifest, never by walking data/raw/."""
        stray = mixed_corpus / "raw" / "indiacode" / "central_acts" / "stray" / "stray.pdf"
        stray.parent.mkdir(parents=True, exist_ok=True)
        from tests import pdfbuild
        pdfbuild.text_pdf(stray, ["A document nobody indexed."])
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert "stray" not in statuses(mixed_corpus)


# --- Resume ----------------------------------------------------------------------


class TestResume:
    def test_a_completed_document_is_not_processed_again(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        marker = output_dir(mixed_corpus, "act-a__handle-1") / config.DOCUMENT_FILENAME
        stamp = marker.stat().st_mtime_ns
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert "act-a__handle-1" not in {
            item.document.document_id
            for item in runner.plan_run(
                Corpus.load(mixed_corpus),
                runner.StatusJournal(runner.journal_path(mixed_corpus)).load()).todo
        }
        assert marker.stat().st_mtime_ns == stamp
        assert report["this_run"]["already_complete"] == 3

    def test_an_interrupted_run_resumes_where_it_stopped(self, mixed_corpus):
        """Process two of four, then re-run: the other two are picked up."""
        first = runner.run(mixed_corpus, workers=1, limit=2, **NO_SPACE_CHECK)
        assert first["this_run"]["attempted"] == 2
        assert first["this_run"]["deferred_by_limit"] == 2

        second = runner.run(mixed_corpus, workers=1, **NO_SPACE_CHECK)
        assert second["this_run"]["attempted"] == 2
        assert second["corpus_totals"]["documents_in_journal"] == 4

    def test_resume_costs_no_reprocessing(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        # Only the FAILED document is retried; the three complete ones are not.
        assert report["this_run"]["selected"] == 1

    def test_force_reprocesses_everything(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        report = runner.run(mixed_corpus, workers=2, force=True, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 4

    def test_a_changed_source_pdf_is_reprocessed(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        manifest_path = mixed_corpus / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["documents"]["act-a__handle-1"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        plan, _ = runner.dry_run(mixed_corpus)
        assert "act-a__handle-1" in {i.document.document_id for i in plan.todo}
        assert "source PDF changed since it was processed" in plan.reasons()


class TestExistingOutputIsVerified:
    def test_a_deleted_output_directory_causes_reprocessing(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        import shutil
        shutil.rmtree(output_dir(mixed_corpus, "act-a__handle-1"))
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert (output_dir(mixed_corpus, "act-a__handle-1")
                / config.DOCUMENT_FILENAME).exists()
        assert report["this_run"]["successful"] == 1

    def test_a_missing_document_json_causes_reprocessing(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        (output_dir(mixed_corpus, "act-a__handle-1")
         / config.DOCUMENT_FILENAME).unlink()
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["successful"] == 1

    def test_a_truncated_document_json_causes_reprocessing(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        path = output_dir(mixed_corpus, "act-a__handle-1") / config.DOCUMENT_FILENAME
        path.write_text('{"schema_version": 1, "sourc', encoding="utf-8")
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["successful"] == 1
        assert json.loads(path.read_text(encoding="utf-8"))["source"]["document_id"]

    def test_output_naming_another_document_causes_reprocessing(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        path = output_dir(mixed_corpus, "act-a__handle-1") / config.DOCUMENT_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["source"]["document_id"] = "somebody-else"
        path.write_text(json.dumps(payload), encoding="utf-8")
        document = Corpus.load(mixed_corpus).get("act-a__handle-1")
        row = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-a__handle-1"]
        assert runner.verify_output(mixed_corpus, document, row) is not None

    def test_only_the_damaged_document_is_reprocessed(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        untouched = (output_dir(mixed_corpus, "act-b__handle-2")
                     / config.DOCUMENT_FILENAME)
        stamp = untouched.stat().st_mtime_ns
        (output_dir(mixed_corpus, "act-a__handle-1")
         / config.DOCUMENT_FILENAME).unlink()
        report = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert report["this_run"]["selected"] == 2      # the damaged one + the failure
        assert untouched.stat().st_mtime_ns == stamp

    def test_directory_existence_alone_is_never_trusted(self, mixed_corpus):
        """An empty output directory must not read as a completed document."""
        directory = output_dir(mixed_corpus, "act-a__handle-1")
        directory.mkdir(parents=True, exist_ok=True)
        document = Corpus.load(mixed_corpus).get("act-a__handle-1")
        assert runner.verify_output(mixed_corpus, document, None) == "never processed"

    def test_fast_verification_does_not_read_the_output(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        import shutil
        shutil.rmtree(output_dir(mixed_corpus, "act-a__handle-1"))
        document = Corpus.load(mixed_corpus).get("act-a__handle-1")
        row = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-a__handle-1"]
        assert runner.verify_output(mixed_corpus, document, row, level="fast") is None
        assert runner.verify_output(mixed_corpus, document, row, level="standard")

    def test_full_verification_reads_pages_json(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        document = Corpus.load(mixed_corpus).get("act-a__handle-1")
        row = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-a__handle-1"]
        assert runner.verify_output(mixed_corpus, document, row, level="full") is None

        path = output_dir(mixed_corpus, "act-a__handle-1") / config.PAGES_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["pages"] = []
        path.write_text(json.dumps(payload), encoding="utf-8")
        # Tell the journal the new size, so 'standard' is satisfied and only the
        # deeper level can see that the page list no longer matches the count.
        row = {**row, "output_bytes": {**row["output_bytes"],
                                       "pages": path.stat().st_size}}
        assert runner.verify_output(mixed_corpus, document, row, level="standard") is None
        assert "pages but claims" in runner.verify_output(
            mixed_corpus, document, row, level="full")

    def test_an_unknown_verification_level_is_refused(self, mixed_corpus):
        document = Corpus.load(mixed_corpus).get("act-a__handle-1")
        with pytest.raises(ValueError):
            runner.verify_output(mixed_corpus, document, None, level="paranoid")


# --- Atomic output ----------------------------------------------------------------


class TestAtomicOutput:
    def test_a_crash_midway_leaves_no_completion_marker(self, monkeypatch, one_document):
        """document.json is the marker: it is deleted first and written last."""
        from processing import process as process_module

        real_write = process_module.write_outputs

        def die_after_pages(result, data_dir):
            directory = output_dir(data_dir, result.document.document_id)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / config.PAGES_FILENAME).write_text("{}", encoding="utf-8")
            raise KeyboardInterrupt("killed mid-document")

        monkeypatch.setattr(runner, "write_outputs", die_after_pages)
        with pytest.raises(KeyboardInterrupt):
            runner.write_document_output(
                ProcessedDocument(document=Corpus.load(one_document)
                                  .get("act-a__handle-1")),
                one_document,
            )
        directory = output_dir(one_document, "act-a__handle-1")
        assert (directory / config.PAGES_FILENAME).exists()
        assert not (directory / config.DOCUMENT_FILENAME).exists()
        assert real_write is process_module.write_outputs

    def test_a_half_written_document_is_reprocessed_not_trusted(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        directory = output_dir(one_document, "act-a__handle-1")
        (directory / config.DOCUMENT_FILENAME).unlink()      # the crash signature
        document = Corpus.load(one_document).get("act-a__handle-1")
        row = runner.StatusJournal(runner.journal_path(one_document)).load()[
            "act-a__handle-1"]
        assert "incomplete" in runner.verify_output(one_document, document, row)

    def test_a_stale_document_json_is_removed_before_rewriting(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        directory = output_dir(one_document, "act-a__handle-1")
        (directory / config.DOCUMENT_FILENAME).write_text(
            '{"stale": true}', encoding="utf-8")
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        payload = json.loads(
            (directory / config.DOCUMENT_FILENAME).read_text(encoding="utf-8"))
        assert "stale" not in payload
        assert payload["source"]["document_id"] == "act-a__handle-1"

    def test_stale_temp_files_are_cleaned_up(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        directory = output_dir(one_document, "act-a__handle-1")
        stale = directory / f".{config.PAGES_FILENAME}.tmp"
        stale.write_text("half a file", encoding="utf-8")
        runner.run(one_document, workers=1, force=True, **NO_SPACE_CHECK)
        assert not stale.exists()

    def test_no_temp_file_survives_a_normal_run(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        leftovers = list((mixed_corpus / "processed").rglob(".*.tmp"))
        assert leftovers == []


# --- The journal itself -----------------------------------------------------------


class TestStatusJournal:
    def test_it_is_append_only(self, tmp_path):
        journal = runner.StatusJournal(tmp_path / "j.jsonl")
        journal.append({"document_id": "a", "status": "FAILED"})
        journal.append({"document_id": "a", "status": "SUCCESS"})
        assert len(list(journal.iter_records())) == 2
        assert journal.load()["a"]["status"] == "SUCCESS"

    def test_the_last_record_wins(self, tmp_path):
        journal = runner.StatusJournal(tmp_path / "j.jsonl")
        journal.append({"document_id": "a", "status": "SUCCESS"})
        journal.append({"document_id": "a", "status": "FAILED"})
        assert journal.load()["a"]["status"] == "FAILED"

    def test_a_truncated_final_line_does_not_break_the_journal(self, tmp_path):
        path = tmp_path / "j.jsonl"
        journal = runner.StatusJournal(path)
        journal.append({"document_id": "a", "status": "SUCCESS"})
        with open(path, "a", encoding="utf-8") as handle:
            handle.write('{"document_id": "b", "stat')
        assert list(journal.load()) == ["a"]

    def test_a_missing_journal_reads_as_empty(self, tmp_path):
        assert runner.StatusJournal(tmp_path / "nothing.jsonl").load() == {}

    def test_attempt_counts_are_read_from_the_history(self, tmp_path):
        journal = runner.StatusJournal(tmp_path / "j.jsonl")
        journal.append({"document_id": "a", "status": "FAILED"})
        journal.append({"document_id": "a", "status": "FAILED"})
        journal.append({"document_id": "b", "status": "SUCCESS"})
        counts = journal.attempt_counts()
        assert counts["a"] == 2 and counts["b"] == 1

    def test_it_is_written_beside_the_manifest(self, one_document):
        runner.run(one_document, workers=1, **NO_SPACE_CHECK)
        assert runner.journal_path(one_document) == (
            one_document / config.PROCESSING_STATUS_FILENAME)
        assert runner.journal_path(one_document).exists()

    def test_appends_from_several_threads_do_not_interleave(self, tmp_path):
        journal = runner.StatusJournal(tmp_path / "j.jsonl")
        payload = {"filler": "x" * 4000}

        def append(index):
            for step in range(20):
                journal.append({"document_id": f"{index}-{step}", **payload})

        threads = [threading.Thread(target=append, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(list(journal.iter_records())) == 80
        assert len(journal.load()) == 80


# --- Concurrency -------------------------------------------------------------------


class TestConcurrency:
    def test_no_document_is_processed_twice(self, monkeypatch, mixed_corpus):
        seen: list[str] = []
        lock = threading.Lock()
        real = runner.process_document

        def counting(document, data_dir, **kwargs):
            with lock:
                seen.append(document.document_id)
            return real(document, data_dir, **kwargs)

        monkeypatch.setattr(runner, "process_document", counting)
        runner.run(mixed_corpus, workers=4, **NO_SPACE_CHECK)
        assert sorted(seen) == sorted(set(seen))
        assert len(seen) == 4

    def test_workers_produce_one_journal_line_per_document(self, mixed_corpus):
        runner.run(mixed_corpus, workers=4, **NO_SPACE_CHECK)
        rows = journal_rows(mixed_corpus)
        assert len(rows) == len({row["document_id"] for row in rows}) == 4

    def test_a_second_runner_is_refused_while_one_holds_the_lock(self, mixed_corpus):
        lock = runner.RunLock(runner.lock_path(mixed_corpus))
        lock.acquire()
        try:
            with pytest.raises(RunLockError):
                runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        finally:
            lock.release()

    def test_the_lock_is_released_when_the_run_finishes(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert not runner.lock_path(mixed_corpus).exists()

    def test_the_lock_is_released_even_when_the_run_raises(self, monkeypatch,
                                                           mixed_corpus):
        def explode(*args, **kwargs):
            raise RuntimeError("planning blew up")

        monkeypatch.setattr(runner, "plan_run", explode)
        with pytest.raises(RuntimeError):
            runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert not runner.lock_path(mixed_corpus).exists()

    def test_a_stale_lock_can_be_broken_deliberately(self, mixed_corpus):
        runner.lock_path(mixed_corpus).write_text('{"pid": 1}', encoding="utf-8")
        report = runner.run(mixed_corpus, workers=2, force_unlock=True,
                            **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 4

    def test_a_run_lock_names_its_holder(self, tmp_path):
        lock = runner.RunLock(tmp_path / "run.lock")
        lock.acquire()
        try:
            assert str(json.loads(lock.describe())["pid"])
        finally:
            lock.release()


# --- Dry run -------------------------------------------------------------------------


class TestDryRun:
    def test_it_writes_nothing(self, mixed_corpus):
        before = sorted(p.name for p in mixed_corpus.iterdir())
        runner.dry_run(mixed_corpus)
        assert sorted(p.name for p in mixed_corpus.iterdir()) == before
        assert not runner.journal_path(mixed_corpus).exists()
        assert not runner.report_path(mixed_corpus).exists()
        assert not (mixed_corpus / "processed").exists()

    def test_it_reports_what_would_be_processed(self, mixed_corpus):
        plan, context = runner.dry_run(mixed_corpus)
        assert len(plan.todo) == 4
        assert plan.manifest_documents == 4
        assert plan.pdfs_on_disk == 4
        assert context["manifest_documents"] == 4

    def test_it_separates_documents_with_no_pdf(self, mixed_corpus):
        Corpus.load(mixed_corpus).get("act-b__handle-2").pdf_path.unlink()
        plan, _ = runner.dry_run(mixed_corpus)
        assert len(plan.todo) == 3
        assert [d.document_id for d in plan.missing_pdf] == ["act-b__handle-2"]

    def test_it_reports_already_processed_documents(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        plan, _ = runner.dry_run(mixed_corpus)
        assert len(plan.already_complete) == 3
        assert len(plan.todo) == 1

    def test_it_names_the_reason_each_document_is_queued(self, mixed_corpus):
        plan, _ = runner.dry_run(mixed_corpus)
        assert plan.reasons() == {"never processed": 4}

    def test_the_rendering_mentions_output_location_and_no_ocr(self, mixed_corpus):
        plan, context = runner.dry_run(mixed_corpus)
        text = runner.render_dry_run(
            plan, context, data_dir=mixed_corpus, workers=4, verify="standard",
            min_free_bytes=0)
        assert "processed" in text and "indiacode" in text
        assert "OCR will NOT run" in text
        assert "ESTIMATED WORKLOAD" in text

    def test_the_cli_dry_run_writes_nothing(self, mixed_corpus, capsys):
        code = runner.main(["--data-dir", str(mixed_corpus), "--dry-run"])
        assert code == 0
        assert "DRY RUN" in capsys.readouterr().out
        assert not runner.journal_path(mixed_corpus).exists()


# --- Selection: --limit and the filters -------------------------------------------


class TestLimit:
    def test_limit_bounds_the_work(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, limit=2, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 2
        assert len(journal_rows(mixed_corpus)) == 2

    def test_limit_is_deterministic(self, tmp_path):
        first = build_corpus(tmp_path / "a", [
            (f"doc-{i}__handle-{i}", {}) for i in range(6)])
        second = build_corpus(tmp_path / "b", [
            (f"doc-{i}__handle-{i}", {}) for i in reversed(range(6))])
        plan_a, _ = runner.dry_run(first, limit=3)
        plan_b, _ = runner.dry_run(second, limit=3)
        assert ([i.document.document_id for i in plan_a.todo]
                == [i.document.document_id for i in plan_b.todo])

    def test_limit_zero_processes_nothing(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, limit=0, **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 0

    def test_the_rest_stay_outstanding(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, limit=1, **NO_SPACE_CHECK)
        plan, _ = runner.dry_run(mixed_corpus)
        assert len(plan.todo) == 3


class TestFiltering:
    def test_category_filter(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, categories=["central_acts"],
                            **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 1
        assert list(statuses(mixed_corpus)) == ["act-a__handle-1"]

    def test_document_type_filter(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, document_types=["state_act"],
                            **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 1
        assert list(statuses(mixed_corpus)) == ["act-b__handle-2"]

    def test_only_filter(self, mixed_corpus):
        report = runner.run(mixed_corpus, workers=2, only=["scan-c__handle-3"],
                            **NO_SPACE_CHECK)
        assert report["this_run"]["attempted"] == 1
        assert list(statuses(mixed_corpus)) == ["scan-c__handle-3"]

    def test_filtered_documents_are_counted_not_hidden(self, mixed_corpus):
        plan, _ = runner.dry_run(mixed_corpus, categories=["central_acts"])
        assert plan.filtered_out == 3


# --- The report ---------------------------------------------------------------------


class TestReport:
    @pytest.fixture()
    def report(self, mixed_corpus):
        return runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)

    def test_it_is_written_where_the_docs_say(self, mixed_corpus, report):
        path = runner.report_path(mixed_corpus)
        assert path == mixed_corpus / config.PROCESSING_REPORT_FILENAME
        assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == (
            config.PROCESSING_REPORT_SCHEMA_VERSION)

    def test_it_holds_every_required_total(self, report):
        this_run = report["this_run"]
        for key in ("selected", "attempted", "successful", "successful_with_ocr",
                    "quarantined", "failed", "skipped", "pages_processed",
                    "duration_seconds", "documents_per_hour", "characters"):
            assert key in this_run
        assert report["corpus_totals"]["total_extracted_characters"] > 0

    def test_it_breaks_down_by_the_dimensions_that_matter(self, report):
        for key in ("documents_by_status", "documents_by_category",
                    "documents_by_ocr_action", "documents_by_quality_status",
                    "documents_by_language_status", "errors_by_type"):
            assert key in report
        assert report["documents_by_category"]["central_acts"] == 1

    def test_it_reports_the_corpus_context(self, report):
        assert report["corpus"]["manifest_documents"] == 4
        assert report["corpus"]["pdfs_on_disk"] == 4

    def test_it_is_rebuilt_from_the_journal_on_a_resumed_run(self, mixed_corpus):
        runner.run(mixed_corpus, workers=2, limit=2, **NO_SPACE_CHECK)
        second = runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert second["this_run"]["attempted"] == 2
        assert second["corpus_totals"]["documents_in_journal"] == 4
        assert sum(second["documents_by_status"].values()) == 4

    def test_it_never_claims_ocr_happened(self, report):
        assert report["corpus_totals"]["pages_ocr"] == 0
        assert report["this_run"]["successful_with_ocr"] == 0

    def test_the_summary_renders(self, report):
        text = runner.render_summary(report)
        assert "FULL-CORPUS PROCESSING" in text
        assert "eligible for indexing" in text


# --- Progress -----------------------------------------------------------------------


class TestProgress:
    def test_progress_shows_the_numbers_a_long_run_needs(self):
        stats = runner.RunStats(total=100)
        stats.record({"status": "SUCCESS", "pages": 10, "chars": 500})
        text = runner.render_progress(stats)
        for label in ("successful", "quarantined", "failed", "skipped",
                      "remaining", "pages", "elapsed", "rate", "ETA"):
            assert label in text

    def test_progress_is_reported_during_a_run(self, mixed_corpus):
        seen: list[int] = []
        corpus = Corpus.load(mixed_corpus)
        journal = runner.StatusJournal(runner.journal_path(mixed_corpus))
        plan = runner.plan_run(corpus, {})
        runner.execute(plan, mixed_corpus, journal, workers=2, progress_every=1,
                       on_progress=lambda stats: seen.append(stats.completed),
                       min_headroom_bytes=0)
        assert seen == [1, 2, 3, 4]

    def test_no_extracted_text_is_retained_in_memory(self, mixed_corpus):
        """The runner keeps counters, not documents."""
        stats = runner.RunStats(total=1)
        stats.record({"status": "SUCCESS", "pages": 3, "chars": 1000})
        assert not hasattr(stats, "results")
        assert stats.characters == 1000


# --- The raw corpus stays untouched --------------------------------------------------


class TestRawCorpusIsUntouched:
    def test_no_raw_pdf_is_modified(self, mixed_corpus):
        from ingestion.utils import sha256_file
        raw = mixed_corpus / "raw"
        before = {p: sha256_file(p) for p in sorted(raw.rglob("*.pdf"))}
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        after = {p: sha256_file(p) for p in sorted(raw.rglob("*.pdf"))}
        assert before == after

    def test_nothing_new_appears_under_data_raw(self, mixed_corpus):
        raw = mixed_corpus / "raw"
        before = sorted(p.relative_to(raw) for p in raw.rglob("*"))
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert sorted(p.relative_to(raw) for p in raw.rglob("*")) == before

    def test_the_manifest_is_not_rewritten(self, mixed_corpus):
        path = mixed_corpus / "manifest.json"
        before = path.read_bytes()
        runner.run(mixed_corpus, workers=2, **NO_SPACE_CHECK)
        assert path.read_bytes() == before


# --- Free space ------------------------------------------------------------------------


class TestFreeSpace:
    def test_it_refuses_to_start_without_room(self, mixed_corpus):
        with pytest.raises(runner.InsufficientSpaceError):
            runner.run(mixed_corpus, workers=1, min_free_bytes=10 ** 18)

    def test_a_run_that_runs_out_of_room_stops_rather_than_fills_the_disk(
            self, mixed_corpus):
        corpus = Corpus.load(mixed_corpus)
        journal = runner.StatusJournal(runner.journal_path(mixed_corpus))
        plan = runner.plan_run(corpus, {})
        stats = runner.execute(plan, mixed_corpus, journal, workers=1,
                               min_headroom_bytes=10 ** 18)
        assert stats.interrupted


# --- CLI --------------------------------------------------------------------------------


# --- A document that will not finish ------------------------------------------------


def slow_process_document(slow_ids, seconds, *, only_while=None):
    """A ``process_document`` that stalls on *slow_ids*.

    *only_while* is an event: while it is set the delay applies, so a test can
    make the first run overrun and the retry finish normally.
    """
    real = runner.process_document

    def process(document, data_dir, **kwargs):
        if document.document_id in slow_ids and (
                only_while is None or only_while.is_set()):
            time.sleep(seconds)
        return real(document, data_dir, **kwargs)

    return process


class TestDocumentTimeout:
    """One PDF must not be able to hold up an unattended overnight run.

    Python cannot kill a thread, so the runner abandons the document rather than
    stopping it. What is being tested is the part that matters: the run
    continues, the outcome is journalled, and the document stays retryable.
    """

    def test_a_document_that_overruns_is_recorded_failed(
            self, monkeypatch, mixed_corpus):
        monkeypatch.setattr(runner, "process_document",
                            slow_process_document({"act-b__handle-2"}, 1.0))
        report = runner.run(mixed_corpus, workers=4, document_timeout=0.2,
                            **NO_SPACE_CHECK)
        record = runner.StatusJournal(runner.journal_path(mixed_corpus)).load()[
            "act-b__handle-2"]
        assert record["status"] == "FAILED"
        assert record["error_type"] == "DocumentTimeout"
        assert record["error_stage"] == "timeout"
        assert "abandoned" in record["error_message"]
        assert report["this_run"]["timed_out"] == 1

    def test_the_rest_of_the_run_still_completes(self, monkeypatch, mixed_corpus):
        monkeypatch.setattr(runner, "process_document",
                            slow_process_document({"act-b__handle-2"}, 1.0))
        runner.run(mixed_corpus, workers=4, document_timeout=0.2, **NO_SPACE_CHECK)
        recorded = statuses(mixed_corpus)
        assert len(recorded) == 4
        assert recorded["act-a__handle-1"] == "SUCCESS"

    def test_a_timed_out_document_is_retried_on_the_next_run(
            self, monkeypatch, mixed_corpus):
        stall = threading.Event()
        stall.set()
        monkeypatch.setattr(
            runner, "process_document",
            slow_process_document({"act-b__handle-2"}, 1.0, only_while=stall))
        runner.run(mixed_corpus, workers=4, document_timeout=0.2, **NO_SPACE_CHECK)
        assert statuses(mixed_corpus)["act-b__handle-2"] == "FAILED"

        stall.clear()                                  # it behaves this time
        runner.run(mixed_corpus, workers=4, document_timeout=5.0, **NO_SPACE_CHECK)
        assert statuses(mixed_corpus)["act-b__handle-2"] != "FAILED"

    def test_a_queued_document_is_not_timed_out_for_waiting(
            self, monkeypatch, mixed_corpus):
        """The clock starts when a document starts, not when it is submitted.

        With one worker the last document sits in the queue for longer than the
        timeout before it ever runs. Timing from submission would abandon work
        that never had a chance.
        """
        monkeypatch.setattr(
            runner, "process_document",
            slow_process_document(
                {"act-a__handle-1", "act-b__handle-2", "scan-c__handle-3",
                 "broken-d__handle-4"}, 0.3))
        report = runner.run(mixed_corpus, workers=1, document_timeout=0.8,
                            **NO_SPACE_CHECK)
        assert report["this_run"]["timed_out"] == 0
        assert len(statuses(mixed_corpus)) == 4

    def test_zero_disables_the_timeout(self, monkeypatch, mixed_corpus):
        monkeypatch.setattr(runner, "process_document",
                            slow_process_document({"act-b__handle-2"}, 0.5))
        report = runner.run(mixed_corpus, workers=4, document_timeout=0,
                            **NO_SPACE_CHECK)
        assert report["this_run"]["timed_out"] == 0
        assert statuses(mixed_corpus)["act-b__handle-2"] != "FAILED"

    def test_timeout_record_carries_the_documents_provenance(self, mixed_corpus):
        corpus = Corpus.load(mixed_corpus)
        document = corpus.documents[0]
        record = runner.timeout_record(document, 1801.0, attempt=2)
        assert record["document_id"] == document.document_id
        assert record["sha256"] == document.sha256
        assert record["status"] == "FAILED"
        assert record["attempt"] == 2
        assert "1801" in record["error_message"]

    def test_the_default_is_generous_enough_for_the_largest_documents(self):
        # 18 corpus documents are over 100 MB and the largest is 300 MB; at the
        # benchmark's rate those need 450-1,350 s. A tighter default would
        # abandon real work and call it a failure.
        assert config.RUN_DOCUMENT_TIMEOUT_SECONDS >= 1500
        assert runner.build_parser().parse_args([]).document_timeout == (
            config.RUN_DOCUMENT_TIMEOUT_SECONDS)


# --- Running a subset that --limit cannot express -----------------------------------


class TestOnlyFrom:
    """``--limit N`` is deterministic but alphabetical. A representative subset
    has to be named, and a thousand names do not fit on a command line."""

    def test_ids_are_read_from_a_file(self, tmp_path, mixed_corpus):
        listing = tmp_path / "pilot.txt"
        listing.write_text("scan-c__handle-3\nact-a__handle-1\n", encoding="utf-8")
        assert runner.read_document_ids(listing) == [
            "scan-c__handle-3", "act-a__handle-1"]

    def test_comments_blanks_and_duplicates_are_ignored(self, tmp_path):
        listing = tmp_path / "pilot.txt"
        listing.write_text(
            "# the pilot sample\n"
            "act-a__handle-1\n"
            "\n"
            "act-a__handle-1\n"
            "scan-c__handle-3   # a scanned one\n",
            encoding="utf-8")
        assert runner.read_document_ids(listing) == [
            "act-a__handle-1", "scan-c__handle-3"]

    def test_a_missing_file_is_an_error_not_an_empty_run(self, tmp_path):
        with pytest.raises(ProcessingError):
            runner.read_document_ids(tmp_path / "nope.txt")

    def test_an_empty_file_is_an_error_not_an_empty_run(self, tmp_path):
        listing = tmp_path / "empty.txt"
        listing.write_text("# nothing but a comment\n", encoding="utf-8")
        with pytest.raises(ProcessingError):
            runner.read_document_ids(listing)

    def test_the_cli_restricts_the_run_to_the_listed_documents(
            self, tmp_path, mixed_corpus):
        listing = tmp_path / "pilot.txt"
        listing.write_text("act-a__handle-1\nscan-c__handle-3\n", encoding="utf-8")
        code = runner.main([
            "--data-dir", str(mixed_corpus), "--only-from", str(listing),
            "--workers", "2", "--min-free-gb", "0",
        ])
        assert code == 0
        assert sorted(statuses(mixed_corpus)) == [
            "act-a__handle-1", "scan-c__handle-3"]

    def test_it_combines_with_only(self, tmp_path, mixed_corpus):
        listing = tmp_path / "pilot.txt"
        listing.write_text("act-a__handle-1\n", encoding="utf-8")
        runner.main([
            "--data-dir", str(mixed_corpus), "--only-from", str(listing),
            "--only", "scan-c__handle-3", "--workers", "2", "--min-free-gb", "0",
        ])
        assert sorted(statuses(mixed_corpus)) == [
            "act-a__handle-1", "scan-c__handle-3"]

    def test_a_bad_file_exits_non_zero_without_running_anything(
            self, tmp_path, mixed_corpus):
        code = runner.main([
            "--data-dir", str(mixed_corpus),
            "--only-from", str(tmp_path / "nope.txt"), "--min-free-gb", "0",
        ])
        assert code == 2
        assert not runner.journal_path(mixed_corpus).exists()

    def test_a_dry_run_honours_it(self, tmp_path, mixed_corpus, capsys):
        listing = tmp_path / "pilot.txt"
        listing.write_text("act-a__handle-1\n", encoding="utf-8")
        code = runner.main([
            "--data-dir", str(mixed_corpus), "--only-from", str(listing),
            "--dry-run",
        ])
        assert code == 0
        assert not runner.journal_path(mixed_corpus).exists()
        assert "1" in capsys.readouterr().out


class TestCli:
    def test_the_default_worker_count_is_four(self):
        assert runner.build_parser().parse_args([]).workers == 4
        assert config.RUN_DEFAULT_WORKERS == 4

    def test_a_small_integration_run_works_end_to_end(self, mixed_corpus, capsys):
        code = runner.main([
            "--data-dir", str(mixed_corpus), "--limit", "2", "--workers", "2",
            "--min-free-gb", "0",
        ])
        assert code == 0
        assert "FULL-CORPUS PROCESSING" in capsys.readouterr().out
        assert len(journal_rows(mixed_corpus)) == 2

    def test_a_failure_is_reported_in_the_exit_code(self, mixed_corpus):
        code = runner.main([
            "--data-dir", str(mixed_corpus), "--workers", "2", "--min-free-gb", "0",
        ])
        assert code == 1                      # the corrupt PDF

    def test_a_held_lock_exits_non_zero_without_a_traceback(self, mixed_corpus):
        lock = runner.RunLock(runner.lock_path(mixed_corpus))
        lock.acquire()
        try:
            assert runner.main([
                "--data-dir", str(mixed_corpus), "--min-free-gb", "0",
            ]) == 2
        finally:
            lock.release()

    def test_unknown_categories_are_rejected(self):
        with pytest.raises(SystemExit):
            runner.build_parser().parse_args(["--category", "judgments"])
