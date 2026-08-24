"""Tests for the benchmark run, its metrics and its report.

The benchmark is the deliverable this phase is judged on, so what is tested here
is mostly that it cannot flatter itself: failures are counted as failures, a
scanned document is reported as scanned, and every metric the project spec asks for is
actually present in the report.
"""

from __future__ import annotations

import json

import pytest

from processing import benchmark, config
from tests import corpusbuild

SECTIONED = (
    "THE SAMPLE ACT, 1999\n"
    "An Act to provide for the regulation of samples.\n"
    "CHAPTER I\n"
    "PRELIMINARY\n"
    "1. Short title and extent. This Act may be called the Sample Act, 1999 and "
    "extends to the whole of India in every respect.\n"
    "(1) It shall come into force at once in the territories to which it extends.\n"
    "(2) It applies to every person resident within those territories.\n"
    "2. Definitions. In this Act, unless the context otherwise requires, the "
    "expressions used shall have the meanings assigned to them by this section.\n"
    "3. Power to make rules. The Central Government may, by notification in the "
    "Official Gazette, make rules for carrying out the purposes of this Act.\n"
)

# A second page that continues the numbering, as a real document's would.
CONTINUED = (
    "CHAPTER II\n"
    "PENALTIES\n"
    "4. Penalty. Whoever contravenes any provision of this Act shall be punished "
    "with imprisonment which may extend to one year, or with fine, or with both.\n"
    "5. Repeal and savings. The corresponding law in force immediately before the "
    "commencement of this Act is hereby repealed to the extent stated below.\n"
    "6. Power to remove difficulties. If any difficulty arises in giving effect "
    "to the provisions of this Act, the Central Government may make an order.\n"
)


@pytest.fixture()
def data_dir(tmp_path):
    entries = [
        corpusbuild.add_document(tmp_path, f"central-{n}__handle-{n}",
                                 category="central_acts", year=1950 + n,
                                 pages=[SECTIONED, CONTINUED])
        for n in range(4)
    ]
    entries += [
        corpusbuild.add_document(tmp_path, f"state-{n}__handle-1{n}",
                                 category="state_acts", year=1890 + n * 40,
                                 jurisdiction=["Assam", "Kerala", "Goa", "Bihar"][n],
                                 pages=[SECTIONED])
        for n in range(4)
    ]
    entries += [
        corpusbuild.add_document(tmp_path, "rule-scanned__rule-1",
                                 category="rules", kind="scanned", year=1967),
        corpusbuild.add_document(tmp_path, "rule-mixed__rule-2",
                                 category="rules", kind="mixed",
                                 pages=[SECTIONED, CONTINUED], year=1985),
        corpusbuild.add_document(tmp_path, "rule-broken__rule-3",
                                 category="rules", kind="corrupt", year=2001),
        corpusbuild.add_document(tmp_path, "regulation-blank__regulation-1",
                                 category="regulations", kind="blank", year=2015),
        corpusbuild.add_document(tmp_path, "regulation-ok__regulation-2",
                                 category="regulations", pages=[SECTIONED],
                                 year=2020),
    ]
    corpusbuild.write_corpus(tmp_path, entries)
    return tmp_path


@pytest.fixture()
def report(data_dir):
    _, _, payload = benchmark.run(data_dir, sample_size=13, workers=2)
    return payload


class TestReportCompleteness:
    """Every metric the project spec names must actually be in the report."""

    REQUIRED_TOTALS = (
        "documents_selected", "documents_processed", "successful_extraction",
        "extraction_failures", "pages_extracted", "average_pages_per_document",
        "average_characters_per_document", "empty_pages",
    )
    REQUIRED_STRUCTURE = (
        "documents_with_sections", "documents_with_chapters",
        "documents_with_subsections",
    )

    def test_totals_are_reported(self, report):
        for key in self.REQUIRED_TOTALS:
            assert key in report["totals"], key

    def test_structure_detection_rates_are_reported(self, report):
        for key in self.REQUIRED_STRUCTURE:
            assert key in report["structure"], key

    def test_pdf_types_and_statuses_are_reported(self, report):
        assert set(report["pdf_types"]) <= set(config.PDF_TYPES)
        assert set(report["text_extraction_status"]) <= \
            set(config.TEXT_EXTRACTION_STATUSES)

    def test_tables_and_problems_are_reported(self, report):
        assert "documents_with_tables" in report["tables"]
        assert "documents_with_problems" in report["problems"]

    def test_thresholds_used_are_recorded(self, report):
        assert report["thresholds"]["min_page_alpha_chars"] == \
            config.MIN_PAGE_ALPHA_CHARS

    def test_the_threshold_snapshot_can_be_taken_at_all(self):
        """It reads every name off :mod:`processing.config` by hand.

        Renaming a threshold therefore breaks the snapshot and nothing else —
        and because it is taken at the very end of ``build_report``, the break
        surfaces only after a hundred documents have been processed. It did
        exactly that once. This is a one-line guard against a nine-minute
        failure.
        """
        snapshot = benchmark._threshold_snapshot()
        assert snapshot
        assert all(value is not None for value in snapshot.values())

    def test_the_orientation_and_language_thresholds_are_recorded(self, report):
        for key in ("orientation_sideways_line_ratio",
                    "orientation_suspect_page_ratio",
                    "language_page_min_letters",
                    "language_mixed_non_en_page_ratio"):
            assert key in report["thresholds"], key

    def test_every_example_slot_is_addressed(self, report):
        for slot, _ in benchmark.EXAMPLE_SLOTS:
            assert slot in report["examples"]


class TestHonestCounting:
    def test_a_broken_pdf_counts_as_a_failure(self, report):
        assert report["totals"]["extraction_failures"] >= 1
        failures = [r for r in report["documents"] if not r["ok"]]
        assert any(r["document_id"] == "rule-broken__rule-3" for r in failures)

    def test_processed_equals_successes_plus_failures(self, report):
        totals = report["totals"]
        assert totals["documents_processed"] == (
            totals["successful_extraction"] + totals["extraction_failures"]
        )

    def test_a_scanned_document_is_reported_as_scanned(self, report):
        row = next(r for r in report["documents"]
                   if r["document_id"] == "rule-scanned__rule-1")
        assert row["pdf_type"] == "scanned"
        assert row["text_extraction_status"] == "requires_ocr"

    def test_a_mixed_document_is_reported_as_mixed(self, report):
        row = next(r for r in report["documents"]
                   if r["document_id"] == "rule-mixed__rule-2")
        assert row["pdf_type"] == "mixed"

    def test_blank_pages_are_counted(self, report):
        assert report["totals"]["empty_pages"] >= 2

    def test_documents_needing_ocr_are_flagged_as_problems(self, report):
        flags = report["problems"]["by_flag"]
        assert flags.get("text_extraction_requires_ocr", 0) >= 1
        assert flags.get("extraction_failed:PDFOpenError", 0) >= 1

    def test_structure_is_detected_where_it_exists(self, report):
        assert report["structure"]["documents_with_sections"] >= 5
        assert report["structure"]["documents_with_chapters"] >= 5


class TestProblemFlags:
    def test_a_clean_document_has_no_flags(self, data_dir):
        _, results, _ = benchmark.run(data_dir, sample_size=13, workers=1)
        clean = next(r for r in results
                     if r.document.document_id.startswith("central-"))
        assert benchmark.problem_flags(clean) == []

    def test_a_failure_flag_names_the_error(self, data_dir):
        _, results, _ = benchmark.run(data_dir, sample_size=13, workers=1)
        broken = next(r for r in results
                      if r.document.document_id == "rule-broken__rule-3")
        assert benchmark.problem_flags(broken) == ["extraction_failed:PDFOpenError"]


class TestArtifacts:
    def test_sample_file_records_the_exact_sample_and_method(self, data_dir):
        records, _, _ = benchmark.run(data_dir, sample_size=13, workers=1)
        path = benchmark.write_sample(data_dir, records)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert len(payload["documents"]) == len(records)
        assert "sha256(document_id)" in payload["method"]
        assert payload["coverage"]["by_category"]

    def test_report_files_are_written(self, data_dir, report):
        json_path, md_path = benchmark.write_report(data_dir, report)
        assert json.loads(json_path.read_text(encoding="utf-8"))["totals"]
        text = md_path.read_text(encoding="utf-8")
        assert "PDF extraction benchmark" in text
        assert "Documents with unusual extraction problems" in text

    def test_examples_are_written_for_the_slots_that_have_a_document(self, data_dir):
        _, results, payload = benchmark.run(data_dir, sample_size=13, workers=1)
        directory = benchmark.write_examples(data_dir, payload, results)
        written = {p.stem for p in directory.glob("*.md")}
        assert "central_act" in written
        assert "scanned" in written

    def test_an_example_shows_real_extracted_text(self, data_dir):
        _, results, payload = benchmark.run(data_dir, sample_size=13, workers=1)
        directory = benchmark.write_examples(data_dir, payload, results)
        text = (directory / "central_act.md").read_text(encoding="utf-8")
        assert "Raw extracted text" in text
        assert "Short title and extent" in text

    def test_per_document_output_is_written_for_the_sample(self, data_dir, report):
        processed = data_dir / config.PROCESSED_SUBDIR
        written = {p.name for p in processed.iterdir()}
        assert "central-0__handle-0" in written
        # The failure wrote nothing.
        assert "rule-broken__rule-3" not in written


class TestCli:
    def test_dry_run_selects_without_processing(self, data_dir, capsys):
        code = benchmark.main(["--data-dir", str(data_dir), "--sample-size", "5",
                               "--dry-run", "-q"])
        assert code == 0
        assert (benchmark.benchmark_dir(data_dir) / config.SAMPLE_FILENAME).exists()
        assert not (data_dir / config.PROCESSED_SUBDIR).exists()

    def test_full_run_writes_everything_and_summarises(self, data_dir, capsys):
        code = benchmark.main(["--data-dir", str(data_dir), "--sample-size", "13",
                               "--workers", "2", "-q"])
        assert code == 0
        out = capsys.readouterr().out
        assert "PDF EXTRACTION BENCHMARK" in out
        directory = benchmark.benchmark_dir(data_dir)
        assert (directory / config.REPORT_JSON_FILENAME).exists()
        assert (directory / config.REPORT_MARKDOWN_FILENAME).exists()
        assert (directory / config.EXAMPLES_DIRNAME).is_dir()

    def test_a_missing_manifest_exits_non_zero(self, tmp_path):
        assert benchmark.main(["--data-dir", str(tmp_path), "-q"]) == 2

    def test_limit_truncates_the_run(self, data_dir):
        _, results, payload = benchmark.run(data_dir, sample_size=13, limit=3)
        assert len(results) == 3
        assert payload["totals"]["documents_processed"] == 3
