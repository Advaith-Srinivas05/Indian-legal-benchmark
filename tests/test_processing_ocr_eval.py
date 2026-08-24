"""Tests for the OCR engine evaluation.

No real OCR engine is exercised here: engines are heavy, slow and downloaded at
first use, and none of that belongs in an offline suite. What is tested is
everything around them — that the pages are chosen deterministically and cover
the failure modes, that the measurements say what they claim, and that an engine
which cannot be loaded is *reported* rather than silently dropped, because "the
standard choice could not be installed" is a result.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from processing import ocr_eval, orientation
from processing.corpus import Corpus
from tests import corpusbuild

BODY = (
    "1. Short title and extent. This Act may be called the Sample Act, 1999 and "
    "it extends to the whole of India in every respect whatsoever.\n"
    "(1) The Central Government may, by notification in the Official Gazette, "
    "make rules for carrying out the purposes of this Act.\n"
    "(a) prescribing the form of an application; and\n"
    "(b) prescribing the fee payable therefor.\n"
)


#: One entry per failure mode the evaluation is supposed to cover. Three
#: documents of each, so a mode is not starved by an earlier mode taking its
#: only candidate.
MODES = [
    dict(pdf_type="vector_outlined", text_extraction_status="partial",
         extraction_quality="good", tables=0, year=1990),
    dict(pdf_type="scanned", text_extraction_status="requires_ocr",
         extraction_quality="questionable", tables=0, year=1930),
    dict(pdf_type="scanned", text_extraction_status="ok",
         extraction_quality="bad", tables=0, year=2015),
    dict(pdf_type="mixed", text_extraction_status="partial",
         extraction_quality="questionable", tables=0, year=1975),
    dict(pdf_type="text_based", text_extraction_status="ok",
         extraction_quality="good", tables=4, year=2020),
    dict(pdf_type="scanned", text_extraction_status="ok",
         extraction_quality="questionable", tables=0, year=1899),
    dict(pdf_type="scanned", text_extraction_status="ok",
         extraction_quality="questionable", tables=0, year=2018),
    dict(pdf_type="text_based", text_extraction_status="ok",
         extraction_quality="good", tables=0, year=1960),
]
DOCUMENT_COUNT = len(MODES) * 3


@pytest.fixture()
def corpus(tmp_path):
    entries = [
        corpusbuild.add_document(tmp_path, f"doc-{n}__handle-{n}",
                                 category="state_acts", year=1900 + n,
                                 pages=[BODY, BODY])
        for n in range(DOCUMENT_COUNT)
    ]
    corpusbuild.write_corpus(tmp_path, entries)
    return Corpus.load(tmp_path)


def _report(**overrides) -> dict:
    """A benchmark report shaped like the real one."""
    documents = []
    for index in range(DOCUMENT_COUNT):
        documents.append({
            "document_id": f"doc-{index}__handle-{index}",
            "title": f"Doc {index}",
            "category": "state_acts",
            "ok": True,
            "pages": 2,
            **MODES[index % len(MODES)],
        })
    return {"documents": documents, **overrides}


class TestPageSelection:
    def test_the_same_report_always_selects_the_same_pages(self, corpus):
        report = _report()
        first = ocr_eval.select_pages(report, corpus, count=6)
        second = ocr_eval.select_pages(report, corpus, count=6)
        assert [(p.document_id, p.page_number) for p in first] == \
            [(p.document_id, p.page_number) for p in second]

    def test_selection_does_not_depend_on_report_ordering(self, corpus):
        report = _report()
        reversed_report = {"documents": list(reversed(report["documents"]))}
        forward = {p.document_id for p in ocr_eval.select_pages(report, corpus, count=6)}
        backward = {p.document_id
                    for p in ocr_eval.select_pages(reversed_report, corpus, count=6)}
        assert forward == backward

    def test_the_failure_modes_are_covered(self, corpus):
        specs = ocr_eval.select_pages(_report(), corpus, count=8)
        reasons = {spec.reason for spec in specs}
        assert any("vector_outlined" in r for r in reasons)
        assert any("no text layer at all" in r for r in reasons)
        assert any("OCR layer is bad" in r for r in reasons)
        assert any("table" in r for r in reasons)

    def test_old_and_modern_documents_are_both_represented(self, corpus):
        specs = ocr_eval.select_pages(_report(), corpus, count=8)
        reasons = " ".join(spec.reason for spec in specs)
        assert "old law" in reasons
        assert "modern law" in reasons

    def test_no_page_is_selected_twice(self, corpus):
        specs = ocr_eval.select_pages(_report(), corpus, count=8)
        keys = [(s.document_id, s.page_number) for s in specs]
        assert len(keys) == len(set(keys))

    def test_a_page_from_the_middle_of_the_document_is_used(self, corpus):
        # The first page is a cover or a masthead across much of this corpus.
        rows = _report()["documents"]
        for row in rows:
            row["pages"] = 40
        assert all(spec.page_number > 1
                   for spec in ocr_eval.select_pages({"documents": rows},
                                                     corpus, count=4))

    def test_the_count_is_respected(self, corpus):
        assert len(ocr_eval.select_pages(_report(), corpus, count=3)) == 3

    def test_a_report_with_nothing_to_fix_selects_nothing_unusable(self, corpus):
        clean = {"documents": [{
            "document_id": "doc-0__handle-0", "title": "Doc 0",
            "category": "state_acts", "ok": True, "pages": 10,
            "pdf_type": "text_based", "text_extraction_status": "ok",
            "extraction_quality": "good", "tables": 0, "year": 2000,
        }]}
        assert ocr_eval.select_pages(clean, corpus, count=5) == []


class TestMeasurement:
    def test_clean_legal_text_measures_well(self):
        measured = ocr_eval.measure(BODY * 3)
        assert measured["words"] > 50
        assert measured["content_language"] == "en"
        assert measured["space_ratio"] > 0.1

    def test_section_and_clause_markers_are_counted(self):
        measured = ocr_eval.measure(
            "1. Short title.\n(1) first\n(2) second\n(a) alpha\n(b) beta\n")
        assert measured["section_markers"] == 1
        assert measured["subsection_markers"] == 2
        assert measured["clause_markers"] == 2

    def test_lost_spaces_are_visible_in_the_space_ratio(self):
        # The characteristic failure of one of the candidate recognisers:
        # "THEGAZETTEOFINDIAEXTRAORDINARY".
        run_together = "THEGAZETTEOFINDIAEXTRAORDINARY" * 10
        spaced = "THE GAZETTE OF INDIA EXTRAORDINARY " * 10
        assert ocr_eval.measure(run_together)["space_ratio"] < \
            ocr_eval.measure(spaced)["space_ratio"]

    def test_punctuation_preservation_is_measured(self):
        with_punctuation = "Section 3(1)(a), read with section 4; and section 5."
        without = "Section 3 1 a read with section 4 and section 5"
        assert ocr_eval.measure(with_punctuation)["punctuation_ratio"] > \
            ocr_eval.measure(without)["punctuation_ratio"]

    def test_empty_output_does_not_divide_by_zero(self):
        measured = ocr_eval.measure("")
        assert measured["characters"] == 0
        assert measured["space_ratio"] == 0.0
        assert measured["punctuation_ratio"] == 0.0


class FakeEngine(ocr_eval.Engine):
    pass


def _fake(name: str, text: str, *, fail: bool = False) -> ocr_eval.Engine:
    def loader():
        def run(image: bytes) -> str:
            if fail:
                raise RuntimeError("engine exploded")
            return text
        return run
    return ocr_eval.Engine(name, f"fake engine {name}", 200, loader)


class TestEngineAvailability:
    def test_an_engine_that_cannot_load_is_reported_not_dropped(self):
        def loader():
            raise ImportError("no such module")
        engines = ocr_eval.available_engines([
            ocr_eval.Engine("missing", "not installed", 200, loader)])
        assert len(engines) == 1
        assert not engines[0].available
        assert "no such module" in engines[0].unavailable_reason

    def test_an_engine_that_loads_is_available(self):
        engines = ocr_eval.available_engines([_fake("fake", "text")])
        assert engines[0].available

    def test_unavailable_engines_appear_in_the_summary(self):
        def loader():
            raise ImportError("needs a system install")
        engines = ocr_eval.available_engines([
            ocr_eval.Engine("tesseract@300", "Tesseract", 300, loader)])
        summary = ocr_eval.summarise({"pages": []}, engines)
        assert summary["unavailable_engines"][0]["name"] == "tesseract@300"
        assert "system install" in summary["unavailable_engines"][0]["reason"]

    def test_the_candidate_list_names_real_engines(self):
        names = {engine.name for engine in ocr_eval.candidate_engines()}
        assert {"rapidocr@200", "easyocr@200", "tesseract@300"} <= names


class TestEvaluationRun:
    @pytest.fixture()
    def specs(self, corpus):
        return ocr_eval.select_pages(_report(), corpus, count=2)

    def test_every_page_gets_a_baseline_and_every_engine(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", BODY * 2)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        assert len(result["pages"]) == 2
        for page in result["pages"]:
            assert "embedded" in page["engines"]
            assert "fake" in page["engines"]

    def test_the_baseline_is_the_existing_text_layer(self, corpus, specs):
        result = ocr_eval.evaluate(specs, [], corpus.data_dir)
        embedded = result["pages"][0]["engines"]["embedded"]
        assert "Short title" in embedded["text"]
        assert embedded["seconds"] == 0.0

    def test_an_engine_failure_is_recorded_not_raised(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("boom", "", fail=True)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        entry = result["pages"][0]["engines"]["boom"]
        assert entry["error"]
        assert not entry["page_aligned"]

    def test_the_summary_compares_engines(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", BODY * 2)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        summary = ocr_eval.summarise(evaluation, engines)["by_engine"]
        # Every engine is reported twice — as it read the page, and with the
        # word spacing repaired (see TestRespacedVariants).
        assert set(summary) == {
            "embedded", "embedded+respace", "fake", "fake+respace"}
        for stats in summary.values():
            assert stats["pages"] == 2
            assert "mean_seconds_per_page" in stats
            assert "section_markers" in stats

    def test_reports_are_written_without_the_full_page_text(self, corpus, specs, tmp_path):
        engines = ocr_eval.available_engines([_fake("fake", BODY * 2)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        payload = {
            "schema_version": 1, "generated_at": "now", "processor": "test",
            "evaluation": evaluation,
            "summary": ocr_eval.summarise(evaluation, engines),
        }
        json_path, md_path = ocr_eval.write_report(corpus.data_dir, payload)
        import json
        written = json.loads(json_path.read_text(encoding="utf-8"))
        entry = written["evaluation"]["pages"][0]["engines"]["fake"]
        assert "text" not in entry
        assert "text_preview" in entry
        assert "OCR engine evaluation" in md_path.read_text(encoding="utf-8")

    def test_samples_show_before_and_after(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", BODY * 2)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        payload = {
            "schema_version": 1, "generated_at": "now", "processor": "test",
            "evaluation": evaluation,
            "summary": ocr_eval.summarise(evaluation, engines),
        }
        directory = ocr_eval.write_samples(corpus.data_dir, payload)
        written = list(Path(directory).glob("*.md"))
        assert written
        text = written[0].read_text(encoding="utf-8")
        assert "## embedded" in text and "## fake" in text


class TestCli:
    def test_list_engines_reports_state(self, capsys):
        # Restricted to the engine that cannot load, so the test does not pull a
        # 100 MB model into an offline suite just to print a line.
        assert ocr_eval.main(["--list-engines", "--engine", "tesseract@300"]) == 0
        assert "tesseract@300" in capsys.readouterr().out

    def test_a_missing_benchmark_report_exits_non_zero(self, tmp_path):
        assert ocr_eval.main(["--data-dir", str(tmp_path), "--engine", "nope"]) == 2


GLUED = (
    "bysection65of theMunicipalActand thepowertoamend the assesment list "
    "conferred by sub-clause (l) of section 67 shallbeexercised by a "
    "sub-committee consisting of theExecutiveOfficerand the members "
    "appointedbythe committee for the purpose of this section.\n"
) * 3


class TestTesseractDiscovery:
    """Tesseract was reported unavailable on the first run because the Windows
    installer does not put it on ``PATH``.

    Reporting the standard document-OCR engine as unevaluable is a finding.
    Reporting it as unevaluable when it is in fact installed is a bug, and it
    cost the first benchmark its most important comparison.
    """

    def test_the_binary_is_looked_for_beyond_path(self, monkeypatch, tmp_path):
        binary = tmp_path / "tesseract.exe"
        binary.write_text("")
        monkeypatch.setattr(orientation.shutil, "which", lambda name: None)
        monkeypatch.setattr(orientation, "_TESSERACT_CANDIDATES", (str(binary),))
        assert orientation.tesseract_binary() == str(binary)

    def test_path_is_preferred_when_it_has_one(self, monkeypatch):
        monkeypatch.setattr(orientation.shutil, "which",
                            lambda name: "/usr/bin/tess")
        assert orientation.tesseract_binary() == "/usr/bin/tess"

    def test_a_genuinely_absent_binary_is_still_absent(self, monkeypatch):
        monkeypatch.setattr(orientation.shutil, "which", lambda name: None)
        monkeypatch.setattr(orientation, "_TESSERACT_CANDIDATES", ())
        assert orientation.tesseract_binary() is None

    def test_the_loader_refuses_rather_than_guesses(self, monkeypatch):
        monkeypatch.setattr(orientation, "tesseract_binary", lambda: None)
        with pytest.raises(Exception) as caught:
            ocr_eval._pytesseract_loader()
        assert "tesseract" in str(caught.value).lower()

    def test_both_dpi_configurations_are_candidates(self):
        names = {engine.name for engine in ocr_eval.candidate_engines()}
        assert {"tesseract@200", "tesseract@300"} <= names


class TestRespacedVariants:
    """Every engine is reported twice: as it read the page, and after
    :mod:`processing.respace` has put the dropped word boundaries back."""

    @pytest.fixture()
    def specs(self, corpus):
        return ocr_eval.select_pages(_report(), corpus, count=1)

    def test_each_engine_gains_a_repaired_twin(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        names = set(result["pages"][0]["engines"])
        assert {"fake", "fake+respace", "embedded", "embedded+respace"} <= names

    def test_the_repair_improves_the_glued_token_rate(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        page = result["pages"][0]["engines"]
        assert page["fake+respace"]["glued_token_rate"] < \
            page["fake"]["glued_token_rate"]
        assert page["fake+respace"]["space_ratio"] > page["fake"]["space_ratio"]

    def test_the_repair_costs_no_extra_ocr(self, corpus, specs):
        """It is text processing, not a second pass, so the time reported is the
        original engine's."""
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        page = result["pages"][0]["engines"]
        assert page["fake+respace"]["seconds"] == page["fake"]["seconds"]

    def test_what_the_repair_changed_is_recorded(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        repaired = result["pages"][0]["engines"]["fake+respace"]["respace"]
        assert repaired["spaces_inserted"] > 0
        assert repaired["examples"]

    def test_the_summary_reports_both_rows(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        summary = ocr_eval.summarise(evaluation, engines)["by_engine"]
        assert {"fake", "fake+respace"} <= set(summary)
        assert summary["fake+respace"]["spaces_inserted"] > 0
        assert summary["fake"]["spaces_inserted"] == 0

    def test_a_repaired_row_is_not_repaired_again(self, corpus, specs):
        engines = ocr_eval.available_engines([_fake("fake", GLUED)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        names = list(result["pages"][0]["engines"])
        assert not any(name.endswith("+respace+respace") for name in names)


class TestOrientationCorrection:
    """A page is turned the right way up before an engine reads it.

    OCR does not fail loudly on a sideways page: it returns confident text that
    is wrong, which is the most expensive failure this pipeline can produce.
    """

    @pytest.fixture()
    def specs(self, corpus):
        return ocr_eval.select_pages(_report(), corpus, count=1)

    @staticmethod
    def _detection(monkeypatch, degrees, confidence=15.0,
                   source="tesseract_osd"):
        turned = []
        monkeypatch.setattr(
            orientation, "detect_image_rotation",
            lambda image: orientation.ImageOrientation(
                rotate_degrees=degrees, confidence=confidence, source=source))

        def rotate(image, degrees_applied):
            turned.append(degrees_applied)
            return image

        monkeypatch.setattr(orientation, "rotate_image", rotate)
        return turned

    def test_no_rotation_is_applied_when_none_could_be_detected(
        self, corpus, specs, monkeypatch
    ):
        self._detection(monkeypatch, 0, source="unavailable")
        engines = ocr_eval.available_engines([_fake("fake", BODY)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        assert result["pages"][0]["engines"]["fake"]["rotation_applied"] is None

    def test_a_detected_rotation_is_applied_and_recorded(
        self, corpus, specs, monkeypatch
    ):
        turned = self._detection(monkeypatch, 90)
        engines = ocr_eval.available_engines([_fake("fake", BODY)])
        result = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        page = result["pages"][0]
        assert page["engines"]["fake"]["rotation_applied"] == 90
        assert turned == [90]
        assert page["orientation"][200]["rotate_degrees"] == 90

    def test_rotated_pages_are_summarised(self, corpus, specs, monkeypatch):
        self._detection(monkeypatch, 270)
        engines = ocr_eval.available_engines([_fake("fake", BODY)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        summary = ocr_eval.summarise(evaluation, engines)["orientation"]
        assert summary["pages_rotated_before_ocr"] == 1

    def test_an_upright_page_is_not_counted_as_rotated(
        self, corpus, specs, monkeypatch
    ):
        turned = self._detection(monkeypatch, 0)
        engines = ocr_eval.available_engines([_fake("fake", BODY)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        summary = ocr_eval.summarise(evaluation, engines)["orientation"]
        assert summary["pages_rotated_before_ocr"] == 0
        assert turned == []

    def test_a_low_confidence_detection_does_not_rotate(
        self, corpus, specs, monkeypatch
    ):
        turned = self._detection(monkeypatch, 90, confidence=0.1)
        engines = ocr_eval.available_engines([_fake("fake", BODY)])
        evaluation = ocr_eval.evaluate(specs, engines, corpus.data_dir)
        assert turned == []
        summary = ocr_eval.summarise(evaluation, engines)["orientation"]
        assert summary["pages_rotated_before_ocr"] == 1     # detected, not applied
