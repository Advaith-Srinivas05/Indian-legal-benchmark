"""The scoring half must be copyable: a kit that only works inside this
repository is a benchmark nobody else can run.

These tests exist because the scorer used to drag in ``processing`` (through
``config``) and the corpus builder (through ``splits``), so copying the files
that matter produced an ImportError rather than a score.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmark import config, predictions, score, splits
from benchmark.baselines.oracle import Oracle
from benchmark.corpus import build_corpus
from benchmark.export import KIT_MODULES, export_kit
from benchmark.questions import Corpus, location
from tests.processedbuild import make_page, write_processed

ACT = """THE LICENSING AUTHORITY ACT, 2001
1. Short title.—This Act may be called the Licensing Authority Act, 2001, and it applies to every district.
2. Definitions.—In this Act "authority" means the licensing authority appointed under section 3 for a district.
3. Authority.—The Government shall appoint a licensing authority for each district, who shall decide every application within thirty days of its receipt."""


@pytest.fixture
def kit(tmp_path):
    """A miniature corpus, two questions, a split — exported as a kit."""
    data = tmp_path / "data"
    write_processed(data, "licensing-act__handle-1", [make_page(1, ACT)],
                    title="The Licensing Authority Act, 2001")
    corpus_dir = data / "corpus"
    build_corpus(data, corpus_dir)
    corpus = Corpus(corpus_dir)

    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()
    made = []
    for i, (key, status) in enumerate([("section:2", "verified"), ("section:3", "verified"),
                                       ("section:1", "rejected")], start=1):
        q = {
            "question_id": f"IN-STAT-{i:04d}", "question": f"question {i}",
            "category": "situational", "answer_type": "abstractive", "jurisdiction_hint": None,
            "gold_evidence": [{"group_id": "g1", "requirement": "sufficient",
                               "locations": [location(corpus, "licensing-act__handle-1", key, "sampled")]}],
            "proposed_alternatives": [], "gold_answer": "an answer", "required_facts": [],
            "must_cite": [], "unanswerable": False, "difficulty": {},
            "provenance": {"status": status},
        }
        (questions_dir / f"{q['question_id']}.json").write_text(json.dumps(q), encoding="utf-8")
        made.append(q)

    splits_dir = tmp_path / "splits"
    splits.write_split([q for q in made if q["provenance"]["status"] == "verified"],
                       splits_dir / "split-s1.json", seed=1, corpus=corpus)

    out = tmp_path / "kit"
    report = export_kit(out, questions_dir=questions_dir, splits_dir=splits_dir)
    return {"out": out, "report": report, "questions": made, "corpus": corpus_dir, "tmp": tmp_path}


def run_in_kit(kit, *args):
    """Run the kit's own CLI with the repository nowhere on the path."""
    env = {"PATH": "", "SYSTEMROOT": "C:\\Windows", "PYTHONIOENCODING": "utf-8", "PYTHONPATH": ""}
    return subprocess.run([sys.executable, "-m", "benchmark", *args], cwd=kit["out"],
                          capture_output=True, text=True, env=env)


def test_the_scoring_modules_import_without_processing_or_ingestion(kit):
    code = ("import sys;"
            "import benchmark.score, benchmark.splits, benchmark.predictions, benchmark.baselines;"
            "print(sorted(m for m in sys.modules if m.split('.')[0] in ('processing', 'ingestion')))")
    done = subprocess.run([sys.executable, "-c", code], cwd=kit["out"], capture_output=True,
                          text=True, env={"PATH": "", "SYSTEMROOT": "C:\\Windows", "PYTHONPATH": ""})
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"


def test_the_config_fallbacks_match_the_processing_values():
    """The copied package hardcodes three constants; they must not drift."""
    from processing import config as processing_config
    assert config.PROCESSED_SUBDIR == processing_config.PROCESSED_SUBDIR
    assert config.DOCUMENT_FILENAME == processing_config.DOCUMENT_FILENAME
    assert config.PAGES_FILENAME == processing_config.PAGES_FILENAME


def test_the_kit_leaves_the_build_and_authoring_code_behind(kit):
    present = {p.name for p in (kit["out"] / "benchmark").glob("*.py")}
    assert present == {Path(m).name for m in KIT_MODULES if "/" not in m} | {"__main__.py"}
    for absent in ("corpus.py", "source.py", "canonical.py", "provenance.py",
                   "authoring.py", "sample.py", "export.py"):
        assert not (kit["out"] / "benchmark" / absent).exists(), absent


def test_the_kit_ships_only_verified_questions(kit):
    shipped = sorted(p.stem for p in (kit["out"] / "benchmark" / "data" / "questions").glob("*.json"))
    assert shipped == ["IN-STAT-0001", "IN-STAT-0002"]
    assert kit["report"]["questions"] == 2


def test_the_kit_scores_a_predictions_file_on_its_own(kit):
    verified = [q for q in kit["questions"] if q["provenance"]["status"] == "verified"]
    run = {"system": "stub", "split": "all",
           "predictions": {q["question_id"]: {"2000": [], "8000": [], "32000": []} for q in verified}}
    (kit["out"] / "run.json").write_text(json.dumps(run), encoding="utf-8")
    done = run_in_kit(kit, "score", "--predictions", "run.json",
                      "--corpus", str(kit["corpus"]), "--split", "all")
    assert done.returncode == 0, done.stderr
    assert "system: stub" in done.stdout
    assert "situational" in done.stdout


def test_the_kit_ships_a_readme_and_a_runnable_example(kit):
    assert (kit["out"] / "README.md").read_text(encoding="utf-8").startswith("# IndiaStatRAG")
    example = (kit["out"] / "example_system.py").read_text(encoding="utf-8")
    compile(example, "example_system.py", "exec")           # it must at least parse
    assert "def retrieve" in example


# --- The predictions file ------------------------------------------------------------


def _questions(kit):
    return [q for q in kit["questions"] if q["provenance"]["status"] == "verified"]


def test_a_predictions_file_scores_the_same_as_running_the_system(kit):
    questions = _questions(kit)
    direct = score.run_system(Oracle(questions), questions, (8000,))
    payload = {"system": "oracle", "predictions": {
        qid: {"8000": [list(s) for s in spans[8000]]} for qid, spans in direct.items()}}
    path = kit["tmp"] / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = predictions.load(path, questions, (8000,))
    a = score.score_run(questions, direct, system="oracle", corpus_dir=kit["corpus"], budgets=(8000,))
    b = score.score_run(questions, loaded["run"], system="oracle", corpus_dir=kit["corpus"], budgets=(8000,))
    assert a["retrieval"] == b["retrieval"]
    assert a["retrieval"]["by_category"]["situational"]["8000"]["evidence_recall"] == 1.0


def test_a_prediction_for_an_unknown_question_is_refused(kit):
    path = kit["tmp"] / "bad.json"
    path.write_text(json.dumps({"predictions": {"IN-STAT-9999": {"8000": []}}}), encoding="utf-8")
    with pytest.raises(predictions.PredictionsError, match="not in the split"):
        predictions.load(path, _questions(kit), (8000,))


def test_a_backwards_span_is_refused(kit):
    q = _questions(kit)[0]["question_id"]
    path = kit["tmp"] / "bad.json"
    path.write_text(json.dumps({"predictions": {q: {"8000": [["doc", 500, 100]]}}}), encoding="utf-8")
    with pytest.raises(predictions.PredictionsError, match="not a forward span"):
        predictions.load(path, _questions(kit), (8000,))


def test_a_missing_budget_says_what_to_do(kit):
    q = _questions(kit)[0]["question_id"]
    path = kit["tmp"] / "bad.json"
    path.write_text(json.dumps({"predictions": {q: {"8000": []}}}), encoding="utf-8")
    with pytest.raises(predictions.PredictionsError, match="--budgets"):
        predictions.load(path, _questions(kit), (2000, 8000))


def test_a_question_with_no_prediction_is_an_abstention_and_is_reported(kit):
    questions = _questions(kit)
    path = kit["tmp"] / "partial.json"
    path.write_text(json.dumps({"predictions": {questions[0]["question_id"]: {"8000": []}}}),
                    encoding="utf-8")
    loaded = predictions.load(path, questions, (8000,))
    assert loaded["missing"] == [questions[1]["question_id"]]
    report = score.score_run(questions, loaded["run"], system="partial",
                             corpus_dir=kit["corpus"], budgets=(8000,))
    assert report["abstention"]["false_abstention"] == 1.0
