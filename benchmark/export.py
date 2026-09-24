"""``python -m benchmark export --out DIR`` — the scoring kit, ready to copy.

The repository holds four things at once: the tools that downloaded India Code,
the tools that parsed the PDFs, the tools that built the corpus and wrote the
questions, and the benchmark itself. Only the last is of any use to someone
scoring a RAG system, and asking them to work out which files those are is how
a benchmark goes unused.

This writes a self-contained directory:

```
<out>/
  benchmark/            the scoring package — no PDF, ingestion or authoring code
    data/questions/     the 502 verified questions and their gold evidence
    data/splits/        the public dev/test assignment
  README.md             how to run it
  example_system.py     a runnable stub of the four-line system contract
```

Drop ``benchmark/`` into your project, point ``--corpus`` at the published
corpus, and score. The copied package imports nothing outside the standard
library, which is checked by ``tests/test_benchmark_portable.py``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from . import config
from .jsonio import dumps, write_atomic

#: The scoring half of the package. Every module here imports only these and the
#: standard library; the build, ingestion and authoring modules are left behind.
KIT_MODULES = ("__init__.py", "config.py", "jsonio.py", "terms.py", "duplicates.py",
               "evidence.py", "questions.py", "score.py", "splits.py", "predictions.py",
               "baselines/__init__.py", "baselines/oracle.py", "baselines/bm25.py")
#: Corpus files a scorer needs. ``structure/`` and the duplicate files are for
#: building and validating gold, not for scoring, and are not copied.
CORPUS_PARTS = ("text", "meta", config.DOCUMENTS_FILENAME, config.CHECKSUMS_FILENAME)

_MAIN = '''"""Entry point for the copied-out scoring kit: ``python -m benchmark score …``."""

import sys

from .score import main

if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "score":                 # accept the repository's spelling
        argv = argv[1:]
    sys.exit(main(argv))
'''

_EXAMPLE = '''"""A runnable stub of the system contract. Replace retrieve() with your pipeline.

    python example_system.py --corpus /path/to/corpus --split test
"""

import argparse
import json
from pathlib import Path

from benchmark import score, splits
from benchmark.questions import load_questions


class MySystem:
    name = "my-system"

    def retrieve(self, question: str, budget_chars: int):
        """Ranked (document_id, char_start, char_end), best first, within the budget.

        Returning [] is a real answer: it is how you abstain, and the
        unanswerable questions are where abstaining earns credit.
        """
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    questions = splits.select(load_questions(verified_only=True),
                              sorted(splits.SPLITS_DIR.glob("split-s*.json"))[0], args.split)
    run = score.run_system(MySystem(), questions, score.DEFAULT_BUDGETS)
    report = score.score_run(questions, run, system=MySystem.name,
                             corpus_dir=args.corpus, split=args.split)
    print(score.format_report(report))
    Path("report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Or write a predictions file and score it without importing this package:
    #   {"system": "my-system", "split": "test",
    #    "predictions": {"IN-STAT-0001": {"2000": [["doc-id", 0, 1500]], ...}}}
    #   python -m benchmark score --predictions run.json --corpus /path/to/corpus
'''


_KIT_README = '''# IndiaStatRAG — scoring kit

Everything needed to score a retrieval system against the IndiaStatRAG question
set. Copy `benchmark/` into your project; it needs Python 3.9+ and nothing else.

## What is here

| path | what it is |
| --- | --- |
| `benchmark/data/questions/` | the questions, gold answers and gold evidence (one JSON per question) |
| `benchmark/data/splits/` | the public dev/test assignment — a result must say which it used |
| `benchmark/score.py` | the scorer |
| `benchmark/baselines/` | oracle, bm25-windows, bm25-two-stage |
| `example_system.py` | a runnable stub of the system contract |

You also need the **published corpus** (the `text/` and `meta/` directories,
`documents.jsonl` and `CHECKSUMS.txt`). Pass it with `--corpus`.

## The contract

```python
def retrieve(self, question: str, budget_chars: int) -> list[tuple[str, int, int]]:
    """Ranked (document_id, char_start, char_end), best first. [] means abstain."""
```

Offsets are character offsets into `corpus/text/<document_id>.txt`. Chunk
however you like — the benchmark never asks about your chunks.

## Score a run

Write a predictions file and score it, importing nothing:

```json
{"system": "my-rag", "split": "test",
 "predictions": {"IN-STAT-0001": {"2000": [["act__handle-1", 120, 1500]],
                                  "8000": [], "32000": []}}}
```

```
python -m benchmark score --predictions run.json --corpus ./corpus --split test
```

Or call the scorer directly — see `example_system.py`.

## Reading the report

Recall and precision are always reported together, per category, at three
budgets. There is deliberately no single averaged number: the categories
measure different things. `oracle` is the ceiling; at B=2,000 the ceiling is
below 1.0 because a gold provision can be longer than the budget.
'''


def export_kit(out_dir: Path, *, corpus_dir: Optional[Path] = None,
               package_dir: Optional[Path] = None, questions_dir: Optional[Path] = None,
               splits_dir: Optional[Path] = None) -> dict:
    """Write the kit. With *corpus_dir*, the corpus files a scorer needs go too.

    Only ``verified`` questions are copied: a rejected draft is part of the
    repository's record of how the set was made, not something to score against.
    """
    out_dir = Path(out_dir)
    package_dir = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    target = out_dir / "benchmark"
    if target.exists():
        shutil.rmtree(target)
    (target / "baselines").mkdir(parents=True)

    for name in KIT_MODULES:
        shutil.copy2(package_dir / name, target / name)
    write_atomic(target / "__main__.py", _MAIN)

    questions_dir = Path(questions_dir) if questions_dir else package_dir / "data" / "questions"
    splits_dir = Path(splits_dir) if splits_dir else package_dir / "data" / "splits"
    (target / "data" / "questions").mkdir(parents=True)
    for path in sorted(questions_dir.glob("IN-STAT-*.json")):
        if json.loads(path.read_text(encoding="utf-8"))["provenance"]["status"] == "verified":
            shutil.copy2(path, target / "data" / "questions" / path.name)
    shutil.copytree(splits_dir, target / "data" / "splits")
    write_atomic(out_dir / "example_system.py", _EXAMPLE)
    write_atomic(out_dir / "README.md", _KIT_README)

    report = {
        "kit": str(target),
        "modules": len(KIT_MODULES) + 1,
        "questions": len(list((target / "data" / "questions").glob("IN-STAT-*.json"))),
        "splits": len(list((target / "data" / "splits").glob("*.json"))),
        "corpus": None,
    }
    if corpus_dir is not None:
        corpus_dir = Path(corpus_dir)
        destination = out_dir / "corpus"
        copied = []
        for part in CORPUS_PARTS:
            source = corpus_dir / part
            if not source.exists():
                raise FileNotFoundError(f"{source} is missing; build the corpus first")
            if source.is_dir():
                shutil.copytree(source, destination / part, dirs_exist_ok=True)
            else:
                destination.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination / part)
            copied.append(part)
        report["corpus"] = {"path": str(destination), "parts": copied}
    write_atomic(out_dir / "kit.json", dumps(report))
    return report
