# IndiaStatRAG

A benchmark for Retrieval-Augmented Generation over Indian statutory law:
15,325 Acts, rules and regulations from India Code, 502 verified questions, and
ground-truth evidence that says *where in the law the answer is*.

Ground truth is a character span in published canonical text plus a legal
citation — never a chunk id. That is the whole point: chunk the corpus however
you like, with whatever embedder, index or reranker you like, and your spans can
still be scored against the same answer key. The benchmark prescribes no
chunking strategy, no embedding model, no vector database, no retriever and no
reranker.

---

## Quick start

### 1. Get the scoring kit into your project

```
python -m benchmark export --out ../my-rag-project
```

That writes a self-contained folder — the scorer, the questions, the split, a
README and a runnable example — with no PDF, ingestion or authoring code in it.
Copy the `benchmark/` folder it produces wherever you like; it imports nothing
outside the Python standard library.

Add `--with-corpus` to copy the corpus files a scorer needs (~0.8 GB).

### 2. Implement one method

```python
class MySystem:
    name = "my-rag"

    def retrieve(self, question: str, budget_chars: int) -> list[tuple[str, int, int]]:
        """Ranked (document_id, char_start, char_end), best first, within the budget.

        Offsets index corpus/text/<document_id>.txt.
        Returning [] is a real answer — it is how you abstain.
        """
```

### 3. Score it

Either call the scorer directly:

```python
from benchmark import score, splits
from benchmark.questions import load_questions

questions = splits.select(load_questions(verified_only=True),
                          "benchmark/data/splits/split-s20260923.json", "test")
run = score.run_system(MySystem(), questions, score.DEFAULT_BUDGETS)
report = score.score_run(questions, run, system="my-rag",
                         corpus_dir="corpus", split="test")
print(score.format_report(report))
```

…or write a **predictions file** and import nothing at all:

```json
{"system": "my-rag", "split": "test",
 "predictions": {
   "IN-STAT-0001": {"2000":  [["right-to-information-act-2005__handle-1400", 4120, 5600]],
                    "8000":  [["right-to-information-act-2005__handle-1400", 4120, 5600]],
                    "32000": []}
 }}
```

```
python -m benchmark score --predictions run.json --corpus ./corpus --split test --report report.json
```

Any language can produce that file. Unknown question ids, backwards spans and
missing budgets are refused with the reason; a question with no prediction is
scored as an abstention and reported, so a half-finished run cannot look like a
modest one.

---

## Which files *are* the benchmark

### You need these to score a system

| path | size | what it is |
| --- | ---: | --- |
| `benchmark/data/questions/` | 2.3 MB | the questions: text, gold answer, gold evidence spans, required facts, citations. **The irreplaceable artefact** |
| `benchmark/data/splits/split-s20260923.json` | 16 KB | public dev/test assignment (dev 150, test 352) |
| `benchmark/score.py` | — | the scorer |
| `benchmark/splits.py`, `predictions.py`, `questions.py`, `terms.py`, `evidence.py`, `duplicates.py`, `config.py`, `jsonio.py` | — | what the scorer imports |
| `benchmark/baselines/` | — | oracle, bm25-windows, bm25-two-stage |

`python -m benchmark export` copies exactly this list.

### And these, from the corpus

| path | size | needed for |
| --- | ---: | --- |
| `corpus/text/` | 723 MB | **retrieval** — the canonical text gold offsets point into |
| `corpus/meta/` | 92 MB | titles, jurisdiction, the path to each text file |
| `corpus/documents.jsonl` | 6.8 MB | the document list |
| `corpus/CHECKSUMS.txt` | 6.3 MB | the hash a published result must quote |

Not needed for scoring: `corpus/structure/` (545 MB), `duplicates.json`,
`provision_equivalents.jsonl` — those build and validate gold. Scoring itself
needs no corpus at all beyond the two fingerprint files; the gold spans are in
the question files.

### Not the benchmark

`ingestion/` downloaded India Code. `processing/` parsed the PDFs.
`benchmark/corpus.py`, `source.py`, `canonical.py`, `provenance.py`,
`sample.py`, `authoring.py` built the corpus and wrote the questions. None of it
is needed to run the benchmark, and `export` leaves all of it behind.

---

## What a question looks like

```jsonc
{
  "question_id": "IN-STAT-0456",
  "question": "Before a state converts forest land to non-forest use, whose approval is needed…",
  "category": "cross_reference",          // and situational, definitional, numeric_threshold,
  "answer_type": "abstractive",           //     jurisdictional, provision_lookup, unanswerable
  "gold_evidence": [                       // groups; hitting ANY location in a group hits it
    {"group_id": "g1", "requirement": "required",
     "locations": [{"document_id": "van-sanrakshan-evam-samvardhan-adhiniyam-1980__handle-1760",
                    "key": "section:2", "citation": "Van (Sanrakshan…) Adhiniyam, 1980, s. 2",
                    "char_start": 4832, "char_end": 7792, "page_start": 3, "page_end": 4,
                    "text_sha256": "09538f…", "evidence_confidence": "high"}]}
  ],
  "gold_answer": "The prior approval of the Central Government, which may constitute…",
  "required_facts": [["prior approval of the Central Government"], ["Advisory Committee"]],
  "must_cite": [{"document_id": "van-…__handle-1760", "key": "section:2"}],
  "unanswerable": false,
  "provenance": {"status": "verified", "drafting_method": "language_model", …}
}
```

A **group** holds interchangeable locations — the same Act under two handles, a
state's verbatim re-enactment — because the corpus really does contain the same
Act many times, and a system that finds the right law in the wrong copy must not
score zero. `requirement` is `sufficient` (one group answers it) or `required`
(a multi-hop question: every group must be hit).

The 50 **unanswerable** questions have no gold evidence. Returning nothing is
the correct answer.

---

## How scoring works

- **The budget, not top-*k*.** A system is asked for its best spans within *B*
  characters, at B = 2,000 / 8,000 / 32,000. Overlapping spans are merged before
  the budget is counted, so padding with near-duplicates buys nothing.
- **A hit** is coverage ≥ τ of a gold location's characters (τ = 0.5 primary;
  0.8 and 1.0 also reported). Coverage is measured over the union of what you
  returned for that document, so a provision split across two of your chunks
  still counts.
- **Reported per category, always in pairs:** evidence recall, evidence
  precision, all-groups (multi-hop), document recall, MRR. There is deliberately
  **no single averaged number** — the categories measure different things.
- **Abstention** is scored only on the unanswerable questions, and always beside
  false abstention on answerable ones. A system that abstains on everything
  scores 1.0 and 1.0, which is visibly useless.
- **Answer metrics** (optional, if you submit generated answers): fact coverage,
  citation precision/recall — right Act with wrong section is wrong — and quote
  grounding.

### The ceiling is not always 1.0

The `oracle` baseline returns the gold spans. It scores 1.0 recall at B = 8,000
and B = 32,000; if it ever does not, the scorer is broken, not your system. At
B = 2,000 it scores 0.83–1.00, because a gold provision can be 6,000 characters
and half of it does not fit in a 2,000-character budget. **Read a B = 2,000
score against the oracle's, not against 1.0.**

## Baselines to beat (test split, B = 8,000)

| category | oracle | bm25-windows | bm25-two-stage |
| --- | ---: | ---: | ---: |
| provision_lookup | 1.000 | 0.491 | **0.528** |
| definitional | 1.000 | **0.382** | 0.294 |
| numeric_threshold | 1.000 | **0.417** | 0.250 |
| cross_reference | 1.000 | **0.235** | 0.196 |
| jurisdictional | 1.000 | 0.100 | **0.120** |
| situational | 1.000 | **0.075** | 0.065 |

Evidence recall. Precision at the same budget runs 0.009–0.055 for both BM25
baselines, and neither ever abstains (0.000 on the unanswerable questions).
`situational` — the questions that never name the Act — is the gap this
benchmark exists to measure. Full tables: `docs/RESULTS.md`.

## Reporting a result

State all six, or the result is not comparable: corpus version and
`CHECKSUMS.txt` hash; split (`dev` or `test`); budget; τ if not 0.5; whether the
system was tuned on `test`; and for judged scores, the judge model id and
protocol version. The scorer puts the first five in every report it writes.

---

## Building from source

Only needed if you are rebuilding the corpus rather than using a published one.

```
python -m benchmark build          # canonical text from data/processed/ (hours)
python -m benchmark duplicates     # copies of the same instrument
python -m benchmark evidence       # the gold evidence pool
python -m benchmark sample --seed 20260922 --size 1000
python -m benchmark author check   # validate every question
python -m benchmark split --seed 20260923
python -m benchmark index          # BM25 index for the baselines (~25 min, 693 MB)
```

Requirements: Python 3.9+. The scorer needs no third-party package at all;
`requirements.txt` covers the ingestion and processing stages only. Tests:
`pytest` (1,260 offline tests; network tests are opt-in via `-m network`).


