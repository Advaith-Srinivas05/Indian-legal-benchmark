"""Scoring the human labels against the system's own verdicts.

:mod:`processing.validation` builds a review set. This scores it — but only once
a person has actually labelled it. With an empty label file it reports that
nothing has been validated and exits non-zero, because "the gates scored 100%
against zero labels" is precisely the kind of vacuous number this whole exercise
exists to replace.

The asymmetry that matters
--------------------------
Two ways to be wrong, and they are not equally bad for a legal corpus:

**False acceptance** — a human calls the page ``BAD`` and the system was going to
index it. Corrupted text enters the corpus, gets retrieved, and is quoted as law.
Nothing downstream can detect it, because by then the only record of the page is
the corrupted text.

**False quarantine** — a human calls the page ``GOOD`` and the system quarantined
it. A document is missing from the index. Recoverable at any time: the text is
still on disk, the reasons are recorded, and re-admitting it costs a threshold
change.

So the headline number here is the false-acceptance count, and this module
reports it first, separately, and by name. A calibration that trades three false
quarantines for one false acceptance is a bad trade whatever it does to the
totals.

Page labels against document gates
----------------------------------
The human labels **pages**; ``extraction_quality``, ``content_language`` and
``eligible_for_indexing`` are verdicts on **documents**. That mismatch is not
papered over here, and it cuts both ways:

* it is the *right* comparison for the safety question — eligibility is what
  admits text into the index, so "was the document holding this bad page
  eligible?" is exactly what a false acceptance means; but
* it is the *wrong* comparison for judging the quality panel — one bad page in a
  sound 300-page act is not evidence that the document-level verdict was wrong.

Both are therefore reported, and a page-level system verdict is derived from the
page-level signals (see :func:`system_page_verdict`) so there is also a
like-for-like comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from ingestion import config as ingestion_config
from ingestion.utils import atomic_write_text, utcnow_iso

from . import __version__, config
from .errors import ProcessingError
from .validation import (
    ERROR_NAMES,
    HUMAN_ACCEPTABLE,
    HUMAN_REJECT,
    LABEL_NAMES,
    MODE_NAMES,
    validation_dir,
)

log = logging.getLogger("processing.validation_score")


# --- Reading the labels -------------------------------------------------------------


def read_labels(path: Path) -> dict[tuple[str, int], dict]:
    """Read the reviewer's CSV. Rows with no label are *pending*, not GOOD."""
    labels: dict[tuple[str, int], dict] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            document_id = (row.get("document_id") or "").strip()
            page_raw = (row.get("page_number") or "").strip()
            if not document_id or not page_raw:
                continue
            label = (row.get("label") or "").strip().upper()
            errors = [
                e.strip() for e in (row.get("legal_critical_errors") or "")
                .replace(",", ";").split(";") if e.strip()
            ]
            labels[(document_id, int(page_raw))] = {
                "label": label,
                "legal_critical_errors": errors,
                "notes": (row.get("notes") or "").strip(),
            }
    return labels


def validate_labels(labels: dict[tuple[str, int], dict]) -> list[str]:
    """Complaints about the label file itself, before anything is scored."""
    problems: list[str] = []
    for (document_id, page_number), entry in sorted(labels.items()):
        where = f"{document_id} page {page_number}"
        label = entry["label"]
        if label and label not in LABEL_NAMES:
            problems.append(
                f"{where}: unknown label {label!r} (expected one of "
                f"{', '.join(LABEL_NAMES)})")
        unknown = [e for e in entry["legal_critical_errors"]
                   if e not in ERROR_NAMES]
        if unknown:
            problems.append(
                f"{where}: unknown legal-critical error(s) "
                f"{', '.join(unknown)}")
        if entry["legal_critical_errors"] and label in ("GOOD", "MINOR_ERROR"):
            # The label definitions make any legal-critical error decisive. A row
            # that records one and then says the page is fine is a contradiction
            # that must be resolved by the reviewer, not silently resolved here.
            problems.append(
                f"{where}: labelled {label} but records legal-critical error(s) "
                f"{', '.join(entry['legal_critical_errors'])}; by the label "
                "definitions such a page is BAD")
    return problems


# --- The system's side --------------------------------------------------------------


def system_page_verdict(entry: dict) -> str:
    """The label the system would give this **page**, in the human vocabulary.

    Derived rather than stored: the pipeline does not emit a page-level verdict,
    it emits page-level *signals* plus document-level classifications. This
    composes them into something directly comparable with a human label, using
    the same precedence the OCR router uses — unreadable first, then wrong-way-up,
    then corrupt, then not-English, then unsure.

    Being derived, it is a statement about how the signals *would* be read, and a
    disagreement here is a finding about the signals, not proof of a bug.
    """
    signals = entry["signals"]
    system = entry["system"]

    if signals["is_vector_outlined"] or (not signals["has_text"]
                                         and not signals["is_empty"]):
        return "UNUSABLE"
    if signals["is_empty"]:
        return "UNUSABLE"
    if system["page_direction_inconsistent"]:
        return "BAD"
    if system["page_orientation"] == "sideways":
        return "BAD"
    if system["extraction_quality"] == "bad" or signals["text_quality_suspect"]:
        return "BAD"
    if system["content_language"] == "non_en":
        return "NON_ENGLISH"
    if (system["extraction_quality"] == "questionable"
            or system["content_language"] == "uncertain"):
        return "NEEDS_REVIEW"
    return "GOOD"


def system_admits(entry: dict) -> bool:
    """Whether the text of this page would reach the index as things stand."""
    return bool(entry["system"]["eligible_for_indexing"])


def human_rejects(label: str) -> bool:
    return label in HUMAN_REJECT


def human_accepts(label: str) -> bool:
    return label in HUMAN_ACCEPTABLE


# --- Matrices -------------------------------------------------------------------------


def confusion(pairs: Iterable[tuple[str, str]]) -> dict:
    """A confusion matrix as ``{human: {system: count}}``, plus its margins."""
    matrix: dict[str, Counter] = {}
    human_totals: Counter = Counter()
    system_totals: Counter = Counter()
    for human, system in pairs:
        matrix.setdefault(human, Counter())[system] += 1
        human_totals[human] += 1
        system_totals[system] += 1
    return {
        "matrix": {h: dict(row) for h, row in sorted(matrix.items())},
        "human_totals": dict(sorted(human_totals.items())),
        "system_totals": dict(sorted(system_totals.items())),
        "n": sum(human_totals.values()),
    }


def safety(labelled: list[tuple[dict, dict]]) -> dict:
    """The headline: what the gate does with pages a human rejected.

    ``NEEDS_REVIEW`` is excluded from both rates rather than assigned to a side.
    It means the reviewer could not decide, and folding it into either number
    would manufacture a verdict nobody gave.
    """
    false_acceptances: list[dict] = []
    false_quarantines: list[dict] = []
    rejected = accepted = undecided = 0

    for entry, label in labelled:
        verdict = label["label"]
        admitted = system_admits(entry)
        if verdict == "NEEDS_REVIEW":
            undecided += 1
            continue
        if human_rejects(verdict):
            rejected += 1
            if admitted:
                false_acceptances.append(_case(entry, label))
        elif human_accepts(verdict):
            accepted += 1
            if not admitted:
                false_quarantines.append(_case(entry, label))

    return {
        "pages_scored": rejected + accepted,
        "pages_undecided": undecided,
        "human_rejected": rejected,
        "human_accepted": accepted,
        "false_acceptances": len(false_acceptances),
        "false_acceptance_rate": (
            round(len(false_acceptances) / rejected, 4) if rejected else None),
        "false_quarantines": len(false_quarantines),
        "false_quarantine_rate": (
            round(len(false_quarantines) / accepted, 4) if accepted else None),
        "false_acceptance_cases": false_acceptances,
        "false_quarantine_cases": false_quarantines,
        "definitions": {
            "false_acceptance":
                "a human rejected the page (BAD / UNUSABLE / NON_ENGLISH) and "
                "the system would have indexed the document containing it. The "
                "dangerous direction: corrupted text enters the corpus and is "
                "quoted as law, and nothing downstream can detect it.",
            "false_quarantine":
                "a human accepted the page (GOOD / MINOR_ERROR) and the system "
                "quarantined the document containing it. Recoverable: the text "
                "is still on disk and re-admitting it costs a threshold change.",
            "denominators":
                "rates are over the pages a human placed on that side, not over "
                "the whole set; NEEDS_REVIEW is excluded from both.",
        },
    }


def _case(entry: dict, label: dict) -> dict:
    page, system = entry["page"], entry["system"]
    return {
        "document_id": page["document_id"],
        "page_number": page["page_number"],
        "title": page["title"],
        "human_label": label["label"],
        "legal_critical_errors": label["legal_critical_errors"],
        "notes": label["notes"],
        "selected_for": page["selected_for"],
        "modes": page["modes"],
        "system_page_verdict": system_page_verdict(entry),
        "system_extraction_quality": system["extraction_quality"],
        "system_quality_score": system["quality_score"],
        "system_quality_failed_checks": system["quality_failed_checks"],
        "system_content_language": system["content_language"],
        "system_page_orientation": system["page_orientation"],
        "system_pdf_type": system["pdf_type"],
        "system_ocr_action": system["ocr_action"],
        "system_eligible_for_indexing": system["eligible_for_indexing"],
    }


CLASSIFIERS = (
    ("extraction_quality", "document-level extraction quality"),
    ("content_language", "document-level content language"),
    ("page_orientation", "page-level orientation"),
    ("ocr_action", "document-level OCR routing decision"),
    ("pdf_type", "document-level PDF type"),
)


def score(payload: dict, labels: dict[tuple[str, int], dict]) -> dict:
    """Compare every labelled page against every classification of it."""
    labelled: list[tuple[dict, dict]] = []
    pending: list[dict] = []
    for entry in payload["entries"]:
        page = entry["page"]
        key = (page["document_id"], page["page_number"])
        label = labels.get(key)
        if label and label["label"]:
            labelled.append((entry, label))
        else:
            pending.append({"document_id": page["document_id"],
                            "page_number": page["page_number"]})

    result = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "pages_in_set": len(payload["entries"]),
        "pages_labelled": len(labelled),
        "pages_pending": len(pending),
        "pending": pending[:60],
        "human_label_counts": dict(
            Counter(label["label"] for _, label in labelled).most_common()),
    }
    if not labelled:
        result["status"] = "NOT_VALIDATED"
        result["note"] = (
            "No page in this set carries a human label, so nothing has been "
            "validated and no rate below would mean anything. The quality gates "
            "remain uncalibrated against anything but themselves."
        )
        return result

    result["status"] = ("VALIDATED" if not pending
                        else "PARTIALLY_LABELLED")
    result["safety"] = safety(labelled)
    result["page_verdict"] = confusion(
        (label["label"], system_page_verdict(entry))
        for entry, label in labelled
    )
    result["by_classifier"] = {
        name: {
            "description": description,
            **confusion((label["label"], str(entry["system"][name]))
                        for entry, label in labelled),
        }
        for name, description in CLASSIFIERS
    }
    result["by_failure_mode"] = {
        mode: dict(Counter(
            label["label"] for entry, label in labelled
            if mode in entry["page"]["modes"]).most_common())
        for mode in MODE_NAMES
    }
    result["legal_critical_errors"] = dict(Counter(
        error for _, label in labelled
        for error in label["legal_critical_errors"]).most_common())
    result["notes_from_reviewer"] = [
        {"document_id": entry["page"]["document_id"],
         "page_number": entry["page"]["page_number"],
         "label": label["label"], "notes": label["notes"]}
        for entry, label in labelled if label["notes"]
    ]
    return result


# --- Reporting -------------------------------------------------------------------


def _matrix_markdown(block: dict, *, title: str, note: str = "") -> list[str]:
    systems = sorted(block["system_totals"])
    lines = [f"### {title}", ""]
    if note:
        lines += [f"> {note}", ""]
    lines.append("| human \\ system | " + " | ".join(f"`{s}`" for s in systems)
                 + " | total |")
    lines.append("| --- | " + " | ".join("---:" for _ in systems) + " | ---: |")
    for human in sorted(block["matrix"]):
        row = block["matrix"][human]
        cells = " | ".join(str(row.get(s, 0)) for s in systems)
        lines.append(f"| `{human}` | {cells} | {block['human_totals'][human]} |")
    totals = " | ".join(str(block["system_totals"][s]) for s in systems)
    lines.append(f"| **total** | {totals} | {block['n']} |")
    lines.append("")
    return lines


def render_markdown(result: dict) -> str:
    lines = [
        "# Quality-gate validation — human labels vs system verdicts",
        "",
        f"Generated {result['generated_at']} by {result['processor']}.",
        "",
        f"- pages in the validation set: **{result['pages_in_set']}**",
        f"- pages carrying a human label: **{result['pages_labelled']}**",
        f"- pages still unlabelled: **{result['pages_pending']}**",
        f"- status: **`{result['status']}`**",
        "",
    ]

    if result["status"] == "NOT_VALIDATED":
        lines += [
            "## Nothing has been validated",
            "",
            result["note"],
            "",
            "The set is built and waiting. Until a person labels it, the honest "
            "answer to *are the quality gates trustworthy* is **unknown**, and "
            "the 78/100 eligibility figure from the benchmark remains the "
            "pipeline's opinion of itself.",
            "",
            "```bash",
            "# 1. open the review sheet",
            f"#    {config.VALIDATION_SUBDIR / config.VALIDATION_REVIEW_FILENAME}",
            "# 2. read LABELS.md",
            f"# 3. fill in {config.VALIDATION_LABELS_TEMPLATE}",
            "# 4. re-run:",
            "python -m processing.validation_score",
            "```",
            "",
        ]
        return "\n".join(lines)

    if result["pages_pending"]:
        lines += [
            f"> **Partial.** {result['pages_pending']} of "
            f"{result['pages_in_set']} pages are unlabelled. Every rate below is "
            "over the labelled subset only, and that subset is whatever the "
            "reviewer reached first — not a random sample of the set.",
            "",
        ]

    lines += ["## Human labels", "", "| label | pages |", "| --- | ---: |"]
    for label, count in result["human_label_counts"].items():
        lines.append(f"| `{label}` | {count} |")
    lines.append("")

    safety_block = result["safety"]
    lines += [
        "## The number that decides this",
        "",
        "| | count | rate |",
        "| --- | ---: | ---: |",
        f"| pages a human rejected (BAD / UNUSABLE / NON_ENGLISH) | "
        f"{safety_block['human_rejected']} | |",
        f"| **false acceptances** — human rejected, system would index | "
        f"**{safety_block['false_acceptances']}** | "
        f"**{_pct(safety_block['false_acceptance_rate'])}** |",
        f"| pages a human accepted (GOOD / MINOR_ERROR) | "
        f"{safety_block['human_accepted']} | |",
        f"| false quarantines — human accepted, system quarantined | "
        f"{safety_block['false_quarantines']} | "
        f"{_pct(safety_block['false_quarantine_rate'])} |",
        f"| undecided (NEEDS_REVIEW, excluded from both rates) | "
        f"{safety_block['pages_undecided']} | |",
        "",
        "A false acceptance puts corrupted text into a corpus whose whole "
        "purpose is to be quotable, and nothing downstream can detect it. A "
        "false quarantine costs a missing document and is reversible with a "
        "threshold change. They are not interchangeable and must not be traded "
        "off against each other.",
        "",
    ]

    if safety_block["false_acceptance_cases"]:
        lines += [
            "### False acceptances, in full",
            "",
            "Each of these is a page a person read and rejected, inside a "
            "document the system was willing to index.",
            "",
        ]
        for case in safety_block["false_acceptance_cases"]:
            lines += [
                f"**{case['title']}** — page {case['page_number']}  ",
                f"`{case['document_id']}`  ",
                f"human: **{case['human_label']}**"
                + (f" ({', '.join(case['legal_critical_errors'])})"
                   if case["legal_critical_errors"] else ""),
                "",
                f"- system page verdict: `{case['system_page_verdict']}`",
                f"- extraction_quality: `{case['system_extraction_quality']}` "
                f"(score {case['system_quality_score']}, failed "
                f"{case['system_quality_failed_checks'] or 'nothing'})",
                f"- content_language: `{case['system_content_language']}`, "
                f"orientation: `{case['system_page_orientation']}`, "
                f"pdf_type: `{case['system_pdf_type']}`",
                f"- ocr_action: `{case['system_ocr_action']}`",
                f"- selected for: `{case['selected_for']}`",
            ]
            if case["notes"]:
                lines.append(f"- reviewer: {case['notes']}")
            lines.append("")
    else:
        lines += [
            "### No false acceptances",
            "",
            "No page a human rejected sits inside a document the system would "
            "have indexed — over the labelled pages, which are deliberately "
            "richer in damage than the corpus average.",
            "",
        ]

    lines += _matrix_markdown(
        result["page_verdict"],
        title="Page-level: human label vs derived system verdict",
        note=("The system emits page-level *signals* and document-level "
              "classifications, not a page verdict. This composes them into one "
              "— see `validation_score.system_page_verdict`. A disagreement here "
              "is a finding about the signals, not proof of a bug."),
    )

    lines += ["## Per-classifier agreement", ""]
    for name, block in result["by_classifier"].items():
        lines += _matrix_markdown(
            block, title=f"`{name}` — {block['description']}",
            note=("Page labels against a document-level verdict: a single bad "
                  "page in a sound 300-page act is not evidence the document "
                  "verdict was wrong."
                  if name != "page_orientation" else ""),
        )

    if result["legal_critical_errors"]:
        lines += ["## Legal-critical errors found", "",
                  "| error | pages |", "| --- | ---: |"]
        for error, count in result["legal_critical_errors"].items():
            lines.append(f"| `{error}` | {count} |")
        lines.append("")

    lines += ["## Labels by failure mode", "",
              "The mode a page exhibits, against what the human made of it. "
              "Modes where the human disagrees with the system most often are "
              "where the thresholds need work.", "",
              "| failure mode | labels |", "| --- | --- |"]
    for mode, counts in result["by_failure_mode"].items():
        if counts:
            rendered = ", ".join(f"{label} {n}" for label, n in counts.items())
            lines.append(f"| `{mode}` | {rendered} |")
    lines.append("")

    if result["notes_from_reviewer"]:
        lines += ["## Reviewer notes", ""]
        for note in result["notes_from_reviewer"]:
            lines.append(
                f"- `{note['document_id']}` p{note['page_number']} "
                f"(**{note['label']}**): {note['notes']}")
        lines.append("")
    return "\n".join(lines)


def _pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


# --- CLI -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m processing.validation_score",
        description=(
            "Score human validation labels against the system's own quality, "
            "language, orientation, routing and type classifications. Reports "
            "false acceptances first, because for a legal corpus they are the "
            "expensive direction."
        ),
    )
    parser.add_argument("--data-dir", type=Path,
                        default=ingestion_config.DEFAULT_DATA_DIR)
    parser.add_argument("--labels", type=Path, default=None,
                        help="Label CSV (default: the one in data/validation/).")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    directory = validation_dir(args.data_dir)
    selection_path = directory / config.VALIDATION_SELECTION_FILENAME
    labels_path = args.labels or (directory / config.VALIDATION_LABELS_TEMPLATE)

    if not selection_path.exists():
        log.error("No validation set at %s; run `python -m processing.validation` "
                  "first.", selection_path)
        return 2
    if not Path(labels_path).exists():
        log.error("No label file at %s.", labels_path)
        return 2

    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        labels = read_labels(labels_path)
    except (OSError, ValueError) as exc:
        log.error("%s", exc)
        return 2

    problems = validate_labels(labels)
    for problem in problems:
        log.error("label file: %s", problem)
    if problems:
        log.error("Refusing to score %d malformed label(s); fix them and re-run.",
                  len(problems))
        return 2

    result = score(payload, labels)
    result["labels_path"] = str(labels_path)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_text(directory / config.VALIDATION_SCORE_JSON,
                      json.dumps(result, indent=2, ensure_ascii=False))
    atomic_write_text(directory / config.VALIDATION_SCORE_MARKDOWN,
                      render_markdown(result))

    print()
    print("=" * 62)
    if result["status"] == "NOT_VALIDATED":
        print("  NOT VALIDATED — the label file is empty")
        print("=" * 62)
        print(f"  pages awaiting a human label : {result['pages_pending']:4d}")
        print("=" * 62)
        print()
        print("The quality gates remain calibrated only against themselves.")
        print(f"Review sheet: {directory / config.VALIDATION_REVIEW_FILENAME}")
        print(f"Fill in     : {labels_path}")
        return 1

    safety_block = result["safety"]
    print(f"  QUALITY-GATE VALIDATION — {result['status']}")
    print("=" * 62)
    print(f"  pages labelled            : {result['pages_labelled']:4d}"
          f" of {result['pages_in_set']}")
    print(f"  human rejected            : {safety_block['human_rejected']:4d}")
    print(f"  FALSE ACCEPTANCES         : "
          f"{safety_block['false_acceptances']:4d}"
          f"  ({_pct(safety_block['false_acceptance_rate'])})")
    print(f"  false quarantines         : "
          f"{safety_block['false_quarantines']:4d}"
          f"  ({_pct(safety_block['false_quarantine_rate'])})")
    print(f"  undecided (NEEDS_REVIEW)  : {safety_block['pages_undecided']:4d}")
    print("=" * 62)
    print()
    print(f"Report: {directory / config.VALIDATION_SCORE_MARKDOWN}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
