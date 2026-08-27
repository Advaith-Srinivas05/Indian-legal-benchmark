"""Full-corpus processing: the resumable driver around ``process_document()``.

This is the production runner for the 19,802 downloaded India Code PDFs. It adds
nothing to what a single document does — every document still goes through
:func:`processing.process.process_document`, unchanged — and everything it adds
is what 19,802 of them need and 100 of them did not:

* **a durable status journal** (``data/processing_status.jsonl``), appended to
  after every document, so a run interrupted at document 5,000 resumes at 5,001
  instead of starting over;
* **verification of existing output**, because a journal record claiming success
  is a claim, not evidence — the files it names must be on disk, the right size,
  parseable, and about the right source PDF, or the document is processed again;
* **completeness that survives a kill -9**: ``document.json`` is removed before a
  document is rewritten and written last, so a half-written document has no
  ``document.json`` and can never be mistaken for a finished one;
* **streaming aggregation** — counters, never retained results. The 100-document
  benchmark holds every ``ProcessedDocument`` in memory to build its report; at
  19,802 that would be the whole corpus's extracted text in RAM;
* **error isolation**, so one unreadable PDF is a journal record rather than the
  end of a 20-hour run;
* **one runner at a time**, enforced with a lock file, so two invocations cannot
  interleave writes into one journal.

Usage::

    python -m processing.run --dry-run              # inspect and estimate; write nothing
    python -m processing.run --limit 5 --workers 2  # small integration run
    python -m processing.run --workers 4            # the full corpus
    python -m processing.run --workers 4            # …again: resumes where it stopped

What it does **not** do:

* it does not run OCR. :mod:`processing.ocr` decides and never executes;
  documents routed to OCR keep their extracted text, are quarantined, and are
  counted so the OCR step can be scoped. Nothing here may be described as "the
  corpus has been OCR'd".
* it does not touch ``data/raw/``. PDFs are opened read-only and every write
  lands under ``data/processed/`` or on the three ``data/processing_*`` files.
* it does not chunk, embed, index or generate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import threading
import time
import traceback as traceback_module
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from ingestion import config as ingestion_config
from ingestion.utils import atomic_write_text, utcnow_iso

from . import __version__, config
from .backends import PdfBackend, default_backend
from .corpus import Corpus
from .errors import ProcessingError, RunLockError
from . import ocr_engine
from .models import CorpusDocument, ProcessedDocument
from .process import output_dir, process_document, write_outputs

log = logging.getLogger("processing.run")


class InsufficientSpaceError(ProcessingError):
    """Not enough free space to start or continue.

    Operator action, not a document problem: kept separate for the same reason
    :class:`processing.errors.BackendUnavailableError` is, because recording
    19,802 identical failures would be worse than stopping.
    """


# --- Where the run's three artefacts live ---------------------------------------


def journal_path(data_dir: Path) -> Path:
    return Path(data_dir) / config.PROCESSING_STATUS_FILENAME


def report_path(data_dir: Path) -> Path:
    return Path(data_dir) / config.PROCESSING_REPORT_FILENAME


def lock_path(data_dir: Path) -> Path:
    return Path(data_dir) / config.PROCESSING_LOCK_FILENAME


# --- One runner at a time -------------------------------------------------------


def _boot_time() -> Optional[datetime]:
    """When this machine last booted, or ``None`` if it cannot be determined.

    A lock written before the last boot cannot still be held: nothing survives
    a restart. This is the cheapest possible proof that a lock is stale, and it
    covers the case that produced the problem in practice — a run killed by
    shutting the laptop down, whose lock then blocked every later run.
    """
    try:
        if os.name == "nt":
            import ctypes                                       # noqa: PLC0415

            uptime_ms = ctypes.WinDLL("kernel32").GetTickCount64()
            return datetime.now(timezone.utc) - timedelta(milliseconds=uptime_ms)
        for name in ("CLOCK_BOOTTIME", "CLOCK_UPTIME"):
            clock = getattr(time, name, None)
            if clock is not None:
                uptime = time.clock_gettime(clock)
                return datetime.now(timezone.utc) - timedelta(seconds=uptime)
    except Exception:                       # a diagnostic, never worth raising over
        return None
    return None


def _pid_is_running(pid: int) -> Optional[bool]:
    """Whether *pid* is a live process. ``None`` means "cannot tell".

    The three-valued answer is the point. Only a definite ``False`` may break a
    lock, so every path that cannot establish death — an unfamiliar platform, a
    refused query, a malformed pid — answers ``None`` and the lock stands.

    ``os.kill(pid, 0)`` is the POSIX idiom and is **not** used on Windows, where
    CPython implements ``os.kill`` with ``TerminateProcess`` and it would kill
    the very process it was asked about.
    """
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes                                       # noqa: PLC0415
            from ctypes import wintypes                         # noqa: PLC0415

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            ERROR_INVALID_PARAMETER = 87                # no process has this id
            STILL_ACTIVE = 259

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                if ctypes.get_last_error() == ERROR_INVALID_PARAMETER:
                    return False
                return None            # access denied, or something unexplained
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return None
                # A process that has exited reports its exit code instead. An
                # exit code that happens to be 259 reads as alive, which errs
                # towards keeping the lock — the safe direction.
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:                 # alive, just not ours to signal
        return True
    except Exception:
        return None


class RunLock:
    """One runner at a time, per data directory.

    Two concurrent runners would each read the journal at startup, each decide
    the same documents were outstanding, and process every one of them twice —
    interleaving writes into one output directory while they did it. The lock is
    created with ``O_CREAT | O_EXCL``, which is atomic on both Windows and POSIX,
    and carries who holds it so a stale one can be identified rather than guessed
    at.

    ``release()`` only runs when a run ends through its own code, so a runner
    that is killed — Ctrl-C twice, a closed terminal, a shutdown — leaves its
    lock behind and blocks every later run until someone passes
    ``--force-unlock``. :meth:`stale_reason` removes that chore where it can be
    done safely, by breaking a lock only on *positive evidence* that its holder
    is gone. Anything short of proof leaves the lock standing and asks the human.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.held = False

    def stale_reason(self) -> Optional[str]:
        """Why this lock is provably abandoned, or ``None`` if it may be live.

        Deliberately asymmetric: it answers "certainly dead" or "don't know",
        never "probably dead". Two runners writing one journal is a far worse
        outcome than one unnecessary ``--force-unlock``.
        """
        try:
            holder = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None                     # unreadable: assume it means something

        if holder.get("host") != platform.node():
            # A pid from another machine says nothing about a process on this
            # one, and the same data directory can be reached over a share.
            return None

        started_at = holder.get("started_at")
        boot = _boot_time()
        if isinstance(started_at, str) and boot is not None:
            try:
                written = datetime.fromisoformat(started_at)
            except ValueError:
                written = None
            if written is not None:
                if written.tzinfo is None:
                    written = written.replace(tzinfo=timezone.utc)
                if written < boot:
                    return (f"it was taken at {started_at}, before this machine "
                            f"last booted at {boot.replace(microsecond=0).isoformat()}")

        pid = holder.get("pid")
        if _pid_is_running(pid) is False:
            return f"process {pid} on {holder.get('host')} is no longer running"
        return None

    def acquire(self, *, force: bool = False) -> None:
        if self.path.exists():
            if force:
                self.break_lock()
            else:
                reason = self.stale_reason()
                if reason:
                    log.warning(
                        "Reclaiming an abandoned run lock: %s. (%s)",
                        reason, self.describe())
                    self.path.unlink(missing_ok=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "pid": os.getpid(),
            "host": platform.node(),
            "started_at": utcnow_iso(),
            "processor": f"processing v{__version__}",
        }, ensure_ascii=False)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RunLockError(
                f"Another processing run holds {self.path} ({self.describe()}). "
                "Its holder could not be shown to be gone — it is either alive, "
                "on another machine, or not answerable from here — so the lock "
                "stands. Wait for it to finish, or re-run with --force-unlock if "
                "you are certain no runner is alive."
            ) from exc
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(payload)
        self.held = True

    def describe(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8").strip()
        except OSError:
            return "(unreadable lock file)"

    def break_lock(self) -> None:
        log.warning("Breaking run lock %s: %s", self.path, self.describe())
        self.path.unlink(missing_ok=True)

    def release(self) -> None:
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


# --- The status journal ---------------------------------------------------------

#: The per-document fields the resume check and the corpus-wide report need.
#: Aggregation keeps only these, one document at a time, so a report over 19,802
#: journal lines costs a few MB rather than the corpus's extracted text.
_PROJECTED_FIELDS = (
    "document_id", "status", "category", "document_type", "title", "pdf_relpath",
    "sha256", "journal_schema_version", "processing_schema_version",
    "pages", "pages_indexable", "pages_excluded", "chars", "empty_pages",
    "bytes", "seconds",
    "pdf_type", "text_extraction_status", "extraction_quality", "content_language",
    "orientation_suspect", "ocr_action", "ocr_text_source", "ocr_estimated_pages",
    # What OCR did, as opposed to what the router asked for. Without these three
    # the corpus totals cannot tell a document OCR rescued from one it never
    # touched, which is how `corpus_totals.pages_ocr` came to be a hardcoded 0.
    "ocr_executed", "ocr_pages_attempted", "ocr_pages_accepted",
    "structure_confidence", "eligible_for_indexing", "quarantine_reasons",
    "error_type", "error_message", "error_stage", "skip_reason",
    "output_relpath", "output_bytes", "attempt", "processed_at",
)


def project(row: dict) -> dict:
    """Reduce a journal record to the fields resume and reporting need."""
    return {key: row[key] for key in _PROJECTED_FIELDS if key in row}


class StatusJournal:
    """Append-only JSONL record of every processing attempt.

    Append-only rather than rewritten, for the same reason
    :class:`ingestion.bulk.FailureReport` is: a run that dies mid-write must
    leave a readable journal behind, and a 19,802-entry file must not be
    rewritten once per document. The last record for a ``document_id`` is the
    current one; earlier records are the attempt history and are kept.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.appended = 0

    def iter_records(self) -> Iterator[dict]:
        """Yield every well-formed record, oldest first.

        A truncated final line — the signature of a process killed mid-append —
        is skipped with a warning rather than treated as a parse failure of the
        whole journal. Losing the last record costs one document's reprocessing;
        refusing to read the journal costs the whole run.
        """
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    log.warning(
                        "%s line %d is not valid JSON; ignoring it. Any document "
                        "it described will simply be processed again.",
                        self.path, number,
                    )
                    continue
                if isinstance(record, dict) and record.get("document_id"):
                    yield record

    def load(self) -> dict[str, dict]:
        """Current state per document: the last record for each ``document_id``."""
        state: dict[str, dict] = {}
        for record in self.iter_records():
            state[record["document_id"]] = record
        return state

    def attempt_counts(self) -> Counter:
        """How many attempts the journal already records, per document."""
        counts: Counter = Counter()
        for record in self.iter_records():
            counts[record["document_id"]] += 1
        return counts

    def append(self, record: dict) -> None:
        """Append one record durably. Safe to call from several threads."""
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.appended += 1


# --- Turning a result into a record ---------------------------------------------


def quarantine_reasons(result: ProcessedDocument) -> list[str]:
    """Why this document is not eligible for indexing.

    Mirrors the four gates in
    :attr:`processing.models.ProcessedDocument.eligible_for_indexing` one for
    one. The gates decide; this names the decision, so a quarantined document can
    be found by reason later without re-running anything.
    """
    if not result.ok:
        return [f"processing_failed:{result.error_type or 'Unknown'}"]
    reasons: list[str] = []
    if result.language is None:
        reasons.append("language_assessment_missing")
    elif not getattr(result.language, "eligible_for_indexing", False):
        content = getattr(result.language, "content_language", None) or "unknown"
        reasons.append(f"content_language_{content}")
    if result.quality is None:
        reasons.append("quality_assessment_missing")
    elif getattr(result.quality, "classification", "") != "good":
        classification = getattr(result.quality, "classification", None) or "unknown"
        reasons.append(f"extraction_quality_{classification}")
    orientation = getattr(result.extraction, "orientation", None) or {}
    if orientation.get("orientation_suspect"):
        reasons.append("orientation_suspect")
    action = getattr(result.ocr_decision, "action", "") or "unspecified"
    if action in config.OCR_ACTIONS_BLOCKING_INDEX:
        reasons.append(f"ocr_{action}")
    return reasons


def classify(result: ProcessedDocument) -> str:
    """Map a processing result onto :data:`processing.config.PROCESSING_STATUSES`.

    ``SUCCESS_OCR`` means the text written to disk came out of an OCR engine on
    at least one page. It was reserved in the vocabulary long before anything
    could emit it, precisely so the journal and the report would not have to
    change shape when OCR was built; this is where it starts being used.

    It is distinguished from ``SUCCESS`` because the two are not equally
    trustworthy. OCR text has been judged better than what it replaced, which is
    a weaker claim than a clean text layer, and a document indexed on the
    strength of it should be findable later without re-reading 19,802 files.
    """
    if not result.ok:
        return "FAILED"
    if not result.eligible_for_indexing:
        return "QUARANTINED"
    if getattr(result.extraction, "ocr_page_count", 0):
        return "SUCCESS_OCR"
    return "SUCCESS"


def _identity(document: CorpusDocument) -> dict:
    """The provenance every record carries, whatever the outcome.

    Copied from the manifest, never re-derived: the runner is not allowed to
    invent or normalise a document's identity, and a record that cannot be tied
    back to one exact PDF is not worth keeping.
    """
    return {
        "journal_schema_version": config.JOURNAL_SCHEMA_VERSION,
        "document_id": document.document_id,
        "title": document.title,
        "category": document.category,
        "document_type": document.document_type,
        "pdf_relpath": document.pdf_relpath,
        "sha256": document.sha256,
        "bytes": document.bytes,
        "source_url": document.source_url,
        "handle": document.handle,
        "language": document.language,
    }


def skipped_record(document: CorpusDocument, reason: str, *, attempt: int = 1) -> dict:
    """A document deliberately not processed — today, only a PDF not on disk."""
    return {
        **_identity(document),
        "status": "SKIPPED",
        "skip_reason": reason,
        "ok": False,
        "attempt": attempt,
        "processed_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "seconds": 0.0,
    }


def timeout_record(
    document: CorpusDocument, seconds: float, *, attempt: int = 1
) -> dict:
    """A document the runner stopped waiting for.

    Recorded as ``FAILED`` — so it stays retryable, as every failure does — with
    ``error_stage: "timeout"`` so it can be told apart from a PDF that could not
    be read. The two want different responses: an unreadable PDF is a corpus
    finding, an overrun is a scheduling one.

    The worker thread is **abandoned, not killed**; Python cannot kill a thread.
    It runs on until it finishes and anything it writes afterwards belongs to a
    document the journal already calls ``FAILED``, so the next run reprocesses
    it from scratch. Nothing on disk is left in a state that could verify as
    complete.
    """
    result = ProcessedDocument(document=document)
    result.error_type = "DocumentTimeout"
    result.error_message = (
        f"still running after {seconds:.0f}s and was abandoned so the run could "
        "continue. Retry it on its own, or raise --document-timeout."
    )
    result.seconds = seconds
    return result_record(
        document, result, "FAILED", error_stage="timeout", attempt=attempt)


def result_record(
    document: CorpusDocument,
    result: ProcessedDocument,
    status: str,
    *,
    output_bytes: Optional[dict] = None,
    output_relpath: Optional[str] = None,
    error_stage: Optional[str] = None,
    traceback_text: Optional[str] = None,
    attempt: int = 1,
) -> dict:
    """One journal record for one processing attempt.

    Carries the whole provenance chain the invariants require — document id,
    India Code source URL, raw PDF path, SHA-256, page count, and the four
    verdicts — so the corpus's state can be read from this file alone, without
    opening 19,802 ``document.json`` files.
    """
    extraction = result.extraction
    structure = result.structure
    orientation = getattr(extraction, "orientation", None) or {}
    counts = getattr(structure, "counts", None) or {}
    record = {
        **_identity(document),
        "status": status,
        "ok": bool(result.ok),
        "attempt": attempt,
        "processed_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "processing_schema_version": config.PROCESSING_SCHEMA_VERSION,
        "seconds": round(result.seconds, 3),
        "output_relpath": output_relpath,
        "output_bytes": output_bytes,
        # Extraction
        "pages": getattr(extraction, "page_count", 0) or 0,
        "chars": getattr(extraction, "char_count", 0) or 0,
        "empty_pages": getattr(extraction, "empty_page_count", 0) or 0,
        # What this document actually contributes downstream. Documents are the
        # unit the runner schedules; pages are the unit that gets indexed, and
        # since quality is decided per page the two counts diverge. A report
        # that only counted documents would show a partially-recovered document
        # as a whole success.
        "pages_indexable": result.indexable_page_count,
        "pages_excluded": ((getattr(extraction, "page_count", 0) or 0)
                           - result.indexable_page_count),
        "pdf_type": getattr(extraction, "pdf_type", None),
        "text_extraction_status": getattr(extraction, "text_extraction_status", None),
        "backend": getattr(extraction, "backend", None),
        # The four verdicts
        "content_language": getattr(result.language, "content_language", None),
        "extraction_quality": getattr(result.quality, "classification", None),
        "quality_score": getattr(result.quality, "score", None),
        "orientation_suspect": bool(orientation.get("orientation_suspect")),
        "ocr_action": getattr(result.ocr_decision, "action", None),
        "ocr_text_source": getattr(result.ocr_decision, "text_source", None),
        "ocr_estimated_pages": getattr(result.ocr_decision, "estimated_pages", 0) or 0,
        # What the OCR stage actually did, as against what it was told to do.
        # Read by plan_run(): a document recorded as never OCR'd is outstanding
        # work for a run that has OCR enabled, however complete its output is.
        "ocr_executed": bool(getattr(result.ocr_run, "executed", False)),
        "ocr_pages_attempted": getattr(result.ocr_run, "attempted", 0) or 0,
        "ocr_pages_accepted": getattr(result.ocr_run, "accepted", 0) or 0,
        "ocr_seconds": round(getattr(result.ocr_run, "seconds", 0.0) or 0.0, 3),
        "ocr_truncated": bool(getattr(result.ocr_run, "truncated", False)),
        "ocr_pages_in_output": getattr(result.extraction, "ocr_page_count", 0) or 0,
        # Structure
        "structure_confidence": getattr(structure, "confidence", None),
        "unit_vocabulary": getattr(structure, "unit_vocabulary", None),
        "sections": counts.get("section", 0),
        "articles": counts.get("article", 0),
        # Outcome
        "eligible_for_indexing": bool(result.eligible_for_indexing),
        "quarantine_reasons": quarantine_reasons(result) if status != "SUCCESS" else [],
        "error_type": result.error_type,
        "error_message": result.error_message,
        "error_stage": error_stage,
    }
    if traceback_text:
        record["traceback"] = traceback_text
    return record


# --- Is output a journal record claims to exist actually there? -----------------


def needs_ocr_pass(row: Optional[dict], *, ocr_enabled: bool) -> bool:
    """Whether *row* describes output that predates OCR and should be redone.

    Without this a document processed with ``--no-ocr`` verifies as complete
    forever: its files are on disk, the right size, and about the right PDF, so
    every check passes and the pages that needed reading never get read.

    Only documents the router actually sent to OCR qualify. Re-running the
    engine over a clean born-digital act would cost the run and change nothing.
    """
    if not ocr_enabled or not row:
        return False
    if row.get("ocr_executed"):
        return False
    return row.get("ocr_action") in config.OCR_ACTIONS_BLOCKING_INDEX


def verify_output(
    data_dir: Path,
    document: CorpusDocument,
    row: Optional[dict],
    *,
    level: str = config.DEFAULT_VERIFY_LEVEL,
) -> Optional[str]:
    """``None`` when this document is genuinely done; otherwise why it is not.

    The project spec forbids trusting directory existence, and a status journal is only
    a stronger claim, not proof. So the claim is checked against the artefacts:

    ``fast``
        The journal alone — status, journal schema, processing schema, and the
        SHA-256 of the source PDF. Nothing is read from ``data/processed/``.
    ``standard`` (default)
        …and ``pages.json`` exists at exactly its recorded size, and
        ``document.json`` exists at exactly its recorded size, parses, and names
        this document and this SHA-256. Sound because
        :func:`write_document_output` deletes ``document.json`` first and writes
        it last, each file atomically: a present, correct ``document.json``
        therefore implies a complete ``pages.json`` before it.
    ``full``
        …and ``pages.json`` itself parses, names this document, and holds as many
        pages as it claims. Reads every byte of the output, so a corpus-wide
        resume at this level costs minutes rather than seconds.
    """
    if level not in config.VERIFY_LEVELS:
        raise ValueError(f"unknown verification level {level!r}")
    if not row:
        return "never processed"
    status = row.get("status")
    if status not in config.PROCESSING_COMPLETE_STATUSES:
        return f"previous status {status or 'unknown'}"
    if row.get("journal_schema_version") != config.JOURNAL_SCHEMA_VERSION:
        return (
            f"journal schema {row.get('journal_schema_version')} != "
            f"{config.JOURNAL_SCHEMA_VERSION}"
        )
    if row.get("processing_schema_version") != config.PROCESSING_SCHEMA_VERSION:
        return (
            f"output schema {row.get('processing_schema_version')} != "
            f"{config.PROCESSING_SCHEMA_VERSION}"
        )
    if row.get("sha256") and document.sha256 and row["sha256"] != document.sha256:
        return "source PDF changed since it was processed"
    if level == "fast":
        return None

    directory = output_dir(data_dir, document.document_id)
    recorded = row.get("output_bytes") or {}
    pages_file = directory / config.PAGES_FILENAME
    document_file = directory / config.DOCUMENT_FILENAME

    problem = _check_size(pages_file, recorded.get("pages"))
    if problem:
        return problem
    if not document_file.exists():
        return f"{config.DOCUMENT_FILENAME} is missing (output incomplete)"
    problem = _check_size(document_file, recorded.get("document"))
    if problem:
        return problem

    try:
        payload = json.loads(document_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{config.DOCUMENT_FILENAME} is unreadable: {exc}"
    if not isinstance(payload, dict):
        return f"{config.DOCUMENT_FILENAME} is not an object"
    if payload.get("schema_version") != config.PROCESSING_SCHEMA_VERSION:
        return f"{config.DOCUMENT_FILENAME} carries a different schema version"
    source = payload.get("source") or {}
    if source.get("document_id") != document.document_id:
        return f"{config.DOCUMENT_FILENAME} names a different document"
    if document.sha256 and source.get("sha256") != document.sha256:
        return f"{config.DOCUMENT_FILENAME} names a different source PDF"
    if level == "standard":
        return None

    try:
        pages_payload = json.loads(pages_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return f"{config.PAGES_FILENAME} is unreadable: {exc}"
    if not isinstance(pages_payload, dict):
        return f"{config.PAGES_FILENAME} is not an object"
    if pages_payload.get("document_id") != document.document_id:
        return f"{config.PAGES_FILENAME} names a different document"
    pages = pages_payload.get("pages")
    if not isinstance(pages, list):
        return f"{config.PAGES_FILENAME} has no page list"
    claimed = pages_payload.get("page_count")
    if isinstance(claimed, int) and len(pages) != claimed:
        return f"{config.PAGES_FILENAME} holds {len(pages)} pages but claims {claimed}"
    return None


def _check_size(path: Path, recorded: Optional[int]) -> Optional[str]:
    if not path.exists():
        return f"{path.name} is missing"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return f"{path.name} cannot be read: {exc}"
    if size == 0:
        return f"{path.name} is empty"
    if isinstance(recorded, int) and recorded > 0 and size != recorded:
        return f"{path.name} is {size} bytes, journal recorded {recorded}"
    return None


# --- Writing output that never looks finished until it is -----------------------


def write_document_output(
    result: ProcessedDocument, data_dir: Path
) -> tuple[Path, dict[str, int]]:
    """Write one document's output, leaving nothing that could look finished.

    ``document.json`` is the completion marker. It is deleted before anything is
    rewritten and it is written last, so at every instant between those two
    points the directory is *visibly* incomplete — and :func:`verify_output`
    reads exactly that. The alternative, writing into the directory in place,
    allows the pathological pair: a stale ``document.json`` from an earlier run
    sitting beside a fresh ``pages.json``, which verifies and is wrong.

    The individual files are still written through
    :func:`ingestion.utils.atomic_write_text` — temp file plus ``os.replace`` —
    so neither file is ever observed half-written either.
    """
    directory = output_dir(data_dir, result.document.document_id)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / config.DOCUMENT_FILENAME).unlink(missing_ok=True)
    for stale in directory.glob(".*.tmp"):
        try:
            stale.unlink()
        except OSError:                              # pragma: no cover - rare
            log.debug("Could not remove stale temp file %s", stale)

    write_outputs(result, data_dir)
    result.output_dir = directory
    sizes = {
        "pages": (directory / config.PAGES_FILENAME).stat().st_size,
        "document": (directory / config.DOCUMENT_FILENAME).stat().st_size,
    }
    return directory, sizes


# --- Deciding what to process ---------------------------------------------------


@dataclass
class WorkItem:
    """One document to process, and why it is being processed."""

    document: CorpusDocument
    reason: str
    attempt: int = 1


@dataclass
class WorkPlan:
    """What a run would do, decided before it does any of it."""

    todo: list[WorkItem] = field(default_factory=list)
    already_complete: list[CorpusDocument] = field(default_factory=list)
    missing_pdf: list[CorpusDocument] = field(default_factory=list)
    held_back: list[CorpusDocument] = field(default_factory=list)
    filtered_out: int = 0
    deferred_by_limit: int = 0
    manifest_documents: int = 0
    pdfs_on_disk: int = 0

    @property
    def selected(self) -> int:
        return len(self.todo) + len(self.missing_pdf)

    def bytes_to_read(self) -> int:
        return sum(item.document.bytes or 0 for item in self.todo)

    def reasons(self) -> dict[str, int]:
        return dict(Counter(item.reason for item in self.todo))


def plan_run(
    corpus: Corpus,
    state: dict[str, dict],
    *,
    verify: str = config.DEFAULT_VERIFY_LEVEL,
    limit: Optional[int] = None,
    categories: Optional[Iterable[str]] = None,
    document_types: Optional[Iterable[str]] = None,
    only: Optional[Iterable[str]] = None,
    retry_failed: bool = True,
    force: bool = False,
    attempts: Optional[Counter] = None,
    ocr_enabled: bool = False,
) -> WorkPlan:
    """Decide what to process. Reads the journal and the output; writes nothing.

    This is the whole of resumption, and it is deliberately one pure function so
    ``--dry-run`` and a real run cannot disagree about what the run would do.

    Documents are considered in ``document_id`` order — the order
    :meth:`processing.corpus.Corpus.load` sorts them into — so ``--limit N``
    selects the same N documents on every invocation.
    """
    category_filter = {c for c in categories} if categories else None
    type_filter = {t for t in document_types} if document_types else None
    only_filter = {d for d in only} if only else None
    attempts = attempts if attempts is not None else Counter()

    plan = WorkPlan(manifest_documents=len(corpus.documents))
    for document in corpus.documents:
        on_disk = document.pdf_path.exists()
        if on_disk:
            plan.pdfs_on_disk += 1
        if only_filter is not None and document.document_id not in only_filter:
            plan.filtered_out += 1
            continue
        if category_filter is not None and document.category not in category_filter:
            plan.filtered_out += 1
            continue
        if type_filter is not None and document.document_type not in type_filter:
            plan.filtered_out += 1
            continue
        if not on_disk:
            # A manifest entry with no PDF. Recorded as SKIPPED rather than
            # ignored: the corpus and its index disagreeing is a finding.
            plan.missing_pdf.append(document)
            continue

        row = state.get(document.document_id)
        if force:
            reason = "forced"
        else:
            if not retry_failed and (row or {}).get("status") == "FAILED":
                plan.held_back.append(document)
                continue
            reason = verify_output(corpus.data_dir, document, row, level=verify)
            if reason is None and needs_ocr_pass(row, ocr_enabled=ocr_enabled):
                reason = "OCR was never run on this document"
            if reason is None:
                plan.already_complete.append(document)
                continue
        plan.todo.append(WorkItem(
            document=document,
            reason=reason,
            attempt=attempts.get(document.document_id, 0) + 1,
        ))

    if limit is not None and limit >= 0 and len(plan.todo) > limit:
        plan.deferred_by_limit = len(plan.todo) - limit
        plan.todo = plan.todo[:limit]
    return plan


# --- Streaming counters ---------------------------------------------------------


@dataclass
class RunStats:
    """Streaming counters for one run. No result object is ever retained."""

    total: int = 0
    completed: int = 0
    successful: int = 0
    successful_with_ocr: int = 0
    quarantined: int = 0
    failed: int = 0
    skipped: int = 0
    timed_out: int = 0
    pages: int = 0
    #: Pages that survived the language and quality gates -- what this run
    #: actually contributes downstream, as against what it read.
    pages_indexable: int = 0
    pages_ocr: int = 0
    ocr_documents: int = 0
    ocr_pages_accepted: int = 0
    ocr_seconds: float = 0.0
    documents_pending_ocr: int = 0
    pages_pending_ocr: int = 0
    characters: int = 0
    bytes_read: int = 0
    errors_by_type: Counter = field(default_factory=Counter)
    interrupted: bool = False
    started_at: float = field(default_factory=time.monotonic)
    started_at_iso: str = field(default_factory=utcnow_iso)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def elapsed(self) -> float:
        return max(1e-6, time.monotonic() - self.started_at)

    @property
    def rate(self) -> float:
        """Documents per second over the run so far."""
        return self.completed / self.elapsed

    @property
    def eta_seconds(self) -> Optional[float]:
        if self.remaining <= 0:
            return 0.0
        if self.completed == 0:
            return None
        return self.remaining / self.rate

    def record(self, record: dict) -> None:
        self.completed += 1
        status = record.get("status")
        if status == "SUCCESS":
            self.successful += 1
        elif status == "SUCCESS_OCR":
            self.successful_with_ocr += 1
        elif status == "QUARANTINED":
            self.quarantined += 1
        elif status == "SKIPPED":
            self.skipped += 1
        else:
            self.failed += 1
            self.errors_by_type[record.get("error_type") or "Unknown"] += 1
        if record.get("ocr_executed"):
            self.ocr_documents += 1
        self.ocr_pages_accepted += record.get("ocr_pages_accepted") or 0
        self.ocr_seconds += record.get("ocr_seconds") or 0.0
        self.pages_ocr += record.get("ocr_pages_in_output") or 0
        self.pages += record.get("pages") or 0
        self.pages_indexable += record.get("pages_indexable") or 0
        self.characters += record.get("chars") or 0
        self.bytes_read += record.get("bytes") or 0
        if record.get("ocr_action") in config.OCR_ACTIONS_BLOCKING_INDEX:
            self.documents_pending_ocr += 1
            self.pages_pending_ocr += record.get("ocr_estimated_pages") or 0


def format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:,.0f} B" if unit == "B" else f"{value:,.2f} {unit}"
        value /= 1024
    return f"{value:,.2f} TB"                        # pragma: no cover - unreachable


def render_progress(stats: RunStats) -> str:
    """The periodic progress block."""
    return "\n".join([
        f"[{stats.completed}/{stats.total}]",
        f"  successful    : {stats.successful:,}",
        f"  with OCR text : {stats.successful_with_ocr:,}",
        f"  quarantined   : {stats.quarantined:,}",
        f"  failed        : {stats.failed:,}",
        f"  skipped       : {stats.skipped:,}",
        f"  remaining     : {stats.remaining:,}",
        f"  still need OCR: {stats.documents_pending_ocr:,} documents "
        f"({stats.pages_pending_ocr:,} pages) — routed to OCR, reading not adopted",
        f"  pages         : {stats.pages:,} ({stats.pages_indexable:,} indexable)",
        f"  characters    : {stats.characters:,}",
        f"  elapsed       : {format_duration(stats.elapsed)}",
        f"  rate          : {stats.rate * 60:.1f} docs/min",
        f"  ETA           : {format_duration(stats.eta_seconds)}",
    ])


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


# --- Reporting ------------------------------------------------------------------


def corpus_context(data_dir: Path, corpus: Corpus) -> dict:
    """Counts that describe the corpus rather than the run.

    ``inventory_records`` and ``unavailable_upstream`` come from the ingestion
    phase's own artefacts. When one is absent the field is ``None`` — the
    invariants forbid guessing, and an inferred 55 would be a guess.
    """
    data_dir = Path(data_dir)
    inventory_records: Optional[int] = None
    inventory_file = (
        data_dir / ingestion_config.DISCOVERY_SUBDIR / ingestion_config.INVENTORY_FILENAME
    )
    if inventory_file.exists():
        try:
            payload = json.loads(inventory_file.read_text(encoding="utf-8"))
            records = payload.get("documents", []) if isinstance(payload, dict) else payload
            inventory_records = len(records)
        except (OSError, json.JSONDecodeError):
            log.warning(
                "Could not read %s; inventory_records reported as null.", inventory_file)

    unavailable: Optional[int] = None
    status_file = data_dir / "download_status.jsonl"
    if status_file.exists():
        try:
            unavailable = 0
            with open(status_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("final_state") == "UNAVAILABLE_UPSTREAM":
                        unavailable += 1
        except (OSError, json.JSONDecodeError):
            unavailable = None
            log.warning(
                "Could not read %s; unavailable_upstream reported as null.", status_file)

    return {
        "inventory_records": inventory_records,
        "manifest_documents": len(corpus.documents),
        "pdfs_on_disk": sum(1 for d in corpus.documents if d.pdf_path.exists()),
        "unavailable_upstream": unavailable,
    }


def build_report(
    journal: StatusJournal,
    stats: RunStats,
    *,
    workers: int,
    backend_name: str,
    verify_level: str,
    context: Optional[dict] = None,
    plan: Optional[WorkPlan] = None,
) -> dict:
    """Aggregate the corpus's state from the journal, streaming.

    Two things are reported and they are not the same: ``this_run`` is what this
    invocation did, and ``corpus_totals`` is the state of the whole corpus after
    it — which on a resumed run includes every document an earlier invocation
    finished. The second is read back from the journal rather than kept in
    memory, one line at a time, holding only the projection in
    :data:`_PROJECTED_FIELDS` per document.
    """
    current: dict[str, dict] = {}
    for row in journal.iter_records():
        current[row["document_id"]] = project(row)

    by_status: Counter = Counter()
    by_category: Counter = Counter()
    by_category_status: dict[str, Counter] = {}
    by_ocr_action: Counter = Counter()
    by_text_source: Counter = Counter()
    by_quality: Counter = Counter()
    by_language: Counter = Counter()
    by_pdf_type: Counter = Counter()
    by_extraction_status: Counter = Counter()
    by_structure: Counter = Counter()
    quarantine_by_reason: Counter = Counter()
    errors: dict[str, dict] = {}
    failure_rows: list[dict] = []
    pages = characters = 0
    pages_indexable = 0
    pages_indexable_documents = 0
    pages_pending_ocr = documents_pending_ocr = 0
    pages_ocr = ocr_pages_attempted = ocr_documents = 0
    eligible = 0
    processing_seconds = 0.0

    for row in current.values():
        status = row.get("status") or "UNKNOWN"
        by_status[status] += 1
        category = row.get("category") or "unknown"
        by_category[category] += 1
        by_category_status.setdefault(category, Counter())[status] += 1
        pages += row.get("pages") or 0
        # Counted only over the records that carry the field. A journal written
        # across a schema change holds both, and summing the absences as zero
        # would report a corpus-wide page count that is silently a subtotal.
        if "pages_indexable" in row:
            pages_indexable += row["pages_indexable"] or 0
            pages_indexable_documents += 1
        characters += row.get("chars") or 0
        processing_seconds += row.get("seconds") or 0.0
        if row.get("pdf_type"):
            by_pdf_type[row["pdf_type"]] += 1
        if row.get("text_extraction_status"):
            by_extraction_status[row["text_extraction_status"]] += 1
        if row.get("extraction_quality"):
            by_quality[row["extraction_quality"]] += 1
        if row.get("content_language"):
            by_language[row["content_language"]] += 1
        if row.get("structure_confidence"):
            by_structure[row["structure_confidence"]] += 1
        action = row.get("ocr_action")
        if action:
            by_ocr_action[action] += 1
            if action in config.OCR_ACTIONS_BLOCKING_INDEX:
                documents_pending_ocr += 1
                pages_pending_ocr += row.get("ocr_estimated_pages") or 0
        if row.get("ocr_text_source"):
            by_text_source[row["ocr_text_source"]] += 1
        # What OCR actually did, corpus-wide. Summed from the journal like every
        # other total here — it used to be hardcoded to 0, left over from when
        # the engine did not exist, and it reported 0 for a corpus with 33,677
        # accepted OCR pages in it.
        if row.get("ocr_executed"):
            ocr_documents += 1
            ocr_pages_attempted += row.get("ocr_pages_attempted") or 0
            pages_ocr += row.get("ocr_pages_accepted") or 0
        if row.get("eligible_for_indexing"):
            eligible += 1
        for reason in row.get("quarantine_reasons") or []:
            quarantine_by_reason[reason] += 1
        if status == "FAILED":
            error_type = row.get("error_type") or "Unknown"
            bucket = errors.setdefault(error_type, {"count": 0, "documents": []})
            bucket["count"] += 1
            if len(failure_rows) < config.RUN_REPORT_FAILURE_SAMPLE:
                failure = {
                    "document_id": row.get("document_id"),
                    "title": row.get("title"),
                    "document_type": row.get("document_type"),
                    "pdf_relpath": row.get("pdf_relpath"),
                    "error_type": error_type,
                    "error_message": row.get("error_message"),
                    "error_stage": row.get("error_stage"),
                    "attempt": row.get("attempt"),
                    "timestamp": row.get("processed_at"),
                }
                failure_rows.append(failure)
                bucket["documents"].append(row.get("document_id"))
        elif status == "SKIPPED":
            quarantine_by_reason[f"skipped:{row.get('skip_reason') or 'unknown'}"] += 1

    context = context or {}
    manifest_documents = context.get("manifest_documents")
    never_attempted = None
    if isinstance(manifest_documents, int):
        never_attempted = max(0, manifest_documents - len(current))

    duration = stats.elapsed
    return {
        "schema_version": config.PROCESSING_REPORT_SCHEMA_VERSION,
        "processing_schema_version": config.PROCESSING_SCHEMA_VERSION,
        "journal_schema_version": config.JOURNAL_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "backend": backend_name,
        "note": (
            "'pages_ocr' counts pages whose stored reading came from the OCR "
            "engine; 'pages_pending_ocr' counts pages processing.ocr routed to "
            "OCR whose reading was not adopted — because the run was launched "
            "with --no-ocr, because no engine was available, or because the "
            "engine's reading did not improve on the text layer. Neither number "
            "says the readings are correct: OCR acceptance has never been "
            "reviewed by a human (KNOWN_ISSUES C1)."
        ),
        "corpus": {
            "inventory_records": context.get("inventory_records"),
            "manifest_documents": manifest_documents,
            "pdfs_on_disk": context.get("pdfs_on_disk"),
            "unavailable_upstream": context.get("unavailable_upstream"),
        },
        "this_run": {
            "started_at": stats.started_at_iso,
            "finished_at": utcnow_iso(),
            "duration_seconds": round(duration, 2),
            "duration": format_duration(duration),
            "workers": workers,
            "verify_level": verify_level,
            "interrupted": stats.interrupted,
            "selected": stats.total,
            "attempted": stats.completed,
            "successful": stats.successful,
            "successful_with_ocr": stats.successful_with_ocr,
            "quarantined": stats.quarantined,
            "failed": stats.failed,
            "timed_out": stats.timed_out,
            "skipped": stats.skipped,
            "already_complete": len(plan.already_complete) if plan else None,
            "held_back_failed": len(plan.held_back) if plan else None,
            "deferred_by_limit": plan.deferred_by_limit if plan else None,
            "pages_processed": stats.pages,
            "pages_indexable": stats.pages_indexable,
            "pages_ocr": stats.pages_ocr,
            "ocr_documents": stats.ocr_documents,
            "ocr_pages_accepted": stats.ocr_pages_accepted,
            "ocr_seconds": round(stats.ocr_seconds, 1),
            "characters": stats.characters,
            "bytes_read": stats.bytes_read,
            "documents_per_hour": round(stats.rate * 3600, 1),
            "seconds_per_document": round(1 / stats.rate, 3) if stats.rate else None,
            "errors_by_type": dict(stats.errors_by_type),
        },
        "corpus_totals": {
            "documents_in_journal": len(current),
            "never_attempted": never_attempted,
            "pages_processed": pages,
            # The count that matters downstream: pages, not documents, are what
            # chunking and retrieval consume. Reported with the number of
            # documents it was measured over, because it is a subtotal until
            # every record carries the field, and a bare number would read as a
            # corpus total.
            "pages_indexable": pages_indexable,
            "pages_indexable_measured_over_documents": pages_indexable_documents,
            "pages_ocr": pages_ocr,
            "ocr_pages_attempted": ocr_pages_attempted,
            "ocr_documents": ocr_documents,
            "total_extracted_characters": characters,
            "total_processing_seconds": round(processing_seconds, 2),
            "documents_pending_ocr": documents_pending_ocr,
            "pages_pending_ocr": pages_pending_ocr,
        },
        "eligibility": {
            "eligible_for_indexing": eligible,
            "quarantined_by_reason": dict(quarantine_by_reason.most_common()),
        },
        "documents_by_status": dict(by_status.most_common()),
        "documents_by_category": dict(by_category.most_common()),
        "documents_by_category_and_status": {
            category: dict(counts.most_common())
            for category, counts in sorted(by_category_status.items())
        },
        "documents_by_pdf_type": dict(by_pdf_type.most_common()),
        "documents_by_text_extraction_status": dict(by_extraction_status.most_common()),
        "documents_by_ocr_action": dict(by_ocr_action.most_common()),
        "documents_by_ocr_text_source": dict(by_text_source.most_common()),
        "documents_by_quality_status": dict(by_quality.most_common()),
        "documents_by_language_status": dict(by_language.most_common()),
        "documents_by_structure_confidence": dict(by_structure.most_common()),
        "errors_by_type": {
            name: bucket["count"] for name, bucket in sorted(errors.items())
        },
        "failures": failure_rows,
        "artifacts": {
            "journal": config.PROCESSING_STATUS_FILENAME,
            "output": str(config.PROCESSED_SUBDIR).replace("\\", "/") + "/<document_id>/",
        },
    }


def write_report(data_dir: Path, report: dict) -> Path:
    path = report_path(data_dir)
    atomic_write_text(path, json.dumps(report, indent=2, ensure_ascii=False))
    return path


def render_summary(report: dict) -> str:
    corpus = report["corpus"]
    this_run = report["this_run"]
    totals = report["corpus_totals"]
    eligibility = report["eligibility"]
    return "\n".join([
        "",
        "============= FULL-CORPUS PROCESSING =============",
        f"  manifest documents      : {_count(corpus['manifest_documents'])}",
        f"  PDFs on disk            : {_count(corpus['pdfs_on_disk'])}",
        "  ----------------------------- this run ---------",
        f"  selected                : {this_run['selected']:>7,}",
        f"  attempted               : {this_run['attempted']:>7,}",
        f"  successful              : {this_run['successful']:>7,}",
        f"  successful with OCR     : {this_run['successful_with_ocr']:>7,}",
        f"  quarantined             : {this_run['quarantined']:>7,}",
        f"  failed                  : {this_run['failed']:>7,}",
        f"  skipped                 : {this_run['skipped']:>7,}",
        f"  already complete        : {_count(this_run['already_complete'])}",
        f"  pages processed         : {this_run['pages_processed']:>7,}",
        f"  duration                : {this_run['duration']:>7}",
        f"  rate                    : {this_run['documents_per_hour']:>7,.0f} docs/hour",
        "  --------------------------- whole corpus -------",
        f"  documents in journal    : {totals['documents_in_journal']:>7,}",
        f"  never attempted         : {_count(totals['never_attempted'])}",
        f"  pages processed         : {totals['pages_processed']:>7,}",
        f"  pages indexable         : {totals['pages_indexable']:>7,}"
        + (f"  (over {totals['pages_indexable_measured_over_documents']:,} of "
           f"{totals['documents_in_journal']:,} documents)"
           if totals['pages_indexable_measured_over_documents']
           != totals['documents_in_journal'] else ""),
        f"  characters extracted    : {totals['total_extracted_characters']:>7,}",
        f"  eligible for indexing   : {eligibility['eligible_for_indexing']:>7,}",
        f"  pages read by OCR       : {totals['pages_ocr']:>7,}"
        + (f"  (of {totals['ocr_pages_attempted']:,} attempted over "
           f"{totals['ocr_documents']:,} documents)"
           if totals.get('ocr_pages_attempted') else ""),
        f"  still need OCR          : {totals['documents_pending_ocr']:>7,} "
        f"({totals['pages_pending_ocr']:,} pages, reading not adopted)",
        "==================================================",
    ])


def _count(value) -> str:
    return "unknown" if value is None else f"{value:>7,}"


def _ocr_line(run_ocr: bool) -> str:
    """What this run will do about OCR, checked rather than assumed.

    A dry run that promised OCR the machine cannot perform would be worse than
    one that said nothing, so the engine is probed here exactly as ``run()``
    probes it.
    """
    if not run_ocr:
        return ("  OCR will NOT run (--no-ocr). Documents needing it are routed "
                "and quarantined, not read.")
    available, why = ocr_engine.engine_available()
    if not available:
        return (f"  OCR was requested but no engine is available ({why}). "
                "Documents needing it will be quarantined as before.")
    return (f"  OCR will run ({config.OCR_ENGINE_NAME}, at most "
            f"{config.OCR_MAX_PAGES_PER_DOCUMENT} pages per document) over pages "
            "extraction could not read.")


def render_dry_run(plan: WorkPlan, context: dict, *, data_dir: Path,
                   workers: int, verify: str, min_free_bytes: int,
                   run_ocr: bool = False) -> str:
    """What a run would do, and roughly what it would cost. Writes nothing."""
    # Extrapolated from the 100-document benchmark two ways, because the two
    # disagree and the honest answer is a range. Per document it is 6.14 s at 4
    # workers; per byte it is far cheaper, because the benchmark's stratified
    # sample averages 4.4 MB/document against a corpus average of 2.1 MB. The
    # truth is somewhere between: extraction cost is neither purely per-document
    # nor purely per-byte.
    scale = 4 / max(1, workers)
    by_count = len(plan.todo) * _BENCHMARK_SECONDS_PER_DOCUMENT * scale
    by_bytes = plan.bytes_to_read() / _BENCHMARK_BYTES_PER_SECOND * scale
    low, high = sorted((by_count, by_bytes))
    output_bytes = len(plan.todo) * _BENCHMARK_OUTPUT_BYTES_PER_DOCUMENT
    free = free_bytes(Path(data_dir)) if Path(data_dir).exists() else 0
    lines = [
        "",
        "================= DRY RUN (nothing was written) =================",
        f"  data directory          : {Path(data_dir)}",
        f"  verification level      : {verify}",
        "  CORPUS",
        f"    inventory records     : {_plain(context.get('inventory_records'))}",
        f"    manifest documents    : {plan.manifest_documents:,}",
        f"    PDFs on disk          : {plan.pdfs_on_disk:,}",
        f"    manifest without PDF  : {len(plan.missing_pdf):,}",
        f"    unavailable upstream  : {_plain(context.get('unavailable_upstream'))}"
        "   (never processed; excluded by absence from the manifest)",
        "  WORK",
        f"    already complete      : {len(plan.already_complete):,}",
        f"    to process            : {len(plan.todo):,}",
        f"    to record as SKIPPED  : {len(plan.missing_pdf):,}  (PDF not on disk)",
        f"    failed, not retried   : {len(plan.held_back):,}  (--skip-failed)",
        f"    excluded by filters   : {plan.filtered_out:,}",
        f"    deferred by --limit   : {plan.deferred_by_limit:,}",
    ]
    if plan.todo:
        lines.append("  WHY EACH DOCUMENT IS QUEUED")
        for reason, count in sorted(plan.reasons().items(), key=lambda kv: -kv[1]):
            lines.append(f"    {count:>7,}  {reason}")
    lines += [
        "  EXPECTED OUTPUT",
        "    location              : "
        f"{(Path(data_dir) / config.PROCESSED_SUBDIR).as_posix()}/<document_id>/",
        f"    files per document    : {config.DOCUMENT_FILENAME}, {config.PAGES_FILENAME}",
        f"    journal               : {journal_path(data_dir)}",
        f"    report                : {report_path(data_dir)}",
        "  ESTIMATED WORKLOAD (extrapolated from the 100-document benchmark)",
        f"    PDF bytes to read     : {format_bytes(plan.bytes_to_read())} (read-only)",
        f"    output to write       : ~{format_bytes(output_bytes)}",
        f"    duration at {workers} workers : ~{format_duration(low)} - "
        f"{format_duration(high)}",
        f"    free space available  : {format_bytes(free)}",
        f"    free space required   : {format_bytes(min_free_bytes)}",
        _ocr_line(run_ocr),
        "=================================================================",
    ]
    return "\n".join(lines)


def _plain(value) -> str:
    return "unknown" if value is None else f"{value:,}"


#: 614.4 s for 100 documents at 4 workers (data/benchmark/pdf_extraction/report.json).
_BENCHMARK_SECONDS_PER_DOCUMENT = 6.144
#: 48.1 MB of processed output for those same 100 documents.
_BENCHMARK_OUTPUT_BYTES_PER_DOCUMENT = int(48.1 * 1024 * 1024 / 100)
#: Those 100 documents were 4.4 MB each on average, so 440 MB in 614.4 s.
_BENCHMARK_BYTES_PER_SECOND = 440 * 1024 ** 2 / 614.4


# --- Doing the work -------------------------------------------------------------


def process_one(
    document: CorpusDocument,
    data_dir: Path,
    *,
    backend: Optional[PdfBackend] = None,
    detect_tables: bool = True,
    attempt: int = 1,
    run_ocr: bool = False,
) -> dict:
    """Process one document and return its journal record. Never raises.

    :func:`processing.process.process_document` already turns a per-document
    problem into a result rather than an exception. What it cannot cover is the
    runner's own write step — a full disk, a permission error, a path the
    filesystem rejects — so that is wrapped too. Either way one bad document
    costs one record, not the run.
    """
    started = time.monotonic()
    try:
        result = process_document(
            document, data_dir, backend=backend,
            detect_tables=detect_tables, write=False, run_ocr=run_ocr,
        )
    except Exception as exc:                         # anything the backend raises
        log.warning("%s raised out of process_document: %s: %s",
                    document.document_id, type(exc).__name__, exc)
        result = ProcessedDocument(document=document)
        result.error_type = type(exc).__name__
        result.error_message = str(exc)
        result.seconds = time.monotonic() - started
        return result_record(
            document, result, "FAILED", error_stage="unexpected",
            traceback_text=traceback_module.format_exc(), attempt=attempt,
        )

    if not result.ok:
        return result_record(
            document, result, "FAILED", error_stage="extract", attempt=attempt)

    try:
        directory, sizes = write_document_output(result, data_dir)
    except Exception as exc:                         # a full disk, a bad path…
        log.warning("%s: could not write output: %s: %s",
                    document.document_id, type(exc).__name__, exc)
        result.ok = False
        result.error_type = type(exc).__name__
        result.error_message = f"writing output failed: {exc}"
        result.seconds = time.monotonic() - started
        return result_record(
            document, result, "FAILED", error_stage="write",
            traceback_text=traceback_module.format_exc(), attempt=attempt,
        )

    result.seconds = time.monotonic() - started
    relpath = str(directory.relative_to(Path(data_dir))).replace("\\", "/")
    return result_record(
        document, result, classify(result),
        output_bytes=sizes, output_relpath=relpath, attempt=attempt,
    )


def execute(
    plan: WorkPlan,
    data_dir: Path,
    journal: StatusJournal,
    *,
    backend: Optional[PdfBackend] = None,
    detect_tables: bool = True,
    workers: int = config.RUN_DEFAULT_WORKERS,
    progress_every: int = config.RUN_PROGRESS_EVERY,
    on_progress: Optional[Callable[[RunStats], None]] = None,
    stop: Optional[threading.Event] = None,
    min_headroom_bytes: int = config.RUN_MIN_HEADROOM_BYTES,
    document_timeout: Optional[float] = config.RUN_DOCUMENT_TIMEOUT_SECONDS,
    poll_seconds: float = config.RUN_POLL_SECONDS,
    run_ocr: bool = False,
) -> RunStats:
    """Process every document in *plan*, journalling each outcome as it lands.

    No result object outlives the loop iteration that produced it: the journal is
    the record, the counters are the summary, and nothing accumulates.

    One thread claims one document. The journal is the only shared writer and it
    serialises its own appends, so two workers can never interleave a line.

    A document that overruns *document_timeout* is journalled ``FAILED`` and
    dropped from the wait set, so an unattended run cannot be held up
    indefinitely by one PDF. Its thread is abandoned rather than killed — see
    :func:`timeout_record` — which is why the executor is shut down without
    waiting once that has happened: the report must reach disk whether or not
    the stray thread ever finishes.
    """
    stop = stop or threading.Event()
    stats = RunStats(total=len(plan.todo) + len(plan.missing_pdf))
    data_dir = Path(data_dir)
    # A timeout can only be as sharp as the interval that checks it. Left
    # unclamped, a poll longer than the timeout means the check never runs
    # before the document finishes and the setting silently does nothing.
    if document_timeout and document_timeout > 0:
        poll_seconds = min(poll_seconds, document_timeout / 2)
    poll_seconds = max(0.01, poll_seconds)

    for document in plan.missing_pdf:
        record = skipped_record(document, "pdf_not_on_disk")
        journal.append(record)
        stats.record(record)
        log.warning("%s is in the manifest but its PDF is not on disk; SKIPPED.",
                    document.document_id)

    if not plan.todo:
        return stats

    claimed: set[str] = set()
    claim_lock = threading.Lock()
    progress_lock = threading.Lock()
    # When each document's thread actually began. A document still queued has
    # not started, and timing it out for the queue's sake would abandon work
    # that never had a chance to run.
    began: dict[str, float] = {}

    def work(item: WorkItem) -> Optional[dict]:
        if stop.is_set():
            return None
        with claim_lock:
            if item.document.document_id in claimed:
                log.warning("%s was claimed twice; the second claim is ignored.",
                            item.document.document_id)
                return None
            claimed.add(item.document.document_id)
            began[item.document.document_id] = time.monotonic()
        return process_one(
            item.document, data_dir, backend=backend,
            detect_tables=detect_tables, attempt=item.attempt, run_ocr=run_ocr,
        )

    executor = ThreadPoolExecutor(max_workers=max(1, workers))
    abandoned = 0
    try:
        futures = {executor.submit(work, item): item for item in plan.todo}
        pending = set(futures)
        try:
            while pending:
                finished, pending = wait(
                    pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
                for future in finished:
                    record = future.result()
                    if record is None:
                        continue
                    journal.append(record)
                    with progress_lock:
                        stats.record(record)
                        done = stats.completed
                    if progress_every and (done % progress_every == 0):
                        (on_progress or _log_progress)(stats)
                    if free_bytes(data_dir) < min_headroom_bytes:
                        stats.interrupted = True
                        stop.set()
                        log.error(
                            "Less than %s free on the data drive; stopping. Free "
                            "space and re-run the same command to continue.",
                            format_bytes(min_headroom_bytes),
                        )
                if not document_timeout or document_timeout <= 0:
                    continue
                now = time.monotonic()
                for future in list(pending):
                    item = futures[future]
                    with claim_lock:
                        started_at = began.get(item.document.document_id)
                    if started_at is None or now - started_at <= document_timeout:
                        continue
                    pending.discard(future)
                    abandoned += 1
                    record = timeout_record(
                        item.document, now - started_at, attempt=item.attempt)
                    journal.append(record)
                    with progress_lock:
                        stats.record(record)
                        stats.timed_out += 1
                    log.error(
                        "%s has run for %s without finishing; recording it FAILED "
                        "and moving on. Its thread is abandoned, not stopped.",
                        item.document.document_id,
                        format_duration(now - started_at),
                    )
        except KeyboardInterrupt:
            stats.interrupted = True
            stop.set()
            log.warning(
                "Interrupted. Letting running documents finish; %d of %d done. "
                "Re-run the same command to resume.", stats.completed, stats.total)
    finally:
        if abandoned:
            # Waiting here would hand the run straight back to the document the
            # timeout existed to escape, and the report would never be written.
            log.warning(
                "%d document(s) were abandoned and may still be running; not "
                "waiting for them. The journal and the report are complete.",
                abandoned)
            executor.shutdown(wait=False, cancel_futures=True)
        else:
            executor.shutdown(wait=True)
    return stats


def _log_progress(stats: RunStats) -> None:
    log.info("\n%s", render_progress(stats))


# --- The whole run --------------------------------------------------------------


def run(
    data_dir: Path,
    *,
    workers: int = config.RUN_DEFAULT_WORKERS,
    limit: Optional[int] = None,
    verify: str = config.DEFAULT_VERIFY_LEVEL,
    categories: Optional[Iterable[str]] = None,
    document_types: Optional[Iterable[str]] = None,
    only: Optional[Iterable[str]] = None,
    retry_failed: bool = True,
    force: bool = False,
    force_unlock: bool = False,
    detect_tables: bool = True,
    progress_every: int = config.RUN_PROGRESS_EVERY,
    min_free_bytes: int = config.RUN_MIN_FREE_BYTES,
    min_headroom_bytes: int = config.RUN_MIN_HEADROOM_BYTES,
    document_timeout: Optional[float] = config.RUN_DOCUMENT_TIMEOUT_SECONDS,
    run_ocr: bool = False,
    backend: Optional[PdfBackend] = None,
    write_report_file: bool = True,
) -> dict:
    """Plan, process and report. The whole run, minus the CLI.

    The lock is held for the planning as well as the processing: a second runner
    that planned while this one worked would decide the same documents were
    outstanding.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    free = free_bytes(data_dir)
    if free < min_free_bytes:
        raise InsufficientSpaceError(
            f"Only {format_bytes(free)} free on the data drive; this run wants at "
            f"least {format_bytes(min_free_bytes)}. Free space, or lower the "
            "requirement with --min-free-gb if you know the run is small."
        )

    backend = backend or default_backend()
    lock = RunLock(lock_path(data_dir))
    lock.acquire(force=force_unlock)
    try:
        corpus = Corpus.load(data_dir)
        journal = StatusJournal(journal_path(data_dir))
        state = journal.load()
        engine_missing = False
        if run_ocr:
            available, why = ocr_engine.engine_available()
            if not available:
                # Said once, at the start, rather than 19,802 times. The run is
                # still worth doing: everything except OCR still happens.
                log.warning(
                    "OCR was requested but no engine is available (%s). "
                    "Documents needing OCR will be processed and quarantined as "
                    "before. Install Tesseract, or pass --no-ocr to say so "
                    "deliberately.", why)
                run_ocr = False
                engine_missing = True
        plan = plan_run(
            corpus, state, verify=verify, limit=limit, categories=categories,
            document_types=document_types, only=only, retry_failed=retry_failed,
            force=force, attempts=journal.attempt_counts(), ocr_enabled=run_ocr,
        )
        if engine_missing:
            # Processing without an engine is a fair first pass. Reprocessing a
            # document that already holds accepted OCR text is not: it replaces a
            # reading that beat extraction on evidence with the reading it beat,
            # and the record afterwards looks like an ordinary complete document.
            # Nothing downstream could tell the difference, so the run stops here
            # rather than quietly undoing work.
            at_risk = [item.document.document_id for item in plan.todo
                       if (state.get(item.document.document_id) or {})
                       .get("ocr_pages_accepted")]
            if at_risk:
                raise ProcessingError(
                    f"{len(at_risk)} of the {len(plan.todo)} selected documents "
                    f"already hold OCR text that was accepted over their "
                    f"extracted text ({', '.join(at_risk[:3])}"
                    f"{', ...' if len(at_risk) > 3 else ''}), and no OCR engine "
                    f"is available now ({why}). Reprocessing them would discard "
                    "those readings. Install Tesseract and run again, or pass "
                    "--no-ocr to accept the loss deliberately."
                )
        log.info("%d documents to process, %d already complete, %d without a PDF.",
                 len(plan.todo), len(plan.already_complete), len(plan.missing_pdf))
        stats = execute(
            plan, data_dir, journal, backend=backend, detect_tables=detect_tables,
            workers=workers, progress_every=progress_every,
            min_headroom_bytes=min_headroom_bytes,
            document_timeout=document_timeout, run_ocr=run_ocr,
        )
        report = build_report(
            journal, stats, workers=workers, backend_name=backend.name,
            verify_level=verify, context=corpus_context(data_dir, corpus), plan=plan,
        )
    finally:
        lock.release()
    if write_report_file:
        write_report(data_dir, report)
    return report


def dry_run(
    data_dir: Path,
    *,
    limit: Optional[int] = None,
    verify: str = config.DEFAULT_VERIFY_LEVEL,
    categories: Optional[Iterable[str]] = None,
    document_types: Optional[Iterable[str]] = None,
    only: Optional[Iterable[str]] = None,
    retry_failed: bool = True,
    force: bool = False,
    run_ocr: bool = False,
) -> tuple[WorkPlan, dict]:
    """Inspect and estimate. Touches nothing: no journal, no output, no lock.

    Deliberately no lock either — inspecting a corpus while a run works on it is
    harmless, and a dry run that could be blocked by a stale lock would be worse
    than useless when the stale lock is the thing being diagnosed.
    """
    data_dir = Path(data_dir)
    corpus = Corpus.load(data_dir)
    journal = StatusJournal(journal_path(data_dir))
    plan = plan_run(
        corpus, journal.load(), verify=verify, limit=limit, categories=categories,
        document_types=document_types, only=only, retry_failed=retry_failed,
        force=force, ocr_enabled=run_ocr and ocr_engine.engine_available()[0],
    )
    return plan, corpus_context(data_dir, corpus)


# --- CLI ------------------------------------------------------------------------


def read_document_ids(path: Path) -> list[str]:
    """One document id per line. Blank lines and ``#`` comments are ignored.

    Exists because a stratified subset of the corpus cannot be expressed any
    other way: ``--limit N`` takes the first N documents in ``document_id``
    order, which is deterministic but alphabetical rather than representative,
    and a thousand ``--only`` arguments do not fit on a command line. Pair it
    with :func:`processing.sample.select` to run a proportional pilot.
    """
    path = Path(path)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProcessingError(f"Cannot read document ids from {path}: {exc}") from exc
    ids = []
    seen = set()
    for line in lines:
        entry = line.split("#", 1)[0].strip()
        if not entry or entry in seen:
            continue
        seen.add(entry)
        ids.append(entry)
    if not ids:
        raise ProcessingError(f"{path} lists no document ids.")
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m processing.run",
        description=(
            "Process the whole downloaded India Code corpus into "
            "data/processed/indiacode/. Resumable: re-running the same command "
            "after an interruption continues from where it stopped. Reads "
            "data/raw/ read-only and never modifies it. Does not run OCR."
        ),
        epilog=(
            "Examples:\n"
            "  python -m processing.run --dry-run\n"
            "  python -m processing.run --limit 5 --workers 2\n"
            "  python -m processing.run --workers 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path,
                        default=ingestion_config.DEFAULT_DATA_DIR,
                        help="Root data directory (default: ./data).")
    parser.add_argument("--workers", type=int, default=config.RUN_DEFAULT_WORKERS,
                        metavar="N",
                        help=f"Concurrent documents (default: {config.RUN_DEFAULT_WORKERS}).")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="Process at most N outstanding documents, then stop. "
                             "For integration runs; the rest stay outstanding.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Inspect the corpus and report the work. Writes "
                             "nothing: no output, no journal, no report.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess every selected document, including ones "
                             "the journal records as complete.")
    parser.add_argument("--verify", choices=config.VERIFY_LEVELS,
                        default=config.DEFAULT_VERIFY_LEVEL,
                        help="How hard to check output a journal record claims "
                             f"exists (default: {config.DEFAULT_VERIFY_LEVEL}). "
                             "'full' re-reads every pages.json.")
    parser.add_argument("--category", action="append", metavar="CATEGORY",
                        choices=list(ingestion_config.CATEGORIES),
                        help="Restrict to a corpus category. Repeatable.")
    parser.add_argument("--type", action="append", metavar="DOCUMENT_TYPE",
                        dest="document_types",
                        choices=sorted(set(ingestion_config.CATEGORY_TO_DOCUMENT_TYPE.values())),
                        help="Restrict to a document type (central_act, state_act, "
                             "rule, regulation). Repeatable.")
    parser.add_argument("--only", action="append", metavar="DOCUMENT_ID",
                        help="Restrict to one document id. Repeatable.")
    parser.add_argument("--only-from", type=Path, metavar="FILE",
                        dest="only_from",
                        help="Restrict to the document ids listed in FILE, one "
                             "per line ('#' comments and blank lines ignored). "
                             "Combines with --only. Use it to run a stratified "
                             "subset that --limit cannot express.")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Do not retry documents that previously FAILED "
                             "(default: they are retried).")
    parser.add_argument("--no-tables", action="store_true",
                        help="Skip table detection (faster; tables unreported).")
    ocr_group = parser.add_mutually_exclusive_group()
    ocr_group.add_argument("--ocr", dest="run_ocr", action="store_true",
                           default=config.OCR_ENABLED_DEFAULT,
                           help="Run OCR over pages extraction could not read "
                                "(default). Needs Tesseract installed.")
    ocr_group.add_argument("--no-ocr", dest="run_ocr", action="store_false",
                           help="Route documents to OCR but do not perform it. "
                                "Faster; documents needing OCR stay quarantined.")
    parser.add_argument("--progress-every", type=int,
                        default=config.RUN_PROGRESS_EVERY, metavar="N",
                        help="Documents between progress blocks (default: "
                             f"{config.RUN_PROGRESS_EVERY}). 0 silences them.")
    parser.add_argument("--document-timeout", type=float, metavar="SECONDS",
                        default=config.RUN_DOCUMENT_TIMEOUT_SECONDS,
                        help="Stop waiting for a document after this long, record "
                             "it FAILED and carry on (default: "
                             f"{config.RUN_DOCUMENT_TIMEOUT_SECONDS:.0f}). Its "
                             "thread is abandoned, not killed. 0 waits forever.")
    parser.add_argument("--min-free-gb", type=float,
                        default=config.RUN_MIN_FREE_BYTES / 1024 ** 3, metavar="GB",
                        help="Refuse to start with less free space than this "
                             f"(default: {config.RUN_MIN_FREE_BYTES / 1024 ** 3:.0f}).")
    parser.add_argument("--force-unlock", action="store_true",
                        help="Break a stale run lock left by a killed process. "
                             "Never use it while another runner is alive.")
    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("-v", "--verbose", action="store_true",
                               help="Debug logging.")
    logging_group.add_argument("-q", "--quiet", action="store_true",
                               help="Warnings and errors only.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose, args.quiet)
    min_free_bytes = int(args.min_free_gb * 1024 ** 3)

    try:
        only = list(args.only or [])
        if args.only_from:
            only.extend(read_document_ids(args.only_from))
            log.info("%d document ids selected from %s.",
                     len(only), args.only_from)
        only = only or None
    except ProcessingError as exc:
        log.error("%s", exc)
        return 2

    try:
        if args.dry_run:
            plan, context = dry_run(
                args.data_dir, limit=args.limit, verify=args.verify,
                categories=args.category, document_types=args.document_types,
                only=only, retry_failed=not args.skip_failed, force=args.force,
                run_ocr=args.run_ocr,
            )
            print(render_dry_run(
                plan, context, data_dir=args.data_dir, workers=args.workers,
                verify=args.verify, min_free_bytes=min_free_bytes,
                run_ocr=args.run_ocr,
            ))
            return 0

        report = run(
            args.data_dir,
            workers=args.workers, limit=args.limit, verify=args.verify,
            categories=args.category, document_types=args.document_types,
            only=only, retry_failed=not args.skip_failed, force=args.force,
            force_unlock=args.force_unlock, detect_tables=not args.no_tables,
            progress_every=args.progress_every, min_free_bytes=min_free_bytes,
            document_timeout=args.document_timeout, run_ocr=args.run_ocr,
        )
    except (RunLockError, InsufficientSpaceError) as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.error("Interrupted before any document was processed.")
        return 130

    print(render_summary(report))
    print(f"\nJournal: {journal_path(args.data_dir)}")
    print(f"Report:  {report_path(args.data_dir)}")
    if report["this_run"]["interrupted"]:
        print("\nThe run was interrupted. Re-run the same command to resume.")
        return 130
    return 1 if report["this_run"]["failed"] else 0


if __name__ == "__main__":                           # pragma: no cover
    raise SystemExit(main())
