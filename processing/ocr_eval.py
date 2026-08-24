"""OCR engine evaluation on a small, deterministic set of problem pages.

The project spec forbids OCRing the corpus, and picking an engine by reputation would
be exactly the kind of unevidenced decision this phase exists to avoid. So this
module renders **about 25 pages** — the ones the benchmark says need OCR — runs
every engine available in the current Python environment over them, and measures
what comes out against the text layer already in the PDF.

What is compared
----------------
``embedded`` is the baseline: whatever text the PDF already carries. For a
scanned gazette with a bad OCR layer that is the thing a new engine would have
to beat; for a page with no text layer at all it is empty, and any output is an
improvement.

Engines are discovered, not assumed. An engine that is not installed is reported
as unavailable with the reason, rather than silently skipped — the absence of a
candidate is itself a result, and the most standard document-OCR engine
(Tesseract) is precisely one that cannot be installed from PyPI.

What is measured
----------------
completeness, English-language quality, preservation of section numbers, of
subsection/clause markers and of punctuation, page alignment, time and output
size. Legal text is not prose: a run that reads the words correctly but drops
every ``(2)(a)`` marker has destroyed the citation structure, and a run that
loses spaces has destroyed retrievability. Both fail differently and both need
their own number.

Nothing here is written back into the corpus. This produces a report.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from ingestion import config as ingestion_config
from ingestion.utils import atomic_write_text, sha256_bytes, utcnow_iso

from . import __version__, config, language, orientation, quality, respace
from .corpus import Corpus
from .errors import ProcessingError

#: Suffix marking an engine's output after :mod:`processing.respace` has put the
#: dropped word boundaries back. Both rows are reported for every engine, so the
#: repair is measured rather than assumed — including on ``embedded``, where the
#: text is already well spaced and the useful result is that nothing changes.
RESPACE_SUFFIX = "+respace"

log = logging.getLogger("processing.ocr_eval")

EVAL_SUBDIR = Path("benchmark") / "ocr_evaluation"
REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"
SAMPLES_DIRNAME = "samples"

#: Pages to evaluate. Small on purpose: this is an engine comparison, not a
#: processing run, and every engine costs 10–30 seconds per page on CPU.
DEFAULT_PAGE_COUNT = 25

_SECTION_MARKER = re.compile(r"(?m)^\s*\d{1,3}[A-Z]?\s*\.")
_SUBSECTION_MARKER = re.compile(r"\(\s*\d{1,3}[A-Z]?\s*\)")
_CLAUSE_MARKER = re.compile(r"\(\s*[a-z]{1,3}\s*\)")
_LEGAL_PUNCTUATION = set(".,;:()[]—–-")
_WORD = re.compile(r"[A-Za-z]+")


# --- Page selection --------------------------------------------------------------


@dataclass
class PageSpec:
    """One page to put through every engine."""

    document_id: str
    category: str
    title: Optional[str]
    pdf_relpath: str
    page_number: int
    reason: str                       # why this page is in the evaluation
    pdf_type: str = ""
    quality: str = ""


def select_pages(
    report: dict, corpus: Corpus, *, count: int = DEFAULT_PAGE_COUNT
) -> list[PageSpec]:
    """Choose the evaluation pages from the benchmark report, deterministically.

    Coverage is by *failure mode* rather than by document: the point is to learn
    where each engine breaks, so the set deliberately spans scanned legal text,
    an existing bad OCR layer, mixed documents, vector-outlined pages, pages
    with tables, and old and modern typesetting. Ordering within each mode is by
    ``sha256(document_id)`` so the set is reproducible.
    """
    rows = [r for r in report.get("documents", []) if r.get("ok")]
    by_id = {d.document_id: d for d in corpus}

    def rank(row: dict) -> str:
        return sha256_bytes(row["document_id"].encode("utf-8"))

    buckets: list[tuple[str, list[dict]]] = [
        ("vector_outlined page (no text layer, no image)",
         [r for r in rows if r.get("pdf_type") == "vector_outlined"]),
        ("scanned page with no text layer at all",
         [r for r in rows if r.get("text_extraction_status") == "requires_ocr"]),
        ("scanned page whose existing OCR layer is bad",
         [r for r in rows if r.get("pdf_type") in ("scanned", "mixed")
          and r.get("extraction_quality") == "bad"]),
        ("mixed document, partial text layer",
         [r for r in rows if r.get("pdf_type") == "mixed"]),
        ("page with a detected table",
         [r for r in rows if r.get("tables")]),
        ("old law (pre-1950) on a scan",
         [r for r in rows if (r.get("year") or 9999) < 1950
          and r.get("pdf_type") in ("scanned", "mixed")]),
        ("modern law (2011+) on a scan",
         [r for r in rows if (r.get("year") or 0) >= 2011
          and r.get("pdf_type") in ("scanned", "mixed")]),
        ("questionable text layer",
         [r for r in rows if r.get("extraction_quality") == "questionable"]),
    ]

    chosen: list[PageSpec] = []
    seen: set[tuple[str, int]] = set()
    exhausted = False
    while len(chosen) < count and not exhausted:
        exhausted = True
        for reason, candidates in buckets:
            if len(chosen) >= count:
                break
            for row in sorted(candidates, key=rank):
                document = by_id.get(row["document_id"])
                if document is None:
                    continue
                page_number = _representative_page(row)
                key = (row["document_id"], page_number)
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(PageSpec(
                    document_id=row["document_id"],
                    category=row["category"],
                    title=row.get("title"),
                    pdf_relpath=document.pdf_relpath,
                    page_number=page_number,
                    reason=reason,
                    pdf_type=row.get("pdf_type", ""),
                    quality=row.get("extraction_quality", ""),
                ))
                exhausted = False
                break
    return chosen[:count]


def _representative_page(row: dict) -> int:
    """A page from the middle of the document.

    The first page is a cover or a masthead in a large share of this corpus and
    would flatter every engine equally; the middle of the document is where the
    legal text is.
    """
    pages = row.get("pages") or 1
    return max(1, pages // 2)


# --- Engines ---------------------------------------------------------------------


@dataclass
class Engine:
    """One OCR configuration under test."""

    name: str
    description: str
    dpi: int
    loader: Callable[[], Callable[[bytes], str]]
    available: bool = True
    unavailable_reason: str = ""
    _run: Optional[Callable[[bytes], str]] = field(default=None, repr=False)

    def prepare(self) -> None:
        if self._run is None:
            self._run = self.loader()

    def run(self, image: bytes) -> str:
        assert self._run is not None
        return self._run(image)


def _rapidocr_loader() -> Callable[[bytes], str]:
    from rapidocr_onnxruntime import RapidOCR              # noqa: PLC0415

    engine = RapidOCR()

    def run(image: bytes) -> str:
        result, _ = engine(image)
        if not result:
            return ""
        return "\n".join(line[1] for line in result)

    return run


def _easyocr_loader() -> Callable[[bytes], str]:
    import easyocr                                         # noqa: PLC0415

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def run(image: bytes) -> str:
        return "\n".join(reader.readtext(image, detail=0, paragraph=True))

    return run


def _pytesseract_loader() -> Callable[[bytes], str]:
    """Tesseract, found wherever it was installed.

    ``pytesseract`` only looks on ``PATH``, and the Windows installer does not
    put it there — which is how the first evaluation ended up reporting the
    standard document-OCR engine as unavailable. The binary is located the same
    way :mod:`processing.orientation` locates it, so the two agree about whether
    Tesseract exists on this machine.
    """
    import io                                              # noqa: PLC0415

    import pytesseract                                     # noqa: PLC0415
    from PIL import Image                                  # noqa: PLC0415

    binary = orientation.tesseract_binary()
    if binary is None:
        raise FileNotFoundError(
            "no tesseract executable on PATH or in a standard install location"
        )
    pytesseract.pytesseract.tesseract_cmd = binary
    pytesseract.get_tesseract_version()                    # raises if unusable

    def run(image: bytes) -> str:
        # --psm 3 (fully automatic page segmentation, no OSD) is the right mode
        # for a page of statute: it finds columns and blocks, which is what a
        # gazette reproduction needs, and it does not re-decide orientation —
        # that is settled before the image gets here.
        return pytesseract.image_to_string(
            Image.open(io.BytesIO(image)), lang="eng", config="--psm 3"
        )

    return run


def candidate_engines() -> list[Engine]:
    """Every engine configuration this evaluation knows how to try."""
    return [
        Engine("rapidocr@200", "RapidOCR (PP-OCRv4, ONNX Runtime, CPU) at 200 dpi",
               200, _rapidocr_loader),
        Engine("rapidocr@300", "RapidOCR (PP-OCRv4, ONNX Runtime, CPU) at 300 dpi",
               300, _rapidocr_loader),
        Engine("easyocr@200", "EasyOCR (CRAFT detector + CRNN recogniser, torch CPU) "
               "at 200 dpi", 200, _easyocr_loader),
        Engine("tesseract@300", "Tesseract 5 (LSTM, --psm 3) at 300 dpi", 300,
               _pytesseract_loader),
        Engine("tesseract@200", "Tesseract 5 (LSTM, --psm 3) at 200 dpi", 200,
               _pytesseract_loader),
    ]


def available_engines(engines: Optional[Iterable[Engine]] = None) -> list[Engine]:
    """Try to load each engine; record why any of them cannot run.

    An unavailable engine stays in the list. "Tesseract is the obvious choice
    and could not be evaluated because it needs a system-level install" is a
    finding, not an omission.
    """
    result = []
    for engine in engines or candidate_engines():
        try:
            engine.prepare()
        except Exception as exc:
            engine.available = False
            engine.unavailable_reason = f"{type(exc).__name__}: {exc}"
            log.warning("Engine %s unavailable: %s", engine.name,
                        engine.unavailable_reason)
        result.append(engine)
    return result


# --- Measurement -----------------------------------------------------------------


def measure(text: str) -> dict:
    """Everything measured about one page of output."""
    words = _WORD.findall(text)
    characters = len(text)
    punctuation = sum(1 for character in text if character in _LEGAL_PUNCTUATION)
    signals = quality.text_signals(text)
    return {
        "characters": characters,
        "words": len(words),
        "mean_word_length": signals["mean_word_length"],
        "single_char_rate": signals["single_char_rate"],
        "common_word_rate": signals["common_word_rate"],
        "mixed_case_rate": signals["mixed_case_rate"],
        "alpha_ratio": signals["alpha_ratio"],
        "space_ratio": round(text.count(" ") / characters, 4) if characters else 0.0,
        # Words run together are invisible to the space ratio once a page is
        # mostly one long token, so the rate of implausibly long tokens is
        # measured directly. Real statutory English runs about 2.5%.
        "glued_token_rate": respace.glued_token_rate(text),
        "punctuation_ratio": round(punctuation / characters, 4) if characters else 0.0,
        "section_markers": len(_SECTION_MARKER.findall(text)),
        "subsection_markers": len(_SUBSECTION_MARKER.findall(text)),
        "clause_markers": len(_CLAUSE_MARKER.findall(text)),
        "content_language": language.assess_text(text).content_language,
        "english_function_word_rate":
            language.assess_text(text).signals.get("function_word_rate", 0.0),
    }


def _page_index(document, page_number: int) -> int:
    """Clamp a 1-based page number to a page the document actually has.

    The page numbers come from a benchmark report, which can be older than the
    corpus. Clamping keeps the evaluation running on a real page instead of
    failing, and the page it lands on is still representative.
    """
    return max(0, min(page_number - 1, document.page_count - 1))


def render_page(pdf_path: Path, page_number: int, dpi: int) -> bytes:
    import pymupdf                                         # noqa: PLC0415

    document = pymupdf.open(pdf_path)
    try:
        page = document.load_page(_page_index(document, page_number))
        return page.get_pixmap(dpi=dpi).tobytes("png")
    finally:
        document.close()


def render_page_upright(
    pdf_path: Path, page_number: int, dpi: int
) -> tuple[bytes, orientation.ImageOrientation]:
    """Render a page and turn it the right way up before any engine sees it.

    OCR engines do not fail loudly on a sideways page — they return confident
    text that is wrong, which is the most expensive failure this pipeline can
    produce. The turn is decided from the pixels (Tesseract's OSD), because the
    text layer cannot say which of the four rotations a page needs, and it is
    applied only when the detection is confident.
    """
    return orientation.upright_image(render_page(pdf_path, page_number, dpi))


def embedded_text(pdf_path: Path, page_number: int) -> str:
    import pymupdf                                         # noqa: PLC0415

    document = pymupdf.open(pdf_path)
    try:
        page = document.load_page(_page_index(document, page_number))
        return page.get_text("text")
    finally:
        document.close()


def evaluate(
    specs: list[PageSpec],
    engines: list[Engine],
    data_dir: Path,
) -> dict:
    """Run every available engine over every page and collect the measurements."""
    results: list[dict] = []
    for index, spec in enumerate(specs, start=1):
        pdf_path = Path(data_dir) / spec.pdf_relpath
        baseline = embedded_text(pdf_path, spec.page_number)
        entry = {
            "document_id": spec.document_id,
            "title": spec.title,
            "category": spec.category,
            "page_number": spec.page_number,
            "reason": spec.reason,
            "pdf_type": spec.pdf_type,
            "existing_quality": spec.quality,
            "engines": {
                "embedded": {
                    "seconds": 0.0,
                    "text": baseline,
                    **measure(baseline),
                    "page_aligned": True,
                },
            },
        }
        rotations: dict[int, dict] = {}
        for engine in engines:
            if not engine.available:
                continue
            image, detected = render_page_upright(
                pdf_path, spec.page_number, engine.dpi)
            rotations[engine.dpi] = detected.to_dict()
            started = time.monotonic()
            try:
                text = engine.run(image)
                error = None
            except Exception as exc:                       # engines fail variously
                text, error = "", f"{type(exc).__name__}: {exc}"
            elapsed = time.monotonic() - started
            entry["engines"][engine.name] = {
                "seconds": round(elapsed, 2),
                "error": error,
                "text": text,
                # One rendered page in, one block of text out: page alignment is
                # structural here, and holds as long as the engine returned
                # something for the page it was given.
                "page_aligned": error is None,
                "image_bytes": len(image),
                "rotation_applied": detected.rotate_degrees if detected.known else None,
                **measure(text),
            }
        entry["orientation"] = rotations
        _add_respaced(entry)
        results.append(entry)
        log.info("  … %d/%d pages evaluated", index, len(specs))
    return {"pages": results}


def _add_respaced(entry: dict) -> None:
    """Add a repaired-spacing twin of every engine's output on this page.

    Free — it is text processing, not a second OCR pass — and it is what turns
    "RapidOCR loses spaces" from an observation into a measured before/after. The
    baseline gets one too: the repair must be shown not to damage text that was
    already correctly spaced.
    """
    for name in list(entry["engines"]):
        if name.endswith(RESPACE_SUFFIX):
            continue
        source = entry["engines"][name]
        repaired = respace.respace(source.get("text") or "")
        entry["engines"][f"{name}{RESPACE_SUFFIX}"] = {
            "seconds": source["seconds"],
            "error": source.get("error"),
            "text": repaired.text,
            "page_aligned": source.get("page_aligned", False),
            "rotation_applied": source.get("rotation_applied"),
            "respace": repaired.to_dict(example_limit=8),
            **measure(repaired.text),
        }


def summarise(evaluation: dict, engines: list[Engine]) -> dict:
    """Aggregate per-page measurements into a per-engine comparison."""
    pages = evaluation["pages"]
    base_names = ["embedded"] + [e.name for e in engines if e.available]
    names = [
        name
        for base in base_names
        for name in (base, f"{base}{RESPACE_SUFFIX}")
    ]
    summary: dict[str, dict] = {}

    for name in names:
        entries = [p["engines"][name] for p in pages if name in p["engines"]]
        if not entries:
            continue
        measured = [e for e in entries if not e.get("error")]
        total_characters = sum(e["characters"] for e in measured)
        summary[name] = {
            "pages": len(entries),
            "failures": len(entries) - len(measured),
            "total_characters": total_characters,
            "total_words": sum(e["words"] for e in measured),
            "mean_seconds_per_page": round(
                sum(e["seconds"] for e in measured) / len(measured), 2)
            if measured else 0.0,
            "pages_with_text": sum(1 for e in measured if e["words"] >= 20),
            "mean_word_length": _mean(measured, "mean_word_length"),
            "mean_common_word_rate": _mean(measured, "common_word_rate"),
            "mean_space_ratio": _mean(measured, "space_ratio"),
            "mean_glued_token_rate": _mean(measured, "glued_token_rate"),
            "mean_punctuation_ratio": _mean(measured, "punctuation_ratio"),
            "spaces_inserted": sum(
                (e.get("respace") or {}).get("spaces_inserted", 0) for e in measured),
            "section_markers": sum(e["section_markers"] for e in measured),
            "subsection_markers": sum(e["subsection_markers"] for e in measured),
            "clause_markers": sum(e["clause_markers"] for e in measured),
            "pages_reading_as_english": sum(
                1 for e in measured if e["content_language"] == "en"),
            "pages_aligned": sum(1 for e in entries if e.get("page_aligned")),
            "output_bytes": sum(
                len(e["text"].encode("utf-8")) for e in measured),
        }

    rotated = [
        {"document_id": p["document_id"], "page_number": p["page_number"],
         "rotations": {str(dpi): info["rotate_degrees"]
                       for dpi, info in (p.get("orientation") or {}).items()}}
        for p in pages
        if any((info or {}).get("rotate_degrees")
               for info in (p.get("orientation") or {}).values())
    ]

    return {
        "by_engine": summary,
        "orientation": {
            "detector": "tesseract_osd" if orientation.tesseract_binary()
                        else "unavailable",
            "pages_rotated_before_ocr": len(rotated),
            "pages": rotated,
            "note": (
                "Rotation is applied to the rendered image before any engine "
                "reads it. The `embedded` baseline is the PDF's own text layer "
                "and is not rotated — which is the point: it is what the "
                "corpus currently holds for these pages."
            ),
        },
        "unavailable_engines": [
            {"name": e.name, "description": e.description,
             "reason": e.unavailable_reason}
            for e in engines if not e.available
        ],
    }


def _mean(entries: list[dict], key: str) -> float:
    values = [e[key] for e in entries if e["words"] >= 20]
    return round(sum(values) / len(values), 4) if values else 0.0


# --- Reporting -------------------------------------------------------------------


def eval_dir(data_dir: Path) -> Path:
    return Path(data_dir) / EVAL_SUBDIR


def write_report(data_dir: Path, payload: dict) -> tuple[Path, Path]:
    directory = eval_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # The full page texts are large and belong beside the report, not in it.
    slim = json.loads(json.dumps(payload))
    for page in slim["evaluation"]["pages"]:
        for entry in page["engines"].values():
            entry["text_preview"] = entry.pop("text", "")[:400]
    json_path = directory / REPORT_JSON
    md_path = directory / REPORT_MARKDOWN
    atomic_write_text(json_path, json.dumps(slim, indent=2, ensure_ascii=False))
    atomic_write_text(md_path, render_markdown(payload))
    return json_path, md_path


def write_samples(data_dir: Path, payload: dict, limit: int = 6) -> Path:
    """Before/after text for a handful of pages, so the numbers can be checked."""
    directory = eval_dir(data_dir) / SAMPLES_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    for page in payload["evaluation"]["pages"][:limit]:
        lines = [
            f"# {page['title']}",
            "",
            f"- document: `{page['document_id']}`, page **{page['page_number']}**",
            f"- selected because: {page['reason']}",
            f"- pdf_type: `{page['pdf_type']}`, existing text quality: "
            f"`{page['existing_quality']}`",
            "",
        ]
        for name, entry in page["engines"].items():
            lines += [
                f"## {name}",
                "",
                f"{entry['characters']:,} characters, {entry['words']:,} words, "
                f"{entry['seconds']}s, reads as `{entry['content_language']}`",
                "",
                "```text",
                (entry.get("text") or "")[:1400].rstrip() or "(no output)",
                "```",
                "",
            ]
        name = f"{page['document_id'][:50]}-p{page['page_number']}.md"
        atomic_write_text(directory / name, "\n".join(lines))
    return directory


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]["by_engine"]
    pages = payload["evaluation"]["pages"]
    out = [
        "# OCR engine evaluation",
        "",
        f"Generated {payload['generated_at']} by {payload['processor']}.",
        "",
        f"**{len(pages)} pages**, chosen deterministically from the extraction "
        "benchmark as the pages that need OCR. No corpus documents were "
        "processed or modified.",
        "",
        "`embedded` is the baseline: the text layer already in the PDF. For a "
        "page with no text layer it is empty, and any output beats it; for a "
        "page with a bad OCR layer it is what a new engine has to beat.",
        "",
        "## Comparison",
        "",
        "| engine | s/page | chars | words | mean word len | English words | "
        "spaces | glued | punct. | §§ | (n) | (a) | pages as English | aligned |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
        "---: | ---: | ---: | ---: |",
    ]
    for name, stats in summary.items():
        out.append(
            f"| `{name}` | {stats['mean_seconds_per_page']} | "
            f"{stats['total_characters']:,} | {stats['total_words']:,} | "
            f"{stats['mean_word_length']} | {stats['mean_common_word_rate']:.0%} | "
            f"{stats['mean_space_ratio']:.1%} | "
            f"{stats['mean_glued_token_rate']:.1%} | "
            f"{stats['mean_punctuation_ratio']:.1%} | "
            f"{stats['section_markers']} | {stats['subsection_markers']} | "
            f"{stats['clause_markers']} | "
            f"{stats['pages_reading_as_english']}/{stats['pages']} | "
            f"{stats['pages_aligned']}/{stats['pages']} |"
        )

    out += [
        "",
        "Column meanings: **English words** is the share of tokens that are "
        "recognisable English legal vocabulary; **spaces** is the share of "
        "characters that are spaces and **glued** the share of tokens 12+ "
        "characters long, which together detect a recogniser that runs words "
        "together; **§§ / (n) / (a)** count section, subsection and clause "
        "markers recovered, because legal citation structure is destroyed "
        "independently of whether the words are right.",
        "",
        f"Rows ending `{RESPACE_SUFFIX}` are the same OCR pass with "
        "`processing.respace` applied — it inserts spaces and changes nothing "
        "else. The row to read it against is the one directly above it.",
        "",
    ]

    rotation = payload["summary"].get("orientation") or {}
    if rotation:
        out += [
            "## Orientation correction",
            "",
            f"- detector: `{rotation.get('detector')}`",
            f"- pages turned upright before OCR: "
            f"**{rotation.get('pages_rotated_before_ocr', 0)}** of {len(pages)}",
            "",
        ]
        for page in rotation.get("pages", []):
            out.append(
                f"  - `{page['document_id'][:44]}` page {page['page_number']} — "
                f"rotated `{page['rotations']}`"
            )
        out += ["", rotation.get("note", ""), ""]

    unavailable = payload["summary"]["unavailable_engines"]
    if unavailable:
        out += ["## Engines that could not be evaluated", ""]
        for entry in unavailable:
            out.append(f"- **{entry['name']}** — {entry['description']}  ")
            out.append(f"  `{entry['reason']}`")
        out.append("")

    out += ["## Pages evaluated", "",
            "| document | page | why | pdf_type | existing quality |",
            "| --- | ---: | --- | --- | --- |"]
    for page in pages:
        out.append(
            f"| `{page['document_id'][:44]}` | {page['page_number']} | "
            f"{page['reason']} | {page['pdf_type']} | {page['existing_quality']} |"
        )

    out += ["", "## Worked examples", "",
            f"Before/after text for the first pages is in "
            f"`{(EVAL_SUBDIR / SAMPLES_DIRNAME).as_posix()}/`.", ""]
    return "\n".join(out)


# --- CLI ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m processing.ocr_eval",
        description=(
            "Compare OCR engines on a small, deterministic set of pages the "
            "extraction benchmark identified as needing OCR. Processes only "
            "those pages; writes no corpus output."
        ),
    )
    parser.add_argument("--data-dir", type=Path,
                        default=ingestion_config.DEFAULT_DATA_DIR)
    parser.add_argument("--pages", type=int, default=DEFAULT_PAGE_COUNT,
                        help=f"Pages to evaluate (default: {DEFAULT_PAGE_COUNT}).")
    parser.add_argument("--engine", action="append", default=[], metavar="NAME",
                        help="Restrict to named engines (repeatable).")
    parser.add_argument("--list-engines", action="store_true",
                        help="Report which engines are installed and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    candidates = candidate_engines()
    if args.engine:
        wanted = set(args.engine)
        candidates = [e for e in candidates if e.name in wanted]
    engines = available_engines(candidates)

    if args.list_engines:
        for engine in engines:
            state = "available" if engine.available else engine.unavailable_reason
            print(f"{engine.name:<16} {state}")
        return 0

    report_path = (
        Path(args.data_dir) / config.BENCHMARK_SUBDIR / config.REPORT_JSON_FILENAME
    )
    if not report_path.exists():
        log.error("No benchmark report at %s; run python -m processing.benchmark "
                  "first.", report_path)
        return 2
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        corpus = Corpus.load(args.data_dir)
        specs = select_pages(report, corpus, count=args.pages)
        if not specs:
            log.error("No pages needing OCR were found in the benchmark report.")
            return 2
        log.info("Evaluating %d engine(s) over %d pages",
                 sum(1 for e in engines if e.available), len(specs))
        evaluation = evaluate(specs, engines, args.data_dir)
    except (ProcessingError, OSError) as exc:
        log.error("%s", exc)
        return 2

    payload = {
        "schema_version": config.PROCESSING_SCHEMA_VERSION,
        "generated_at": utcnow_iso(),
        "processor": f"processing v{__version__}",
        "evaluation": evaluation,
        "summary": summarise(evaluation, engines),
    }
    json_path, md_path = write_report(args.data_dir, payload)
    write_samples(args.data_dir, payload)
    print(f"\nOCR evaluation: {md_path}")
    print(f"JSON:           {json_path}")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
