"""Running OCR over a page, and deciding whether to believe the result.

:mod:`processing.ocr` decides *that* a document needs OCR. This module is what
actually reads a page, and it is deliberately separate from that decision: the
routing is cheap and runs over every document, while this is expensive and runs
over a minority of pages.

Three rules shape everything here.

**The page text on disk is never replaced.** ``pages.json`` keeps byte-for-byte
what the extraction backend produced, for the same reason the raw PDFs are
immutable: it is the only evidence of what went wrong, and an OCR regression a
year from now has to be diagnosable. Engine output is stored *beside* it and
:attr:`PageText.text_source` names which one later stages should read.

**OCR must earn its place.** A scanned gazette with a bad OCR layer is the most
dangerous document in this corpus — text is present, nothing looks broken, and
the words are wrong. Replacing that text with different wrong text is not an
improvement, it is a second unverifiable claim. So the engine's output is
measured against the text it would replace, on the same panel the quality gate
uses, and is kept only when it is measurably better. **OCR can move a document
out of quarantine; it can never move one in.**

**Rotation happens before reading, or not at all.** An OCR engine does not fail
loudly on a sideways page: it returns confident text that is wrong. The turn is
decided from the pixels by Tesseract's orientation detection, because a text
layer cannot say which of the four rotations a page needs, and it is applied
only when that detection is confident — see
:func:`processing.orientation.upright_image`.

Tesseract is the engine, chosen by measurement against RapidOCR and EasyOCR in
:mod:`processing.ocr_eval` (most text recovered, most clause markers, most pages
reading as English, 3.5-14x faster). Do not replace it without another
evaluation.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import config, language, orientation, quality, respace

log = logging.getLogger(__name__)


# --- Rasterisation ---------------------------------------------------------------
#
# The PdfBackend protocol covers text extraction and says nothing about turning a
# page into pixels, so these reach PyMuPDF directly. Kept here rather than
# duplicated in ocr_eval.py and validation.py, which is where they used to live.


def budgeted_dpi(
    width_points: float,
    height_points: float,
    dpi: int = config.OCR_DPI,
    *,
    max_megapixels: float = config.OCR_MAX_MEGAPIXELS,
    min_dpi: int = config.OCR_MIN_DPI,
) -> int:
    """The resolution a page of this size may be rendered at.

    Ordinary pages come back unchanged; an oversized one is lowered just far
    enough to fit the pixel budget, and never below ``min_dpi`` or above the dpi
    the caller asked for. Pure arithmetic on the page's size in points, so the
    budget is testable without a PDF.

    A page so large that even ``min_dpi`` overruns the budget is rendered at
    ``min_dpi``: losing orientation detection is worse than a large image. See
    :data:`processing.config.OCR_MAX_MEGAPIXELS`.
    """
    if width_points <= 0 or height_points <= 0 or dpi <= 0:
        return dpi
    megapixels = (
        (width_points / 72.0 * dpi) * (height_points / 72.0 * dpi) / 1_000_000.0
    )
    if megapixels <= max_megapixels:
        return dpi
    # Pixel count scales with the square of the dpi, so the shrink factor is the
    # square root of how far over budget the page is.
    lowered = int(dpi * math.sqrt(max_megapixels / megapixels))
    return min(dpi, max(min_dpi, lowered))


def render_page(pdf_path: Path, page_number: int, dpi: int = config.OCR_DPI) -> bytes:
    """Render one 1-based page to PNG bytes, inside the pixel budget."""
    import pymupdf                                          # noqa: PLC0415

    document = pymupdf.open(pdf_path)
    try:
        index = max(0, min(page_number - 1, document.page_count - 1))
        page = document.load_page(index)
        rect = page.rect
        effective_dpi = budgeted_dpi(
            rect.width, rect.height, dpi,
            max_megapixels=config.OCR_MAX_MEGAPIXELS,
            min_dpi=config.OCR_MIN_DPI,
        )
        if effective_dpi != dpi:
            log.debug(
                "%s page %s measures %.0fx%.0f pt: rendering at %d dpi rather "
                "than %d to stay inside the %.0f MP budget",
                pdf_path.name, page_number, rect.width, rect.height,
                effective_dpi, dpi, config.OCR_MAX_MEGAPIXELS,
            )
        return page.get_pixmap(dpi=effective_dpi).tobytes("png")
    finally:
        document.close()


def render_page_upright(
    pdf_path: Path, page_number: int, dpi: int = config.OCR_DPI
) -> tuple[bytes, "orientation.ImageOrientation"]:
    """Render a page and turn it the right way up before any engine sees it."""
    return orientation.upright_image(render_page(pdf_path, page_number, dpi))


# --- The engine ------------------------------------------------------------------


def tesseract_reader() -> Callable[[bytes], str]:
    """A callable that reads PNG bytes and returns text. Raises if unavailable.

    ``pytesseract`` only looks on ``PATH`` and the Windows installer does not put
    Tesseract there, which is how the first engine evaluation reported the
    standard document-OCR engine as missing. The binary is located the same way
    :mod:`processing.orientation` locates it, so the two always agree about
    whether Tesseract exists on this machine.
    """
    import io                                               # noqa: PLC0415

    import pytesseract                                      # noqa: PLC0415
    from PIL import Image                                   # noqa: PLC0415

    binary = orientation.tesseract_binary()
    if binary is None:
        raise FileNotFoundError(
            "no tesseract executable on PATH or in a standard install location"
        )
    pytesseract.pytesseract.tesseract_cmd = binary
    pytesseract.get_tesseract_version()                     # raises if unusable

    def read(image: bytes) -> str:
        # --psm 3 (fully automatic page segmentation, no OSD) is the right mode
        # for a page of statute: it finds columns and blocks, which a gazette
        # reproduction needs, and it does not re-decide orientation — that is
        # settled before the image gets here.
        return pytesseract.image_to_string(
            Image.open(io.BytesIO(image)), lang=config.OCR_LANGUAGE,
            config="--psm 3",
        )

    return read


def engine_available() -> tuple[bool, str]:
    """Whether OCR can run here, and why not when it cannot.

    Reported rather than raised. "Tesseract is not installed" is a finding about
    the machine, and a corpus run must be able to say so once instead of failing
    19,802 times.
    """
    try:
        tesseract_reader()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


# --- Which pages are worth reading -----------------------------------------------


def page_trigger(page) -> Optional[str]:
    """Why this page needs OCR, or ``None`` if it does not.

    Every predicate here was already computed during extraction; none is new.
    That is the point — the pipeline has always known which pages were broken and
    has never done anything about it.
    """
    if getattr(page, "is_vector_outlined", False):
        return "vector_outlined"
    if getattr(page, "is_sideways", False):
        return "sideways"
    if getattr(page, "has_inconsistent_direction", False):
        return "inconsistent_direction"
    if not getattr(page, "has_text", False):
        return "no_text_layer"
    if getattr(page, "text_quality_suspect", False):
        return "text_quality_suspect"
    return None


def should_skip(page) -> Optional[str]:
    """Why this page must not be OCR'd even though it was triggered.

    A page already established as another script *and carrying text* is skipped:
    its language is known, and re-reading it in English produces the same
    unusable text more slowly. A page with no text is not skipped, because
    nothing is established about a page that yielded nothing.
    """
    verdict = (getattr(page, "language", None) or {}).get("verdict")
    if verdict == "non_en" and getattr(page, "has_text", False):
        return "page is in another script"
    return None


# --- Believing the result --------------------------------------------------------


@dataclass
class Judgement:
    """Whether the engine's output should replace what extraction produced."""

    accepted: bool
    reason: str
    signals: dict = field(default_factory=dict)


def _panel_score(text: str) -> float:
    """The quality panel's score for a block of text, on its own.

    :func:`processing.quality.assess` needs pages and a structure; this needs to
    compare two strings. ``assess`` accepts a ``text`` override for exactly this,
    so the comparison is made on the same panel the eligibility gate uses rather
    than on a second, differently-calibrated measure.
    """
    return quality.assess([], text=text).score


def judge(backend_text: str, ocr_text: str) -> Judgement:
    """Compare the engine's reading against the text already on the page.

    Deliberately conservative and deliberately asymmetric. Extraction output is
    the incumbent and OCR has to beat it on the evidence, because the failure
    this is guarding against is not "OCR was not tried" — it is "OCR replaced a
    sound text layer with a plausible-looking wrong one", which is undetectable
    downstream and produces confident false statements of law.
    """
    ocr_stripped = (ocr_text or "").strip()
    backend_stripped = (backend_text or "").strip()

    if not ocr_stripped:
        return Judgement(False, "the engine returned nothing")
    if backend_stripped and not ocr_stripped:
        return Judgement(False, "the engine returned nothing where text existed")

    if not backend_stripped:
        # Nothing to lose: any readable text beats an empty page. This is the
        # scan-without-a-text-layer case, and it is the easy one.
        signals = {"ocr_score": round(_panel_score(ocr_stripped), 4)}
        return Judgement(
            True, "the page had no text layer and the engine produced text",
            signals)

    before = _panel_score(backend_stripped)
    after = _panel_score(ocr_stripped)
    english_before = language.assess_text(backend_stripped).signals.get(
        "function_word_rate", 0.0)
    english_after = language.assess_text(ocr_stripped).signals.get(
        "function_word_rate", 0.0)
    signals = {
        "backend_score": round(before, 4),
        "ocr_score": round(after, 4),
        "backend_function_word_rate": round(english_before, 4),
        "ocr_function_word_rate": round(english_after, 4),
    }

    if after <= before:
        return Judgement(
            False,
            f"quality panel {before:.2f} -> {after:.2f}: the engine did not "
            "improve on the existing text",
            signals)
    if english_after < english_before:
        # The panel can be satisfied by text that is shaped like English without
        # being English. Recovered words are the point; losing them while the
        # shape improves is the failure this catches.
        return Judgement(
            False,
            f"quality panel improved {before:.2f} -> {after:.2f} but English "
            f"function words fell {english_before:.1%} -> {english_after:.1%}",
            signals)
    return Judgement(
        True,
        f"quality panel {before:.2f} -> {after:.2f}, English function words "
        f"{english_before:.1%} -> {english_after:.1%}",
        signals)


# --- Reading one page ------------------------------------------------------------


def ocr_page(
    pdf_path: Path,
    page,
    *,
    reader: Callable[[bytes], str],
    dpi: int = config.OCR_DPI,
    trigger: str = "",
) -> dict:
    """Render, straighten, read and judge one page. Never raises.

    A page that cannot be read is recorded as an attempt that failed, because a
    corpus run has to report what it could not do rather than stop.
    """
    started = time.monotonic()
    record = {
        "attempted": True,
        "trigger": trigger,
        "engine": config.OCR_ENGINE_NAME,
        "dpi": dpi,
        "preprocessing": [],
        "text": None,
        "accepted": False,
        "reason": "",
        "seconds": 0.0,
    }
    try:
        image, detected = orientation.upright_image(
            render_page(pdf_path, page.page_number, dpi))
        if getattr(detected, "known", False) and detected.rotate_degrees:
            record["preprocessing"].append(f"rotate_upright:{detected.rotate_degrees}")
        record["orientation"] = detected.to_dict() if detected else None

        raw = reader(image)
        # Only ever inserts spaces, so the stored text stays auditable against
        # what the engine actually read.
        repaired = respace.respace(raw)
        text = repaired.text
        if repaired.spaces_inserted:
            record["preprocessing"].append(f"respace:{repaired.spaces_inserted}")

        verdict = judge(page.text, text)
        record["text"] = text
        record["accepted"] = verdict.accepted
        record["reason"] = verdict.reason
        record["signals"] = verdict.signals
    except Exception as exc:                                # engine, PIL, PyMuPDF
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["reason"] = f"OCR failed: {record['error']}"
        log.warning("OCR failed on page %s of %s: %s",
                    getattr(page, "page_number", "?"), pdf_path, record["error"])
    record["seconds"] = round(time.monotonic() - started, 3)
    return record


# --- Reading a document ----------------------------------------------------------


@dataclass
class OcrRun:
    """What the OCR stage did to one document."""

    executed: bool = False
    attempted: int = 0
    accepted: int = 0
    seconds: float = 0.0
    truncated: bool = False
    unavailable_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "executed": self.executed,
            "pages_attempted": self.attempted,
            "pages_accepted": self.accepted,
            "seconds": round(self.seconds, 3),
            "truncated": self.truncated,
            "unavailable_reason": self.unavailable_reason,
        }


def ocr_document(
    extraction,
    pdf_path: Path,
    *,
    reader: Optional[Callable[[bytes], str]] = None,
    dpi: int = config.OCR_DPI,
    max_pages: int = config.OCR_MAX_PAGES_PER_DOCUMENT,
    document_language: Optional[str] = None,
) -> OcrRun:
    """Run OCR over the pages of *extraction* that need it, in place.

    Mutates each page's ``ocr`` and ``text_source``; **never** its ``text``.
    Returns what was done, so the runner can journal it without re-deriving it.
    """
    run = OcrRun()
    if document_language == "non_en":
        run.unavailable_reason = "document is not English; OCR would not help"
        return run

    candidates = [
        (page, trigger) for page, trigger in
        ((p, page_trigger(p)) for p in extraction.pages)
        if trigger and not should_skip(page)
    ]
    if not candidates:
        return run

    if reader is None:
        try:
            reader = tesseract_reader()
        except Exception as exc:
            run.unavailable_reason = f"{type(exc).__name__}: {exc}"
            log.warning("OCR unavailable for %s: %s", pdf_path, run.unavailable_reason)
            return run

    if len(candidates) > max_pages:
        # Deferring is honest; letting one 764-page gazette consume a night is
        # not. The document stays quarantined and stays retryable.
        run.truncated = True
        candidates = candidates[:max_pages]

    run.executed = True
    for page, trigger in candidates:
        record = ocr_page(pdf_path, page, reader=reader, dpi=dpi, trigger=trigger)
        page.ocr = record
        run.attempted += 1
        run.seconds += record.get("seconds", 0.0)
        if record.get("accepted"):
            page.text_source = "ocr"
            run.accepted += 1
    return run


def page_text(page) -> str:
    """The text a later stage should read for this page.

    The one place that knows how ``text`` and ``ocr.text`` relate. Everything
    downstream asks here rather than deciding for itself, so there is a single
    answer to "which reading of this page is authoritative".
    """
    if getattr(page, "text_source", "backend") == "ocr":
        record = getattr(page, "ocr", None) or {}
        text = record.get("text")
        if text:
            return text
    return getattr(page, "text", "") or ""
