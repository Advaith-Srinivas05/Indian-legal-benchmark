"""Page orientation: is this page the right way up, and if not, what does OCR need?

The problem, found on the benchmark sample
------------------------------------------
``the-rajasthan-legislative-assembly-secretariat-recruitment…`` page 19 is a
landscape service-conditions table printed sideways on a portrait page. The PDF
declares ``/Rotate 0``, so nothing in the file says anything is wrong, and half
the document's lines are set at 90° to the page. Its OCR layer was produced in
the other orientation, and what comes out of text extraction is this::

    Jo siseq oy3 uQ
    *p UWN[OO Uj PIQLID

That is "On the basis of" and "cribed in column 4." with the characters in
reverse order. It extracts as text, it counts as characters, and it is worthless
— the exact failure mode this phase exists to catch before anything is indexed.

What can and cannot be established from the text layer
------------------------------------------------------
Two different questions hide under "orientation", and only one of them is
answerable without looking at pixels.

**Which axis the text runs along** is answerable. Every extracted line carries a
writing direction, and a page whose lines run vertically on screen is sideways.
The one subtlety is ``/Rotate``: a viewer applies it before displaying the page,
so a 90°-rotated page whose lines are set vertically in file coordinates is
upright on screen. Rotating by 90 or 270 swaps the axis, and rotating by 180
does not — which is the whole of the correction needed, and is why
:func:`assess_page` is an axis test rather than a vector test.

**Which way up** is not answerable from the text layer, and this module does not
pretend otherwise. Upside-down pages were checked for on the sample and the
evidence contradicts itself: a document rotated 180° whose lines read left to
right in file coordinates renders perfectly upright, so the sign of the writing
direction does not survive the round trip through ``/Rotate`` in a way that can
be relied on. Deciding *which* 90° turn a sideways page needs, and whether an
image-only page is upside down, therefore belongs to :func:`detect_image_rotation`,
which reads the rendered pixels through Tesseract's orientation-and-script
detection. When Tesseract is not installed, the answer is ``unknown`` and it is
reported as unknown.

Does it work?
-------------
Measured on the page above. The text layer calls it sideways (95% of its lines
run vertically); Tesseract's OSD, given the page at 300 dpi, asks for 90°
clockwise with confidence 10.3; and OCR of the rotated page reads 35.1% English
legal vocabulary against 2.4% for the same page unrotated. Both halves of the
mechanism agree, and the correction is worth about fifteen times the recovered
vocabulary on that page.

The operational catch is resolution: OSD scores 10–17 at 200–300 dpi and 0.14 at
72 dpi. Under-rendered, it does not answer wrongly — it stops answering — which
is why :data:`~processing.config.ORIENTATION_MIN_OSD_CONFIDENCE` exists and why
a low-confidence reading leaves the page untouched.

Nothing here rewrites extracted text. A sideways page is *labelled* sideways and
routed to rasterise-and-OCR; the text already extracted from it stays on disk
exactly as the backend produced it, because it is the only evidence of what went
wrong.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

from . import config

#: What a page's text is doing.
ORIENTATIONS = ("upright", "sideways", "unknown")

#: Rotations a page may need, in degrees clockwise, to bring it upright.
ROTATIONS = (0, 90, 180, 270)

#: Where an install of Tesseract is looked for when it is not on ``PATH``.
#: Windows installers do not put it there, and an engine that is present but
#: unfindable would be reported as unavailable — which would be a false finding.
_TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)


@dataclass
class PageOrientation:
    """What the text layer says about one page's orientation."""

    page_number: int
    declared_rotation: int = 0
    orientation: str = "unknown"
    #: Share of measured lines whose text runs vertically *on screen*.
    sideways_line_ratio: float = 0.0
    measured_lines: int = 0
    #: True when the page needs rotating before an OCR engine can read it. Which
    #: way it needs rotating is not decided here — see :func:`detect_image_rotation`.
    needs_rotation: bool = False
    source: str = "text_line_geometry"
    #: Share of the page's horizontal lines whose writing direction points the
    #: *opposite* way to the rest of the document. See
    #: :func:`flag_direction_inconsistency` for what this catches.
    reverse_line_ratio: float = 0.0
    #: Lines this page contributed to that ratio. Zero means the page has an
    #: opinion about nothing — a scan, a blank, or a page set entirely sideways —
    #: and it must not vote on which direction the document runs in.
    horizontal_lines: int = 0
    #: Set by :func:`flag_direction_inconsistency`, which needs the whole
    #: document to have an opinion about which way is forward.
    direction_inconsistent: bool = False

    def to_dict(self) -> dict:
        return {
            "declared_rotation": self.declared_rotation,
            "orientation": self.orientation,
            "sideways_line_ratio": round(self.sideways_line_ratio, 4),
            "reverse_line_ratio": round(self.reverse_line_ratio, 4),
            "horizontal_lines": self.horizontal_lines,
            "direction_inconsistent": self.direction_inconsistent,
            "measured_lines": self.measured_lines,
            "needs_rotation": self.needs_rotation,
            "source": self.source,
        }


@dataclass
class ImageOrientation:
    """What the rendered pixels say. Only an OCR engine can answer this."""

    rotate_degrees: int = 0
    confidence: float = 0.0
    script: Optional[str] = None
    source: str = "unavailable"
    detail: str = ""

    @property
    def known(self) -> bool:
        return self.source != "unavailable"

    def to_dict(self) -> dict:
        return {
            "rotate_degrees": self.rotate_degrees,
            "confidence": round(self.confidence, 2),
            "script": self.script,
            "source": self.source,
            "detail": self.detail,
        }


# --- From the text layer ---------------------------------------------------------


def line_is_sideways(direction: Sequence[float], declared_rotation: int) -> bool:
    """Whether one line runs vertically **on screen**.

    *direction* is the writing direction the backend reported, in file
    coordinates. ``/Rotate 90`` and ``/Rotate 270`` swap the screen axis;
    ``/Rotate 0`` and ``/Rotate 180`` leave it alone.
    """
    dx, dy = float(direction[0]), float(direction[1])
    vertical_in_file = abs(dy) > abs(dx)
    axis_swapped = declared_rotation % 360 in (90, 270)
    return vertical_in_file != axis_swapped


def assess_page(
    page_number: int,
    declared_rotation: int,
    line_directions: Iterable[Sequence[float]],
) -> PageOrientation:
    """Judge one page's orientation from its line geometry."""
    directions = [d for d in line_directions if d is not None]
    measured = len(directions)
    if measured < config.ORIENTATION_MIN_LINES:
        # Too few lines to mean anything. A page with no text layer at all lands
        # here, which is correct: its orientation is a question for OCR, not for
        # this function.
        return PageOrientation(
            page_number=page_number,
            declared_rotation=declared_rotation % 360,
            orientation="unknown",
            measured_lines=measured,
            source="insufficient_lines",
        )

    sideways = sum(1 for d in directions if line_is_sideways(d, declared_rotation))
    ratio = sideways / measured
    horizontal = [d for d in directions if not line_is_sideways(d, declared_rotation)]
    reverse = sum(1 for d in horizontal if float(d[0]) < 0)
    reverse_ratio = reverse / len(horizontal) if horizontal else 0.0
    if ratio >= config.ORIENTATION_SIDEWAYS_LINE_RATIO:
        orientation = "sideways"
    elif ratio <= 1.0 - config.ORIENTATION_SIDEWAYS_LINE_RATIO:
        orientation = "upright"
    else:
        # A page genuinely half sideways — a rotated table beside upright prose.
        # Neither label is true of the whole page, so neither is applied.
        orientation = "unknown"

    return PageOrientation(
        page_number=page_number,
        declared_rotation=declared_rotation % 360,
        orientation=orientation,
        sideways_line_ratio=ratio,
        reverse_line_ratio=reverse_ratio,
        horizontal_lines=len(horizontal),
        measured_lines=measured,
        needs_rotation=orientation == "sideways",
    )


def flag_direction_inconsistency(
    orientations: Iterable[PageOrientation],
) -> list[PageOrientation]:
    """Mark pages whose text runs the opposite way to the rest of the document.

    The absolute sign of a writing direction cannot be trusted on its own —
    ``/Rotate`` is applied by the viewer and the sign does not survive that round
    trip in a way this code can rely on. Its *inconsistency within one document*
    can be, and it turned out to matter: 64 of the 309 pages of the Madhya
    Pradesh Goods and Services Tax Act, 2017 are set in Devanagari with a broken
    character map, and they extract as reversed Latin nonsense::

        ele Bl dee Eb Lie

    Those pages, and only those, report a reversed writing direction. The
    document's *average* language still reads as English, so the document-level
    language check clears it; this is the signal that says a fifth of it should
    not be believed.

    Only pages that actually contain horizontal text get a vote. A scan, a blank
    page or a page set entirely sideways has no opinion about which way the
    document runs, and letting two hundred image-only pages outvote the three
    text pages in a scanned gazette would flag those three on no evidence at all.
    """
    entries = list(orientations)
    voting = [o for o in entries if o.horizontal_lines]
    forward = sum(1 for o in voting if o.reverse_line_ratio < 0.5)
    reverse = len(voting) - forward
    if min(forward, reverse) == 0:
        return entries                      # a document consistent with itself
    minority_is_reverse = reverse <= forward
    for entry in voting:
        page_is_reverse = entry.reverse_line_ratio >= 0.5
        entry.direction_inconsistent = page_is_reverse == minority_is_reverse
    return entries


def summarise(orientations: Iterable[PageOrientation], page_count: int) -> dict:
    """Document-level counts, and whether the document as a whole is suspect."""
    entries = list(orientations)
    sideways = [o for o in entries if o.orientation == "sideways"]
    declared = [o for o in entries if o.declared_rotation]
    inconsistent = [o for o in entries if o.direction_inconsistent]
    ratio = len(sideways) / page_count if page_count else 0.0
    inconsistent_ratio = len(inconsistent) / page_count if page_count else 0.0
    return {
        "sideways_pages": len(sideways),
        "sideways_page_numbers": [o.page_number for o in sideways][:50],
        "sideways_page_ratio": round(ratio, 4),
        "pages_with_declared_rotation": len(declared),
        "declared_rotations": sorted({o.declared_rotation for o in declared}),
        "direction_inconsistent_pages": len(inconsistent),
        "direction_inconsistent_page_numbers": [
            o.page_number for o in inconsistent][:50],
        "direction_inconsistent_ratio": round(inconsistent_ratio, 4),
        "orientation_suspect": (
            ratio >= config.ORIENTATION_SUSPECT_PAGE_RATIO
            or inconsistent_ratio >= config.ORIENTATION_SUSPECT_PAGE_RATIO
        ),
        "thresholds": {
            "sideways_line_ratio": config.ORIENTATION_SIDEWAYS_LINE_RATIO,
            "suspect_page_ratio": config.ORIENTATION_SUSPECT_PAGE_RATIO,
            "min_lines": config.ORIENTATION_MIN_LINES,
        },
    }


# --- From the rendered pixels -----------------------------------------------------


def tesseract_binary() -> Optional[str]:
    """Locate a Tesseract executable, or ``None``."""
    found = shutil.which("tesseract")
    if found:
        return found
    for candidate in _TESSERACT_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    return None


def _configured_pytesseract():
    """``pytesseract`` with its binary path set, or ``None`` if unusable."""
    try:
        import pytesseract                                  # noqa: PLC0415
    except ImportError:
        return None
    binary = tesseract_binary()
    if binary is None:
        return None
    pytesseract.pytesseract.tesseract_cmd = binary
    return pytesseract


def detect_image_rotation(image: bytes) -> ImageOrientation:
    """Ask Tesseract's orientation-and-script detection which way up a page is.

    Returns the rotation **clockwise in degrees** that brings the page upright.
    This is the only reliable answer to "which of the four turns does this page
    need", and it needs an OCR engine — so when Tesseract is not installed the
    result says ``unavailable`` rather than defaulting to zero, because "no
    rotation needed" and "nobody looked" must not be the same value.
    """
    pytesseract = _configured_pytesseract()
    if pytesseract is None:
        return ImageOrientation(
            source="unavailable",
            detail="Tesseract (with osd.traineddata) is required for "
                   "orientation detection and was not found",
        )
    try:
        import io                                           # noqa: PLC0415

        from PIL import Image                               # noqa: PLC0415

        data = pytesseract.image_to_osd(
            Image.open(io.BytesIO(image)), output_type=pytesseract.Output.DICT
        )
    except Exception as exc:                                # OSD fails on blank pages
        return ImageOrientation(
            source="unavailable", detail=f"{type(exc).__name__}: {exc}"
        )
    return ImageOrientation(
        rotate_degrees=int(data.get("rotate", 0)) % 360,
        confidence=float(data.get("orientation_conf", 0.0)),
        script=data.get("script"),
        source="tesseract_osd",
    )


def rotate_image(image: bytes, degrees: int) -> bytes:
    """Rotate a PNG *clockwise* by 0/90/180/270 degrees."""
    degrees %= 360
    if degrees == 0:
        return image
    import io                                               # noqa: PLC0415

    from PIL import Image                                   # noqa: PLC0415

    with Image.open(io.BytesIO(image)) as picture:
        # PIL rotates counter-clockwise, so the angle is negated to make the
        # contract here "clockwise", which is what Tesseract's OSD reports.
        turned = picture.rotate(-degrees, expand=True)
        buffer = io.BytesIO()
        turned.save(buffer, format="PNG")
    return buffer.getvalue()


def upright_image(
    image: bytes, *, min_confidence: float = config.ORIENTATION_MIN_OSD_CONFIDENCE
) -> tuple[bytes, ImageOrientation]:
    """Return the page the right way up, plus what was detected.

    Below *min_confidence* the image is returned untouched: a low-confidence OSD
    reading on a sparse page is not better evidence than leaving the page alone,
    and rotating a page that did not need it corrupts an otherwise good OCR pass.
    """
    detected = detect_image_rotation(image)
    if not detected.known or detected.rotate_degrees == 0:
        return image, detected
    if detected.confidence < min_confidence:
        detected.detail = (
            f"detected {detected.rotate_degrees}° but confidence "
            f"{detected.confidence:.1f} is below {min_confidence}; left as is"
        )
        return image, detected
    return rotate_image(image, detected.rotate_degrees), detected
