"""The OCR decision: what should happen to a document's text, and why.

This module *decides*; it does not OCR. Running OCR over the corpus is a
separate, expensive step that the project spec keeps behind an explicit approval, and
the first thing needed to size that step is an honest count of what actually
requires it.

The decision follows the classification and the quality assessment::

    PDF
     └─ what is it?          text_based | scanned | mixed | vector_outlined
         └─ did text come out?      ok | partial | requires_ocr
             └─ is the text usable?    good | questionable | bad
                 └─ action

The distinction that matters most is inside ``scanned``. Three states hide
there, and the first benchmark collapsed them:

``scan_with_good_ocr``
    A scanned gazette that already carries a sound OCR layer. Re-OCRing it costs
    time and would probably make it worse. Use what is there.

``scan_with_bad_ocr``
    Text came out, so nothing looks broken, but it is not quotable. This is the
    dangerous class and the one worth spending OCR on.

``scan_without_ocr``
    No text layer at all. Obvious, cheap to detect, and the smallest of the
    three.

``vector_outlined`` is worse than any of them: the glyphs were converted to
outlines, so there is neither a text layer nor an image to read. It has to be
rasterised before OCR can see anything, which is why it gets its own action.

A **sideways** document outranks all of these, and is checked first. Such a page
is not a low-quality page — it is a page an engine will read confidently and
wrongly, and re-OCRing it as it stands would produce exactly the same worthless
text more slowly. The rotation is not an optimisation of the OCR step; it is the
precondition for it, which is why the decision carries it explicitly in
``preprocessing`` rather than leaving it to whatever runs the engine.

"Sideways document" is not the same as "document containing a sideways page".
The Code of Civil Procedure, 1908 has two landscape schedules among 347 pages;
that is ordinary typesetting, and sending a sound born-digital Central Act
through OCR because of it would be a downgrade as well as an expense. The action
therefore turns on the document being *substantially* sideways, while the two
pages are still recorded individually and named in the decision's reason.
"""

from __future__ import annotations

from typing import Optional

from . import config
from .models import ExtractedDocument, OcrDecision

#: What to do with the document's text.
ACTIONS = (
    "use_extracted_text",       # the text is good; nothing to do
    "review_text",              # usable but questionable; a human should look
    "ocr_recommended",          # text exists and is bad; OCR would improve it
    "ocr_required",             # no usable text at all
    "rasterize_then_ocr",       # vector-outlined: nothing to read without rendering
)

#: Where the current text came from, which is what decides whether re-OCRing
#: could help.
TEXT_SOURCES = (
    "born_digital",
    "scan_with_good_ocr",
    "scan_with_bad_ocr",
    "scan_without_ocr",
    "vector_outlined",
    "partial_text_layer",
)


def text_source(extraction: ExtractedDocument, quality) -> str:
    """Name where this document's text came from and what state it is in."""
    if extraction.pdf_type == "vector_outlined":
        return "vector_outlined"

    scanned = extraction.pdf_type in ("scanned", "mixed")
    if extraction.text_extraction_status == "requires_ocr":
        return "scan_without_ocr" if scanned else "scan_without_ocr"
    if extraction.text_extraction_status == "partial":
        return "partial_text_layer"

    if not scanned:
        return "born_digital"
    classification = getattr(quality, "classification", "questionable")
    return "scan_with_good_ocr" if classification == "good" else "scan_with_bad_ocr"


def _ocr_settled_partial(ocr_run, classification: str) -> bool:
    """Has OCR already answered the question ``partial`` was asking?

    ``partial`` does not say the text is bad. It says *some pages yielded no
    text*, and it recommends OCR to find out whether anything is there. When OCR
    has since run over exactly those pages and adopted **nothing**, that question
    has been answered: there is nothing on them to recover. Continuing to
    quarantine the whole document for pages that are provably unrecoverable —
    and that are already excluded from indexing one by one — blocks a good text
    layer on the strength of a question that is closed.

    Every condition here is load-bearing:

    ``executed`` and ``attempted``
        A run that never happened, or that found no page worth attempting, has
        answered nothing. Absence of OCR is not evidence about the pages.

    ``accepted == 0``
        The moment OCR contributes a single reading, the document's text is
        partly unreviewed OCR output, and KNOWN_ISSUES C1 applies — no human has
        checked OCR acceptance anywhere in this corpus. Those documents keep the
        flag. This is the line that stops the fix spreading from 823 documents to
        1,451, and it is drawn where the evidence stops, not where it would be
        convenient.

    ``not truncated``
        A run stopped by ``OCR_MAX_PAGES_PER_DOCUMENT`` never reached the later
        pages, so it cannot say anything about them.

    ``classification == "good"``
        Quality still gates. A ``bad`` document returned earlier; a
        ``questionable`` one must keep falling through to ``review_text``.

    Measured on the 2026-08-27 corpus run: 823 documents qualify. All are
    ``partial``, all ``good``, 709 of 823 ``text_based``, and **none** has a
    single OCR-sourced indexable page. Their COMMON_WORDS rate is 0.629 against
    0.622 for documents that were never quarantined — indistinguishable. See
    DECISIONS D26.
    """
    return bool(
        ocr_run is not None
        and getattr(ocr_run, "executed", False)
        and getattr(ocr_run, "attempted", 0) > 0
        and getattr(ocr_run, "accepted", 0) == 0
        and not getattr(ocr_run, "truncated", False)
        and classification == "good"
    )


def decide(
    extraction: ExtractedDocument,
    quality,
    *,
    language=None,
    ocr_run=None,
) -> OcrDecision:
    """Choose the action for one document.

    *language* is accepted so a document already known not to be English is not
    queued for OCR: re-reading a Gujarati scan in English would produce the same
    unusable text more slowly. It is quarantined instead, and OCR in the right
    language is a decision for whoever decides whether this corpus should hold
    non-English law at all.

    *ocr_run* is accepted for one narrow case, and only that one: a ``partial``
    document whose good text layer survived a completed OCR pass that adopted
    nothing. See :func:`_ocr_settled_partial`. Every other input to this
    function still describes the document as extraction first found it, which is
    deliberate — the classification and the quality panel are statements about
    the PDF, not about what OCR later did to it.
    """
    source = text_source(extraction, quality)
    pages = extraction.page_count
    classification = getattr(quality, "classification", "questionable")
    content_language = getattr(language, "content_language", None)
    sideways = extraction.sideways_page_count
    orientation_suspect = bool(extraction.orientation.get("orientation_suspect"))

    if sideways and orientation_suspect:
        # A sideways page is not a low-quality page: it is a page an engine will
        # read confidently and wrongly. It is therefore worth OCR *before*
        # anything else is considered, and it is worth nothing at all unless the
        # rotation happens first.
        #
        # Gated on the document being *substantially* sideways. A 347-page
        # Central Act with two landscape fold-out schedules is ordinary
        # typesetting, and routing the whole of it through OCR on that basis
        # would be both expensive and a downgrade — those two pages are recorded
        # individually, which is the right granularity to fix them at.
        return OcrDecision(
            action="rasterize_then_ocr",
            text_source=source,
            reason=(
                f"{sideways} of {pages} pages are set sideways; their existing "
                "text layer was produced in the wrong orientation and extracts "
                "as reversed characters. The pages must be rotated upright and "
                "re-OCR'd"
            ),
            priority=6,
            estimated_pages=sideways,
            preprocessing=["rotate_upright"],
        )
    sideways_note = (
        f" {sideways} of {pages} pages are set sideways and need rotating "
        "before they can be re-read, but the document as a whole does not."
        if sideways else ""
    )

    if content_language == "non_en":
        return OcrDecision(
            action="review_text",
            text_source=source,
            reason=(
                "content-level language check says this document is not English; "
                "it is quarantined rather than queued for OCR, because OCRing it "
                "again in English would reproduce the same unusable text"
            ) + sideways_note,
            priority=1,
            estimated_pages=0,
        )

    if source == "vector_outlined":
        return OcrDecision(
            action="rasterize_then_ocr",
            text_source=source,
            reason=(
                f"{extraction.vector_outlined_page_count} of {pages} pages draw "
                "their text as vector outlines: there is no text layer and no "
                "image, so the pages must be rendered before OCR can read them"
            ) + sideways_note,
            priority=5,
            estimated_pages=extraction.vector_outlined_page_count or pages,
        )

    if extraction.text_extraction_status == "requires_ocr":
        return OcrDecision(
            action="ocr_required",
            text_source=source,
            reason=(
                "no usable text layer "
                f"({extraction.classification_evidence.get('pages_with_text', 0)}"
                f"/{pages} pages had text)"
            ) + sideways_note,
            priority=4,
            estimated_pages=pages,
        )

    if source == "scan_with_bad_ocr" or classification == "bad":
        return OcrDecision(
            action="ocr_recommended",
            text_source=source,
            reason=(
                f"a text layer exists but its quality is {classification} "
                f"(score {getattr(quality, 'score', 0):.2f}); re-OCR is likely "
                "to improve it"
            ) + sideways_note,
            priority=3,
            estimated_pages=pages,
        )

    if extraction.text_extraction_status == "partial":
        missing = pages - extraction.classification_evidence.get("pages_with_text", 0)
        if _ocr_settled_partial(ocr_run, classification):
            return OcrDecision(
                action="use_extracted_text",
                text_source=source,
                reason=(
                    f"{missing} of {pages} pages yielded no text, OCR was run "
                    f"over {ocr_run.attempted} of them and none of its readings "
                    "improved on what was already there; the text layer that "
                    "remains passes every quality check, and the pages OCR "
                    "could not rescue are excluded individually"
                ) + sideways_note,
                priority=0,
                estimated_pages=0,
            )
        return OcrDecision(
            action="ocr_recommended",
            text_source=source,
            reason=f"{missing} of {pages} pages yielded no text" + sideways_note,
            priority=2,
            estimated_pages=missing,
        )

    if classification == "questionable":
        return OcrDecision(
            action="review_text",
            text_source=source,
            reason=(
                "the text is usable but the quality panel is not satisfied "
                f"(score {getattr(quality, 'score', 0):.2f}); a human should "
                "decide before this document is cited"
            ) + sideways_note,
            priority=1,
            estimated_pages=0,
        )

    return OcrDecision(
        action="use_extracted_text",
        text_source=source,
        reason="the extracted text passes every quality check" + sideways_note,
        priority=0,
        estimated_pages=0,
    )


def summarise(decisions: list[OcrDecision]) -> dict:
    """Aggregate decisions into the numbers a corpus-scale plan needs."""
    by_action: dict[str, int] = {}
    by_source: dict[str, int] = {}
    by_preprocessing: dict[str, int] = {}
    pages = 0
    for decision in decisions:
        by_action[decision.action] = by_action.get(decision.action, 0) + 1
        by_source[decision.text_source] = by_source.get(decision.text_source, 0) + 1
        for step in decision.preprocessing:
            by_preprocessing[step] = by_preprocessing.get(step, 0) + 1
        if decision.action in config.OCR_ACTIONS_BLOCKING_INDEX:
            pages += decision.estimated_pages
    ocr_documents = sum(
        by_action.get(action, 0)
        for action in config.OCR_ACTIONS_BLOCKING_INDEX
    )
    return {
        "by_action": by_action,
        "by_text_source": by_source,
        "by_preprocessing": by_preprocessing,
        "documents_needing_ocr": ocr_documents,
        "pages_needing_ocr": pages,
    }
