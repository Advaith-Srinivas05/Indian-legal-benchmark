"""Typed data structures passed between the processing stages.

Three layers, kept apart on purpose:

* :class:`CorpusDocument` — *identity and provenance*, taken from the ingestion
  manifest and inventory. Nothing here is inferred from the PDF.
* :class:`PageText` / :class:`ExtractedDocument` — *what the PDF actually
  contains*, page by page, with the evidence behind every classification.
* :class:`LegalUnit` / :class:`DocumentStructure` — *legal hierarchy*, only
  where it could be identified from the text.

Every structure carries its source page range, because a legal claim that
cannot be pointed back at a page of a specific PDF is not citable.

All of them serialise through :meth:`to_dict`; the JSON written under
``data/processed/`` is exactly these dictionaries, so the on-disk format never
drifts from the in-memory model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from . import config

# --- Corpus identity ------------------------------------------------------------


@dataclass
class CorpusDocument:
    """One successfully-downloaded document, as the ingestion phase left it.

    Built by :mod:`processing.corpus` from ``data/manifest.json`` (the record of
    what is on disk) joined to ``data/discovery/indiacode_inventory.json`` (which
    carries the year and the state/UT a document belongs to — the manifest does
    not). Nothing in here is guessed: a field the artefacts do not establish
    stays ``None``.
    """

    document_id: str
    category: str                     # central_acts | state_acts | rules | regulations
    document_type: str                # central_act | state_act | rule | regulation
    title: Optional[str]
    pdf_path: Path
    pdf_relpath: str
    sha256: str
    bytes: int
    source_url: Optional[str] = None
    primary_pdf_url: Optional[str] = None
    handle: Optional[str] = None
    language: Optional[str] = None
    language_source: Optional[str] = None
    #: State/UT for state legislation, ``"India"`` for central. From the
    #: discovery inventory.
    jurisdiction: Optional[str] = None
    #: Act year from the discovery inventory (``None`` when India Code does not
    #: state one).
    year: Optional[int] = None
    enactment_date: Optional[str] = None
    parent_title: Optional[str] = None
    #: Present only for documents that came through the inventory join.
    collection_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "category": self.category,
            "document_type": self.document_type,
            "title": self.title,
            "pdf_relpath": self.pdf_relpath,
            "sha256": self.sha256,
            "bytes": self.bytes,
            "source_url": self.source_url,
            "primary_pdf_url": self.primary_pdf_url,
            "handle": self.handle,
            "language": self.language,
            "language_source": self.language_source,
            "jurisdiction": self.jurisdiction,
            "year": self.year,
            "enactment_date": self.enactment_date,
            "parent_title": self.parent_title,
            "collection_name": self.collection_name,
        }


# --- Extraction -----------------------------------------------------------------


@dataclass
class ExtractedTable:
    """A table candidate found on a page.

    ``retained`` records whether it passed the quality filter in
    :mod:`processing.extract`. Rejected candidates are kept rather than dropped
    so the benchmark can report the detector's false-positive rate instead of
    hiding it.
    """

    page_number: int
    bbox: tuple[float, float, float, float]
    row_count: int
    col_count: int
    filled_ratio: float
    rows: list[list[Optional[str]]] = field(default_factory=list)
    detector: str = ""
    retained: bool = False
    rejected_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "bbox": [round(v, 2) for v in self.bbox],
            "row_count": self.row_count,
            "col_count": self.col_count,
            "filled_ratio": round(self.filled_ratio, 4),
            "rows": self.rows,
            "detector": self.detector,
            "retained": self.retained,
            "rejected_reason": self.rejected_reason,
        }


@dataclass
class FurnitureLine:
    """A line identified as a running header/footer or a page number.

    Recorded, never removed from :attr:`PageText.text`. Downstream stages skip
    these lines by index; the original page text stays byte-for-byte auditable.
    """

    line_index: int
    text: str
    kind: str                          # header | footer | page_number
    reason: str

    def to_dict(self) -> dict:
        return {
            "line_index": self.line_index,
            "text": self.text,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass
class FootnoteBlock:
    """A footnote block separated out of a page's flowing text.

    Kept as its own unit rather than deleted: the amendment history it carries
    is genuinely useful, and the project spec forbids silently discarding extracted
    text. The page text in ``pages.json`` still contains these lines verbatim;
    this records where they are so the structure parser can keep them out of the
    provision above them.
    """

    page_number: int
    line_start: int
    line_end: int
    text: str
    detected_by: str = ""

    def to_dict(self) -> dict:
        return {
            "type": "footnote",
            "page_number": self.page_number,
            "page_start": self.page_number,
            "page_end": self.page_number,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "text": self.text,
            "detected_by": self.detected_by,
        }


@dataclass
class PageText:
    """One page of a PDF, with the evidence behind its classification.

    ``text`` is verbatim what the backend produced — no normalisation, no
    trimming, no furniture removal — because it is the only thing that makes an
    extraction bug diagnosable after the fact.
    """

    page_number: int                   # 1-based, as a citation would give it
    text: str
    char_count: int
    alpha_char_count: int = 0
    line_count: int = 0
    width: float = 0.0
    height: float = 0.0
    image_count: int = 0
    image_area_ratio: float = 0.0
    is_image_backed: bool = False
    #: No usable text, no page-covering image, and hundreds of vector paths:
    #: the glyphs were converted to outlines. Neither text extraction nor plain
    #: OCR can read this page — it has to be rasterised first.
    drawing_path_count: int = 0
    is_vector_outlined: bool = False
    has_text: bool = False
    is_empty: bool = False
    text_quality_suspect: bool = False
    quality_signals: dict = field(default_factory=dict)
    tables: list[ExtractedTable] = field(default_factory=list)
    furniture: list[FurnitureLine] = field(default_factory=list)
    footnotes: list[FootnoteBlock] = field(default_factory=list)
    #: :class:`processing.orientation.PageOrientation` — whether this page's text
    #: is the right way round, and on what evidence.
    orientation: Optional[object] = None
    #: This page's own extraction-quality verdict -- ``good``, ``suspect`` or
    #: ``unjudged`` -- from :func:`processing.quality.page_verdict`, judged over
    #: its English lines. ``None`` until that has run.
    #:
    #: A page-level fact, kept beside the document-level
    #: :class:`~processing.quality.QualityAssessment` rather than replacing it:
    #: a document can be sound overall and have three ruined pages, and both
    #: halves of that sentence are worth recording.
    quality: Optional[dict] = None
    #: What language this page is written in, judged on script alone by
    #: :func:`processing.language.classify_page`. ``None`` until that has run.
    language: Optional[dict] = None
    #: Lines written in another script, by index into this page's own line
    #: stream. Labelled here and skipped by the structure parser, never removed
    #: from :attr:`text` -- the same contract furniture and footnotes have.
    non_english_lines: list[dict] = field(default_factory=list)
    #: What an OCR engine made of this page, when one was run: the trigger, the
    #: engine, any rotation applied, the text it read, and whether that reading
    #: was accepted over :attr:`text`. ``None`` when OCR was never attempted.
    #:
    #: The engine's text lives here and never in :attr:`text`, which stays
    #: byte-for-byte what the extraction backend produced. Both readings are
    #: kept so a bad OCR pass is diagnosable after the fact instead of having
    #: silently overwritten the evidence.
    ocr: Optional[dict] = None
    #: Which reading later stages should use: ``backend`` or ``ocr``. Resolved
    #: by :func:`processing.ocr_engine.page_text`.
    text_source: str = "backend"

    #: Whether this page's text may be indexed as English law. False for the
    #: other-language pages of a bilingual document, which print the same law in
    #: translation, and for pages whose own extracted text is too damaged to
    #: quote. Set from the language verdict and the page quality verdict
    #: together; every page of an ordinary, soundly extracted English document
    #: stays True.
    #:
    #: Not on its own a licence to index: the *document* must also be eligible.
    #: Chunking reads the intersection.
    indexable: bool = True
    warnings: list[str] = field(default_factory=list)
    #: Per-line geometry from the backend, used to place footnotes. Transient:
    #: it is an order of magnitude larger than the page text and adds nothing
    #: once the footnotes have been located, so it is not serialised.
    line_metrics: list = field(default_factory=list, repr=False)

    @property
    def retained_tables(self) -> list[ExtractedTable]:
        return [t for t in self.tables if t.retained]

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "char_count": self.char_count,
            "alpha_char_count": self.alpha_char_count,
            "line_count": self.line_count,
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "image_count": self.image_count,
            "image_area_ratio": round(self.image_area_ratio, 4),
            "is_image_backed": self.is_image_backed,
            "drawing_path_count": self.drawing_path_count,
            "is_vector_outlined": self.is_vector_outlined,
            "has_text": self.has_text,
            "is_empty": self.is_empty,
            "text_quality_suspect": self.text_quality_suspect,
            "quality_signals": self.quality_signals,
            "tables": [t.to_dict() for t in self.tables],
            "furniture": [f.to_dict() for f in self.furniture],
            "footnotes": [f.to_dict() for f in self.footnotes],
            "orientation": self.orientation.to_dict() if self.orientation else None,
            "quality": self.quality,
            "language": self.language,
            "non_english_lines": self.non_english_lines,
            "ocr": self.ocr,
            "text_source": self.text_source,
            "indexable": self.indexable,
            "warnings": self.warnings,
        }

    @property
    def selected_text(self) -> str:
        """The reading of this page that later stages should use.

        :attr:`text` when extraction's output stands, the engine's text when OCR
        was run and its result was judged better. The one place that resolves
        which reading is authoritative, so no downstream stage has to decide for
        itself -- and so :attr:`text` can stay byte-for-byte what the backend
        produced without every consumer having to remember that.
        """
        if self.text_source == "ocr":
            text = (self.ocr or {}).get("text")
            if text:
                return text
        return self.text

    @property
    def is_sideways(self) -> bool:
        return getattr(self.orientation, "orientation", "") == "sideways"

    @property
    def has_inconsistent_direction(self) -> bool:
        return bool(getattr(self.orientation, "direction_inconsistent", False))


@dataclass
class ExtractedDocument:
    """Everything the extraction stage learned about one PDF."""

    document_id: str
    page_count: int
    pages: list[PageText]
    pdf_type: str                      # text_based | scanned | mixed | vector_outlined
    text_extraction_status: str        # ok | partial | requires_ocr
    backend: str = ""
    #: Pages that read as an arrangement-of-sections / contents listing.
    #: Established during extraction because table retention depends on it.
    toc_pages: list[int] = field(default_factory=list)
    pdf_metadata: dict = field(default_factory=dict)
    is_encrypted: bool = False
    #: Counters behind ``pdf_type``/``text_extraction_status``, so a
    #: classification can be checked without re-reading the PDF.
    classification_evidence: dict = field(default_factory=dict)
    #: Document-level orientation summary from :func:`processing.orientation.summarise`.
    orientation: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    extraction_seconds: float = 0.0

    @property
    def char_count(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def empty_page_count(self) -> int:
        return sum(1 for p in self.pages if p.is_empty)

    @property
    def table_count(self) -> int:
        return sum(len(p.retained_tables) for p in self.pages)

    @property
    def table_candidate_count(self) -> int:
        return sum(len(p.tables) for p in self.pages)

    @property
    def footnote_count(self) -> int:
        return sum(len(p.footnotes) for p in self.pages)

    @property
    def vector_outlined_page_count(self) -> int:
        return sum(1 for p in self.pages if p.is_vector_outlined)

    @property
    def indexable_page_count(self) -> int:
        return sum(1 for p in self.pages if p.indexable)

    @property
    def ocr_page_count(self) -> int:
        """Pages whose stored reading came out of an OCR engine."""
        return sum(1 for p in self.pages if p.text_source == "ocr")

    @property
    def sideways_page_count(self) -> int:
        return sum(1 for p in self.pages if p.is_sideways)

    @property
    def direction_inconsistent_page_count(self) -> int:
        return sum(1 for p in self.pages if p.has_inconsistent_direction)

    def to_dict(self) -> dict:
        return {
            "page_count": self.page_count,
            "char_count": self.char_count,
            "empty_page_count": self.empty_page_count,
            "pdf_type": self.pdf_type,
            "text_extraction_status": self.text_extraction_status,
            "backend": self.backend,
            "is_encrypted": self.is_encrypted,
            "pdf_metadata": self.pdf_metadata,
            "toc_pages": self.toc_pages,
            "table_count": self.table_count,
            "table_candidate_count": self.table_candidate_count,
            "footnote_count": self.footnote_count,
            "indexable_page_count": self.indexable_page_count,
            "ocr_page_count": self.ocr_page_count,
            "vector_outlined_page_count": self.vector_outlined_page_count,
            "sideways_page_count": self.sideways_page_count,
            "direction_inconsistent_page_count": self.direction_inconsistent_page_count,
            "classification_evidence": self.classification_evidence,
            "orientation": self.orientation,
            "warnings": self.warnings,
            "extraction_seconds": round(self.extraction_seconds, 3),
        }


# --- Legal structure ------------------------------------------------------------


@dataclass
class LegalUnit:
    """One identified unit of legal hierarchy, with its page provenance.

    ``page_start``/``page_end`` are what a citation is eventually built from, so
    they are recorded for every unit including nested ones. ``line_start`` and
    ``line_end`` index the document's line stream and exist so any unit can be
    traced back to the exact extracted lines it came from.
    """

    unit_type: str                     # part|chapter|section|subsection|clause|...
    number: Optional[str]
    heading: Optional[str]
    text: str
    page_start: int
    page_end: int
    line_start: int = -1
    line_end: int = -1
    #: Which pattern established this unit — the audit trail for "why does the
    #: parser think this is a section?".
    detected_by: str = ""
    children: list["LegalUnit"] = field(default_factory=list)

    def walk(self):
        """Yield this unit and every descendant, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict:
        data = {
            "unit_type": self.unit_type,
            "number": self.number,
            "heading": self.heading,
            "text": self.text,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "detected_by": self.detected_by,
        }
        if self.children:
            data["children"] = [c.to_dict() for c in self.children]
        return data


@dataclass
class DocumentStructure:
    """The legal hierarchy identified in one document — or the honest absence
    of one.

    ``confidence`` is not a score: it says which of three situations we are in.
    ``structured`` means sections were found and the document is navigable;
    ``partial`` means some units were found but the document is not covered;
    ``unstructured`` means nothing was confidently identified and the extracted
    page text is all we have. Guessing is never the third option.
    """

    title: Optional[str] = None
    title_source: Optional[str] = None
    preamble: Optional[LegalUnit] = None
    units: list[LegalUnit] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    #: Pages that look like an arrangement-of-sections/contents listing. Their
    #: numbered lines are not reported as sections; their text is untouched.
    toc_pages: list[int] = field(default_factory=list)
    #: Footnote blocks, kept as their own units so amendment apparatus is not
    #: presented as statutory text — and not thrown away either.
    footnotes: list[FootnoteBlock] = field(default_factory=list)
    #: Which vocabulary this document numbers its provisions in: ``section``,
    #: ``article``, or ``None`` when neither was established. Recorded rather
    #: than assumed — Articles are never folded into Sections.
    unit_vocabulary: Optional[str] = None
    confidence: str = "unstructured"
    warnings: list[str] = field(default_factory=list)

    def all_units(self):
        for unit in self.units:
            yield from unit.walk()

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "title_source": self.title_source,
            "preamble": self.preamble.to_dict() if self.preamble else None,
            "counts": self.counts,
            "toc_pages": self.toc_pages,
            "unit_vocabulary": self.unit_vocabulary,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "footnotes": [f.to_dict() for f in self.footnotes],
            "units": [u.to_dict() for u in self.units],
        }


# --- Whole-document result ------------------------------------------------------


@dataclass
class OcrDecision:
    """What should happen to this document's text, and why.

    The decision is separate from the classification that produced it because
    the two answer different questions — *what is this PDF* versus *what do we
    do about it* — and because the second is the one that costs money at corpus
    scale. A scanned document that already carries a good OCR layer must not be
    re-OCR'd; a text-based document whose text is corrupt must be.
    """

    action: str                        # see processing.ocr.ACTIONS
    text_source: str                   # scan_with_good_ocr, scan_with_bad_ocr, …
    reason: str
    priority: int = 0                  # higher = more urgent
    estimated_pages: int = 0
    #: Steps that must happen *before* an engine sees the page — currently only
    #: ``rotate_upright``. Carried on the decision rather than left implicit
    #: because OCRing a sideways page produces confident nonsense, which is the
    #: most expensive kind of failure this pipeline can have.
    preprocessing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "text_source": self.text_source,
            "reason": self.reason,
            "priority": self.priority,
            "estimated_pages": self.estimated_pages,
            "preprocessing": self.preprocessing,
        }


@dataclass
class ProcessedDocument:
    """A corpus document plus what processing made of it."""

    document: CorpusDocument
    extraction: Optional[ExtractedDocument] = None
    structure: Optional[DocumentStructure] = None
    language: Optional[object] = None          # processing.language.LanguageAssessment
    quality: Optional[object] = None           # processing.quality.QualityAssessment
    ocr_decision: Optional[OcrDecision] = None
    #: :class:`processing.ocr_engine.OcrRun` -- what the OCR stage actually did,
    #: as against :attr:`ocr_decision`, which is what it was told to do.
    ocr_run: Optional[object] = None
    ok: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    output_dir: Optional[Path] = None
    seconds: float = 0.0

    @property
    def eligible_for_indexing(self) -> bool:
        """Whether this document may go downstream to chunking and retrieval.

        Four gates, all of which must pass: the content must be established
        English, the extracted text must be good enough to quote, the pages must
        be the right way round, and extraction must have produced a document
        rather than a fragment of one. Anything else is quarantined — a routing
        decision, not a deletion.

        Each gate fails independently, which is why there are four.

        *Quality* is measured over a whole document, so a 300-page act with a
        40-page sideways section keeps a respectable average while a seventh of
        it is reversed characters — hence the *orientation* gate. *Language* is a
        property of the words and says nothing about whether they came out in the
        right order.

        The fourth gate came out of building the human validation set: eight
        documents were eligible while the OCR router had already decided their
        text needed replacing. The Nagaland Agriculture Produce Marketing
        regulations yielded text on 2 of 68 pages and were eligible for indexing;
        the UP Stamp Rules on 49 of 148. Nothing in the first three gates asks
        whether extraction actually produced a document, because quality is
        computed over the text that *did* come out — which, for a document that
        is 97% empty, is a small and unrepresentative sample of it.

        The quality gate is deliberately **document-level**, and making it
        page-level was tried and reverted on the evidence. The premise — that a
        document quarantined on quality holds sound pages worth keeping — is true
        of some of them, and the page-level checks cannot tell which: the pages
        they admit are visibly corrupted ("thereil", "concerngd", "recognuon")
        and measurably worse than pages from documents that were never
        quarantined, with no threshold separating the two populations. Recording
        each page's verdict is useful; letting it admit text is not. See
        :attr:`PageText.quality` and docs/KNOWN_ISSUES.md D6.
        """
        if not self.ok or self.language is None or self.quality is None:
            return False
        orientation = getattr(self.extraction, "orientation", None) or {}
        action = getattr(self.ocr_decision, "action", "")
        return bool(
            getattr(self.language, "eligible_for_indexing", False)
            and getattr(self.quality, "classification", "") == "good"
            and not orientation.get("orientation_suspect")
            and action not in config.OCR_ACTIONS_BLOCKING_INDEX
        )

    @property
    def indexable_pages(self) -> list:
        """The pages this document may contribute downstream.

        Marked during processing by the language gate. Quality is judged for
        each page too (:attr:`PageText.quality`) but deliberately does not
        decide this — see :attr:`eligible_for_indexing`.
        """
        pages = getattr(self.extraction, "pages", None) or []
        return [page for page in pages if getattr(page, "indexable", False)]

    @property
    def indexable_page_count(self) -> int:
        return len(self.indexable_pages)

