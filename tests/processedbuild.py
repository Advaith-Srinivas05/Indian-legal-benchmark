"""Builds a miniature ``data/processed/`` tree for the benchmark tests.

The benchmark reads what processing wrote, so its fixtures are *processed
output*, not PDFs. Rather than hand-writing that JSON — which would drift from
the real shape the moment either side changed — these helpers build real
:class:`~processing.models.PageText` objects, run the real
:func:`processing.structure.parse_structure` over them, and serialise through the
same ``to_dict`` methods processing uses.

The payoff: the line indices in a fixture ``document.json`` are produced by the
same code that produces them in production, so a test relying on them tests the
real contract rather than a convenient copy of it.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from processing import config as processing_config
from processing.models import PageText
from processing.structure import parse_structure


def make_page(number: int, text: str, *, ocr_text: Optional[str] = None,
              indexable: bool = True, quality: str = "good") -> PageText:
    """One page as processing would have left it.

    Passing *ocr_text* builds the shape that trips up every naive consumer: the
    backend read nothing, the engine read the page, and ``text_source`` is the
    only thing that says so.
    """
    page = PageText(
        page_number=number,
        text="" if ocr_text is not None else text,
        char_count=len(ocr_text if ocr_text is not None else text),
    )
    if ocr_text is not None:
        page.ocr = {"text": ocr_text, "accepted": True, "engine": "tesseract@300"}
        page.text_source = "ocr"
    page.quality = {"verdict": quality, "reason": "fixture"}
    page.language = {"verdict": "en", "reason": "fixture"}
    page.indexable = indexable
    return page


def write_processed(
    data_dir: Path,
    document_id: str,
    pages: Sequence[PageText],
    *,
    eligible: bool = True,
    title: Optional[str] = "The Sample Act, 1999",
    category: str = "central_acts",
    document_type: str = "central_act",
    jurisdiction: str = "India",
    year: Optional[int] = 1999,
    omit_page_fields: Iterable[str] = (),
    schema_version: int = processing_config.PROCESSING_SCHEMA_VERSION,
    pdf_sha256: Optional[str] = None,
) -> Path:
    """Write ``document.json`` and ``pages.json`` for one fake processed document.

    *omit_page_fields* drops named fields from every page, reproducing the real
    documents written before ``text_source``/``quality``/``ocr`` existed.
    """
    directory = Path(data_dir) / processing_config.PROCESSED_SUBDIR / document_id
    directory.mkdir(parents=True, exist_ok=True)

    assessed = [p for p in pages if p.indexable] or list(pages)
    structure = parse_structure(assessed, metadata_title=title)

    page_payloads = []
    for page in pages:
        payload = page.to_dict()
        for name in omit_page_fields:
            payload.pop(name, None)
        page_payloads.append(payload)

    (directory / processing_config.PAGES_FILENAME).write_text(
        json.dumps({"schema_version": schema_version, "document_id": document_id,
                    "page_count": len(pages), "pages": page_payloads},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (directory / processing_config.DOCUMENT_FILENAME).write_text(
        json.dumps({
            "schema_version": schema_version,
            "processed_at": "2026-08-28T00:00:00+00:00",
            "processor": "processing v0.1.0",
            "source": {
                "provider": "India Code",
                "document_id": document_id,
                "category": category,
                "document_type": document_type,
                "title": title,
                "pdf_relpath": f"raw/indiacode/{category}/{document_id}/{document_id}.pdf",
                "sha256": pdf_sha256 or hashlib.sha256(document_id.encode()).hexdigest(),
                "bytes": 1234,
                "source_url": f"https://www.indiacode.nic.in/handle/123456789/{document_id}",
                "handle": f"123456789/{zlib.crc32(document_id.encode()) % 100000}",
                "jurisdiction": jurisdiction,
                "year": year,
                "enactment_date": "1999-06-24",
            },
            "extraction": {"page_count": len(pages), "pdf_type": "text_based"},
            "content_language": {"content_language": "en"},
            "extraction_quality": {"classification": "good", "score": 0.95},
            "ocr_decision": {"action": "use_extracted_text", "text_source": "born_digital"},
            "eligible_for_indexing": eligible,
            "structure": structure.to_dict(),
            "artifacts": {"pages": processing_config.PAGES_FILENAME},
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return directory


def write_raw_metadata(data_dir: Path, document_id: str, *, category: str = "central_acts",
                       **fields) -> Path:
    """Write the ``metadata.json`` ingestion leaves beside each PDF."""
    directory = Path(data_dir) / "raw" / "indiacode" / category / document_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "document_id": document_id,
        "short_title": "The Sample Act, 1999",
        "long_title": "An Act to provide for the regulation of samples.",
        "act_number": "7",
        "act_year": 1999,
        "india_code_act_id": "199907",
        "ministry": "Ministry of Law and Justice",
        "download": {"downloaded_at": "2026-08-16T12:00:00+00:00", "bytes": 1234,
                     "final_url": "https://www.indiacode.nic.in/bitstream/1/sample.pdf"},
        **fields,
    }
    path = directory / "metadata.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def rewrite_document(data_dir: Path, document_id: str, change: Callable[[dict], None]) -> None:
    """Tamper with a written ``document.json`` in place — for failure-path tests."""
    path = Path(data_dir) / processing_config.PROCESSED_SUBDIR / document_id / processing_config.DOCUMENT_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    change(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def walk_units(units):
    for unit in units:
        yield unit
        yield from walk_units(unit.get("children") or [])


#: A Central Act in India Code house style: dashed section headings, a chapter,
#: subsections and clauses. Structured enough that the parser finds real units.
ACT_PAGE_ONE = """THE SAMPLE ACT, 1999
ACT NO. 7 OF 1999
An Act to provide for the regulation of samples.
BE it enacted by Parliament in the Fiftieth Year of the Republic of India as follows:—
CHAPTER I
PRELIMINARY
1. Short title and extent.—(1) This Act may be called the Sample Act, 1999.
(2) It extends to the whole of India.
2. Definitions.—In this Act, unless the context otherwise requires,—
(a) "appointed day" means the day on which this Act comes into force;
(b) "prescribed" means prescribed by rules made under this Act."""

ACT_PAGE_TWO = """3. Power to make rules.—(1) The Central Government may make rules.
(2) Every rule made under this section shall be laid before Parliament.
Provided that no rule shall take effect before it is published.
4. Repeal and savings.—The Sample Act, 1950 is hereby repealed."""

#: A schedule page of the kind that is often image-only and read by OCR.
SCHEDULE_PAGE = """THE SCHEDULE
(See section 3)
Intelligence Bureau.
Directorate of Revenue Intelligence."""

#: Numbered headings alone on their lines, with no dash: the weak detector.
PLAIN_RULES = """THE PLAIN RULES, 2001
1. Short title
These rules may be called the Plain Rules, 2001.
2. Definitions
In these rules the Act means the Sample Act, 1999.
3. Fees
A fee of ten rupees shall be paid with every application.
4. Appeals
An appeal lies to the Collector within thirty days."""


def make_act(title: str, sections: int = 20, *, changed: Iterable[int] = (),
             per_page: int = 10) -> list[PageText]:
    """A synthetic Act in house style, with every section distinct.

    Sections listed in *changed* get a different body, so two calls differing
    only in *changed* model two versions of one Act: their similarity is set
    exactly by how many sections differ.
    """
    changed = set(changed)
    lines = [title.upper()]
    for n in range(1, sections + 1):
        body = (f"The authority shall revise matter {n} within {n + 90} days of receipt."
                if n in changed else
                f"The authority shall consider matter {n} within {n + 10} days of receipt.")
        lines.append(f"{n}. Subject number {n}.—{body}")
    pages = [lines[i:i + per_page] for i in range(0, len(lines), per_page)]
    return [make_page(i + 1, "\n".join(chunk)) for i, chunk in enumerate(pages)]
