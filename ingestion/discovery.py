"""Corpus discovery: build an inventory of every in-scope English document.

Discovery answers "what should we eventually download?" without downloading
anything. It inspects HTML and metadata only; **no PDF is ever fetched here**.

India Code's actual structure (established by inspecting the live site)
--------------------------------------------------------------------

* There is **no REST API and no OAI-PMH endpoint** (``/rest/*`` and ``/oai/*``
  both 404), so enumeration has to go through the site's own browse indexes.
* Acts are DSpace items grouped into **collections**: one for Central Acts and
  one per State/UT, all linked from the site navigation.
* A collection is enumerated via
  ``/handle/<coll>/browse?type=shorttitle&rpp=200&offset=N``. That listing
  already carries enactment date, act number, short title and the item handle,
  so the browse phase is cheap.
* **Rules and Regulations are not DSpace items.** They hang off their parent
  act's page as tables inside modal dialogs, and are served from
  ``/ViewFileUploaded?path=<act path>/rulesindividualfile/&file=<name>`` — not
  from ``/bitstream/``. Their tables have explicit ``Files(Eng)`` and
  ``Files(Hindi)`` columns, which is the strongest language evidence the site
  offers anywhere.

So discovery runs in two phases:

    browse collections  ->  listings.json      (cheap, one request per page)
    inspect each act    ->  journal.jsonl      (one request per act page)
                            indiacode_inventory.json

Both phases are resumable: listings are cached, and each inspected act appends
one line to an append-only journal. Re-running skips acts already journalled, a
torn final line is discarded, and the inventory itself is only ever written
atomically. Nothing is ever lost or corrupted by an interruption.

Language: every candidate file is put through :mod:`ingestion.language`, the
same gate the downloader uses. A document enters the inventory only when its
English status is *positively established*; Hindi-only and unverifiable
documents are counted and reported, never listed as download jobs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from . import config, indiacode, language
from .errors import IngestionError, LanguageError, MetadataError
from .http_client import get_text
from .indiacode import (          # page parsing lives with the source adapter
    india_code_path as _india_code_path,
    parse_subordinate_rows,
    pdf_url_kind,
)
from .models import ParsedItem, SubordinateRow
from .utils import make_document_id, make_subordinate_document_id, utcnow_iso

log = logging.getLogger(__name__)

_HANDLE_IN_HREF = re.compile(r"/handle/(\d+/\d+)")
_YEAR_RE = re.compile(r"(1[6-9]\d{2}|20\d{2})")


class Outcome:
    """Per-document discovery outcomes (also the summary's counter names)."""

    ENGLISH_CONFIRMED = "english_confirmed"
    HINDI_REJECTED = "hindi_rejected"
    AMBIGUOUS = "ambiguous"
    NO_PDF = "no_pdf"
    ERROR = "error"


# --- data structures -----------------------------------------------------------


@dataclass
class Collection:
    """A browsable India Code collection (Central Acts, or one State/UT)."""

    name: str
    handle: str                 # "123456789/1362"
    document_type: str          # "central_act" | "state_act"
    jurisdiction: str           # "India" | "Andhra Pradesh" | ...
    url: str

    @property
    def handle_id(self) -> str:
        return self.handle.rsplit("/", 1)[-1]


@dataclass
class ActListing:
    """One row of a collection's browse listing (no item page fetched yet)."""

    handle: str
    url: str
    short_title: Optional[str] = None
    act_number: Optional[str] = None
    enactment_date: Optional[str] = None
    collection_name: str = ""
    collection_handle: str = ""
    document_type: str = ""
    jurisdiction: str = ""
    browse_url: str = ""


@dataclass
class InventoryEntry:
    """One download job: a document whose English PDF has been identified."""

    document_id: str
    title: Optional[str]
    short_title: Optional[str]
    document_type: str                 # central_act | state_act | rule | regulation
    jurisdiction: str
    year: Optional[int]
    act_number: Optional[str]
    india_code_url: str                # page this document was found on
    english_pdf_url: str               # what the downloader will fetch
    pdf_url_kind: str                  # "bitstream" | "viewfileuploaded"
    language: str
    language_source: str
    language_evidence: Optional[str] = None
    enactment_date: Optional[str] = None
    handle: Optional[str] = None
    handle_id: Optional[str] = None
    india_code_act_id: Optional[str] = None
    india_code_path: Optional[str] = None
    ministry: Optional[str] = None
    department: Optional[str] = None
    description: Optional[str] = None
    parent_document_id: Optional[str] = None
    parent_handle: Optional[str] = None
    parent_title: Optional[str] = None
    collection_name: str = ""
    collection_handle: str = ""
    discovered_at: str = ""
    #: Every listing/page this document was seen through, so a merged duplicate
    #: can still be traced back to all of its India Code sources.
    sources: list[dict] = field(default_factory=list)


@dataclass
class Rejection:
    """A document deliberately left out of the inventory, kept for the report."""

    reason: str                        # an Outcome value
    document_type: str
    jurisdiction: str
    india_code_url: str
    title: Optional[str] = None
    detail: str = ""


@dataclass
class ActResult:
    """Everything learned from one act page (one journal line)."""

    handle: str
    outcome: str
    entries: list[InventoryEntry] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    error: Optional[str] = None


# --- phase 1: collections ------------------------------------------------------


def parse_collections(html: str, base_url: str) -> list[Collection]:
    """Extract the Central Acts collection and every State/UT collection.

    Read from the site navigation (present on every page) rather than a
    hardcoded list, so a state added or renamed upstream is picked up
    automatically. The "Browse Central Acts" menu points at the central
    collection's browse index; the "State Acts" menu lists one link per
    State/UT.
    """
    soup = BeautifulSoup(html, "html.parser")
    collections: list[Collection] = []
    seen: set[str] = set()

    def add(name: str, handle: str, document_type: str, jurisdiction: str) -> None:
        if handle in seen or not name:
            return
        seen.add(handle)
        collections.append(
            Collection(
                name=name, handle=handle, document_type=document_type,
                jurisdiction=jurisdiction,
                url=f"{config.BASE_URL}/handle/{handle}",
            )
        )

    for menu in soup.find_all("li", class_="dropdown"):
        toggle = menu.find("a", class_="dropdown-toggle")
        if toggle is None:
            continue
        label = toggle.get_text(" ", strip=True).lower()
        links = menu.find_all("a", href=True)

        if "central act" in label:
            for anchor in links:
                match = _HANDLE_IN_HREF.search(anchor["href"])
                if match and "browse?" in anchor["href"]:
                    add("Central Acts", match.group(1), "central_act", "India")
                    break
        elif "state act" in label:
            for anchor in links:
                match = _HANDLE_IN_HREF.search(anchor["href"])
                name = anchor.get_text(" ", strip=True)
                if match and name and "browse?" not in anchor["href"]:
                    add(name, match.group(1), "state_act", name)

    return collections


def discover_collections(session: requests.Session) -> list[Collection]:
    """Fetch the site navigation and parse the in-scope collections from it."""
    url = f"{config.BASE_URL}/community-list"
    collections = parse_collections(get_text(session, url), url)
    if not collections:
        raise MetadataError(
            f"No India Code collections could be found on {url}; the site "
            "navigation may have changed."
        )
    log.info(
        "Found %d collections (%d central, %d state/UT).", len(collections),
        sum(c.document_type == "central_act" for c in collections),
        sum(c.document_type == "state_act" for c in collections),
    )
    return collections


# --- phase 1: browse listings --------------------------------------------------


def browse_url(collection: Collection, offset: int, page_size: int) -> str:
    return (
        f"{config.BASE_URL}/handle/{collection.handle}/browse"
        f"?type={config.BROWSE_INDEX}&sort_by=1&order=ASC"
        f"&rpp={page_size}&etal=-1&null=&offset={offset}"
    )


def parse_browse_page(html: str, base_url: str, collection: Collection) -> list[ActListing]:
    """Parse one browse page into act listings.

    The results table has the columns *Enactment Date | Act Number | Short Title
    | View*, where the "View" cell links to the item's handle page. An empty
    list means the last page has been passed.
    """
    soup = BeautifulSoup(html, "html.parser")
    listings: list[ActListing] = []
    seen: set[str] = set()

    for row in soup.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue
        anchor = next(
            (a for a in row.find_all("a", href=True) if _HANDLE_IN_HREF.search(a["href"])),
            None,
        )
        if anchor is None:
            continue
        handle = _HANDLE_IN_HREF.search(anchor["href"]).group(1)
        if handle in seen or handle == collection.handle:
            continue
        seen.add(handle)
        values = [cell.get_text(" ", strip=True) for cell in cells]
        listings.append(
            ActListing(
                handle=handle,
                url=f"{config.BASE_URL}/handle/{handle}",
                enactment_date=_pick(values, 0),
                act_number=_pick(values, 1),
                short_title=_pick(values, 2),
                collection_name=collection.name,
                collection_handle=collection.handle,
                document_type=collection.document_type,
                jurisdiction=collection.jurisdiction,
                browse_url=base_url,
            )
        )
    return listings


def browse_collection(
    session: requests.Session,
    collection: Collection,
    *,
    page_size: int = config.BROWSE_PAGE_SIZE,
    max_pages: int = 500,
) -> list[ActListing]:
    """Page through a collection's browse index until it runs out of rows."""
    listings: list[ActListing] = []
    seen: set[str] = set()
    offset = 0
    for _page in range(max_pages):
        url = browse_url(collection, offset, page_size)
        page = parse_browse_page(get_text(session, url), url, collection)
        if not page:
            break
        fresh = [item for item in page if item.handle not in seen]
        seen.update(item.handle for item in fresh)
        listings.extend(fresh)
        offset += page_size
    log.info("%-34s %5d acts", collection.name, len(listings))
    return listings


def _pick(values: list[str], index: int) -> Optional[str]:
    if index < len(values):
        value = values[index].strip()
        # The "View" cell and empty cells carry no information.
        if value and value.lower().rstrip(".") != "view":
            return value
    return None


# --- phase 2: inspecting one act ----------------------------------------------


def inspect_act(session: requests.Session, listing: ActListing) -> ActResult:
    """Fetch one act page and derive its inventory entries.

    Returns an :class:`ActResult` for the act *and* its in-scope Rules and
    Regulations. Never raises for a per-act failure: problems become an
    ``ERROR`` outcome so a long run is not killed by one bad page.
    """
    try:
        html = get_text(session, listing.url)
    except IngestionError as exc:
        log.warning("FAILED %s: %s", listing.url, exc)
        return ActResult(listing.handle, Outcome.ERROR, error=str(exc))

    try:
        item = indiacode.parse_item_page(
            html, base_url=listing.url, source_url=listing.url
        )
    except IngestionError as exc:
        log.warning("Could not parse %s: %s", listing.url, exc)
        return ActResult(listing.handle, Outcome.ERROR, error=str(exc))

    result = _act_entry(item, listing)
    result.entries.extend(
        _subordinate_entries(html, item, listing, result)
    )
    return result


def _act_entry(item: ParsedItem, listing: ActListing) -> ActResult:
    """Classify the act's own PDF and build its inventory entry."""
    now = utcnow_iso()
    metadata = item.metadata
    title = item.title or listing.short_title

    if not item.bitstreams:
        return ActResult(
            listing.handle, Outcome.NO_PDF,
            rejections=[_rejection(Outcome.NO_PDF, listing, title,
                                   "the item page lists no downloadable file")],
        )

    try:
        english = language.select_english_bitstream(item)
    except LanguageError as exc:
        outcome = (
            Outcome.HINDI_REJECTED
            if any(b.language == "hi" for b in item.bitstreams)
            and not any(b.language is None for b in item.bitstreams)
            else Outcome.AMBIGUOUS
        )
        return ActResult(
            listing.handle, outcome,
            rejections=[_rejection(outcome, listing, title, str(exc))],
        )

    document_id = make_document_id(title, indiacode.handle_item_id(item.handle))
    entry = InventoryEntry(
        document_id=document_id,
        title=title,
        short_title=metadata.get("short_title") or listing.short_title,
        document_type=listing.document_type,
        jurisdiction=listing.jurisdiction,
        year=_as_year(metadata.get("act_year") or listing.enactment_date),
        act_number=str(metadata.get("act_number") or listing.act_number or "") or None,
        india_code_url=listing.url,
        english_pdf_url=english.url,
        pdf_url_kind=pdf_url_kind(english.url),
        language=language.CORPUS_LANGUAGE,
        language_source=english.language_source,
        language_evidence=english.language_evidence,
        enactment_date=metadata.get("enactment_date") or listing.enactment_date,
        handle=item.handle,
        handle_id=indiacode.handle_item_id(item.handle),
        india_code_act_id=metadata.get("india_code_act_id"),
        india_code_path=None,
        ministry=metadata.get("ministry"),
        department=metadata.get("department"),
        collection_name=listing.collection_name,
        collection_handle=listing.collection_handle,
        discovered_at=now,
        sources=[_source(listing, "collection_browse")],
    )
    return ActResult(listing.handle, Outcome.ENGLISH_CONFIRMED, entries=[entry])


def _subordinate_entries(
    html: str, item: ParsedItem, listing: ActListing, act_result: ActResult
) -> list[InventoryEntry]:
    """Build inventory entries for the act's in-scope Rules and Regulations."""
    parent_id = act_result.entries[0].document_id if act_result.entries else None
    parent_title = item.title or listing.short_title
    entries: list[InventoryEntry] = []

    for row in parse_subordinate_rows(html, listing.url):
        # English rests solely on the column India Code filed the file under
        # ("Files(Eng)"). The row's description is passed only as *link text*,
        # where it can disqualify the file (Devanagari, or the Hindi
        # description) but can never certify it as English — a row must not be
        # able to self-certify just by having a description.
        verdict = language.classify_bitstream(
            filename=row.english_filename or "",
            link_text=row.description,
            explicit_label=row.english_column_label,
            english_titles=(),
            hindi_titles=[row.hindi_description] if row.hindi_description else [],
        )
        if row.english_url is None:
            act_result.rejections.append(
                _rejection(
                    Outcome.HINDI_REJECTED if row.hindi_url else Outcome.NO_PDF,
                    listing, row.description,
                    "only a Hindi file is published for this "
                    f"{row.document_type}" if row.hindi_url
                    else f"no file is published for this {row.document_type}",
                    document_type=row.document_type,
                )
            )
            continue
        if not verdict.is_english:
            outcome = Outcome.HINDI_REJECTED if verdict.language == "hi" else Outcome.AMBIGUOUS
            act_result.rejections.append(
                _rejection(
                    outcome, listing, row.description,
                    f"English status not established for {row.english_filename!r} "
                    f"(language={verdict.language or 'undetermined'})",
                    document_type=row.document_type,
                )
            )
            continue

        entries.append(
            InventoryEntry(
                document_id=_subordinate_document_id(row),
                title=row.description,
                short_title=row.description,
                document_type=row.document_type,
                jurisdiction=listing.jurisdiction,
                year=_as_year(row.year),
                act_number=None,
                india_code_url=listing.url,
                english_pdf_url=row.english_url,
                pdf_url_kind=pdf_url_kind(row.english_url),
                language=language.CORPUS_LANGUAGE,
                language_source=verdict.source,
                language_evidence=verdict.evidence,
                enactment_date=row.year,
                handle=None,
                handle_id=None,
                india_code_act_id=item.metadata.get("india_code_act_id"),
                india_code_path=_india_code_path(row.english_url),
                ministry=item.metadata.get("ministry"),
                department=item.metadata.get("department"),
                description=row.description,
                parent_document_id=parent_id,
                parent_handle=item.handle,
                parent_title=parent_title,
                collection_name=listing.collection_name,
                collection_handle=listing.collection_handle,
                discovered_at=utcnow_iso(),
                sources=[_source(listing, f"act_page_{row.tab_label.lower()}_table")],
            )
        )
    return entries


def _subordinate_document_id(row: SubordinateRow) -> str:
    """A stable id for a rule/regulation, which has no India Code handle.

    Shared with the downloader (:func:`ingestion.utils.make_subordinate_document_id`)
    so an inventory entry and the document it later becomes on disk have one
    identity.
    """
    return make_subordinate_document_id(
        row.document_type, row.description, row.english_url,
        fallback_name=row.english_filename,
    )


def _rejection(reason, listing: ActListing, title, detail, *, document_type=None) -> Rejection:
    return Rejection(
        reason=reason,
        document_type=document_type or listing.document_type,
        jurisdiction=listing.jurisdiction,
        india_code_url=listing.url,
        title=title,
        detail=detail,
    )


def _source(listing: ActListing, via: str) -> dict:
    return {
        "via": via,
        "collection_name": listing.collection_name,
        "collection_handle": listing.collection_handle,
        "india_code_url": listing.url,
        "browse_url": listing.browse_url,
        "handle": listing.handle,
    }


def _as_year(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 1600 <= value <= 2100 else None
    match = _YEAR_RE.search(str(value))
    return int(match.group(1)) if match else None
