"""India Code (DSpace) source adapter.

Responsibilities:

* Validate and normalise a user-supplied URL into a canonical India Code URL.
* Fetch a *landing/handle* page and parse it into a :class:`ParsedItem`
  (metadata + the list of downloadable bitstreams).
* Attach a *language* to every bitstream, from India Code's own metadata, so
  that :mod:`ingestion.language` can pick the English PDF and reject the Hindi
  one. This corpus is English-only.

Why this is not just "GET the URL":
An India Code document URL such as ``/handle/123456789/1372`` is an HTML
landing page, **not** the PDF. The PDF lives at a separate *bitstream* URL like
``/bitstream/123456789/1372/1/196715.pdf``. The landing page advertises the
primary PDF via a ``<meta name="citation_pdf_url">`` tag and lists all
bitstreams as ``<a href=".../bitstream/...">`` links. This module extracts
those, plus Dublin Core ``<meta>`` tags and the visible metadata table.

All parsing is done by the pure function :func:`parse_item_page`, which takes
HTML text and returns a :class:`ParsedItem` — this is what the tests exercise,
so no network is needed to test parsing.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from urllib.parse import parse_qs, quote, urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

from . import config, language
from .errors import FetchError, InvalidURLError, MetadataError
from .http_client import get_text
from .models import BitstreamRef, ParsedItem, SubordinateInfo, SubordinateRow

log = logging.getLogger(__name__)

_HANDLE_RE = re.compile(r"^/handle/(\d+)/(\d+)/?$")
_BITSTREAM_RE = re.compile(r"^/bitstream/(\d+)/(\d+)/(\d+)/([^/]+)$")

#: Visible metadata-table label -> normalised metadata key.
_TABLE_LABEL_MAP = {
    "act id": "india_code_act_id",
    "act number": "act_number",
    "enactment date": "enactment_date",
    "act year": "act_year",
    "short title": "short_title",
    "hindi title": "hindi_title",
    "long title": "long_title",
    "ministry": "ministry",
    "department": "department",
    "enforcement date": "enforcement_date",
}


# --- URL handling --------------------------------------------------------------


def normalise_url(url: str) -> tuple[str, str, str | None, str | None]:
    """Validate a URL and classify it.

    Returns ``(kind, canonical_url, handle, filename)`` where *kind* is
    ``"handle"``, ``"bitstream"`` or ``"viewfile"``.

    * ``handle`` is like ``"123456789/1372"``; it is ``None`` for
      ``/ViewFileUploaded`` URLs, which belong to a parent act rather than to a
      DSpace item of their own.
    * ``filename`` is set for bitstream and ViewFileUploaded URLs.

    Only the scheme and host are ever rewritten. A ViewFileUploaded query
    string is carried through **byte for byte**: its ``path``/``file``
    parameters are percent-encoded by India Code and re-encoding them would
    produce a URL the server does not recognise.

    Raises :class:`InvalidURLError` for anything else.
    """
    if not url or not isinstance(url, str):
        raise InvalidURLError("URL must be a non-empty string.")
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise InvalidURLError(f"URL must be http(s): {url!r}")
    host = parsed.netloc.lower()
    if host not in config.ALLOWED_HOSTS:
        raise InvalidURLError(
            f"Host {host!r} is not an India Code host. "
            f"Expected one of {sorted(config.ALLOWED_HOSTS)}."
        )

    path = parsed.path
    # hdl.handle.net/123456789/1372 -> treat as a handle path.
    if host == "hdl.handle.net":
        path = "/handle" + path if not path.startswith("/handle") else path

    handle_match = _HANDLE_RE.match(path)
    if handle_match:
        handle = f"{handle_match.group(1)}/{handle_match.group(2)}"
        canonical = urlunparse(
            ("https", config.INDIA_CODE_HOST, f"/handle/{handle}", "", "", "")
        )
        return "handle", canonical, handle, None

    bitstream_match = _BITSTREAM_RE.match(path)
    if bitstream_match:
        prefix, item, _seq, filename = bitstream_match.groups()
        handle = f"{prefix}/{item}"
        canonical = urlunparse(
            ("https", config.INDIA_CODE_HOST, path, "", "", "")
        )
        return "bitstream", canonical, handle, filename

    if path.lower().rstrip("/") == config.VIEWFILE_PATH:
        query = parse_qs(parsed.query)
        file_param = _single(query, "file")
        if not _single(query, "path") or not file_param:
            raise InvalidURLError(
                f"ViewFileUploaded URL {url!r} must carry both a 'path' and a "
                "'file' query parameter."
            )
        # Query preserved verbatim - see the note in this function's docstring.
        canonical = urlunparse(
            ("https", config.INDIA_CODE_HOST, parsed.path, "", parsed.query, "")
        )
        return "viewfile", canonical, None, file_param

    raise InvalidURLError(
        f"URL path {path!r} is not an India Code /handle/, /bitstream/ or "
        "/ViewFileUploaded URL."
    )


def _single(query: dict, key: str) -> str | None:
    """Return one decoded query parameter value, or ``None`` if absent/empty."""
    values = query.get(key) or []
    value = values[0].strip() if values else ""
    return value or None


def viewfile_identity(url: str) -> tuple[str, str]:
    """Identity of a ViewFileUploaded URL: its decoded ``(path, file)`` pair.

    Used to match a requested file against the rows of its parent act page.
    Comparing decoded parameters rather than raw URL text means an equivalent
    encoding (``+`` versus ``%20``) still matches.
    """
    query = parse_qs(urlparse(url).query)
    return (
        (_single(query, "path") or "").strip("/"),
        (_single(query, "file") or "").strip(),
    )


#: Parent-act metadata that genuinely also describes its subordinate
#: legislation. Everything else (short title, act number, enactment date, …)
#: describes the *act* and would be misleading on a rule, so it is left behind
#: rather than copied down.
_INHERITABLE_METADATA = ("ministry", "department", "ministry_relation", "india_code_act_id")


def _inherited_metadata(parent_metadata: dict) -> dict:
    return {
        key: parent_metadata[key]
        for key in _INHERITABLE_METADATA
        if parent_metadata.get(key) not in (None, "")
    }


def handle_item_id(handle: str) -> str:
    """Return the unique item id from a handle string (``.../1372`` -> ``1372``)."""
    return handle.rsplit("/", 1)[-1]


# --- Resolution ----------------------------------------------------------------


def resolve(
    session: requests.Session, url: str, *, parent_url: str | None = None
) -> ParsedItem:
    """Turn a user URL into a fully-parsed :class:`ParsedItem`.

    * Handle URLs are fetched and parsed for metadata + bitstreams.
    * Bitstream URLs point straight at a PDF, which carries no language
      information of its own, so the item's landing page is fetched anyway to
      establish the file's language (and its metadata). The result is pinned to
      the requested file via ``requested_bitstream_url``.
    * ViewFileUploaded URLs (Rules/Regulations) belong to a parent act rather
      than to a DSpace item, and likewise carry no language information. Their
      *parent act page* is where India Code states the language, via the
      ``Files(Eng)`` column, so ``parent_url`` must identify that page.
    """
    kind, canonical, handle, filename = normalise_url(url)

    if kind == "viewfile":
        return _resolve_viewfile(session, url.strip(), canonical, parent_url)

    if kind == "bitstream":
        return _resolve_bitstream(session, canonical, handle, filename)

    html = get_text(session, canonical)
    item = parse_item_page(html, base_url=canonical, source_url=canonical)
    if item.primary_pdf_url is None:
        raise MetadataError(
            f"No PDF bitstream could be located on the India Code page {canonical}."
        )
    return item


def _resolve_bitstream(
    session: requests.Session, canonical: str, handle: str, filename: str | None
) -> ParsedItem:
    """Resolve a direct ``/bitstream/`` URL via its item's landing page.

    The landing page is the only place India Code states which language a file
    is in, so it is consulted even though the PDF URL is already known. If it
    cannot be fetched or does not list the file, the bitstream is returned with
    an *undetermined* language — which the language gate turns into a clear
    failure rather than an unverified download.
    """
    handle_url = f"{config.BASE_URL}/handle/{handle}"
    try:
        item = parse_item_page(
            get_text(session, handle_url), base_url=handle_url, source_url=handle_url
        )
    except (FetchError, MetadataError) as exc:
        log.warning(
            "Could not read the landing page %s for %s (%s); the file's language "
            "cannot be verified.", handle_url, canonical, exc,
        )
        return ParsedItem(
            source_url=canonical,
            handle=handle,
            handle_id=handle_item_id(handle),
            primary_pdf_url=canonical,
            bitstreams=[
                BitstreamRef(
                    url=canonical, filename=filename or "document.pdf", is_primary=True
                )
            ],
            requested_bitstream_url=canonical,
        )

    item.requested_bitstream_url = canonical
    item.source_url = canonical
    return item


def _resolve_viewfile(
    session: requests.Session, original_url: str, canonical: str, parent_url: str | None
) -> ParsedItem:
    """Resolve a Rules/Regulations file against its parent act page.

    The file's language is taken from the column India Code filed it under on
    that page. Without the parent page there is no language evidence at all, so
    the request is refused rather than guessed — the corpus is English-only.
    """
    if not parent_url:
        raise MetadataError(
            f"{original_url} is a Rules/Regulations file, which India Code lists "
            "only on its parent act's page. Supply that page (--parent-url, a "
            "'parent_url' column in the input file, or --inventory) so the "
            "file's language can be verified; it is not assumed to be English."
        )

    _kind, parent_canonical, _handle, _filename = normalise_url(parent_url)
    html = get_text(session, parent_canonical)
    parent = parse_item_page(html, base_url=parent_canonical, source_url=parent_canonical)

    wanted = viewfile_identity(canonical)
    matches = [
        r for r in parse_subordinate_rows(html, parent_canonical)
        if r.english_url and viewfile_identity(r.english_url) == wanted
    ]
    if not matches:
        raise MetadataError(
            f"{original_url} is not listed in the Rules/Regulations of "
            f"{parent_canonical}, so its language cannot be verified. Refusing "
            "to download an unverified file (English-only corpus)."
        )

    row, verdict = _classify_subordinate(matches)
    reference = BitstreamRef(
        url=canonical,
        filename=row.english_filename or wanted[1] or "document.pdf",
        is_primary=True,
        link_text=row.description,
        language_label=row.english_column_label,
        language=verdict.language,
        language_source=verdict.source,
        language_evidence=verdict.evidence,
    )
    return ParsedItem(
        source_url=parent_canonical,
        handle=parent.handle,
        handle_id=parent.handle_id,
        title=row.description,
        primary_pdf_url=canonical,
        bitstreams=[reference],
        metadata=_inherited_metadata(parent.metadata),
        dublin_core=parent.dublin_core,
        metadata_table=parent.metadata_table,
        requested_bitstream_url=canonical,
        original_url=original_url,
        subordinate=SubordinateInfo(
            document_type=row.document_type,
            description=row.description,
            parent_handle=parent.handle,
            parent_title=parent.title,
            parent_url=parent_canonical,
            india_code_path=india_code_path(canonical),
        ),
    )


def _classify_subordinate(
    rows: list[SubordinateRow],
) -> tuple[SubordinateRow, language.LanguageVerdict]:
    """Pick the row that decides a file's language when several list it.

    India Code sometimes lists the *same* file on more than one row of an act's
    Rules table — typically once described in English and once in the state
    language. Taking whichever came first in the HTML would make the language
    verdict depend on document order, so every matching row is classified and
    the evidence is combined with the module's own asymmetry (see
    :mod:`ingestion.language`): a Hindi verdict anywhere still rejects the file,
    and English is taken only when some row positively proves it.
    """
    verdicts = [(row, _classify_row(row)) for row in rows]
    for row, verdict in verdicts:
        if verdict.language == "hi":
            return row, verdict
    english = [(row, verdict) for row, verdict in verdicts if verdict.is_english]
    if english:
        # The winning row also supplies the document's description, so among
        # equally-English rows prefer one actually described in English.
        return next(
            (pair for pair in english if language.is_latin_text(pair[0].description)),
            english[0],
        )
    return verdicts[0]


def _classify_row(row: SubordinateRow) -> language.LanguageVerdict:
    """Classify one Rules/Regulations row from the columns India Code filed it in."""
    # A "Hindi Description" that merely repeats the English one is India Code
    # copying a cell, not a statement about the file, so it is not evidence.
    hindi_titles = (
        [row.hindi_description]
        if row.hindi_description and row.hindi_description != row.description
        else []
    )
    return language.classify_bitstream(
        filename=row.english_filename or "",
        link_text=row.description,
        explicit_label=row.english_column_label,
        english_titles=(),
        hindi_titles=hindi_titles,
    )


# --- Pure parsing (unit-tested) ------------------------------------------------


def parse_item_page(html: str, *, base_url: str, source_url: str) -> ParsedItem:
    """Parse India Code landing-page HTML into a :class:`ParsedItem`.

    Pure function: no network access, so it is fully unit-testable against a
    saved HTML fixture.
    """
    soup = BeautifulSoup(html, "html.parser")

    dublin_core = _extract_dublin_core(soup)
    table = _extract_metadata_table(soup)
    handle = _extract_handle(dublin_core, base_url)

    # Metadata is parsed first: India Code's own English/Hindi title fields are
    # the evidence used to decide each file's language.
    metadata = _normalise_metadata(dublin_core, table)
    title = (
        metadata.get("title")
        or metadata.get("short_title")
        or _first(dublin_core.get("DC.title"))
    )

    bitstreams = _extract_bitstreams(
        soup, base_url, dublin_core,
        english_titles=_english_titles(metadata, dublin_core),
        hindi_titles=_hindi_titles(metadata, dublin_core),
    )
    primary = next((b for b in bitstreams if b.is_primary), None)
    primary_url = primary.url if primary else (bitstreams[0].url if bitstreams else None)

    return ParsedItem(
        source_url=source_url,
        handle=handle,
        handle_id=handle_item_id(handle),
        title=title,
        primary_pdf_url=primary_url,
        bitstreams=bitstreams,
        metadata=metadata,
        dublin_core=dublin_core,
        metadata_table=table,
    )


def _extract_dublin_core(soup: BeautifulSoup) -> dict:
    """Collect ``<meta name="DC.*"|"DCTERMS.*"|"citation_*">`` tags.

    A name can legitimately repeat (e.g. two ``DC.title`` values: English and
    Hindi), so repeats are collected into a list.
    """
    result: dict[str, object] = {}
    for meta in soup.find_all("meta"):
        name = meta.get("name")
        content = meta.get("content")
        if not name or content is None:
            continue
        if not (name.startswith(("DC.", "DCTERMS.", "citation_"))):
            continue
        content = content.strip()
        if name in result:
            existing = result[name]
            if isinstance(existing, list):
                existing.append(content)
            else:
                result[name] = [existing, content]
        else:
            result[name] = content
    return result


def _extract_metadata_table(soup: BeautifulSoup) -> dict:
    """Extract the visible label/value metadata table into a flat dict."""
    table: dict[str, str] = {}
    for label_cell in soup.select("td.metadataFieldLabel"):
        value_cell = label_cell.find_next_sibling("td")
        if value_cell is None:
            continue
        label = label_cell.get_text(" ", strip=True).replace("\xa0", " ")
        label = label.rstrip(":").strip().lower()
        value = value_cell.get_text(" ", strip=True).replace("\xa0", " ").strip()
        if label and value:
            table[label] = value
    return table


def _extract_handle(dublin_core: dict, base_url: str) -> str:
    """Determine the handle (``123456789/1372``) from DC identifiers or URL."""
    for value in _as_list(dublin_core.get("DC.identifier")):
        match = re.search(r"(\d+)/(\d+)$", value) if "handle" in value else None
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    # Fall back to the page URL.
    path = urlparse(base_url).path
    match = _HANDLE_RE.match(path)
    if match:
        return f"{match.group(1)}/{match.group(2)}"
    raise MetadataError(f"Could not determine India Code handle for {base_url!r}.")


def _extract_bitstreams(
    soup: BeautifulSoup,
    base_url: str,
    dublin_core: dict,
    *,
    english_titles: list[str],
    hindi_titles: list[str],
) -> list[BitstreamRef]:
    """Find all bitstream (file) links, flag the primary, and set languages.

    Every returned :class:`BitstreamRef` carries the language India Code's own
    metadata supports (``None`` when it supports none). Nothing here decides
    what to download — that is :func:`ingestion.language.select_english_bitstream`.
    """
    primary_pdf_url = _first(dublin_core.get("citation_pdf_url"))
    primary_path = urlparse(primary_pdf_url).path if primary_pdf_url else None

    seen: set[str] = set()
    refs: list[BitstreamRef] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        if "/bitstream/" not in href:
            continue
        absolute = urljoin(base_url, href)
        path = urlparse(absolute).path
        match = _BITSTREAM_RE.match(path)
        if not match:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        _prefix, _item, seq, filename = match.groups()
        refs.append(
            BitstreamRef(
                url=absolute,
                filename=filename,
                sequence=int(seq),
                is_primary=primary_path is not None and path == primary_path,
                link_text=anchor.get_text(" ", strip=True) or None,
                language_label=_language_label_near(anchor),
            )
        )

    # If citation_pdf_url named a PDF we didn't see as a link, add it, and make
    # sure exactly one bitstream is marked primary.
    if primary_path and not any(b.is_primary for b in refs):
        absolute = urljoin(base_url, primary_pdf_url)
        match = _BITSTREAM_RE.match(urlparse(absolute).path)
        if match and absolute not in seen:
            refs.insert(
                0,
                BitstreamRef(
                    url=absolute,
                    filename=match.group(4),
                    sequence=int(match.group(3)),
                    is_primary=True,
                ),
            )
    if refs and not any(b.is_primary for b in refs):
        # No citation hint: assume the lowest-sequence file is primary.
        refs.sort(key=lambda b: (b.sequence is None, b.sequence))
        refs[0].is_primary = True

    for ref in refs:
        verdict = language.classify_bitstream(
            filename=ref.filename,
            link_text=ref.link_text,
            explicit_label=ref.language_label,
            is_citation_pdf=ref.is_primary and primary_path is not None,
            english_titles=english_titles,
            hindi_titles=hindi_titles,
        )
        ref.language = verdict.language
        ref.language_source = verdict.source
        ref.language_evidence = verdict.evidence
        if verdict.is_undetermined:
            log.warning(
                "Language of bitstream %s could not be determined from India Code "
                "metadata; it will not be downloaded.", ref.url,
            )
    return refs


def _language_label_near(anchor) -> str | None:
    """Return an explicit language label attached to a file link, if any.

    India Code renders file lists in two shapes: a table whose row carries a
    language cell, and a plain list of links. Only text that *is* a language
    label is returned — a neighbouring cell holding a size or a title is
    ignored, so it can never be mistaken for language evidence.
    """
    for attribute in ("title", "aria-label"):
        value = anchor.get(attribute)
        if value and language.language_label_of(value):
            return value.strip()

    cell = anchor.find_parent(["td", "th"])
    if cell is not None and cell.parent is not None:
        for sibling in cell.parent.find_all(["td", "th"], recursive=False):
            if sibling is cell:
                continue
            text = sibling.get_text(" ", strip=True)
            if text and language.language_label_of(text):
                return text
    return None


def _english_titles(metadata: dict, dublin_core: dict) -> list[str]:
    """India Code's English title fields, used as language evidence."""
    candidates = [
        metadata.get("short_title"),
        metadata.get("title"),
        _first(dublin_core.get("citation_title")),
        _first(dublin_core.get("DC.title")),
    ]
    hindi = {language.normalise_title(v) for v in _hindi_titles(metadata, dublin_core)}
    return _dedupe(
        value for value in candidates
        if value
        # An English title may quote a Hindi phrase; only a title actually
        # written in Devanagari is disqualified.
        and not language.is_devanagari_text(value)
        and language.normalise_title(value) not in hindi
    )


def _hindi_titles(metadata: dict, dublin_core: dict) -> list[str]:
    """India Code's Hindi title fields (the ``Hindi Title`` row / 2nd DC.title)."""
    candidates = [metadata.get("hindi_title"), *_as_list(dublin_core.get("DC.title"))[1:]]
    return _dedupe(value for value in candidates if language.is_devanagari_text(value))


def _dedupe(values) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = language.normalise_title(value)
        if key and key not in seen:
            seen.add(key)
            result.append(value)
    return result


# --- Subordinate legislation (Rules / Regulations) ------------------------------
#
# These are not DSpace items: India Code lists them in tables on the parent
# act's page and serves them from /ViewFileUploaded. The tables carry explicit
# Files(Eng) / Files(Hindi) columns, which is the strongest language evidence
# anywhere on the site.


def parse_subordinate_rows(html: str, base_url: str) -> list[SubordinateRow]:
    """Extract in-scope Rules/Regulations rows from an act's landing page.

    The tables live inside modal dialogs reached from labelled buttons, so the
    button's own text ("Rules", "Regulations") is what identifies the table —
    never the modal's numeric id, which is incidental markup.

    India Code emits these tables with **unclosed ``<tr>`` tags**, so the parser
    nests every row inside the previous one. Reading only each row's *direct*
    ``<td>`` children is what keeps one row from absorbing all the rows below
    it.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows: list[SubordinateRow] = []

    for button in soup.find_all("a", attrs={"data-target": True}):
        label = button.get_text(" ", strip=True)
        document_type = config.SUBORDINATE_TYPES.get(label.strip().lower())
        if document_type is None:
            continue                       # notifications, orders, ... are out of scope
        modal = soup.find(id=button["data-target"].lstrip("#"))
        if modal is None:
            continue
        table = modal.find("table")
        if table is None:
            continue
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        english_header, hindi_header = _file_column_headers(headers)

        for row in table.find_all("tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 4:
                continue
            english = _cell_file(cells[3], base_url)
            hindi = _cell_file(cells[4], base_url) if len(cells) > 4 else None
            description = cells[1].get_text(" ", strip=True) or None
            if english is None and hindi is None and not description:
                continue      # spacer/blank row, not a document
            # A described row with no file at all is a real gap in India Code's
            # data and is kept so it can be reported as "no PDF available".
            rows.append(
                SubordinateRow(
                    document_type=document_type,
                    tab_label=label,
                    year=cells[0].get_text(" ", strip=True) or None,
                    description=description,
                    hindi_description=cells[2].get_text(" ", strip=True) or None,
                    english_url=english[0] if english else None,
                    english_filename=english[1] if english else None,
                    english_column_label=english_header,
                    hindi_url=hindi[0] if hindi else None,
                    hindi_column_label=hindi_header,
                )
            )
    return rows


def _file_column_headers(headers: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return the (English, Hindi) file-column headers, e.g. ``Files(Eng)``.

    These headers *are* India Code's explicit statement of each column's
    language, so they are handed to the language classifier verbatim rather
    than being second-guessed here.
    """
    by_language: dict[str, list[str]] = {"en": [], "hi": []}
    for header in headers:
        detected = language.language_label_of(header)
        if detected in by_language:
            by_language[detected].append(header)

    def pick(candidates: list[str]) -> Optional[str]:
        # "Hindi Description" is also a Hindi-labelled column; the one that
        # describes the *file* is the column we mean.
        for header in candidates:
            if "file" in header.lower():
                return header
        return candidates[0] if candidates else None

    return pick(by_language["en"]), pick(by_language["hi"])


def _cell_file(cell, base_url: str) -> Optional[tuple[str, str]]:
    """Return ``(absolute_url, filename)`` for a file cell, if it holds a link."""
    anchor = cell.find("a", href=True) if cell is not None else None
    if anchor is None:
        return None
    href = anchor["href"].strip()
    if not href or href.startswith("#"):
        return None
    url = urljoin(base_url, href)
    filename = (anchor.get("title") or "").strip() or _filename_from_url(url)
    return url, filename


def _filename_from_url(url: str) -> str:
    query = parse_qs(urlparse(url).query)
    for key in ("file", "filename"):
        if query.get(key):
            return query[key][0].strip()
    return Path(urlparse(url).path).name


#: ``/ViewFileUploaded``'s ``path`` names the sub-directory a file sits in;
#: India Code's file server takes the same distinction as a ``type`` parameter.
_VIEWFILE_DIRECTORY_TO_TYPE = {
    "rulesindividualfile": "rule",
    "regulationindividualfile": "regulation",
}


def showfile_url(url: str) -> str | None:
    """The address ``/ViewFileUploaded`` itself redirects a file to.

    ``/ViewFileUploaded?path=<actid>/<directory>/&file=<name>`` is a lookup
    servlet: it answers ``302`` to
    ``https://upload.indiacode.nic.in/showfile?actid=…&type=…&filename=…`` and
    the file server there does the actual serving. Both parameters come
    straight out of the request, so the target can be derived without asking —
    which matters because for a file whose name contains a non-ASCII character
    the servlet fails to emit the ``Location`` header at all and there is
    otherwise no way to reach a file India Code is perfectly willing to serve.

    This is India Code's own mapping, not a guessed one: it reproduces the
    recorded redirect target of every ``/ViewFileUploaded`` document in the
    corpus exactly. Returns ``None`` for any URL it does not apply to, and the
    caller still host-checks and PDF-checks whatever comes back.
    """
    parsed = urlparse(url)
    if parsed.path.lower().rstrip("/") != config.VIEWFILE_PATH:
        return None
    query = parse_qs(parsed.query)
    path = (_single(query, "path") or "").strip("/")
    filename = _single(query, "file")
    if not filename or "/" not in path:
        return None
    act_id, _, directory = path.partition("/")
    document_type = _VIEWFILE_DIRECTORY_TO_TYPE.get(directory.strip("/").lower())
    if not act_id or not document_type:
        return None
    return (
        f"https://{config.UPLOAD_HOST}{config.SHOWFILE_PATH}"
        f"?actid={quote(act_id)}&type={document_type}"
        f"&filename={quote(filename, safe=_SHOWFILE_SAFE)}"
    )


#: India Code leaves these unescaped in the ``filename`` it emits, and the file
#: server matches on them, so they are preserved rather than percent-encoded.
_SHOWFILE_SAFE = "()!*"


def pdf_url_kind(url: str) -> str:
    """Classify how a PDF URL must eventually be fetched."""
    path = urlparse(url).path.lower()
    if "/bitstream/" in path:
        return "bitstream"
    if "viewfileuploaded" in path:
        return "viewfileuploaded"
    return "other"


def india_code_path(url: str) -> str | None:
    """The ``AC_CEN_…`` act path India Code serves subordinate files from."""
    value = parse_qs(urlparse(url).query).get("path")
    return value[0].strip("/") if value else None


def _normalise_metadata(dublin_core: dict, table: dict) -> dict:
    """Merge table + Dublin Core into clean keys, omitting anything unknown.

    Table values (curated by India Code) take precedence; Dublin Core fills
    gaps. No value is fabricated: keys are present only when a source had them.
    """
    meta: dict[str, object] = {}

    for label, value in table.items():
        key = _TABLE_LABEL_MAP.get(label)
        if key:
            meta[key] = value

    # Title: first (English) DC.title.
    dc_title = _first(dublin_core.get("DC.title"))
    if dc_title:
        meta.setdefault("title", dc_title)
    # Long title also appears as DCTERMS.alternative.
    alt = _first(dublin_core.get("DCTERMS.alternative"))
    if alt:
        meta.setdefault("long_title", alt)
    # Hindi title: a second DC.title value, if present.
    dc_titles = _as_list(dublin_core.get("DC.title"))
    if len(dc_titles) > 1:
        meta.setdefault("hindi_title", dc_titles[1])
    # Enactment date: DCTERMS.issued (ISO).
    issued = _first(dublin_core.get("DCTERMS.issued"))
    if issued:
        meta.setdefault("enactment_date", issued)
    # Ministry/relation.
    relation = _first(dublin_core.get("DC.relation"))
    if relation:
        meta.setdefault("ministry_relation", relation)

    # Coerce act_year to int when it is a clean 4-digit year.
    year = meta.get("act_year") or _first(dublin_core.get("DC.date"))
    if year and re.fullmatch(r"\d{4}", str(year).strip()):
        meta["act_year"] = int(str(year).strip())
    elif "act_year" in meta and not re.fullmatch(r"\d{4}", str(meta["act_year"]).strip()):
        # Leave as string rather than guess.
        meta["act_year"] = str(meta["act_year"]).strip()

    return meta


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _first(value):
    items = _as_list(value)
    return items[0] if items else None
