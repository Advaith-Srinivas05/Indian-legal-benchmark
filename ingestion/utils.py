"""Pure, side-effect-light helpers: slugs, safe filenames, hashing, atomic IO.

Everything here is deterministic and unit-tested (see ``tests/test_utils.py``).
Keeping this logic free of network/HTTP concerns is what makes document-ID
generation, hashing and duplicate detection straightforward to test.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit

# Windows-reserved device names (case-insensitive, with or without extension).
_WINDOWS_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Characters not allowed in filenames on Windows (superset of POSIX rules).
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


#: A query segment that starts a genuine parameter rather than continuing a
#: value, e.g. ``type=rule``.
_QUERY_PARAM = re.compile(r"[A-Za-z][A-Za-z0-9_]*=")
#: Once a file name has reached its extension, later segments are parameters.
_FILE_SUFFIXES = (".pdf",)
_FILENAME_PARAM = "filename="


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (second precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def repair_location(location: str) -> str:
    """Escape an ``&`` India Code leaves raw inside a ``filename=`` value.

    ``/ViewFileUploaded`` redirects to
    ``…/showfile?actid=…&type=…&filename=<name>.pdf`` and writes the file name
    into the query **unescaped**. A name containing an ampersand — "GOA-IDC
    (Transfer & Sub-Lease Regulations), 2018" — therefore looks like the start
    of another parameter, the file server sees a truncated name, and answers
    HTTP 200 with a small HTML error page instead of the PDF.

    Only the ``filename`` value is touched: a following segment is left alone
    when it looks like a real parameter (``type=rule``) or when the name has
    already reached its ``.pdf`` extension. Every other ``&`` in the URL,
    including the ones separating legitimate parameters, is preserved.
    """
    head, separator, tail = location.rpartition(_FILENAME_PARAM)
    if not separator:
        return location
    segments = tail.split("&")
    name = [segments[0]]
    parameters: list[str] = []
    for segment in segments[1:]:
        ends_name = name[-1].lower().endswith(_FILE_SUFFIXES)
        if parameters or ends_name or _QUERY_PARAM.match(segment):
            parameters.append(segment)
        else:
            name.append(segment)
    repaired = "%26".join(name)
    if parameters:
        repaired += "&" + "&".join(parameters)
    return head + separator + repaired


def canonical_url(url: str) -> str:
    """Fold a URL to a comparable form: case and percent-encoding normalised.

    India Code writes the *same* bitstream URL two ways — a browse listing may
    give ``.../A2000-21 (1).pdf`` while the item page links
    ``.../A2000-21%20%281%29.pdf`` — and comparing those as raw strings makes
    a requested file look absent from the page that lists it. Scheme and host
    are lowercased, the path is decoded and re-encoded once so equivalent
    escapes collapse, and the query is compared by its decoded key/value pairs.

    For comparison only: the URL actually requested is always the one India
    Code published (see :func:`ingestion.indiacode.normalise_url`).
    """
    parsed = urlsplit((url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.port:
        host = f"{host}:{parsed.port}"
    path = quote(unquote(parsed.path), safe="/")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((parsed.scheme.lower(), host, path, query, ""))


def same_url(left: str | None, right: str | None) -> bool:
    """True when two URLs address the same resource (see :func:`canonical_url`)."""
    if left is None or right is None:
        return left is right
    return left == right or canonical_url(left) == canonical_url(right)


def slugify(text: str, max_length: int = 60) -> str:
    """Turn arbitrary text into a lowercase ASCII hyphen-slug.

    Non-ASCII characters (e.g. Devanagari titles) are transliterated where
    possible and otherwise dropped, so the result is always filesystem-safe.
    Returns an empty string if nothing usable remains.
    """
    if not text:
        return ""
    # Normalise and strip accents/decorations down to ASCII.
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii")
    ascii_text = ascii_text.lower()
    # Any run of non-alphanumerics becomes a single hyphen.
    ascii_text = re.sub(r"[^a-z0-9]+", "-", ascii_text)
    ascii_text = ascii_text.strip("-")
    if max_length and len(ascii_text) > max_length:
        ascii_text = ascii_text[:max_length].rstrip("-")
    return ascii_text


def safe_filename(name: str, fallback: str = "document") -> str:
    """Return a filesystem-safe version of *name* valid on Windows and POSIX.

    Replaces reserved characters, trims trailing dots/spaces (illegal on
    Windows) and guards against reserved device names.
    """
    if not name:
        return fallback
    cleaned = _UNSAFE_CHARS.sub("_", name)
    # Collapse whitespace runs and trim characters Windows forbids at the end.
    cleaned = cleaned.strip().rstrip(". ")
    cleaned = cleaned.strip()
    if not cleaned:
        return fallback
    stem = cleaned.split(".")[0]
    if stem.lower() in _WINDOWS_RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned


def make_document_id(title: str | None, handle_id: str) -> str:
    """Build a stable, human-readable, filesystem-safe document ID.

    Uniqueness is guaranteed by the India Code handle item id (which is unique
    per document); the title slug is a human-readable prefix. Example::

        make_document_id("Passports Act, 1967", "1372")
        -> "passports-act-1967__handle-1372"

    ``handle_id`` is required because it is the only reliably-unique key; the
    title alone is not (two items can share a title, or have none).
    """
    handle_id = safe_filename(str(handle_id).strip(), fallback="unknown")
    slug = slugify(title or "")
    if slug:
        return f"{slug}__handle-{handle_id}"
    return f"handle-{handle_id}"


def make_subordinate_document_id(
    document_type: str,
    description: str | None,
    url: str | None,
    *,
    fallback_name: str | None = None,
) -> str:
    """Build a stable id for a Rule/Regulation, which has no India Code handle.

    Uniqueness comes from the file URL — the one thing India Code guarantees is
    distinct per file — with a readable slug in front::

        make_subordinate_document_id("rule", "The Passport Rules, 1967", url)
        -> "the-passport-rules-1967__rule-bf4c863fce"

    Discovery and the downloader both call this, so an inventory entry and the
    document it later becomes on disk share one identity.
    """
    slug = slugify(description or fallback_name or document_type)
    digest = short_hash(sha256_bytes((url or "").encode("utf-8")), 10)
    return f"{slug or document_type}__{document_type}-{digest}"


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 digest of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks (memory-safe)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def short_hash(sha256: str, length: int = 12) -> str:
    """Return the leading *length* characters of a hex digest (for filenames)."""
    return sha256[:length]


def atomic_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Atomically move *src* onto *dst* (same filesystem), replacing any file."""
    os.replace(src, dst)


def atomic_write_text(path: str | os.PathLike[str], text: str, encoding: str = "utf-8") -> None:
    """Write text atomically: write to a sibling temp file then ``os.replace``.

    This guarantees readers never observe a half-written file even if the
    process is interrupted mid-write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w", encoding=encoding, newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
