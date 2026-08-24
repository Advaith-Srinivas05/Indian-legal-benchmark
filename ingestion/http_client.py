"""HTTP layer: a configured :mod:`requests` session and a streaming PDF fetch.

Uses a proper HTTP client (``requests``) rather than shelling out to
curl/wget. Two India-Code-specific realities are handled here:

* A browser ``User-Agent`` is mandatory (the site 403s otherwise).
* Downloads are streamed to a ``.part`` file so an interrupted transfer never
  leaves a half-written PDF in the corpus.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config
from .errors import FetchError, HostNotAllowedError, NotPDFError
from .models import FetchResult
from .utils import repair_location

log = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"


class IndiaCodeSession(requests.Session):
    """A session that survives India Code's malformed ``Location`` headers.

    ``requests`` re-reads a redirect target as UTF-8 (it re-encodes the header,
    which ``http.client`` decoded as latin-1, and decodes it strictly). India
    Code writes file names into that header as raw bytes, so a rule whose name
    contains a non-breaking space or a ``¬`` makes ``requests`` raise
    :class:`UnicodeDecodeError` — from inside ``send()``, *before* the caller
    ever sees the response, and even with ``allow_redirects=False``.

    Rather than decode every page loosely, the strictness is relaxed at exactly
    the point that is wrong: when the header is not valid UTF-8 we keep the
    latin-1 reading ``http.client`` already produced, which is a faithful
    round-trip of the bytes on the wire. :func:`_open_stream` then handles the
    redirect itself, and ``requests`` percent-encodes the name as UTF-8 when it
    requests it — which is what India Code's file server expects.
    """

    def get_redirect_target(self, response):  # noqa: D102 - see class docstring
        try:
            return super().get_redirect_target(response)
        except UnicodeDecodeError:
            location = response.headers.get("location")
            log.debug(
                "Redirect target of %s is not valid UTF-8; keeping the latin-1 "
                "reading of the header: %r", response.url, location,
            )
            return location


def build_session() -> requests.Session:
    """Return a session with a browser UA and automatic retry/backoff."""
    session = IndiaCodeSession()
    session.headers.update(
        {
            "User-Agent": config.USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/pdf,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    retry = Retry(
        total=config.MAX_RETRIES,
        connect=config.MAX_RETRIES,
        read=config.MAX_RETRIES,
        backoff_factor=config.RETRY_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class RateLimiter:
    """A global minimum interval between requests, shared across workers.

    India Code is a public government site: this is what keeps a concurrent run
    polite regardless of how many worker threads are in play.
    """

    def __init__(self, min_interval: float):
        self._min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        if not self._min_interval:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + self._min_interval
        if delay > 0:
            time.sleep(delay)


class SessionPool:
    """One :mod:`requests` session per worker thread.

    A ``requests.Session`` is not documented as thread-safe, so concurrent
    crawls take a session each rather than sharing one.
    """

    def __init__(self):
        self._local = threading.local()

    def get(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = build_session()
            self._local.session = session
        return session


def get_text(session: requests.Session, url: str) -> str:
    """GET a URL expected to return HTML/text. Raises :class:`FetchError`."""
    try:
        response = session.get(url, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:  # network, timeout, HTTP error
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise FetchError(f"Failed to fetch {url}: {exc}", status_code=status) from exc
    return decode_response(response)


#: Encodings tried, in order, when the server did not declare one it means.
#: cp1252 is last and cannot fail on any byte sequence, so decoding always
#: succeeds — but only after the encodings that would be *correct* have been
#: given their chance, so a legitimate UTF-8 page is never read as mojibake.
_TEXT_FALLBACK_ENCODINGS = ("utf-8", "cp1252")


def decode_response(response: requests.Response) -> str:
    """Decode an HTML/text response, preferring what the server declared.

    India Code serves nearly everything as UTF-8 and says so, but a few pages
    arrive with no charset at all — for which ``requests`` assumes latin-1 per
    RFC 2616 and would silently mojibake the Devanagari and Kannada titles the
    language gate depends on. Worse, a handful of pages carry stray cp1252 bytes
    that are not valid UTF-8, so simply forcing UTF-8 (the previous behaviour)
    raised :class:`UnicodeDecodeError`.

    The strategy is therefore: trust a declared charset; otherwise try UTF-8
    strictly and fall back to cp1252, which accepts any byte sequence. Lossy
    ``errors="replace"`` decoding is never used, so no character is silently
    destroyed.
    """
    declared = _declared_charset(response)
    if declared:
        response.encoding = declared
        return response.text
    content = response.content
    for encoding in _TEXT_FALLBACK_ENCODINGS:
        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        response.encoding = encoding
        return text
    # Unreachable with the list above (cp1252 maps every byte), but if the list
    # is ever narrowed, fail loudly rather than return corrupted text.
    raise FetchError(
        f"Could not decode the response from {response.url} using any of "
        f"{', '.join(_TEXT_FALLBACK_ENCODINGS)}."
    )


def _declared_charset(response: requests.Response) -> Optional[str]:
    """The charset the server actually stated, or ``None``.

    ``requests`` reports ``iso-8859-1`` both when the server said so and when it
    said nothing at all, so the header is re-read here instead of trusting
    ``response.encoding``.
    """
    content_type = response.headers.get("Content-Type", "")
    for part in content_type.split(";")[1:]:
        name, _, value = part.partition("=")
        if name.strip().lower() == "charset":
            return value.strip().strip('"\'') or None
    return None


def download_stream(
    session: requests.Session,
    url: str,
    dest_part: Path,
    *,
    max_bytes: int = config.MAX_PDF_BYTES,
    expect_pdf: bool = True,
    fallback_url: Optional[str] = None,
) -> FetchResult:
    """Stream *url* into *dest_part*, or *fallback_url* if *url* cannot be
    resolved, and report what was fetched.

    ``fallback_url`` exists because India Code's ``/ViewFileUploaded`` lookup
    servlet sometimes cannot answer for a file it is perfectly willing to serve:
    for a name containing a non-ASCII character it replies ``302`` with **no
    Location header at all**, and for a few others it replies ``404``. The
    caller supplies the address that servlet would have redirected to
    (:func:`ingestion.indiacode.showfile_url`), and it is tried once — but only
    when the *request* could not be resolved.

    A response that arrives and turns out not to be a PDF is **not** retried:
    the redirect worked, the file server simply does not have the file, and
    asking a second way would only fetch the same error page. Nor is a redirect
    that left India Code, which is a security stop rather than a transport
    failure.
    """
    try:
        return _download_once(
            session, url, dest_part, max_bytes=max_bytes, expect_pdf=expect_pdf,
        )
    except HostNotAllowedError:
        raise
    except FetchError as exc:
        if not fallback_url or fallback_url == url:
            raise
        log.warning(
            "%s could not be resolved (%s); trying India Code's own redirect "
            "target %s.", url, exc, fallback_url,
        )
        result = _download_once(
            session, fallback_url, dest_part, max_bytes=max_bytes, expect_pdf=expect_pdf,
        )
        result.fallback_url = fallback_url
        return result


def _download_once(
    session: requests.Session,
    url: str,
    dest_part: Path,
    *,
    max_bytes: int,
    expect_pdf: bool,
) -> FetchResult:
    """Stream *url* into *dest_part* and report what was fetched.

    Redirects are followed by hand (see :func:`_open_stream`) so India Code's
    malformed ``Location`` header can be repaired before it is used. Two guards
    run *before* any byte reaches the disk:

    * every URL in the redirect chain, and the final one the bytes come from,
      must still be an India Code host — a ``/ViewFileUploaded`` request
      legitimately redirects to India Code's file server, but a redirect
      off-site must never put foreign bytes in the corpus;
    * with ``expect_pdf``, the leading bytes must be the ``%PDF-`` magic number,
      so an HTML error page served with HTTP 200 is rejected immediately instead
      of being written out and inspected afterwards.

    The caller still validates and atomically moves the ``.part`` file into
    place. On any failure the partial file is removed so a retry starts clean.
    """
    dest_part.parent.mkdir(parents=True, exist_ok=True)
    try:
        response, redirects = _open_stream(session, url)
        with response:
            response.raise_for_status()
            _check_download_host(url, response.url)
            content_type = response.headers.get("Content-Type", "")
            status = response.status_code
            total = 0
            checked = not expect_pdf
            preamble = b""
            with open(dest_part, "wb") as handle:
                for chunk in response.iter_content(config.DOWNLOAD_CHUNK_SIZE):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise FetchError(
                            f"Download from {url} exceeded {max_bytes} bytes; aborting."
                        )
                    if not checked:
                        # Hold the first bytes back until they prove to be a PDF.
                        preamble += chunk
                        if len(preamble) < len(_PDF_MAGIC):
                            continue
                        _check_pdf_magic(url, preamble, content_type)
                        checked = True
                        handle.write(preamble)
                        continue
                    handle.write(chunk)
                if not checked and preamble:
                    _check_pdf_magic(url, preamble, content_type)
                    handle.write(preamble)
        return FetchResult(
            content_type=content_type,
            bytes=total,
            http_status=status,
            final_url=response.url,
            redirects=redirects,
        )
    except requests.RequestException as exc:
        _safe_unlink(dest_part)
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise FetchError(
            f"Failed to download {url}: {exc}", status_code=status
        ) from exc
    except BaseException:
        # KeyboardInterrupt / FetchError(size) / NotPDFError / disk errors:
        # never leave a partial file behind.
        _safe_unlink(dest_part)
        raise


def _open_stream(session: requests.Session, url: str):
    """Open a streaming GET, following any redirects ourselves.

    ``requests`` would follow them automatically, but it would follow India
    Code's ``Location`` header *verbatim* — and for a Rule or Regulation whose
    file name contains an ampersand that header is malformed, so the file server
    returns an HTML error page with HTTP 200 instead of the PDF. Resolving each
    hop by hand lets the target be repaired (and re-checked against the allowed
    hosts) *before* it is requested. Returns ``(response, redirect_chain)``.
    """
    redirects: list[str] = []
    current = url
    for _ in range(config.MAX_REDIRECTS + 1):
        response = session.get(
            current, stream=True, allow_redirects=False, timeout=config.REQUEST_TIMEOUT
        )
        if not 300 <= response.status_code < 400:
            return response, redirects
        location = response.headers.get("Location")
        response.close()
        if not location:
            raise FetchError(
                f"{current} answered HTTP {response.status_code} without a "
                f"Location header; cannot follow the redirect.",
                status_code=response.status_code,
            )
        target = urljoin(current, repair_location(location))
        # Validate every hop, not just the last one.
        _check_download_host(url, target)
        redirects.append(target)
        current = target
    raise FetchError(
        f"Download of {url} exceeded {config.MAX_REDIRECTS} redirects; aborting."
    )


def _check_download_host(requested_url: str, final_url: str) -> None:
    """Reject a redirect that leaves India Code."""
    host = (urlparse(final_url).hostname or "").lower()
    if not any(
        host == suffix or host.endswith("." + suffix)
        for suffix in config.ALLOWED_DOWNLOAD_HOST_SUFFIXES
    ):
        raise HostNotAllowedError(
            f"Download of {requested_url} redirected to {final_url!r}, which is "
            f"not an India Code host; refusing to fetch it."
        )


def _check_pdf_magic(url: str, preamble: bytes, content_type: str) -> None:
    if not preamble.startswith(_PDF_MAGIC):
        raise NotPDFError(
            f"Response from {url} is not a PDF (content-type={content_type!r}, "
            f"starts with {preamble[:len(_PDF_MAGIC)]!r}); nothing was written."
        )


def _safe_unlink(path: Path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
