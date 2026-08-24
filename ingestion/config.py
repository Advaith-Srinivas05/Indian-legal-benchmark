"""Static configuration for the India Code ingestion pipeline.

Kept dependency-free and side-effect-free so it can be imported from tests and
from any other module without pulling in the network stack.
"""

from __future__ import annotations

from pathlib import Path

# --- India Code (DSpace) source ------------------------------------------------

INDIA_CODE_HOST = "www.indiacode.nic.in"
#: Hosts we accept in user-supplied URLs. ``hdl.handle.net`` is the persistent
#: handle resolver which redirects into India Code; we rewrite it ourselves so
#: we never depend on following that redirect.
ALLOWED_HOSTS = frozenset(
    {"www.indiacode.nic.in", "indiacode.nic.in", "hdl.handle.net"}
)
BASE_URL = f"https://{INDIA_CODE_HOST}"

#: Rules and Regulations are not DSpace bitstreams. They are served from this
#: endpoint, e.g. ``/ViewFileUploaded?path=AC_CEN_…/rulesindividualfile/&file=x.pdf``.
VIEWFILE_PATH = "/viewfileuploaded"

#: India Code's file server, which ``/ViewFileUploaded`` redirects to. It is a
#: different host but still within ``ALLOWED_DOWNLOAD_HOST_SUFFIXES`` below.
UPLOAD_HOST = "upload.indiacode.nic.in"
SHOWFILE_PATH = "/showfile"

#: ``/ViewFileUploaded`` 302-redirects to India Code's file server
#: (``upload.indiacode.nic.in/showfile?actid=…``). Downloads may follow a
#: redirect only within India Code itself, so a redirect off-site cannot
#: quietly put foreign bytes into the corpus.
ALLOWED_DOWNLOAD_HOST_SUFFIXES = ("indiacode.nic.in",)

#: DSpace handle prefix used by India Code (the ``123456789`` in a handle).
HANDLE_PREFIX = "123456789"

# --- Document taxonomy ---------------------------------------------------------

#: Canonical corpus categories -> the on-disk sub-directory under
#: ``data/raw/indiacode/``. These match the layout mandated in the project spec.
CATEGORIES = ("central_acts", "state_acts", "rules", "regulations")

#: User-facing ``--type`` values (and input-file ``type`` fields) mapped to a
#: canonical category directory. Both singular and plural spellings accepted.
TYPE_TO_CATEGORY = {
    "central_act": "central_acts",
    "central_acts": "central_acts",
    "state_act": "state_acts",
    "state_acts": "state_acts",
    "rule": "rules",
    "rules": "rules",
    "regulation": "regulations",
    "regulations": "regulations",
}

#: Category directory -> the canonical singular ``document_type`` recorded in
#: metadata (used later by retrieval/citation stages).
CATEGORY_TO_DOCUMENT_TYPE = {
    "central_acts": "central_act",
    "state_acts": "state_act",
    "rules": "rule",
    "regulations": "regulation",
}

# --- On-disk layout ------------------------------------------------------------

DEFAULT_DATA_DIR = Path("data")
RAW_SUBDIR = Path("raw") / "indiacode"
PROCESSED_SUBDIR = Path("processed")
CHUNKS_SUBDIR = Path("chunks")
EMBEDDINGS_SUBDIR = Path("embeddings")
MANIFEST_FILENAME = "manifest.json"
METADATA_FILENAME = "metadata.json"
VERSIONS_DIRNAME = "versions"

# --- Discovery (corpus inventory) ----------------------------------------------

DISCOVERY_SUBDIR = Path("discovery")
#: The deliverable: every in-scope English document India Code offers.
INVENTORY_FILENAME = "indiacode_inventory.json"
#: Working files that make a run resumable (see ingestion/discovery.py).
LISTINGS_FILENAME = "listings.json"
JOURNAL_FILENAME = "journal.jsonl"
#: Header-only size estimate for the corpus (see ingestion/sizing.py).
SIZE_ESTIMATE_FILENAME = "size_estimate.json"

#: India Code has no REST/OAI endpoint, so collections are enumerated through
#: the site's own browse index. ``shorttitle`` and ``dateissued`` are the two
#: indexes that cover every item (``title`` is not built on this instance).
BROWSE_INDEX = "shorttitle"
#: Rows per browse page. 200 is accepted by the site and keeps paging cheap.
BROWSE_PAGE_SIZE = 200

#: Subordinate-legislation tabs on an act page. Only these two are in scope;
#: notifications, orders, circulars, ordinances and statutes are not.
SUBORDINATE_TYPES = {
    "rules": "rule",
    "regulations": "regulation",
}

DISCOVERY_SCHEMA_VERSION = 1
#: Polite crawling defaults: India Code is a government site.
DISCOVERY_WORKERS = 4
DISCOVERY_DELAY_SECONDS = 0.0

#: Schema versions embedded in the artefacts so future format changes are
#: detectable and migratable.
MANIFEST_SCHEMA_VERSION = 1
METADATA_SCHEMA_VERSION = 1

# --- HTTP ----------------------------------------------------------------------

#: India Code returns HTTP 403 to non-browser user agents, so a realistic
#: desktop browser UA is required for both the landing page and the PDF.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
REQUEST_TIMEOUT = (15, 120)  # (connect, read) seconds
MAX_RETRIES = 4
RETRY_BACKOFF_FACTOR = 1.0
DOWNLOAD_CHUNK_SIZE = 64 * 1024
#: Redirect hops a download will follow. ``/ViewFileUploaded`` needs one (to
#: India Code's file server), which may itself upgrade http -> https.
MAX_REDIRECTS = 5
#: Guard against a runaway/incorrect download. India Code acts are small; raise
#: this if a legitimately large document is ever rejected.
MAX_PDF_BYTES = 512 * 1024 * 1024
#: Anything smaller than this almost certainly is not a real PDF.
MIN_PDF_BYTES = 100
