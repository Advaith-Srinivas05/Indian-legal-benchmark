"""Live checks against the real India Code website.

Deselected by default (``-m "not network"`` in ``pyproject.toml``) so the normal
suite stays offline and deterministic. Run them explicitly with::

    pytest -m network

They guard the one thing a saved fixture cannot: that the *current* live page
for a document published in both English and Hindi still yields the English PDF
and rejects the Hindi one.
"""

from __future__ import annotations

import pytest

from urllib.parse import urlparse

from ingestion import discovery, http_client, indiacode, language, sizing
from ingestion.errors import LanguageError, MetadataError, NotPDFError
from ingestion.http_client import build_session, get_text
from ingestion.utils import repair_location

pytestmark = pytest.mark.network

# The Passports Act, 1967 — carries an English PDF (196715.pdf) and a Hindi
# PDF (H1967-15.pdf) as two bitstreams of the same item.
HANDLE_URL = "https://www.indiacode.nic.in/handle/123456789/1372"
ENGLISH_PDF = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
HINDI_PDF = "https://www.indiacode.nic.in/bitstream/123456789/1372/2/H1967-15.pdf"
# A Rule of the same act: not a DSpace bitstream, served from /ViewFileUploaded.
LIVE_RULE_PDF = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile/"
    "&file=The+Passport+Rules%2C+1967+%2810.05.1967%29.pdf"
)


@pytest.fixture(scope="module")
def live_item():
    return indiacode.resolve(build_session(), HANDLE_URL)


def test_live_page_offers_both_languages(live_item):
    urls = {b.url for b in live_item.bitstreams}
    assert ENGLISH_PDF in urls, "expected the live item to still offer an English PDF"
    assert HINDI_PDF in urls, "expected the live item to still offer a Hindi PDF"


def test_live_english_pdf_is_selected(live_item):
    selected = language.select_english_bitstream(live_item)
    assert selected.url == ENGLISH_PDF
    assert selected.language == "en"
    assert selected.language_source == "indiacode_metadata_title"


def test_live_hindi_pdf_is_classified_hindi(live_item):
    hindi = next(b for b in live_item.bitstreams if b.url == HINDI_PDF)
    assert hindi.language == "hi"
    assert hindi.language_source in {"indiacode_hindi_title", "indiacode_devanagari_script"}


def test_asking_for_the_live_hindi_pdf_directly_is_refused():
    item = indiacode.resolve(build_session(), HINDI_PDF)
    with pytest.raises(LanguageError, match="No English version"):
        language.select_english_bitstream(item)


# --- discovery: the site structure the inventory depends on ---------------------


@pytest.fixture(scope="module")
def session():
    return build_session()


def test_live_collections_are_still_enumerable(session):
    collections = discovery.discover_collections(session)
    central = [c for c in collections if c.document_type == "central_act"]
    states = [c for c in collections if c.document_type == "state_act"]
    assert len(central) == 1
    assert central[0].handle == "123456789/1362"
    assert len(states) > 30, "expected a collection per State/UT"


def test_live_browse_index_still_paginates(session):
    collection = discovery.Collection(
        "Central Acts", "123456789/1362", "central_act", "India", HANDLE_URL
    )
    url = discovery.browse_url(collection, 0, 5)
    listings = discovery.parse_browse_page(get_text(session, url), url, collection)
    assert len(listings) == 5
    assert all(item.handle.startswith("123456789/") for item in listings)
    assert all(item.short_title for item in listings)


def test_live_viewfileuploaded_rule_resolves_as_english(session):
    """A real Rule, verified against its parent act page rather than assumed."""
    item = indiacode.resolve(session, LIVE_RULE_PDF, parent_url=HANDLE_URL)
    english = language.select_english_bitstream(item)
    assert english.url == LIVE_RULE_PDF
    assert english.language == "en"
    assert english.language_source == "indiacode_bitstream_label"
    assert "Files(Eng)" in english.language_evidence
    assert item.subordinate.document_type == "rule"
    assert item.subordinate.parent_handle == "123456789/1372"


def test_live_viewfileuploaded_without_a_parent_page_is_refused(session):
    with pytest.raises(MetadataError, match="parent act"):
        indiacode.resolve(session, LIVE_RULE_PDF, parent_url=None)


def test_live_viewfileuploaded_serves_a_pdf_from_an_india_code_host(session):
    """It 302s to India Code's file server; only the first bytes are read."""
    with session.get(LIVE_RULE_PDF, stream=True, timeout=(15, 60)) as response:
        response.raise_for_status()
        host = urlparse(response.url).hostname or ""
        assert host.endswith("indiacode.nic.in"), f"redirected off-site to {response.url}"
        assert "application/pdf" in response.headers.get("Content-Type", "")
        assert next(response.iter_content(8), b"").startswith(b"%PDF-")


# --- the malformed-redirect repair, against the real file server ----------------

# "Goa-IDC (Transfer & Sub-Lease Regulations), 2018": India Code's redirect for
# this file writes the ampersand into the query unescaped, so following the
# Location verbatim yields an HTML error page instead of the PDF.
AMPERSAND_PARENT = "https://www.indiacode.nic.in/handle/123456789/19802"
AMPERSAND_PDF = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_GA_65_853_00005_00005_1709710299779/regulationindividualfile/"
    "&file=goa-idc_%28transfer_%26_sub-lease_regulations%29%2C_2018_"
    "%28amended_upto_2021%29_%281%29.pdf"
)


def test_live_indiacode_still_emits_the_malformed_location(session):
    """The bug this repair exists for is still present upstream."""
    with session.get(AMPERSAND_PDF, allow_redirects=False, stream=True, timeout=(15, 60)) as r:
        location = r.headers.get("Location", "")
    assert "filename=" in location
    raw_name = location.rpartition("filename=")[2]
    assert "&" in raw_name, "India Code no longer emits an unescaped '&'; repair may be moot"
    assert repair_location(location) != location


def test_live_ampersand_document_downloads_as_a_real_pdf(session, tmp_path):
    item = indiacode.resolve(session, AMPERSAND_PDF, parent_url=AMPERSAND_PARENT)
    english = language.select_english_bitstream(item)
    assert english.language == "en"

    part = tmp_path / "goa.pdf.part"
    fetched = http_client.download_stream(session, english.url, part)
    assert part.read_bytes().startswith(b"%PDF-")
    assert "application/pdf" in fetched.content_type
    assert fetched.bytes > 1000
    # The repaired target is what was actually fetched.
    assert fetched.redirects and "%26" in fetched.redirects[-1]
    assert urlparse(fetched.final_url).hostname.endswith("indiacode.nic.in")


# --- sizing: header-only probing of the live site -------------------------------


def test_live_head_reports_the_size_of_a_bitstream_pdf(session):
    probe = sizing.probe_size(session, ENGLISH_PDF, kind="bitstream")
    assert probe.status == sizing.OK
    assert probe.content_length > 0
    assert probe.method == "head"


def test_live_viewfileuploaded_size_comes_from_headers_not_a_download(session):
    """HEAD is answered with a Location-less 302 here, so headers are read from
    a streaming GET whose body is never touched."""
    probe = sizing.probe_size(session, LIVE_RULE_PDF, kind="viewfileuploaded")
    assert probe.status == sizing.OK
    assert probe.content_length > 0
    assert probe.method == "redirect+head"
    assert urlparse(probe.final_url or "").hostname.endswith("indiacode.nic.in")


def test_live_act_page_still_yields_english_act_and_rules(session):
    listing = discovery.ActListing(
        handle="123456789/1372", url=HANDLE_URL, short_title="The Passports Act, 1967",
        collection_name="Central Acts", collection_handle="123456789/1362",
        document_type="central_act", jurisdiction="India",
    )
    result = discovery.inspect_act(session, listing)
    assert result.outcome == discovery.Outcome.ENGLISH_CONFIRMED

    act = next(e for e in result.entries if e.document_type == "central_act")
    assert act.english_pdf_url == ENGLISH_PDF

    rules = [e for e in result.entries if e.document_type == "rule"]
    assert rules, "the live Passports Act still publishes Rules"
    assert all(e.language == "en" for e in result.entries)
    assert all("hindifile" not in e.english_pdf_url.lower() for e in result.entries)


# --- recovering the corpus download's failures ----------------------------------
#
# The full-corpus run left 177 failures. These pin the live behaviour behind the
# ones that turned out to be recoverable, so a future regression is visible.

# A regulation whose file name contains a right single quotation mark. India
# Code's /ViewFileUploaded servlet answers 302 with *no Location header* for it.
NON_ASCII_PARENT = "https://www.indiacode.nic.in/handle/123456789/1373"
NON_ASCII_PDF = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_2_33_00044_193804_1523351752525/regulationindividualfile/"
    "&file=irdai_%28protection_of_policyholders%E2%80%99_interests%29_regulations%2C_2017.pdf"
)
# A rule whose name carries a raw cp1252 byte, which makes requests' own
# redirect handling raise UnicodeDecodeError.
UNDECODABLE_PARENT = "https://www.indiacode.nic.in/handle/123456789/11616"
UNDECODABLE_PDF = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_GA_65_1354_00001_00001_1565679096686/rulesindividualfile/"
    "&file=the_goa_ferries_%28regu%C2%AClation_of_issue_of_tickets%29_rules%2C_1990.pdf"
)
# A DSpace item whose browse listing writes the bitstream URL unencoded.
UNENCODED_BITSTREAM = (
    "https://www.indiacode.nic.in/bitstream/123456789/1999/1/A2000-21 (1).pdf"
)


def test_live_lookup_still_fails_to_redirect_a_non_ascii_filename(session):
    """The upstream fault the derived URL exists for is still present."""
    with session.get(
        NON_ASCII_PDF, allow_redirects=False, stream=True, timeout=(15, 60)
    ) as response:
        assert 300 <= response.status_code < 400
        assert not response.headers.get("Location"), (
            "India Code now emits a Location for non-ASCII names; the fallback "
            "may no longer be needed"
        )


def test_live_derived_showfile_url_serves_the_pdf(session, tmp_path):
    derived = indiacode.showfile_url(NON_ASCII_PDF)
    assert urlparse(derived).hostname.endswith("indiacode.nic.in")

    part = tmp_path / "irdai.pdf.part"
    fetched = http_client.download_stream(
        session, NON_ASCII_PDF, part, fallback_url=derived
    )
    assert part.read_bytes().startswith(b"%PDF-")
    assert "application/pdf" in fetched.content_type
    assert fetched.fallback_url == derived
    assert fetched.bytes > 1000


def test_live_derivation_reproduces_a_real_redirect(session):
    """On a document India Code *can* redirect, the derived URL is the target."""
    with session.get(
        LIVE_RULE_PDF, allow_redirects=False, stream=True, timeout=(15, 60)
    ) as response:
        location = response.headers.get("Location", "")
    assert location, "expected India Code to redirect this rule"
    derived = indiacode.showfile_url(LIVE_RULE_PDF)
    from urllib.parse import unquote

    assert unquote(urlparse(derived).query) == unquote(urlparse(location).query)


def test_live_undecodable_location_header_no_longer_crashes(session, tmp_path):
    part = tmp_path / "goa-ferries.pdf.part"
    fetched = http_client.download_stream(
        session, UNDECODABLE_PDF, part,
        fallback_url=indiacode.showfile_url(UNDECODABLE_PDF),
    )
    assert part.read_bytes().startswith(b"%PDF-")
    assert "application/pdf" in fetched.content_type
    assert urlparse(fetched.final_url).hostname.endswith("indiacode.nic.in")


def test_live_unencoded_bitstream_url_matches_the_page(session):
    """The item page links this file percent-encoded; both name the same file."""
    item = indiacode.resolve(session, UNENCODED_BITSTREAM)
    selected = language.select_english_bitstream(item)
    assert selected.language == "en"
    assert "%20%281%29" in selected.url, "expected the page's encoded form"


def test_live_missing_file_is_still_rejected_not_stored(session, tmp_path):
    """A rule India Code indexes but no longer holds: the file server answers
    HTTP 200 with an HTML page, and it must never reach the corpus."""
    missing = (
        "https://www.indiacode.nic.in/ViewFileUploaded"
        "?path=AC_CEN_2_11_00026_187305_1523269408953/rulesindividualfile/&file=2003.pdf"
    )
    part = tmp_path / "missing.pdf.part"
    with pytest.raises(NotPDFError):
        http_client.download_stream(
            session, missing, part, fallback_url=indiacode.showfile_url(missing)
        )
    assert not part.exists()
