"""Tests for discovery persistence: dedup, resumability and the summary.

The whole discovery run is exercised end to end with the network replaced by a
fixture router, so ``pytest`` needs no connectivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion import discovery, inventory
from ingestion.discovery import ActListing, ActResult, Collection, InventoryEntry, Outcome, Rejection
from ingestion.inventory import InventoryStore, deduplicate, run_discovery, summarise

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def entry(url="https://ic/bitstream/1/2/3/a.pdf", document_id="a", **kwargs) -> InventoryEntry:
    defaults = dict(
        document_id=document_id, title="A", short_title="A", document_type="central_act",
        jurisdiction="India", year=1967, act_number="15",
        india_code_url="https://ic/handle/123456789/1372",
        english_pdf_url=url, pdf_url_kind="bitstream", language="en",
        language_source="indiacode_metadata_title",
        sources=[{"via": "collection_browse", "handle": "123456789/1372"}],
    )
    defaults.update(kwargs)
    return InventoryEntry(**defaults)


def result(handle="123456789/1372", outcome=Outcome.ENGLISH_CONFIRMED, entries=None, rejections=None):
    return ActResult(handle, outcome, entries or [], rejections or [])


class TestDeduplicate:
    def test_same_pdf_url_becomes_one_download_job(self):
        first = entry(document_id="a")
        second = entry(document_id="b", sources=[{"via": "act_page_rules_table"}])
        entries, duplicates = deduplicate([result(entries=[first]), result("x", entries=[second])])
        assert len(entries) == 1
        assert duplicates == 1

    def test_every_route_to_a_duplicate_is_preserved(self):
        first = entry(sources=[{"via": "collection_browse", "handle": "1"}])
        second = entry(sources=[{"via": "act_page_rules_table", "handle": "2"}])
        entries, _ = deduplicate([result(entries=[first, second])])
        assert len(entries) == 1
        assert {s["via"] for s in entries[0].sources} == {
            "collection_browse", "act_page_rules_table",
        }

    def test_identical_sources_are_not_repeated(self):
        source = {"via": "collection_browse", "handle": "1"}
        entries, _ = deduplicate([result(entries=[entry(sources=[source]),
                                                  entry(sources=[dict(source)])])])
        assert len(entries[0].sources) == 1

    def test_duplicate_fills_in_gaps_of_the_first_sighting(self):
        sparse = entry(title=None, ministry=None)
        richer = entry(title="Passports Act, 1967", ministry="Ministry of External Affairs")
        entries, _ = deduplicate([result(entries=[sparse, richer])])
        assert entries[0].title == "Passports Act, 1967"
        assert entries[0].ministry == "Ministry of External Affairs"

    def test_different_pdfs_stay_separate(self):
        entries, duplicates = deduplicate(
            [result(entries=[entry(url="https://ic/a.pdf"), entry(url="https://ic/b.pdf")])]
        )
        assert len(entries) == 2
        assert duplicates == 0

    def test_trailing_space_in_a_url_is_not_a_new_document(self):
        entries, duplicates = deduplicate(
            [result(entries=[entry(url="https://ic/a.pdf"), entry(url="https://ic/a.pdf ")])]
        )
        assert len(entries) == 1
        assert duplicates == 1


class TestJournal:
    def test_append_and_reload_round_trip(self, tmp_path):
        store = InventoryStore(tmp_path / "data")
        store.ensure_layout()
        store.append_journal(result(entries=[entry()]))
        store.append_journal(result("123456789/9001", Outcome.HINDI_REJECTED,
                                    rejections=[Rejection("hindi_rejected", "central_act",
                                                          "India", "https://ic/x")]))
        reloaded = store.load_journal()
        assert set(reloaded) == {"123456789/1372", "123456789/9001"}
        assert reloaded["123456789/1372"].entries[0].english_pdf_url.endswith("a.pdf")
        assert reloaded["123456789/9001"].rejections[0].reason == "hindi_rejected"

    def test_interrupted_final_line_is_discarded(self, tmp_path):
        store = InventoryStore(tmp_path / "data")
        store.ensure_layout()
        store.append_journal(result(entries=[entry()]))
        # Simulate a process killed mid-append.
        with open(store.journal_path, "a", encoding="utf-8") as handle:
            handle.write('{"handle": "123456789/9999", "outcome": "eng')
        reloaded = store.load_journal()
        assert set(reloaded) == {"123456789/1372"}

    def test_corrupt_middle_line_is_skipped_not_fatal(self, tmp_path):
        store = InventoryStore(tmp_path / "data")
        store.ensure_layout()
        store.append_journal(result(entries=[entry()]))
        with open(store.journal_path, "a", encoding="utf-8") as handle:
            handle.write("not json at all\n")
        store.append_journal(result("123456789/2000"))
        reloaded = store.load_journal()
        assert set(reloaded) == {"123456789/1372", "123456789/2000"}

    def test_missing_journal_is_an_empty_start(self, tmp_path):
        assert InventoryStore(tmp_path / "data").load_journal() == {}


class TestSummary:
    def test_counts_every_reported_bucket(self):
        entries = [
            entry(document_id="c1", document_type="central_act"),
            entry(document_id="s1", document_type="state_act", jurisdiction="Kerala"),
            entry(document_id="s2", document_type="state_act", jurisdiction="Kerala"),
            entry(document_id="s3", document_type="state_act", jurisdiction="Goa"),
            entry(document_id="r1", document_type="rule"),
            entry(document_id="g1", document_type="regulation"),
        ]
        results = [
            result(entries=entries),
            result("h2", Outcome.HINDI_REJECTED,
                   rejections=[Rejection(Outcome.HINDI_REJECTED, "central_act", "India", "u")]),
            result("h3", Outcome.AMBIGUOUS,
                   rejections=[Rejection(Outcome.AMBIGUOUS, "central_act", "India", "u")]),
            result("h4", Outcome.NO_PDF,
                   rejections=[Rejection(Outcome.NO_PDF, "rule", "India", "u")]),
            result("h5", Outcome.ERROR),
        ]
        summary = summarise(results, entries, [], [], duplicates=3, complete=True)
        assert summary["total_discovered"] == 6
        assert summary["central_acts"] == 1
        assert summary["state_acts"] == 3
        assert summary["state_acts_by_state"] == {"Goa": 1, "Kerala": 2}
        assert summary["rules"] == 1
        assert summary["regulations"] == 1
        assert summary["english_confirmed"] == 6
        assert summary["hindi_rejected"] == 1
        assert summary["ambiguous_rejected"] == 1
        assert summary["no_pdf_available"] == 1
        assert summary["duplicates_merged"] == 3
        assert summary["errors"] == 1


class TestWriteInventory:
    def test_inventory_file_shape(self, tmp_path):
        store = InventoryStore(tmp_path / "data")
        store.ensure_layout()
        collection = Collection("Central Acts", "123456789/1362", "central_act", "India", "u")
        payload = store.write_inventory([result(entries=[entry()])], [collection], [], complete=True)

        assert store.inventory_path.name == "indiacode_inventory.json"
        on_disk = json.loads(store.inventory_path.read_text(encoding="utf-8"))
        assert on_disk == payload
        assert on_disk["scope"]["language"] == "en"
        assert on_disk["complete"] is True
        assert len(on_disk["documents"]) == 1
        assert on_disk["documents"][0]["language"] == "en"
        assert on_disk["documents"][0]["language_source"]

    def test_every_document_carries_its_provenance(self, tmp_path):
        store = InventoryStore(tmp_path / "data")
        store.ensure_layout()
        payload = store.write_inventory([result(entries=[entry()])], [], [], complete=True)
        document = payload["documents"][0]
        for key in ("india_code_url", "english_pdf_url", "sources", "document_id",
                    "document_type", "jurisdiction", "language", "language_source"):
            assert document.get(key), f"missing {key}"


# --- full run, with the network replaced by fixtures ----------------------------


ACT_PAGES = {
    "123456789/1372": "act_with_subordinate_1372.html",   # English + Hindi + rules
    "123456789/8100": "act_subordinate_edge_cases.html",  # mixed subordinate cases
    "123456789/9001": "hindi_only_9001.html",             # Hindi only
    "123456789/9002": "ambiguous_language_9002.html",     # no language evidence
}


@pytest.fixture
def offline(monkeypatch):
    """Route every request to a fixture and count the pages fetched."""
    fetched: list[str] = []

    def fake_get_text(session, url):
        fetched.append(url)
        if "/community-list" in url:
            return read("indiacode_nav.html")
        if "/browse?" in url:
            offset = int(url.rsplit("offset=", 1)[1])
            if offset > 0 or "1362" not in url:
                return "<html></html>"
            rows = "".join(
                '<tr><td>1-Jan-1967</td><td>%d</td><td>Act %d</td>'
                '<td><a href="/handle/%s?view_type=browse">View...</a></td></tr>'
                % (i, i, handle)
                for i, handle in enumerate(ACT_PAGES, 1)
            )
            return "<html><body><table>%s</table></body></html>" % rows
        for handle, fixture in ACT_PAGES.items():
            if handle in url:
                return read(fixture)
        raise AssertionError("unexpected URL: %s" % url)

    monkeypatch.setattr(discovery, "get_text", fake_get_text)
    monkeypatch.setattr(inventory, "build_session", lambda: object())
    return fetched


class TestRunDiscovery:
    def test_full_run_produces_the_inventory(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        payload = run_discovery(store, workers=1, checkpoint_every=0)

        summary = payload["summary"]
        assert summary["complete"] is True
        assert summary["acts_inspected"] == 4
        assert summary["central_acts"] == 2          # 1372 and 8100
        assert summary["rules"] == 6                 # 5 from 1372, 1 from 8100
        assert summary["hindi_rejected"] >= 1
        assert summary["ambiguous_rejected"] >= 1
        assert summary["no_pdf_available"] >= 1
        assert store.inventory_path.exists()

    def test_no_pdf_is_requested_during_discovery(self, tmp_path, offline):
        run_discovery(InventoryStore(tmp_path / "data"), workers=1, checkpoint_every=0)
        assert not [url for url in offline if url.lower().endswith(".pdf")]
        assert not [url for url in offline if "ViewFileUploaded" in url]
        assert not [url for url in offline if "/bitstream/" in url]

    def test_inventory_contains_no_hindi_document(self, tmp_path, offline):
        payload = run_discovery(InventoryStore(tmp_path / "data"), workers=1, checkpoint_every=0)
        assert all(d["language"] == "en" for d in payload["documents"])
        assert all(d["language_source"] for d in payload["documents"])
        blob = json.dumps(payload["documents"])
        assert "ruleshindifile" not in blob
        assert "H1967-15.pdf" not in blob

    def test_second_run_resumes_and_refetches_nothing(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, checkpoint_every=0)
        offline.clear()
        payload = run_discovery(store, workers=1, checkpoint_every=0)
        # Listings are cached and every act is journalled: no page is re-read.
        assert offline == []
        assert payload["summary"]["acts_inspected"] == 4

    def test_interrupted_run_continues_from_the_journal(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        first = run_discovery(store, workers=1, limit=2, checkpoint_every=0)
        assert first["summary"]["acts_inspected"] == 2
        assert first["summary"]["complete"] is False

        act_pages_read = [u for u in offline if "/handle/" in u and "browse" not in u]
        assert len(act_pages_read) == 2

        offline.clear()
        second = run_discovery(store, workers=1, checkpoint_every=0)
        assert second["summary"]["acts_inspected"] == 4
        assert second["summary"]["complete"] is True
        # Only the two acts left over are fetched the second time.
        assert len([u for u in offline if "/handle/" in u and "browse" not in u]) == 2

    def test_partial_inventory_is_still_valid_json(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, limit=1, checkpoint_every=0)
        payload = json.loads(store.inventory_path.read_text(encoding="utf-8"))
        assert payload["complete"] is False
        assert payload["documents"]

    def test_restart_discards_previous_progress(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, limit=1, checkpoint_every=0)
        assert store.journal_path.exists()
        offline.clear()
        run_discovery(store, workers=1, limit=1, restart=True, checkpoint_every=0)
        # The nav and browse pages are read again after a restart.
        assert any("/community-list" in url for url in offline)

    def test_recheck_reinspects_only_unconfirmed_acts(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, checkpoint_every=0)
        unconfirmed = [
            handle for handle, result in store.load_journal().items()
            if result.outcome != Outcome.ENGLISH_CONFIRMED
        ]
        assert unconfirmed, "fixtures include Hindi-only and ambiguous acts"

        offline.clear()
        run_discovery(store, workers=1, recheck=True, checkpoint_every=0)
        refetched = [u for u in offline if "/handle/" in u and "browse" not in u]
        assert len(refetched) == len(unconfirmed)
        assert all(any(h in u for h in unconfirmed) for u in refetched)

    def test_recheck_keeps_the_latest_result_for_an_act(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, checkpoint_every=0)
        payload = run_discovery(store, workers=1, recheck=True, checkpoint_every=0)
        # Re-inspection appends fresh journal lines; the newest wins, so nothing
        # is double-counted.
        assert payload["summary"]["acts_inspected"] == 4
        assert len({d["document_id"] for d in payload["documents"]}) == len(payload["documents"])

    def test_collection_filter(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        payload = run_discovery(
            store, workers=1, collections_filter=["Central Acts"], checkpoint_every=0
        )
        assert payload["summary"]["collections"] == 1
        assert all(d["jurisdiction"] == "India" for d in payload["documents"])

    def test_checkpoint_writes_inventory_mid_run(self, tmp_path, offline):
        store = InventoryStore(tmp_path / "data")
        run_discovery(store, workers=1, checkpoint_every=1)
        assert store.inventory_path.exists()
