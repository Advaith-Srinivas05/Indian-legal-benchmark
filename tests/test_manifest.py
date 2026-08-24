"""Tests for manifest load/save, upsert idempotency and hash lookup."""

from __future__ import annotations

import json

from ingestion.manifest import Manifest


def _entry(document_id: str, sha256: str) -> dict:
    return {"document_id": document_id, "sha256": sha256, "title": "X"}


class TestManifestRoundTrip:
    def test_load_missing_creates_empty(self, tmp_path):
        m = Manifest.load(tmp_path / "manifest.json")
        assert m.documents == {}
        assert "schema_version" in m._data

    def test_save_then_load_preserves_documents(self, tmp_path):
        path = tmp_path / "manifest.json"
        m = Manifest.load(path)
        m.upsert(_entry("doc-1", "abc"))
        m.save()

        reloaded = Manifest.load(path)
        assert "doc-1" in reloaded.documents
        assert reloaded.get("doc-1")["sha256"] == "abc"
        # File is valid JSON on disk.
        json.loads(path.read_text(encoding="utf-8"))


class TestUpsertAndLookup:
    def test_upsert_is_idempotent_by_id(self, tmp_path):
        m = Manifest.load(tmp_path / "manifest.json")
        m.upsert(_entry("doc-1", "abc"))
        m.upsert(_entry("doc-1", "abc"))
        assert len(m.documents) == 1

    def test_upsert_replaces_same_id(self, tmp_path):
        m = Manifest.load(tmp_path / "manifest.json")
        m.upsert(_entry("doc-1", "abc"))
        m.upsert(_entry("doc-1", "def"))
        assert m.get("doc-1")["sha256"] == "def"

    def test_find_by_sha256(self, tmp_path):
        m = Manifest.load(tmp_path / "manifest.json")
        m.upsert(_entry("doc-1", "abc"))
        m.upsert(_entry("doc-2", "xyz"))
        assert m.find_by_sha256("xyz")["document_id"] == "doc-2"
        assert m.find_by_sha256("nope") is None
