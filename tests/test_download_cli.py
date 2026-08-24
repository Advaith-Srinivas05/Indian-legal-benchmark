"""Tests for CLI helpers: type resolution and JSON/CSV input parsing."""

from __future__ import annotations

import json

import pytest

from ingestion import download
from ingestion.errors import DocumentTypeError, IngestionError


class TestResolveCategory:
    @pytest.mark.parametrize("value,expected", [
        ("central_act", "central_acts"),
        ("central_acts", "central_acts"),
        ("STATE_ACT", "state_acts"),
        ("rule", "rules"),
        ("regulation", "regulations"),
    ])
    def test_valid_types(self, value, expected):
        assert download.resolve_category(value) == expected

    def test_missing_type_raises(self):
        with pytest.raises(DocumentTypeError):
            download.resolve_category(None)

    def test_unknown_type_raises(self):
        with pytest.raises(DocumentTypeError):
            download.resolve_category("judgment")


class TestLoadInputFile:
    def test_json_list(self, tmp_path):
        path = tmp_path / "urls.json"
        path.write_text(json.dumps([
            {"url": "https://www.indiacode.nic.in/handle/123456789/1372", "type": "central_act"},
            {"url": "https://www.indiacode.nic.in/handle/123456789/2000", "type": "rule"},
        ]), encoding="utf-8")
        targets = download.load_input_file(path, default_type=None)
        assert [(t.url, t.category) for t in targets] == [
            ("https://www.indiacode.nic.in/handle/123456789/1372", "central_acts"),
            ("https://www.indiacode.nic.in/handle/123456789/2000", "rules"),
        ]
        assert all(t.parent_url is None for t in targets)

    def test_json_object_with_documents_key(self, tmp_path):
        path = tmp_path / "urls.json"
        path.write_text(json.dumps({"documents": [
            {"url": "https://www.indiacode.nic.in/handle/123456789/1372"},
        ]}), encoding="utf-8")
        targets = download.load_input_file(path, default_type="central_act")
        assert [(t.url, t.category) for t in targets] == [
            ("https://www.indiacode.nic.in/handle/123456789/1372", "central_acts")
        ]

    def test_csv(self, tmp_path):
        path = tmp_path / "urls.csv"
        path.write_text(
            "url,type\n"
            "https://www.indiacode.nic.in/handle/123456789/1372,central_act\n",
            encoding="utf-8",
        )
        targets = download.load_input_file(path, default_type=None)
        assert [(t.url, t.category) for t in targets] == [
            ("https://www.indiacode.nic.in/handle/123456789/1372", "central_acts")
        ]

    def test_missing_url_raises(self, tmp_path):
        path = tmp_path / "urls.json"
        path.write_text(json.dumps([{"type": "central_act"}]), encoding="utf-8")
        with pytest.raises(IngestionError):
            download.load_input_file(path, default_type=None)

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "urls.txt"
        path.write_text("whatever", encoding="utf-8")
        with pytest.raises(IngestionError):
            download.load_input_file(path, default_type="central_act")
