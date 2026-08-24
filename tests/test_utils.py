"""Tests for the pure helpers: slugs, safe filenames, IDs and hashing."""

from __future__ import annotations

import hashlib

import pytest

from ingestion import utils


class TestSlugify:
    def test_basic_title(self):
        assert utils.slugify("Passports Act, 1967") == "passports-act-1967"

    def test_collapses_and_trims_separators(self):
        assert utils.slugify("  The   Code!! of --- Conduct  ") == "the-code-of-conduct"

    def test_transliterates_non_ascii_away(self):
        # Devanagari has no ASCII form -> empty slug (caller falls back to handle).
        assert utils.slugify("पासपोर्ट अधिनियम") == ""

    def test_respects_max_length_without_trailing_hyphen(self):
        slug = utils.slugify("a" * 50 + " " + "b" * 50, max_length=20)
        assert len(slug) <= 20
        assert not slug.endswith("-")


class TestSafeFilename:
    @pytest.mark.parametrize("bad", ['a/b', 'a\\b', 'a:b', 'a*b', 'a?b', 'a"b', 'a<b>c', "a|b"])
    def test_replaces_reserved_characters(self, bad):
        result = utils.safe_filename(bad)
        for ch in '<>:"/\\|?*':
            assert ch not in result

    def test_strips_trailing_dot_and_space(self):
        assert utils.safe_filename("report. ") == "report"

    def test_guards_windows_reserved_names(self):
        assert utils.safe_filename("CON").lower() != "con"
        assert utils.safe_filename("con.pdf").lower().startswith("_")

    def test_empty_falls_back(self):
        assert utils.safe_filename("") == "document"


class TestMakeDocumentId:
    def test_combines_slug_and_handle(self):
        assert utils.make_document_id("Passports Act, 1967", "1372") == \
            "passports-act-1967__handle-1372"

    def test_falls_back_to_handle_when_title_unusable(self):
        assert utils.make_document_id("पासपोर्ट", "1372") == "handle-1372"
        assert utils.make_document_id(None, "1372") == "handle-1372"

    def test_is_deterministic(self):
        a = utils.make_document_id("Some Act, 2020", "9999")
        b = utils.make_document_id("Some Act, 2020", "9999")
        assert a == b

    def test_result_is_filesystem_safe(self):
        doc_id = utils.make_document_id('Weird/Name: "Act"', "12/34")
        for ch in '<>:"/\\|?*':
            assert ch not in doc_id


class TestHashing:
    def test_sha256_bytes_matches_hashlib(self):
        data = b"hello legal world"
        assert utils.sha256_bytes(data) == hashlib.sha256(data).hexdigest()

    def test_sha256_file_matches_bytes(self, tmp_path):
        data = b"%PDF-1.4 dummy content %%EOF"
        path = tmp_path / "x.pdf"
        path.write_bytes(data)
        assert utils.sha256_file(path) == utils.sha256_bytes(data)

    def test_sha256_file_streaming_consistent_for_large_input(self, tmp_path):
        data = b"A" * (5 * 1024 * 1024) + b"B"
        path = tmp_path / "big.bin"
        path.write_bytes(data)
        assert utils.sha256_file(path, chunk_size=4096) == hashlib.sha256(data).hexdigest()


class TestAtomicWrite:
    def test_atomic_write_text_creates_parents_and_content(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.txt"
        utils.atomic_write_text(target, "content")
        assert target.read_text(encoding="utf-8") == "content"

    def test_atomic_write_leaves_no_temp_file(self, tmp_path):
        target = tmp_path / "c.txt"
        utils.atomic_write_text(target, "content")
        assert list(tmp_path.glob(".*.tmp")) == []
