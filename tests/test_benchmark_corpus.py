"""The corpus runner and verifier, end to end over a miniature data directory."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark import config
from benchmark.corpus import build_corpus, main, verify_corpus
from tests.processedbuild import (ACT_PAGE_ONE, ACT_PAGE_TWO, make_page, rewrite_document,
                                  write_processed, write_raw_metadata)

GOOD = ["alpha-act-1999__handle-1", "beta-act-2001__handle-2"]
RULE = "sample-rules-2002__rule-abc1234567"
QUARANTINED = "gamma-act-2003__handle-3"


def make_data(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    for doc in GOOD:
        write_processed(data, doc, [make_page(1, ACT_PAGE_ONE), make_page(2, ACT_PAGE_TWO)])
        write_raw_metadata(data, doc)
    write_processed(data, RULE, [make_page(1, ACT_PAGE_ONE)], category="rules",
                    document_type="rule", title="The Sample Rules, 2002")
    write_raw_metadata(data, RULE, category="rules", handle="123456789/1876",
                       parent_handle="123456789/1876", parent_title="Sample Act, 1999",
                       parent_url="https://www.indiacode.nic.in/handle/123456789/1876")
    write_processed(data, QUARANTINED, [make_page(1, ACT_PAGE_ONE)], eligible=False)
    return data


def snapshot(directory: Path) -> dict[str, bytes]:
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in sorted(directory.rglob("*")) if p.is_file()}


def test_build_writes_text_structure_and_meta_for_each_eligible_document(tmp_path):
    data = make_data(tmp_path)
    report = build_corpus(data, data / "corpus")
    assert report["documents"] == 3 and report["failures"] == []
    for doc in GOOD + [RULE]:
        for sub, ext in (("text", "txt"), ("meta", "json"), ("structure", "json")):
            assert (data / "corpus" / sub / f"{doc}.{ext}").exists()
    rows = (data / "corpus" / config.DOCUMENTS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert [json.loads(r)["document_id"] for r in rows] == sorted(GOOD + [RULE])


def test_a_quarantined_document_is_not_in_the_corpus(tmp_path):
    data = make_data(tmp_path)
    report = build_corpus(data, data / "corpus")
    assert report["this_run"]["not_eligible"] == 1
    assert not (data / "corpus" / "text" / f"{QUARANTINED}.txt").exists()


def test_a_broken_document_fails_loudly_and_writes_nothing(tmp_path):
    data = make_data(tmp_path)
    rewrite_document(data, GOOD[0], lambda d: d["structure"]["counts"].update(line_count=1))
    report = build_corpus(data, data / "corpus")
    assert [f["document_id"] for f in report["failures"]] == [GOOD[0]]
    assert report["failures"][0]["error_type"] == "LineStreamMismatchError"
    for sub in ("text", "meta", "structure"):
        assert not list((data / "corpus" / sub).glob(f"{GOOD[0]}.*"))
    assert (data / "corpus" / "meta" / f"{GOOD[1]}.json").exists()
    assert main(["--data-dir", str(data), "build", "--workers", "1", "--force"]) == 1


def test_rebuilding_is_byte_identical(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, tmp_path / "first")
    build_corpus(data, tmp_path / "second", workers=2)
    assert snapshot(tmp_path / "first") == snapshot(tmp_path / "second")


def test_a_resumed_build_skips_documents_already_built(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    report = build_corpus(data, data / "corpus")
    assert report["this_run"]["already_built"] == 3
    assert report["documents"] == 3


def test_text_is_written_with_unix_newlines_on_every_platform(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    raw = (data / "corpus" / "text" / f"{GOOD[0]}.txt").read_bytes()
    assert b"\r" not in raw and b"\n" in raw


def test_a_rule_does_not_claim_its_parent_acts_handle(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    rule = json.loads((data / "corpus" / "meta" / f"{RULE}.json").read_text(encoding="utf-8"))
    act = json.loads((data / "corpus" / "meta" / f"{GOOD[0]}.json").read_text(encoding="utf-8"))
    assert rule["source"]["handle"] is None and rule["source"]["handle_uri"] is None
    assert rule["parent"]["handle"] == "123456789/1876"
    assert act["source"]["handle_uri"].startswith("http://hdl.handle.net/")
    assert act["long_title"] == "An Act to provide for the regulation of samples."


def test_verify_passes_on_a_fresh_build(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    result = verify_corpus(data / "corpus")
    assert result["problems"] == [] and result["documents"] == 3 and result["provisions"] > 0


def test_verify_catches_a_tampered_text_file(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    path = data / "corpus" / "text" / f"{GOOD[0]}.txt"
    path.write_bytes(path.read_bytes().replace(b"Sample", b"Simple", 1))
    problems = verify_corpus(data / "corpus")["problems"]
    assert any("checksum mismatch" in p for p in problems)
    assert any("text hash differs" in p for p in problems)


def test_checksums_cover_every_published_file(tmp_path):
    data = make_data(tmp_path)
    build_corpus(data, data / "corpus")
    corpus = data / "corpus"
    listed = {line.split("  ", 1)[1] for line in
              (corpus / config.CHECKSUMS_FILENAME).read_text(encoding="utf-8").splitlines()}
    expected = {p.relative_to(corpus).as_posix() for sub in ("text", "meta", "structure")
                for p in (corpus / sub).iterdir()} | {config.DOCUMENTS_FILENAME}
    assert listed == expected
