"""Offline tests for the header-only size estimate.

Every HTTP interaction is faked. The point of these is not only that the
arithmetic is right, but that the probe *never* asks for a body: the fake
responses raise if anything tries to read one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from ingestion import estimate_size, sizing
from ingestion.sizing import ERROR, OK, UNKNOWN, Probe

BITSTREAM = "https://www.indiacode.nic.in/bitstream/123456789/1372/1/196715.pdf"
VIEWFILE = (
    "https://www.indiacode.nic.in/ViewFileUploaded"
    "?path=AC_CEN_10_10_00008_196715_1517807321481/rulesindividualfile/"
    "&file=The+Passport+Rules%2C+1967+%2810.05.1967%29.pdf"
)
REDIRECT_TARGET = (
    "http://upload.indiacode.nic.in/showfile"
    "?actid=AC_CEN_10_10_00008_196715_1517807321481&type=rule"
    "&filename=The Passport Rules, 1967 (10.05.1967).pdf"
)


class FakeResponse:
    """A response whose body cannot be read without failing the test."""

    def __init__(self, status_code=200, headers=None, url=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_content(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the size probe must not read any response body")

    @property
    def content(self):  # pragma: no cover - must never run
        raise AssertionError("the size probe must not read any response body")

    def raise_for_status(self):  # pragma: no cover - probe checks the code itself
        raise AssertionError("the probe inspects status_code rather than raising")


class FakeSession:
    """Serves canned responses and records the calls it was asked to make."""

    def __init__(self, head=None, get=None):
        self._head = head or {}
        self._get = get or {}
        self.calls: list[tuple[str, str, dict]] = []

    def head(self, url, **kwargs):
        self.calls.append(("HEAD", url, kwargs))
        response = self._head.get(url)
        if response is None:
            raise AssertionError(f"unexpected HEAD {url}")
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        response = self._get.get(url)
        if response is None:
            raise AssertionError(f"unexpected GET {url}")
        if isinstance(response, Exception):
            raise response
        return response


def pdf_head(length, url=BITSTREAM, content_type="application/pdf"):
    headers = {"Content-Type": content_type}
    if length is not None:
        headers["Content-Length"] = str(length)
    return FakeResponse(200, headers, url)


# --- sampling -------------------------------------------------------------------


def make_documents():
    documents = []
    documents += [
        {"document_id": f"c{i}", "document_type": "central_act", "jurisdiction": "India",
         "english_pdf_url": f"{BITSTREAM}?i={i}", "pdf_url_kind": "bitstream"}
        for i in range(50)
    ]
    for state in ("Assam", "Kerala", "Punjab"):
        documents += [
            {"document_id": f"s{state}{i}", "document_type": "state_act", "jurisdiction": state,
             "english_pdf_url": f"{BITSTREAM}?{state}{i}", "pdf_url_kind": "bitstream"}
            for i in range(40)
        ]
    documents += [
        {"document_id": f"r{i}", "document_type": "rule", "jurisdiction": "India",
         "english_pdf_url": f"{VIEWFILE}&i={i}", "pdf_url_kind": "viewfileuploaded"}
        for i in range(30)
    ]
    documents += [
        {"document_id": f"g{i}", "document_type": "regulation", "jurisdiction": "India",
         "english_pdf_url": f"{VIEWFILE}&g={i}", "pdf_url_kind": "viewfileuploaded"}
        for i in range(5)
    ]
    return documents


class TestStratifiedSample:
    def test_every_document_type_is_represented(self):
        sample = sizing.stratified_sample(make_documents(), per_type=10)
        types = {d["document_type"] for d in sample}
        assert types == {"central_act", "state_act", "rule", "regulation"}

    def test_per_type_cap_is_respected_and_small_types_are_not_padded(self):
        sample = sizing.stratified_sample(make_documents(), per_type=10)
        counts = {}
        for document in sample:
            counts[document["document_type"]] = counts.get(document["document_type"], 0) + 1
        assert counts["central_act"] == 10
        assert counts["state_act"] == 10
        assert counts["rule"] == 10
        # Only five regulations exist; the sample takes all of them, not more.
        assert counts["regulation"] == 5

    def test_jurisdictions_are_spread_rather_than_sorted_into_the_first_state(self):
        sample = sizing.stratified_sample(make_documents(), per_type=6)
        states = {d["jurisdiction"] for d in sample if d["document_type"] == "state_act"}
        assert states == {"Assam", "Kerala", "Punjab"}

    def test_sampling_is_reproducible_for_a_seed(self):
        documents = make_documents()
        first = sizing.stratified_sample(documents, per_type=15, seed=7)
        second = sizing.stratified_sample(documents, per_type=15, seed=7)
        other = sizing.stratified_sample(documents, per_type=15, seed=8)
        assert [d["document_id"] for d in first] == [d["document_id"] for d in second]
        assert [d["document_id"] for d in first] != [d["document_id"] for d in other]

    def test_no_document_is_sampled_twice(self):
        sample = sizing.stratified_sample(make_documents(), per_type=1000)
        ids = [d["document_id"] for d in sample]
        assert len(ids) == len(set(ids)) == len(make_documents())


# --- probing --------------------------------------------------------------------


class TestProbeBitstream:
    def test_head_returns_the_content_length(self):
        session = FakeSession(head={BITSTREAM: pdf_head(254962)})
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == OK
        assert probe.content_length == 254962
        assert probe.method == "head"
        assert [call[0] for call in session.calls] == ["HEAD"]

    def test_head_without_a_content_length_falls_back_to_headers_only_get(self):
        session = FakeSession(
            head={BITSTREAM: pdf_head(None)},
            get={BITSTREAM: pdf_head(4242)},
        )
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == OK
        assert probe.content_length == 4242
        assert probe.method == "stream-headers"
        assert session.calls[1][2]["stream"] is True

    def test_a_locationless_302_on_head_falls_back_to_headers_only_get(self):
        """India Code answers HEAD for some bitstreams with a bare 302."""
        session = FakeSession(
            head={BITSTREAM: FakeResponse(302, {}, BITSTREAM)},
            get={BITSTREAM: pdf_head(106025237)},
        )
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == OK
        assert probe.content_length == 106025237
        assert probe.method == "stream-headers"
        assert probe.reason is None
        assert "no followable Location" in probe.fallback_reason

    def test_size_stays_unknown_rather_than_zero_when_nothing_reports_it(self):
        session = FakeSession(
            head={BITSTREAM: pdf_head(None)},
            get={BITSTREAM: pdf_head(None)},
        )
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == UNKNOWN
        assert probe.content_length is None
        assert "no Content-Length" in probe.reason

    def test_non_pdf_response_is_unknown(self):
        session = FakeSession(head={BITSTREAM: pdf_head(1200, content_type="text/html")})
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == UNKNOWN
        assert "not a PDF" in probe.reason

    def test_http_error_is_reported_as_an_error(self):
        session = FakeSession(head={BITSTREAM: FakeResponse(404, {}, BITSTREAM)})
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == ERROR
        assert "HTTP 404" in probe.reason

    def test_network_failure_is_captured_not_raised(self):
        session = FakeSession(head={BITSTREAM: requests.ConnectionError("boom")})
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == ERROR
        assert "boom" in probe.reason

    def test_a_redirect_off_india_code_is_refused(self):
        elsewhere = FakeResponse(200, {"Content-Type": "application/pdf",
                                       "Content-Length": "10"}, "https://evil.example/x.pdf")
        session = FakeSession(head={BITSTREAM: elsewhere})
        probe = sizing.probe_size(session, BITSTREAM, kind="bitstream")
        assert probe.status == ERROR
        assert "off India Code" in probe.reason


class TestProbeViewFileUploaded:
    def test_redirect_is_resolved_then_headed(self):
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": REDIRECT_TARGET}, VIEWFILE)},
            head={REDIRECT_TARGET: pdf_head(30515565, url=REDIRECT_TARGET)},
        )
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == OK
        assert probe.content_length == 30515565
        assert probe.method == "redirect+head"
        # The redirect is read with a streaming, non-following GET; the body of
        # the PDF itself is only ever asked about with HEAD.
        assert [call[0] for call in session.calls] == ["GET", "HEAD"]
        get_kwargs = session.calls[0][2]
        assert get_kwargs["allow_redirects"] is False
        assert get_kwargs["stream"] is True

    def test_query_string_is_passed_through_untouched(self):
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": REDIRECT_TARGET}, VIEWFILE)},
            head={REDIRECT_TARGET: pdf_head(1, url=REDIRECT_TARGET)},
        )
        sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert session.calls[0][1] == VIEWFILE  # not re-encoded or reassembled

    def test_relative_location_is_resolved_against_the_request(self):
        relative = "/showfile?actid=X&type=rule"
        absolute = "https://www.indiacode.nic.in/showfile?actid=X&type=rule"
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": relative}, VIEWFILE)},
            head={absolute: pdf_head(4096, url=absolute)},
        )
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == OK
        assert probe.content_length == 4096

    def test_redirect_without_a_location_is_unknown(self):
        session = FakeSession(get={VIEWFILE: FakeResponse(302, {}, VIEWFILE)})
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == UNKNOWN
        assert "without a Location" in probe.reason

    def test_redirect_leaving_india_code_is_refused(self):
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": "https://evil.example/x.pdf"}, VIEWFILE)}
        )
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == ERROR
        assert "leaves India Code" in probe.reason
        assert [call[0] for call in session.calls] == ["GET"]

    def test_http_error_on_the_viewfile_url_is_an_error(self):
        session = FakeSession(get={VIEWFILE: FakeResponse(500, {}, VIEWFILE)})
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == ERROR
        assert "HTTP 500" in probe.reason

    def test_inline_pdf_without_a_redirect_is_sized_from_its_headers(self):
        inline = FakeResponse(
            200, {"Content-Type": "application/pdf", "Content-Length": "8192"}, VIEWFILE
        )
        session = FakeSession(get={VIEWFILE: inline})
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == OK
        assert probe.content_length == 8192
        assert probe.method == "stream-headers"

    def test_an_ampersand_in_the_redirect_filename_is_escaped(self):
        """India Code writes the file name into the redirect unescaped, so a
        name containing '&' would otherwise truncate and 404."""
        raw = (
            "http://upload.indiacode.nic.in/showfile?actid=AC_GA_65&type=regulation"
            "&filename=goa-idc_(transfer_&_sub-lease_regulations),_2018.pdf"
        )
        repaired = raw.replace("_&_sub", "_%26_sub")
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": raw}, VIEWFILE)},
            head={repaired: pdf_head(813904, url=repaired)},
        )
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == OK
        assert probe.content_length == 813904

    def test_a_redirect_without_an_ampersand_is_left_alone(self):
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": REDIRECT_TARGET}, VIEWFILE)},
            head={REDIRECT_TARGET: pdf_head(10, url=REDIRECT_TARGET)},
        )
        assert sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded").status == OK
        assert session.calls[1][1] == REDIRECT_TARGET

    def test_repair_only_touches_the_filename_value(self):
        assert sizing.repair_location("https://h/x?a=1&b=2") == "https://h/x?a=1&b=2"
        assert sizing.repair_location("https://h/x?a=1&filename=p&q.pdf") == (
            "https://h/x?a=1&filename=p%26q.pdf"
        )

    def test_a_redirect_loop_is_abandoned_rather_than_followed_forever(self):
        loop = "https://www.indiacode.nic.in/loop"
        session = FakeSession(
            get={VIEWFILE: FakeResponse(302, {"Location": loop}, VIEWFILE),
                 loop: FakeResponse(302, {"Location": loop}, loop)},
            head={loop: FakeResponse(302, {}, loop)},
        )
        probe = sizing.probe_size(session, VIEWFILE, kind="viewfileuploaded")
        assert probe.status == ERROR
        assert "redirects" in probe.reason


def test_probing_writes_no_files(tmp_path, monkeypatch):
    """A whole probing run must leave the filesystem untouched."""
    monkeypatch.chdir(tmp_path)
    session = FakeSession(head={BITSTREAM: pdf_head(1000)})
    monkeypatch.setattr(sizing.SessionPool, "get", lambda self: session)
    documents = [
        {"document_id": "c1", "document_type": "central_act", "jurisdiction": "India",
         "english_pdf_url": BITSTREAM, "pdf_url_kind": "bitstream"}
    ]
    probes = sizing.probe_documents(documents, workers=1, delay=0)
    assert [p.status for p in probes] == [OK]
    assert list(tmp_path.iterdir()) == []


def test_rate_limiter_spaces_requests_out():
    limiter = sizing.RateLimiter(0.05)
    import time

    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    assert time.monotonic() - start >= 0.09  # two enforced gaps


# --- statistics -----------------------------------------------------------------


class TestStatistics:
    def test_percentiles_interpolate(self):
        values = [1, 2, 3, 4]
        assert sizing.percentile(values, 0) == 1
        assert sizing.percentile(values, 50) == 2.5
        assert sizing.percentile(values, 100) == 4
        assert sizing.percentile([], 50) is None

    def test_describe_reports_the_full_shape(self):
        stats = sizing.describe([10, 20, 30, 40, 1000])
        assert stats["count"] == 5
        assert stats["min_bytes"] == 10
        assert stats["max_bytes"] == 1000
        assert stats["median_bytes"] == 30
        assert stats["mean_bytes"] == 220
        assert stats["total_bytes"] == 1100

    def test_describe_of_nothing_is_empty_rather_than_zero(self):
        assert sizing.describe([]) == {"count": 0}


class TestEstimate:
    def _probes(self, document_type, sizes, kind="bitstream"):
        return [
            Probe(document_id=f"{document_type}{i}", document_type=document_type,
                  jurisdiction="India", pdf_url_kind=kind, url="u",
                  status=OK, content_length=size)
            for i, size in enumerate(sizes)
        ]

    def test_each_type_is_scaled_by_its_own_mean(self):
        result = sizing.estimate(
            {"central_act": 100, "rule": 10},
            self._probes("central_act", [100, 200, 300]) + self._probes("rule", [1000, 3000]),
        )
        assert result["by_document_type"]["central_act"]["estimated_bytes"] == 100 * 200
        assert result["by_document_type"]["rule"]["estimated_bytes"] == 10 * 2000
        assert result["estimated_total_bytes"] == 100 * 200 + 10 * 2000

    def test_unsampled_types_are_flagged_not_silently_zero(self):
        result = sizing.estimate(
            {"central_act": 100, "regulation": 50}, self._probes("central_act", [10, 20])
        )
        assert result["by_document_type"]["regulation"]["estimated_bytes"] is None
        assert result["types_without_a_sample"] == ["regulation"]
        assert result["population_covered"] == 100

    def test_failed_probes_do_not_count_as_zero_bytes(self):
        probes = self._probes("central_act", [100, 200])
        probes.append(Probe("x", "central_act", "India", "bitstream", "u", status=ERROR))
        result = sizing.estimate({"central_act": 10}, probes)
        assert result["by_document_type"]["central_act"]["sample"]["count"] == 2
        assert result["estimated_total_bytes"] == 10 * 150

    def test_confidence_interval_shrinks_as_the_sample_covers_the_population(self):
        wide = sizing.estimate({"central_act": 10000}, self._probes("central_act", [1, 500, 1000]))
        narrow = sizing.estimate({"central_act": 3}, self._probes("central_act", [1, 500, 1000]))
        assert narrow["estimated_total_bytes_ci95"] < wide["estimated_total_bytes_ci95"]

    def test_jurisdiction_estimate_marks_borrowed_means(self):
        probes = [
            Probe(f"a{i}", "state_act", "Assam", "bitstream", "u", status=OK, content_length=100)
            for i in range(6)
        ] + [
            Probe("k1", "state_act", "Kerala", "bitstream", "u", status=OK, content_length=500)
        ]
        result = sizing.estimate_by_jurisdiction(
            {("Assam", "state_act"): 10, ("Kerala", "state_act"): 10}, probes
        )
        assert result["Assam"]["fully_measured"] is True
        assert result["Assam"]["estimated_bytes"] == 1000
        # Kerala had one probe: it borrows the state_act mean and says so.
        assert result["Kerala"]["fully_measured"] is False


# --- report + CLI ---------------------------------------------------------------


def test_report_is_json_serialisable_and_carries_the_method(tmp_path):
    documents = make_documents()
    sample = sizing.stratified_sample(documents, per_type=5)
    probes = [
        Probe(d["document_id"], d["document_type"], d["jurisdiction"],
              d["pdf_url_kind"], d["english_pdf_url"], status=OK, content_length=1000)
        for d in sample
    ]
    report = sizing.build_report(
        inventory_path="inv.json", documents=documents, sample=sample, probes=probes,
        per_type=5, seed=1, workers=2, delay=0.25,
    )
    text = json.dumps(report)
    assert "no PDF body was downloaded" in report["method"].lower() or \
           "No PDF body was downloaded" in report["method"]
    assert report["inventory"]["documents"] == len(documents)
    assert report["outcomes"]["known_content_length"] == len(probes)
    assert json.loads(text)["estimate"]["population"] == len(documents)


def test_cli_writes_the_estimate_and_downloads_nothing(tmp_path, monkeypatch, capsys):
    inventory = tmp_path / "discovery" / "indiacode_inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({"documents": make_documents()}), encoding="utf-8")

    session = FakeSession(
        head={
            **{f"{BITSTREAM}?i={i}": pdf_head(1000, url=f"{BITSTREAM}?i={i}") for i in range(50)},
            **{f"{BITSTREAM}?{s}{i}": pdf_head(2000, url=f"{BITSTREAM}?{s}{i}")
               for s in ("Assam", "Kerala", "Punjab") for i in range(40)},
            REDIRECT_TARGET: pdf_head(4000, url=REDIRECT_TARGET),
        },
        get={
            **{f"{VIEWFILE}&i={i}": FakeResponse(302, {"Location": REDIRECT_TARGET})
               for i in range(30)},
            **{f"{VIEWFILE}&g={i}": FakeResponse(302, {"Location": REDIRECT_TARGET})
               for i in range(5)},
        },
    )
    monkeypatch.setattr(sizing.SessionPool, "get", lambda self: session)

    code = estimate_size.main([
        "--data-dir", str(tmp_path), "--per-type", "4", "--workers", "1", "--delay", "0",
    ])
    assert code == 0

    written = json.loads((tmp_path / "discovery" / "size_estimate.json").read_text(encoding="utf-8"))
    assert written["sampling"]["sampled"] == 16
    assert written["outcomes"]["known_content_length"] == 16
    assert written["estimate"]["estimated_total_bytes"] > 0
    # 50 central acts x 1000 B, exactly.
    assert written["estimate"]["by_document_type"]["central_act"]["estimated_bytes"] == 50_000
    assert "Size estimate" in capsys.readouterr().out
    # Nothing that looks like a PDF was produced anywhere.
    assert not list(tmp_path.rglob("*.pdf"))
    assert not list(tmp_path.rglob("*.part"))


def test_cli_fails_loudly_when_no_size_can_be_determined(tmp_path, monkeypatch, capsys):
    """If HEAD is unsupported we report it; we never fall back to downloading."""
    inventory = tmp_path / "discovery" / "indiacode_inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        json.dumps({"documents": [
            {"document_id": "c1", "document_type": "central_act", "jurisdiction": "India",
             "english_pdf_url": BITSTREAM, "pdf_url_kind": "bitstream"}
        ]}),
        encoding="utf-8",
    )
    session = FakeSession(head={BITSTREAM: pdf_head(None)}, get={BITSTREAM: pdf_head(None)})
    monkeypatch.setattr(sizing.SessionPool, "get", lambda self: session)

    code = estimate_size.main([
        "--data-dir", str(tmp_path), "--per-type", "5", "--workers", "1", "--delay", "0",
    ])
    assert code == 1
    written = json.loads((tmp_path / "discovery" / "size_estimate.json").read_text(encoding="utf-8"))
    assert written["outcomes"]["unknown"] == 1
    assert written["estimate"]["estimated_total_bytes"] is None


def test_cli_reports_a_missing_inventory(tmp_path, caplog):
    assert estimate_size.main(["--data-dir", str(tmp_path)]) == 2


def test_existing_inventory_and_manifest_are_never_opened_for_writing(tmp_path, monkeypatch):
    """The sizing pass must not touch the inventory or the manifest."""
    inventory = tmp_path / "discovery" / "indiacode_inventory.json"
    inventory.parent.mkdir(parents=True)
    central_only = [d for d in make_documents() if d["document_type"] == "central_act"]
    payload = json.dumps({"documents": central_only})
    inventory.write_text(payload, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")

    session = FakeSession(head={f"{BITSTREAM}?i={i}": pdf_head(1000, url=f"{BITSTREAM}?i={i}")
                                for i in range(50)})
    monkeypatch.setattr(sizing.SessionPool, "get", lambda self: session)
    estimate_size.main([
        "--data-dir", str(tmp_path), "--per-type", "2", "--workers", "1", "--delay", "0",
        "--output", str(tmp_path / "estimate.json"),
    ])
    assert inventory.read_text(encoding="utf-8") == payload
    assert manifest.read_text(encoding="utf-8") == "{}"
