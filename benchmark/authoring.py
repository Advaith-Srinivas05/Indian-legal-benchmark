"""The authoring workflow: sample row → draft → written → verified.

``python -m benchmark author new``      draft a question skeleton from a sample row
``python -m benchmark author check``    validate every question; report the set
``python -m benchmark author review``   build the verification page
``python -m benchmark author verdicts`` apply a verifier's exported decisions

A draft arrives with its evidence already resolved: the sampled provision, and
every word-for-word identical copy in the same instrument as an interchangeable
alternative. Copies in *other* instruments (another state's identical section)
are only proposed — for a jurisdictional question they are the wrong answer, so
the author decides. For ``cross_reference`` the provision the sampled one cites
is added as a second required group.

Verification is done against the **official PDF page**, reached by link: the
review page never renders a PDF, and nothing here opens one.
"""

from __future__ import annotations

import html
import json
from datetime import date
from pathlib import Path
from typing import Optional

from . import config
from .questions import CHECKLIST, CATEGORIES, Corpus, evidence_text, location, validate_set

QUESTIONS_DIR = config.SAMPLES_DIR.parent / "questions"


def load_questions(directory: Path = QUESTIONS_DIR) -> list[dict]:
    directory = Path(directory)
    if not directory.exists():
        return []
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("IN-STAT-*.json"))]


def save_question(q: dict, directory: Path = QUESTIONS_DIR) -> Path:
    from .corpus import _json, _write_atomic
    path = Path(directory) / f"{q['question_id']}.json"
    _write_atomic(path, _json(q))
    return path


def next_id(directory: Path = QUESTIONS_DIR) -> str:
    numbers = [int(q["question_id"].split("-")[-1]) for q in load_questions(directory)]
    return f"IN-STAT-{(max(numbers) + 1) if numbers else 1:04d}"


def _equivalents(corpus_dir: Path, did: str, key: str) -> tuple[list[tuple], bool]:
    """Members of this provision's equivalence class, other than itself."""
    with open(Path(corpus_dir) / config.PROVISION_EQUIVALENTS_FILENAME, "r", encoding="utf-8") as handle:
        for line in handle:
            if f'"{did}"' not in line:
                continue
            c = json.loads(line)
            members = [(m["document_id"], m["key"]) for m in c["members"]]
            if (did, key) in members:
                return [m for m in members if m != (did, key)], c["same_instrument"]
    return [], True


def new_draft(corpus_dir: Path, sample_path: Path, index: int, category: str, *,
              directory: Path = QUESTIONS_DIR, question_id: Optional[str] = None) -> dict:
    """A question skeleton for one sample row, evidence resolved, text blank."""
    if category not in CATEGORIES or category == "unanswerable":
        raise ValueError(f"category must be one of {CATEGORIES[:-1]} (unanswerable questions have no sample row)")
    sample = json.loads(Path(sample_path).read_text(encoding="utf-8"))
    row = sample["rows"][index]
    corpus = Corpus(corpus_dir)
    did, key = row["document_id"], row["key"]

    locations = [location(corpus, did, key, "sampled")]
    proposed = []
    others, same_instrument = _equivalents(corpus_dir, did, key)
    for other in others:
        loc = location(corpus, *other, "same_instrument_equivalent" if same_instrument
                       else "cross_instrument_equivalent")
        (locations if same_instrument else proposed).append(loc)

    groups = [{"group_id": "g1", "requirement": "sufficient", "locations": locations}]
    must_cite = [{"document_id": did, "key": key}]
    if category == "cross_reference":
        if not row["references"]:
            raise ValueError(f"sample row {index} cites no other provision; pick a row tagged cross_reference")
        groups[0]["requirement"] = "required"
        ref = row["references"][0]
        groups.append({"group_id": "g2", "requirement": "required",
                       "locations": [location(corpus, did, ref, "referenced")]})
        must_cite.append({"document_id": did, "key": ref})

    return {
        "question_id": question_id or next_id(directory),
        "question": "",
        "category": category,
        "answer_type": "abstractive",
        "jurisdiction_hint": row["jurisdiction"] if category == "jurisdictional" else None,
        "gold_evidence": groups,
        "proposed_alternatives": proposed,
        "gold_answer": "",
        "required_facts": [],
        "must_cite": must_cite,
        "unanswerable": False,
        "difficulty": row["signals"],
        "provenance": {
            "status": "draft",
            "sample": Path(sample_path).name,
            "sample_index": index,
            "tags": row["tags"],
            "author": None,
            "verified_by": None,
            "verified_at": None,
            "verification": {},
        },
    }


def check(corpus_dir: Path, directory: Path = QUESTIONS_DIR) -> dict:
    """Validate the whole set. Returns a report; problems keyed by question_id."""
    questions = load_questions(directory)
    problems = validate_set(questions, Corpus(corpus_dir))
    counts: dict[str, dict[str, int]] = {}
    for q in questions:
        c = counts.setdefault(q["category"], {s: 0 for s in ("draft", "verified", "rejected")})
        c[q["provenance"]["status"]] += 1
    return {
        "questions": len(questions),
        "by_category": dict(sorted(counts.items())),
        "with_problems": sum(1 for v in problems.values() if v),
        "problems": {k: v for k, v in problems.items() if v},
    }


# --- The verification page -------------------------------------------------------------


def _pdf_link(meta: dict, page: int) -> str:
    url = meta["source"].get("final_pdf_url") or meta["source"].get("source_url") or ""
    return f"{url}#page={page}" if url else ""


def build_review_page(corpus_dir: Path, out_path: Path, directory: Path = QUESTIONS_DIR) -> int:
    """A self-contained HTML page for verifying written drafts. Returns the count.

    Verdicts are kept in the browser as they are ticked and exported as a JSON
    file for ``author verdicts``. Only drafts with text are shown.
    """
    corpus = Corpus(corpus_dir)
    questions = [q for q in load_questions(directory)
                 if q["provenance"]["status"] == "draft" and q["question"].strip()]
    cards = []
    for q in questions:
        evidence = []
        for g in q["gold_evidence"]:
            for loc in g["locations"]:
                meta = corpus.meta(loc["document_id"])
                link = _pdf_link(meta, loc["page_start"])
                evidence.append(
                    f'<div class="loc"><div class="cite">{html.escape(loc["citation"])} '
                    f'<span class="muted">· {html.escape(g["group_id"])} {g["requirement"]} · '
                    f'{loc["source"]} · pages {loc["page_start"]}–{loc["page_end"]} · '
                    f'chars {loc["char_start"]}–{loc["char_end"]}</span> '
                    + (f'<a href="{html.escape(link)}" target="_blank" rel="noopener">open official PDF page</a>' if link else "")
                    + f'</div><pre>{html.escape(evidence_text(corpus, loc))}</pre></div>')
        facts = " · ".join(" / ".join(html.escape(v) for v in group) for group in q["required_facts"])
        checks = "".join(f'<label><input type="checkbox" data-item="{c}"> {c.replace("_", " ")}</label>'
                         for c in CHECKLIST)
        cards.append(
            f'<section class="card" data-id="{q["question_id"]}">'
            f'<h2>{q["question_id"]} <span class="muted">{q["category"]}</span></h2>'
            f'<p class="q">{html.escape(q["question"])}</p>'
            f'<p><b>Gold answer.</b> {html.escape(q["gold_answer"])}</p>'
            f'<p><b>Required facts.</b> {facts}</p>{"".join(evidence)}'
            f'<div class="checks">{checks}</div>'
            f'<div class="decide"><label><input type="radio" name="v-{q["question_id"]}" value="verified"> verified</label>'
            f'<label><input type="radio" name="v-{q["question_id"]}" value="rejected"> rejected</label>'
            f'<input class="reason" placeholder="reason (required to reject)"></div></section>')

    page = _REVIEW_TEMPLATE.replace("{{CARDS}}", "\n".join(cards)).replace("{{COUNT}}", str(len(questions)))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8", newline="")
    return len(questions)


def apply_verdicts(verdicts_path: Path, directory: Path = QUESTIONS_DIR) -> dict:
    """Write exported verdicts into the question files. Returns counts."""
    payload = json.loads(Path(verdicts_path).read_text(encoding="utf-8"))
    verifier = (payload.get("verifier") or "").strip()
    if not verifier:
        raise ValueError("the verdicts file does not name its verifier")
    by_id = {q["question_id"]: q for q in load_questions(directory)}
    applied = {"verified": 0, "rejected": 0, "skipped": 0}
    for qid, v in sorted(payload.get("verdicts", {}).items()):
        q = by_id.get(qid)
        if q is None or v.get("decision") not in ("verified", "rejected"):
            applied["skipped"] += 1
            continue
        checklist = {item: bool(v.get("checklist", {}).get(item)) for item in CHECKLIST}
        if v["decision"] == "verified" and not all(checklist.values()):
            applied["skipped"] += 1          # an unticked item is not a verification
            continue
        if v["decision"] == "rejected" and not (v.get("reason") or "").strip():
            applied["skipped"] += 1
            continue
        q["provenance"].update({
            "status": v["decision"],
            "verified_by": verifier,
            "verified_at": v.get("date") or date.today().isoformat(),
            "verification": checklist,
        })
        if v["decision"] == "rejected":
            q["provenance"]["rejection_reason"] = v["reason"].strip()
        save_question(q, directory)
        applied[v["decision"]] += 1
    return applied


_REVIEW_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Question verification</title>
<style>
:root{--ink:#1d1d1f;--muted:#6b6b70;--line:#dcdce0;--bg:#fafaf8;--card:#fff;--accent:#1f5fbf}
body{margin:0;font:15px/1.5 system-ui,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:12px 20px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
main{max-width:960px;margin:0 auto;padding:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 20px;margin-bottom:20px}
h2{font-size:16px;margin:0 0 8px}.muted{color:var(--muted);font-weight:400;font-size:13px}
.q{font-size:17px;font-weight:600}.loc{border-top:1px solid var(--line);padding-top:8px;margin-top:8px}
.cite{font-size:13px}pre{white-space:pre-wrap;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;font:13px/1.45 ui-monospace,monospace;max-height:320px;overflow:auto}
.checks,.decide{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:10px;font-size:14px}
.reason{flex:1;min-width:220px;padding:4px 8px}a{color:var(--accent)}button{padding:6px 12px}
</style></head><body>
<header><b>Verify {{COUNT}} questions</b>
<input id="verifier" placeholder="your name (recorded as verified_by)">
<button id="export">Export verdicts</button><span id="status" class="muted"></span></header>
<main>
<p class="muted">Open each official PDF page, confirm the evidence is the provision it claims and reads exactly as the source,
then tick every item. A question is verified only with every box ticked. Progress is kept in this browser.</p>
{{CARDS}}
</main>
<script>
const KEY="question-verdicts";
let saved={};try{saved=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const verifier=document.getElementById("verifier");verifier.value=saved.__verifier||"";
function collect(){const out={};document.querySelectorAll(".card").forEach(c=>{const id=c.dataset.id;const checklist={};
c.querySelectorAll("input[data-item]").forEach(i=>checklist[i.dataset.item]=i.checked);
const d=c.querySelector("input[type=radio]:checked");out[id]={checklist,decision:d?d.value:null,reason:c.querySelector(".reason").value,
date:new Date().toISOString().slice(0,10)}});return out}
function persist(){const v=collect();v.__verifier=verifier.value;try{localStorage.setItem(KEY,JSON.stringify(v))}catch(e){}
document.getElementById("status").textContent=Object.values(v).filter(x=>x&&x.decision).length+" decided"}
document.querySelectorAll(".card").forEach(c=>{const s=saved[c.dataset.id];if(s){
c.querySelectorAll("input[data-item]").forEach(i=>i.checked=!!(s.checklist||{})[i.dataset.item]);
if(s.decision){const r=c.querySelector(`input[value=${s.decision}]`);if(r)r.checked=true}c.querySelector(".reason").value=s.reason||""}});
document.addEventListener("input",persist);persist();
document.getElementById("export").onclick=()=>{if(!verifier.value.trim()){alert("Enter your name first.");return}
const blob=new Blob([JSON.stringify({verifier:verifier.value.trim(),verdicts:collect()},null,2)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="verdicts.json";a.click()};
</script></body></html>
"""
