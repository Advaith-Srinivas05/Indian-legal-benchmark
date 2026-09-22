"""The question record, and every rule a question must satisfy before it ships.

One JSON file per question under ``benchmark/data/questions/``, tracked in git:
it is the irreplaceable artefact. A question moves ``draft`` → ``verified`` (or
``rejected``, kept with its reason — never deleted).

:func:`validate` is the single gate. It resolves every evidence location against
the published corpus and checks the rules of ``docs/BENCHMARK_DESIGN.md`` and
``docs/EVIDENCE_MODEL.md`` mechanically, so no rule depends on an author
remembering it:

* evidence is character spans in canonical text that still reproduce their
  hashes; each group has at least one gold-eligible anchor (decision B19);
* a question that must paraphrase does not name the Act, give the provision
  number, or reuse most of the provision's words;
* every required fact is actually in the evidence, and none is the provision's
  own number;
* a ``verified`` question carries who verified it, when, and a complete checklist.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Optional

from . import config, terms
from .duplicates import _title_words, places_in
from .evidence import exclusion

CATEGORIES = ("provision_lookup", "situational", "definitional", "numeric_threshold",
              "cross_reference", "jurisdictional", "unanswerable")
ANSWER_TYPES = ("extractive", "abstractive", "boolean", "numeric", "list")
STATUSES = ("draft", "verified", "rejected")
REQUIREMENTS = ("required", "sufficient")
LOCATION_SOURCES = ("sampled", "same_instrument_equivalent", "same_instrument_version",
                    "cross_instrument_equivalent", "successor_or_predecessor", "referenced",
                    "author_added")
#: What a verifier confirms, one tick each.
CHECKLIST = ("span_matches_source", "text_undamaged", "answer_follows",
             "facts_present", "question_fair", "alternatives_checked")
#: How a question was drafted, stated as fact so the published set can say so.
DRAFTING_METHODS = ("human", "language_model")
#: What a verification was checked against. ``official_pdf``: a person opened the
#: official India Code page (the review page's links). ``pdf_page_image``: the
#: stored PDF page was rendered and inspected. ``text_review``: the extracted
#: text was reviewed — sufficient for born-digital gold, whose text *is* the
#: PDF's own text layer, but weaker, and recorded as such.
VERIFICATION_METHODS = ("official_pdf", "pdf_page_image", "text_review")
#: Categories whose question must not name the Act or the provision.
PARAPHRASE_CATEGORIES = frozenset({"situational", "definitional", "numeric_threshold",
                                   "cross_reference", "jurisdictional"})
LEAKAGE_MAX = 0.5
QUESTION_ID = re.compile(r"^IN-STAT-\d{4}$")
_CITATION_WORDS = r"(?:sections?|s\.|ss\.|rules?|r\.|regulations?|reg\.|articles?|art\.|clauses?|paragraphs?)"
#: Words too generic to show that a question names its Act.
_TITLE_GENERIC = frozenset({"act", "acts", "rules", "rule", "regulations", "regulation", "code",
                            "amendment", "order", "india", "indian", "state", "general"})
_UNIT_ABBREVIATION = {"article": "art."}
_CATEGORY_ABBREVIATION = {"rules": "r.", "regulations": "reg."}


# --- Reading the corpus -------------------------------------------------------------


class Corpus:
    """Cached, read-only access to the published corpus for validation."""

    def __init__(self, corpus_dir: Path) -> None:
        self.dir = Path(corpus_dir)
        self._meta: dict[str, dict] = {}
        self._structure: dict[str, dict] = {}
        self._text: dict[str, str] = {}
        self._conflicted: Optional[set[str]] = None
        self._mates: dict[str, list[str]] = {}

    def meta(self, did: str) -> Optional[dict]:
        if did not in self._meta:
            path = self.dir / config.META_DIRNAME / f"{did}.json"
            self._meta[did] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        return self._meta[did]

    def provision(self, did: str, key: str) -> Optional[dict]:
        if did not in self._structure:
            path = self.dir / config.STRUCTURE_DIRNAME / f"{did}.json"
            self._structure[did] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"provisions": []}
        return next((p for p in self._structure[did]["provisions"] if p["key"] == key), None)

    def provisions(self, did: str) -> list[dict]:
        self.provision(did, "")
        return self._structure[did]["provisions"]

    def text(self, did: str) -> str:
        if did not in self._text:
            meta = self.meta(did)
            with open(self.dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
                self._text[did] = handle.read()
        return self._text[did]

    def _duplicates(self) -> None:
        dup = json.loads((self.dir / config.DUPLICATES_FILENAME).read_text(encoding="utf-8"))
        self._conflicted = {d for e in dup["title_conflicts"] for d in (e["a"], e["b"])}
        self._mates = {}
        for c in dup["clusters"]:
            docs = [d["document_id"] for d in c["documents"]]
            for d in docs:
                self._mates[d] = [x for x in docs if x != d]

    def conflicted(self) -> set[str]:
        if self._conflicted is None:
            self._duplicates()
        return self._conflicted

    def cluster_mates(self, did: str) -> list[str]:
        """Other documents that are copies of the same instrument."""
        if self._conflicted is None:
            self._duplicates()
        return self._mates.get(did, [])


def citation(meta: dict, provision: dict) -> str:
    """"Right to Information Act, 2005, s. 8" — the label a person would write."""
    abbreviation = (_UNIT_ABBREVIATION.get(provision["unit_type"])
                    or _CATEGORY_ABBREVIATION.get(meta["category"]) or "s.")
    return f"{meta['title']}, {abbreviation} {provision['number']}"


def location(corpus: Corpus, did: str, key: str, source: str) -> dict:
    """An evidence location record for one provision, read from the corpus."""
    meta, p = corpus.meta(did), corpus.provision(did, key)
    if meta is None or p is None:
        raise KeyError(f"{did} {key}: no such provision in the corpus")
    return {
        "document_id": did,
        "key": key,
        "citation": citation(meta, p),
        "unit_path": p["unit_path"],
        "char_start": p["char_start"],
        "char_end": p["char_end"],
        "page_start": p["page_start"],
        "page_end": p["page_end"],
        "text_sha256": p["text_sha256"],
        "evidence_confidence": p["evidence_confidence"],
        "source": source,
    }


def evidence_text(corpus: Corpus, loc: dict) -> str:
    return corpus.text(loc["document_id"])[loc["char_start"]:loc["char_end"]]


# --- Validation ----------------------------------------------------------------------


def validate(q: dict, corpus: Corpus) -> list[str]:
    """Every rule the question breaks, as readable sentences. Empty means valid."""
    problems: list[str] = []
    add = problems.append
    for field in ("question_id", "question", "category", "answer_type", "gold_evidence",
                  "gold_answer", "required_facts", "must_cite", "unanswerable", "provenance"):
        if field not in q:
            add(f"missing field '{field}'")
    if problems:
        return problems

    if not QUESTION_ID.match(q["question_id"]):
        add(f"question_id {q['question_id']!r} is not of the form IN-STAT-0001")
    if q["category"] not in CATEGORIES:
        add(f"category {q['category']!r} is not one of {CATEGORIES}")
    if q["answer_type"] not in ANSWER_TYPES:
        add(f"answer_type {q['answer_type']!r} is not one of {ANSWER_TYPES}")
    status = q["provenance"].get("status")
    if status not in STATUSES:
        add(f"provenance.status {status!r} is not one of {STATUSES}")
    if status == "rejected":
        if not q["provenance"].get("rejection_reason"):
            add("a rejected question must say why (provenance.rejection_reason)")
        return problems                     # a rejected draft need not be complete
    if not (q["question"] or "").strip():
        add("the question text is empty")
    if not (q["gold_answer"] or "").strip():
        add("gold_answer is empty")

    if q["category"] == "unanswerable" or q["unanswerable"]:
        if q["category"] != "unanswerable" or not q["unanswerable"]:
            add("unanswerable is true exactly when category is 'unanswerable'")
        if q["gold_evidence"] or q["must_cite"] or q["required_facts"]:
            add("an unanswerable question has no evidence, citations or required facts")
        _check_verified(q, add)
        return problems

    texts = _check_evidence(q, corpus, add)
    if texts is None:
        return problems
    locations = [loc for g in q["gold_evidence"] for loc in g["locations"]]
    _check_citations(q, locations, add)
    _check_facts(q, locations, texts, add)
    if q["category"] in PARAPHRASE_CATEGORIES:
        _check_paraphrase(q, corpus, locations, texts, add)
    if q["category"] == "numeric_threshold" and not any(
            terms.QUANTITY.search(v) for group in q["required_facts"] for v in group):
        add("a numeric_threshold question needs a required fact that is a quantity with a unit")
    if q["category"] == "jurisdictional":
        hint = q.get("jurisdiction_hint")
        if not hint:
            add("a jurisdictional question sets jurisdiction_hint")
        elif not any(corpus.meta(loc["document_id"])["jurisdiction"] == hint for loc in locations):
            add(f"no evidence location is in the jurisdiction the question names ({hint})")
    if q.get("proposed_alternatives"):
        if status == "verified":
            add("proposed_alternatives must be decided (moved into a group or removed) before verifying")
    _check_verified(q, add)
    return problems


def _check_evidence(q: dict, corpus: Corpus, add) -> Optional[dict]:
    groups = q["gold_evidence"]
    if not groups:
        add("an answerable question needs at least one evidence group")
        return None
    ids = [g.get("group_id") for g in groups]
    if len(set(ids)) != len(ids):
        add("evidence group_ids are not unique")
    requirements = [g.get("requirement") for g in groups]
    if q["category"] == "cross_reference":
        if len(groups) < 2 or set(requirements) != {"required"}:
            add("a cross_reference question has two or more groups, all 'required'")
    elif requirements != ["sufficient"]:
        add("a single-provision question has exactly one group, 'sufficient'")

    texts: dict[tuple, str] = {}
    unresolved = False
    for g in groups:
        if g.get("requirement") not in REQUIREMENTS:
            add(f"group {g.get('group_id')}: requirement must be one of {REQUIREMENTS}")
        if not g.get("locations"):
            add(f"group {g.get('group_id')} has no locations")
            continue
        anchored = False
        # The length floor exists so the *sampled* provision has something to ask
        # about; a one-line commencement clause is perfectly good second-hop
        # evidence. The ceiling still applies: a span over the budget cannot be
        # covered at all.
        tolerated = {"too_short"} if g is not groups[0] else set()
        for loc in g["locations"]:
            label = f"{loc.get('document_id')} {loc.get('key')}"
            if loc.get("source") not in LOCATION_SOURCES:
                add(f"{label}: source must be one of {LOCATION_SOURCES}")
            meta, p = corpus.meta(loc.get("document_id", "")), corpus.provision(loc.get("document_id", ""), loc.get("key", ""))
            if meta is None or p is None:
                add(f"{label}: not a provision in the corpus")
                unresolved = True
                continue
            for field in ("char_start", "char_end", "page_start", "page_end", "text_sha256"):
                if loc.get(field) != p[field]:
                    add(f"{label}: {field} is {loc.get(field)!r}, the corpus says {p[field]!r}")
            text = evidence_text(corpus, {**loc, "char_start": p["char_start"], "char_end": p["char_end"]})
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != p["text_sha256"]:
                add(f"{label}: span no longer reproduces its text")
            texts[(loc["document_id"], loc["key"])] = text
            reason = exclusion(p, meta, corpus.conflicted(), text)
            if reason is None or reason in tolerated:
                anchored = True
        if not anchored:
            add(f"group {g.get('group_id')}: no location is gold-eligible (high tier, born digital, "
                "uniquely citable, 150-6,000 chars, no title conflict)")
    # Checks after this read the evidence; with a location missing they would
    # judge the question against less than its whole evidence.
    return None if unresolved else texts


def _check_citations(q: dict, locations: list[dict], add) -> None:
    if not q["must_cite"]:
        add("must_cite is empty")
    known = {(loc["document_id"], loc["key"]) for loc in locations}
    for cite in q["must_cite"]:
        if (cite.get("document_id"), cite.get("key")) not in known:
            add(f"must_cite {cite.get('document_id')} {cite.get('key')} is not an evidence location")


def _check_facts(q: dict, locations: list[dict], texts: dict, add) -> None:
    facts = q["required_facts"]
    if not facts:
        add("required_facts is empty")
    numbers = {str(loc["key"].rsplit(":", 1)[-1]).split("#")[0].lower() for loc in locations}
    corpus_text = " ".join(texts.values())
    for i, group in enumerate(facts):
        if not group or not all(isinstance(v, str) and v.strip() for v in group):
            add(f"required_facts[{i}] must be a non-empty list of non-empty strings")
            continue
        if not any(terms.appears_in(v, corpus_text) for v in group):
            add(f"required_facts[{i}] {group}: no variant appears in the evidence text")
        for v in group:
            bare = re.sub(rf"^\s*{_CITATION_WORDS}\s*", "", v.strip().lower())
            if bare in numbers:
                add(f"required_facts[{i}] {v!r} is the provision's own number, which any answer "
                    "satisfies just by citing it")


def _check_paraphrase(q: dict, corpus: Corpus, locations: list[dict], texts: dict, add) -> None:
    question = q["question"]
    for loc in locations:
        number = re.escape(str(corpus.provision(loc["document_id"], loc["key"])["number"]))
        if re.search(rf"\b{_CITATION_WORDS}\s*{number}\b", question, re.IGNORECASE):
            add(f"the question gives the provision number ({loc['citation']}); a "
                f"{q['category']} question must not")
            break
    asked = set(terms.tokens(question))
    for loc in locations:
        title = corpus.meta(loc["document_id"])["title"]
        allowed = set()
        if q["category"] == "jurisdictional":
            # The state may be named; that is the point of the category.
            for place in places_in(title):
                allowed |= set(place.split())
        words = {w for w in _title_words(title)
                 if len(w) >= 4 and w not in _TITLE_GENERIC and w not in allowed}
        if len(words) >= 2 and words <= asked:
            add(f"the question names the Act ({title}); a {q['category']} question must not")
            break
    anchor = next(iter(texts.values()), "")
    overlap = terms.leakage(question, anchor)
    if overlap > LEAKAGE_MAX:
        add(f"{overlap:.0%} of the question's content words are the provision's own "
            f"(limit {LEAKAGE_MAX:.0%}): paraphrase, or keyword search finds it for free")


def _check_verified(q: dict, add) -> None:
    prov = q["provenance"]
    if prov.get("drafting_method") not in DRAFTING_METHODS:
        add(f"provenance.drafting_method must be one of {DRAFTING_METHODS}")
    if prov.get("status") != "verified":
        return
    if not prov.get("verified_by") or not prov.get("verified_at"):
        add("a verified question records verified_by and verified_at")
    if prov.get("verification_method") not in VERIFICATION_METHODS:
        add(f"provenance.verification_method must be one of {VERIFICATION_METHODS}")
    checklist = prov.get("verification") or {}
    missing = [item for item in CHECKLIST if checklist.get(item) is not True]
    if missing:
        add(f"verification checklist incomplete: {missing}")


def validate_set(questions: list[dict], corpus: Corpus) -> dict[str, list[str]]:
    """Validate every question, plus the rules that span the set."""
    problems = {q.get("question_id", f"#{i}"): validate(q, corpus) for i, q in enumerate(questions)}
    seen_ids: dict[str, int] = {}
    anchors: dict[tuple, str] = {}
    for q in questions:
        qid = q.get("question_id")
        seen_ids[qid] = seen_ids.get(qid, 0) + 1
        if q.get("provenance", {}).get("status") == "rejected":
            continue
        for g in q.get("gold_evidence") or []:
            for loc in g.get("locations") or []:
                if loc.get("source") == "sampled":
                    k = (loc["document_id"], loc["key"])
                    if k in anchors and anchors[k] != qid:
                        problems[qid].append(f"{k[0]} {k[1]} is already the sampled evidence of {anchors[k]}")
                    anchors.setdefault(k, qid)
    for qid, n in seen_ids.items():
        if n > 1:
            problems[qid].append(f"question_id {qid} is used {n} times")
    return problems
