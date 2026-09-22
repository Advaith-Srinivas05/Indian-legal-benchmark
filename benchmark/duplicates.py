"""Which documents are copies of the same instrument, and which provisions are word-for-word the same.

India Code stores the same law many times. Union Territory collections re-host
Central Acts that extend to them — the Bharatiya Nagarik Suraksha Sanhita, 2023
appears under India and three states/UTs — and one PDF is sometimes listed twice
(an Act and its Rules printed together). Without this, a system that retrieves
the right law from a different copy scores zero.

Two levels, because they answer different questions:

**Documents.** Each document is fingerprinted as the set of its normalised text
lines (five or more tokens). Pairs sharing lines are scored by Jaccard
similarity, then classified:

========================  =====================================================
``identical``             same PDF bytes; or same canonical text, unless the
                          titles name different states
``same_instrument``       Jaccard >= 0.95, or >= 0.5 with matching titles —
                          and the titles do not name different states
``parallel``              Jaccard >= 0.5 otherwise: a *different law* drafted
                          from a shared model (Bihar's GST Act and Uttar
                          Pradesh's), or adopted verbatim by another state.
                          Recorded, never clustered.
``contains``              one document holds >= 90 % of the other's lines
========================  =====================================================

Titles alone are not trusted — India Code's are noisy ("MP IRRGATION RULES",
"HUMEN RIGHTS"), so long words match through a typo — and text alone is not
enough, because parallel state laws share most wording. ``identical`` and
``same_instrument`` edges form clusters. Identical copies whose titles disagree
are listed separately as ``title_conflicts``: some are India Code attaching the
wrong PDF to a listing.

**Provisions.** Copies of one instrument can be different amended versions, so
a document cluster does not prove a given section is the same in both. The
provision index groups provisions whose text is identical after whitespace and
Unicode normalisation, across documents. That is what gold evidence groups are
built from; each class says whether all its documents are one instrument.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from . import config

#: Fingerprint lines need this many tokens: short lines ("(a)", headings) are
#: shared by unrelated documents and say nothing about identity.
MIN_LINE_TOKENS = 5
#: Lines found in more documents than this are boilerplate and are not used to
#: propose pairs (they still count once a pair is being scored).
MAX_LINE_DOCUMENT_FREQUENCY = 50
#: A pair must share at least this many fingerprint lines to be scored.
MIN_SHARED_LINES = 3

SAME_INSTRUMENT_JACCARD = 0.95
VERSION_JACCARD = 0.5
TITLE_MATCH = 0.8
CONTAINS = 0.9

_TOKEN = re.compile(r"[a-z0-9]+")
_TITLE_WORD = re.compile(r"[a-z]+")
_TITLE_STOP = frozenset({"the", "of", "and", "no", "a", "an", "for", "in", "to"})
_SPACE = re.compile(r"\s+")


def _hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()


def fingerprint(text: str) -> set[str]:
    """The set of normalised lines that identify a document's content."""
    lines = set()
    for line in text.split("\n"):
        tokens = _TOKEN.findall(unicodedata.normalize("NFKC", line).lower())
        if len(tokens) >= MIN_LINE_TOKENS:
            lines.add(_hash(" ".join(tokens)))
    return lines


def _title_words(title: str) -> list[str]:
    return [w for w in _TITLE_WORD.findall((title or "").lower()) if w not in _TITLE_STOP]


def _same_word(a: str, b: str) -> bool:
    """Equal, or a one-slip typo of a long word ("humen" / "human", "compansation")."""
    if a == b:
        return True
    return min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def title_similarity(a: str, b: str) -> float:
    """Token Jaccard of two titles, ignoring case, punctuation, numbers and stopwords.

    Long words that differ by a typo count as the same word: India Code's titles
    carry "PROTECTION OF HUMEN RIGHTS" and "Right to fair Compansation".
    """
    ta, tb = set(_title_words(a)), set(_title_words(b))
    if not ta | tb:
        return 0.0
    unmatched_b = set(tb)
    matched = 0
    for word in sorted(ta):
        hit = next((w for w in sorted(unmatched_b) if _same_word(word, w)), None)
        if hit is not None:
            matched += 1
            unmatched_b.discard(hit)
    return matched / (len(ta) + len(tb) - matched)


#: States and Union Territories, as titles name them, including older names and
#: common abbreviations. Stopwords ("and") are dropped, as in titles.
PLACES = {
    "andhra pradesh": ("andhra pradesh",), "arunachal pradesh": ("arunachal pradesh",),
    "assam": ("assam",), "bihar": ("bihar",), "chhattisgarh": ("chhattisgarh", "chhatisgarh"),
    "goa": ("goa",), "gujarat": ("gujarat",), "haryana": ("haryana",),
    "himachal pradesh": ("himachal pradesh", "h p"), "jharkhand": ("jharkhand",),
    "karnataka": ("karnataka", "mysore"), "kerala": ("kerala",),
    "madhya pradesh": ("madhya pradesh", "madhaya pradesh", "m p", "mp"),
    "maharashtra": ("maharashtra", "bombay"), "manipur": ("manipur",), "meghalaya": ("meghalaya",),
    "mizoram": ("mizoram",), "nagaland": ("nagaland",), "odisha": ("odisha", "orissa"),
    "punjab": ("punjab",), "rajasthan": ("rajasthan",), "sikkim": ("sikkim",),
    "tamil nadu": ("tamil nadu", "madras"), "telangana": ("telangana",), "tripura": ("tripura",),
    "uttar pradesh": ("uttar pradesh", "u p", "up"), "uttarakhand": ("uttarakhand", "uttaranchal"),
    "west bengal": ("west bengal",), "andaman nicobar": ("andaman nicobar",),
    "chandigarh": ("chandigarh",), "dadra nagar haveli": ("dadra nagar haveli",),
    "daman diu": ("daman diu",), "delhi": ("delhi",), "jammu kashmir": ("jammu kashmir",),
    "ladakh": ("ladakh",), "lakshadweep": ("lakshadweep",), "puducherry": ("puducherry", "pondicherry"),
}


def places_in(title: str) -> set[str]:
    """The states and Union Territories a title names."""
    joined = f" {' '.join(_title_words(title))} "
    return {place for place, variants in PLACES.items()
            if any(f" {v} " in joined for v in variants)}


def places_differ(a: str, b: str) -> bool:
    """Both titles name a place, and no place in common.

    "Union Territory of Chandigarh Street Vendors Scheme" and the Daman and Diu
    scheme share 96 % of their lines and are two instruments. A title naming no
    place ("Explosive Substances Act, 1908") never conflicts.
    """
    pa, pb = places_in(a), places_in(b)
    return bool(pa) and bool(pb) and not (pa & pb)


def normalise_provision(text: str) -> str:
    """Provision text with Unicode and whitespace differences removed.

    Two copies of an Act wrap lines differently; that must not make an identical
    provision look different. Case, punctuation and numbers are kept: "30 days"
    and "60 days" are different law.
    """
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def classify(jaccard: float, title_sim: float, containment: float, *, same_file: bool = False,
             same_text: bool = False, different_places: bool = False) -> str:
    """The relation between two documents. See the module docstring for the table.

    The same PDF bytes are one file whatever the listings say. Otherwise, titles
    naming different places mean two laws even when the text is identical — the
    Chhattisgarh Acts adopted verbatim from Madhya Pradesh in 2000 are in force
    in a different state — so they are ``parallel``, and their identical
    provisions are still linked, as crossing instruments.
    """
    if same_file or (same_text and not different_places):
        return "identical"
    if not different_places and (
            jaccard >= SAME_INSTRUMENT_JACCARD
            or (jaccard >= VERSION_JACCARD and title_sim >= TITLE_MATCH)):
        return "same_instrument"
    if jaccard >= VERSION_JACCARD:
        return "parallel"
    if containment >= CONTAINS:
        return "contains"
    return "unrelated"


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic: the smaller id is always the root.
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


def _load(corpus_dir: Path):
    metas, texts, provisions = {}, {}, {}
    for path in sorted((corpus_dir / config.META_DIRNAME).glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        did = meta["document_id"]
        metas[did] = meta
        with open(corpus_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
            texts[did] = handle.read()
        structure = json.loads((corpus_dir / config.STRUCTURE_DIRNAME / f"{did}.json").read_text(encoding="utf-8"))
        provisions[did] = structure["provisions"]
    return metas, texts, provisions


def find_duplicates(corpus_dir: Path) -> tuple[dict, list[dict]]:
    """Return (``duplicates.json`` payload, provision-equivalence rows)."""
    corpus_dir = Path(corpus_dir)
    metas, texts, provisions = _load(corpus_dir)
    ids = sorted(metas)
    prints = {d: fingerprint(texts[d]) for d in ids}

    postings: dict[str, list[str]] = defaultdict(list)
    for d in ids:
        for h in prints[d]:
            postings[h].append(d)
    shared: dict[tuple[str, str], int] = defaultdict(int)
    for docs in postings.values():
        if 2 <= len(docs) <= MAX_LINE_DOCUMENT_FREQUENCY:
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    shared[(docs[i], docs[j])] += 1

    # Identical text or PDF is decisive even when the fingerprint is too small to
    # propose a pair (a short notification, an empty text).
    for key in ("text", "pdf"):
        groups: dict[str, list[str]] = defaultdict(list)
        for d in ids:
            if key == "text":
                # Empty texts all share the hash of "" and are not copies of anything.
                value = metas[d]["text"]["sha256"] if metas[d]["text"]["char_count"] else None
            else:
                value = metas[d]["source"]["pdf_sha256"]
            if value:
                groups[value].append(d)
        for docs in groups.values():
            for i in range(len(docs)):
                for j in range(i + 1, len(docs)):
                    shared.setdefault((docs[i], docs[j]), 0)

    edges = []
    for (a, b) in sorted(shared):
        n = len(prints[a] & prints[b])
        same_text = (metas[a]["text"]["char_count"] > 0
                     and metas[a]["text"]["sha256"] == metas[b]["text"]["sha256"])
        same_file = (metas[a]["source"]["pdf_sha256"] is not None
                     and metas[a]["source"]["pdf_sha256"] == metas[b]["source"]["pdf_sha256"])
        if n < MIN_SHARED_LINES and not (same_text or same_file):
            continue
        union = len(prints[a] | prints[b])
        jaccard = n / union if union else 1.0
        containment = max(n / len(prints[a]) if prints[a] else 0.0,
                          n / len(prints[b]) if prints[b] else 0.0)
        tsim = title_similarity(metas[a]["title"], metas[b]["title"])
        differ = places_differ(metas[a]["title"], metas[b]["title"])
        relation = classify(jaccard, tsim, containment, same_file=same_file,
                            same_text=same_text, different_places=differ)
        if relation == "unrelated":
            continue
        edges.append({"a": a, "b": b, "relation": relation, "jaccard": round(jaccard, 4),
                      "containment": round(containment, 4), "title_similarity": round(tsim, 4),
                      "shared_lines": n, "same_file": same_file, "same_text": same_text,
                      "places_differ": differ})

    uf = _UnionFind()
    for e in edges:
        if e["relation"] in ("identical", "same_instrument"):
            uf.union(e["a"], e["b"])
    members: dict[str, list[str]] = defaultdict(list)
    for d in sorted(uf.parent):
        members[uf.find(d)].append(d)

    def rank(d: str):
        m = metas[d]
        return (m["category"] != "central_acts", -m["provisions"]["count"], d)

    clusters = []
    cluster_of: dict[str, str] = {}
    for i, root in enumerate(sorted(members, key=lambda r: min(members[r])), 1):
        docs = sorted(members[root])
        cid = f"dup-{i:05d}"
        for d in docs:
            cluster_of[d] = cid
        clusters.append({
            "cluster_id": cid,
            "representative": min(docs, key=rank),
            "documents": [{"document_id": d, "title": metas[d]["title"], "category": metas[d]["category"],
                           "jurisdiction": metas[d]["jurisdiction"], "year": metas[d]["year"]} for d in docs],
            "edges": [e for e in edges if e["relation"] in ("identical", "same_instrument")
                      and e["a"] in docs],
        })

    classes = _provision_classes(ids, texts, provisions, cluster_of)
    title_conflicts = [e for e in edges if e["relation"] == "identical" and e["title_similarity"] < 0.5]
    by_relation: dict[str, int] = defaultdict(int)
    for e in edges:
        by_relation[e["relation"]] += 1
    payload = {
        "corpus_schema_version": config.CORPUS_SCHEMA_VERSION,
        "method": {
            "fingerprint": f"set of NFKC-lowercased token lines with >= {MIN_LINE_TOKENS} tokens",
            "max_line_document_frequency": MAX_LINE_DOCUMENT_FREQUENCY,
            "min_shared_lines": MIN_SHARED_LINES,
            "same_instrument_jaccard": SAME_INSTRUMENT_JACCARD,
            "version_jaccard": VERSION_JACCARD,
            "title_match": TITLE_MATCH,
            "contains": CONTAINS,
        },
        "summary": {
            "documents": len(ids),
            "clusters": len(clusters),
            "documents_in_clusters": sum(len(c["documents"]) for c in clusters),
            "edges": dict(sorted(by_relation.items())),
            "provision_classes": len(classes),
            "provisions_in_classes": sum(len(c["members"]) for c in classes),
            "cross_instrument_classes": sum(1 for c in classes if not c["same_instrument"]),
            "title_conflicts": len(title_conflicts),
        },
        "clusters": clusters,
        "parallel": [e for e in edges if e["relation"] == "parallel"],
        "contains": [e for e in edges if e["relation"] == "contains"],
        # The same text published under titles that disagree. Usually an
        # abbreviation or a translation; sometimes India Code attached the wrong
        # PDF to a listing (Rajasthan's "registration act, 1908" holds the
        # Unlawful Activities (Prevention) Act). Unverified: a document here must
        # not supply gold evidence until a person has checked its title.
        "title_conflicts": title_conflicts,
    }
    return payload, classes


def _provision_classes(ids: Iterable[str], texts: dict, provisions: dict,
                       cluster_of: dict[str, str]) -> list[dict]:
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for d in ids:
        text = texts[d]
        for p in provisions[d]:
            norm = normalise_provision(text[p["char_start"]:p["char_end"]])
            groups[hashlib.sha256(norm.encode("utf-8")).hexdigest()].append((d, p["key"]))
    classes = []
    for digest in sorted(groups):
        members = sorted(groups[digest])
        documents = sorted({d for d, _ in members})
        if len(documents) < 2:
            continue
        clusters = {cluster_of.get(d, d) for d in documents}
        classes.append({
            "normalised_sha256": digest,
            "documents": len(documents),
            "same_instrument": len(clusters) == 1,
            "members": [{"document_id": d, "key": k} for d, k in members],
        })
    for i, c in enumerate(classes, 1):
        c["class_id"] = f"prov-{i:06d}"
    return classes


#: Review strata, riskiest first. Each is sampled separately so that a rare but
#: dangerous band (text-only merges) is not drowned out by the common safe one.
REVIEW_BANDS = (
    ("identical, titles disagree", lambda e: e["relation"] == "identical" and e["title_similarity"] < 0.5),
    ("identical, titles agree", lambda e: e["relation"] == "identical" and e["title_similarity"] >= 0.5),
    ("same_instrument on text alone", lambda e: e["relation"] == "same_instrument" and e["title_similarity"] < TITLE_MATCH),
    ("same_instrument, amended version", lambda e: e["relation"] == "same_instrument" and e["title_similarity"] >= TITLE_MATCH),
    ("parallel, high overlap", lambda e: e["relation"] == "parallel" and e["jaccard"] >= 0.8),
    ("parallel, moderate overlap", lambda e: e["relation"] == "parallel" and e["jaccard"] < 0.8),
    ("contains", lambda e: e["relation"] == "contains"),
)
REVIEW_COLUMNS = ("band", "relation", "jaccard", "title_similarity", "a_document_id", "a_title",
                  "a_jurisdiction", "a_opening", "b_document_id", "b_title", "b_jurisdiction",
                  "b_opening", "verdict", "notes")


def write_review_sample(corpus_dir: Path, out_path: Path, *, per_band: int = 15,
                        seed: int = 20260922) -> int:
    """A stratified sample of edges for a human to label, as a UTF-8 CSV Excel opens.

    ``verdict`` is left blank for the reviewer: ``same`` (one instrument),
    ``different`` (two laws), or ``unsure``. Returns the number of rows.
    """
    import csv
    import random

    corpus_dir = Path(corpus_dir)
    payload = json.loads((corpus_dir / config.DUPLICATES_FILENAME).read_text(encoding="utf-8"))
    edges = [e for c in payload["clusters"] for e in c["edges"]] + payload["parallel"] + payload["contains"]
    edges.sort(key=lambda e: (e["a"], e["b"]))
    rng = random.Random(seed)

    def opening(did: str) -> tuple[dict, str]:
        meta = json.loads((corpus_dir / config.META_DIRNAME / f"{did}.json").read_text(encoding="utf-8"))
        with open(corpus_dir / meta["text"]["path"], "r", encoding="utf-8", newline="") as handle:
            head = handle.read(400)
        return meta, " ".join(head.split())

    rows = []
    for band, test in REVIEW_BANDS:
        pool = [e for e in edges if test(e)]
        for e in sorted(rng.sample(pool, min(per_band, len(pool))), key=lambda e: (e["a"], e["b"])):
            (ma, ha), (mb, hb) = opening(e["a"]), opening(e["b"])
            rows.append({"band": band, "relation": e["relation"], "jaccard": e["jaccard"],
                         "title_similarity": e["title_similarity"],
                         "a_document_id": e["a"], "a_title": ma["title"], "a_jurisdiction": ma["jurisdiction"],
                         "a_opening": ha, "b_document_id": e["b"], "b_title": mb["title"],
                         "b_jurisdiction": mb["jurisdiction"], "b_opening": hb, "verdict": "", "notes": ""})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            # A leading = + - @ would be run as a formula by a spreadsheet.
            # (Not ``v[:1] in ...``: the empty string is "in" every string.)
            writer.writerow({k: ("'" + v if isinstance(v, str) and v and v[0] in "=+-@" else v)
                             for k, v in row.items()})
    return len(rows)


def write_duplicates(corpus_dir: Path) -> dict:
    """Compute and write ``duplicates.json`` and ``provision_equivalents.jsonl``."""
    from .corpus import _json, _write_atomic, write_checksums  # local: avoids an import cycle

    corpus_dir = Path(corpus_dir)
    payload, classes = find_duplicates(corpus_dir)
    _write_atomic(corpus_dir / config.DUPLICATES_FILENAME, _json(payload))
    _write_atomic(corpus_dir / config.PROVISION_EQUIVALENTS_FILENAME,
                  "".join(_json({"class_id": c["class_id"], **{k: v for k, v in c.items() if k != "class_id"}},
                                compact=True) for c in classes))
    write_checksums(corpus_dir)
    return payload["summary"]
