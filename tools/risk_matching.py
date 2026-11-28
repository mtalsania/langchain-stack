"""Match risk factors across two years and find what changed. No LLM here.

Most risk factors carry over from one 10-K to the next with small edits, so the
cheap signals do most of the work and the LLM judge (graphs/risk_changes.py) only
sees the pairs where wording really changed:

1. Identical text (after normalizing spaces and quote marks) -> unchanged.
2. Score every (this year, last year) pair by
     - embedding similarity (local fastembed, free), and
     - text overlap: share of a risk's 6-word phrases that also appear in the other.
   Pairs are taken greedily, best score first, one partner each.
     strong match:   overlap >= 0.5 or similarity >= MATCH_SIM
     possible match: similarity >= POSSIBLE_SIM -> the judge decides if it's the same risk
3. A matched pair whose words barely changed (or only numbers/dates changed) is "unchanged";
   otherwise it goes to the judge with a compact before/after of each changed passage.
4. Anything left unmatched is "new" (this year only) or "dropped" (last year only), unless
   most of its text sits in another risk of the other year: then it was "reorganized"
   (split, merged or moved), not truly new or dropped.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

import numpy as np

MATCH_SIM = 0.90
POSSIBLE_SIM = 0.80
OVERLAP_MATCH = 0.5
REORG_OVERLAP = 0.5
SHINGLE = 6
MIN_CHANGED_WORDS = 4  # fewer changed words than this (or only numbers) = unchanged
MAX_HUNKS = 6
MAX_HUNK_CHARS = 450  # per side; keeps judge prompts inside the free-tier token budget

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
                         "­": None, " ": " "})
_WORD = re.compile(r"\S+")
_NUMERIC = re.compile(r"^[\W\d]*\d[\W\d]*$|^(fiscal|january|february|march|april|may|june|july|august|september|"
                      r"october|november|december)\W*$", re.I)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTES)).strip().lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['.-][a-z0-9]+)*", normalize(text))


def shingles(text: str, n: int = SHINGLE) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


def overlap(a: set, b: set) -> float:
    """Share of a's phrases found in b."""
    return len(a & b) / len(a) if a else 0.0


def contains_quote(haystack: str, quote: str) -> bool:
    """Is quote verbatim in haystack? Ignores case, spacing, curly quotes; '...' may skip text."""
    hay = normalize(haystack)
    parts = [normalize(p).strip(" .,;:\"'") for p in re.split(r"\.\.\.|…", quote)]
    parts = [p for p in parts if p]
    if not parts:
        return False
    pos = 0
    for p in parts:
        i = hay.find(p, pos)
        if i < 0:
            return False
        pos = i + len(p)
    return True


# ------------------------------------------------------------------ diffs ----
@dataclass
class Hunk:
    before: str  # last year's sentence(s) around the change, verbatim
    after: str  # this year's sentence(s), verbatim
    changed_words: int


def _sentence_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen [start, end) to whole sentences (capped at MAX_HUNK_CHARS)."""
    s = max(text.rfind(". ", 0, start), text.rfind("\n", 0, start))
    s = 0 if s < 0 else s + 1
    frm = max(end - 1, s)  # a change ending on a full stop ends its sentence right there
    e_candidates = [i for i in (text.find(". ", frm), text.find("\n", frm)) if i >= 0]
    e = min(e_candidates) + 1 if e_candidates else len(text)
    if e - s > MAX_HUNK_CHARS:  # very long sentence: keep a window around the change
        s = max(s, start - MAX_HUNK_CHARS // 3)
        e = min(e, max(end, start) + MAX_HUNK_CHARS // 3 * 2)
    return s, e


def diff_hunks(prior: str, current: str) -> list[Hunk]:
    """Changed passages between two versions of a risk, each widened to whole sentences."""
    pt, ct = [m for m in _WORD.finditer(prior)], [m for m in _WORD.finditer(current)]
    pw, cw = [normalize(m.group()) for m in pt], [normalize(m.group()) for m in ct]
    sm = difflib.SequenceMatcher(None, pw, cw, autojunk=False)
    spans: list[list[int]] = []  # [p_start, p_end, c_start, c_end, changed_words]
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        changed = [w for w in pw[i1:i2] + cw[j1:j2] if not _NUMERIC.match(w)]
        p0 = pt[i1].start() if i1 < len(pt) else len(prior)
        p1 = pt[i2 - 1].end() if i2 > i1 else p0
        c0 = ct[j1].start() if j1 < len(ct) else len(current)
        c1 = ct[j2 - 1].end() if j2 > j1 else c0
        ps, pe = _sentence_span(prior, p0, p1)
        cs, ce = _sentence_span(current, c0, c1)
        if spans and (ps <= spans[-1][1] or cs <= spans[-1][3]):  # same sentence as the previous change
            last = spans[-1]
            last[1], last[3], last[4] = max(last[1], pe), max(last[3], ce), last[4] + len(changed)
        else:
            spans.append([ps, pe, cs, ce, len(changed)])
    return [Hunk(prior[a:b].strip(), current[c:d].strip(), n) for a, b, c, d, n in spans if n]


def changed_words(hunks: list[Hunk]) -> int:
    return sum(h.changed_words for h in hunks)


# --------------------------------------------------------------- matching ----
@dataclass
class Pair:
    id: str  # P1, P2...
    current: dict  # risk dict (id, heading, body, category, text)
    prior: dict
    similarity: float
    overlap: float
    kind: str  # "identical", "minor", "changed", "possible"
    hunks: list[Hunk] = field(default_factory=list)


@dataclass
class MatchResult:
    pairs: list[Pair]
    new: list[dict]  # risk dicts with optional "reorganized_from"
    dropped: list[dict]

    def to_judge(self) -> list[Pair]:
        return [p for p in self.pairs if p.kind in ("changed", "possible")]


def _unit(m: np.ndarray) -> np.ndarray:
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-9, None)


def embed_texts(risks: list[dict], embeddings) -> np.ndarray:
    # bge-small reads ~512 tokens: the heading and the start of the body carry the topic.
    texts = [f"{r['heading']}\n{r['body'][:1500]}" for r in risks]
    return _unit(np.array(embeddings.embed_documents(texts), dtype=float)) if texts else np.zeros((0, 1))


def match_risks(current: list[dict], prior: list[dict], embeddings) -> MatchResult:
    cur_sh = [shingles(r["text"]) for r in current]
    pri_sh = [shingles(r["text"]) for r in prior]
    pairs: list[Pair] = []
    used_c, used_p = set(), set()

    def add(i: int, j: int, sim: float, ov: float, kind: str) -> None:
        pairs.append(Pair(f"P{len(pairs) + 1}", current[i], prior[j], round(sim, 3), round(ov, 3), kind))
        used_c.add(i)
        used_p.add(j)

    # 1. identical text
    by_text = {}
    for j, r in enumerate(prior):
        by_text.setdefault(normalize(r["text"]), j)
    for i, r in enumerate(current):
        j = by_text.get(normalize(r["text"]))
        if j is not None and j not in used_p:
            add(i, j, 1.0, 1.0, "identical")

    # 2. greedy on the best of similarity and overlap
    if current and prior:
        sim = embed_texts(current, embeddings) @ embed_texts(prior, embeddings).T
    else:
        sim = np.zeros((len(current), len(prior)))
    cands = []
    for i in range(len(current)):
        if i in used_c:
            continue
        for j in range(len(prior)):
            if j in used_p:
                continue
            ov = max(overlap(cur_sh[i], pri_sh[j]), overlap(pri_sh[j], cur_sh[i]))
            s = float(sim[i, j])
            if ov >= OVERLAP_MATCH or s >= MATCH_SIM:
                cands.append((1 + ov + s, i, j, s, ov, "match"))
            elif s >= POSSIBLE_SIM:
                cands.append((s, i, j, s, ov, "possible"))
    for _, i, j, s, ov, kind in sorted(cands, reverse=True):
        if i in used_c or j in used_p:
            continue
        add(i, j, s, ov, kind)

    # 3. what changed inside each pair
    for p in pairs:
        if p.kind == "identical":
            continue
        p.hunks = diff_hunks(p.prior["text"], p.current["text"])
        if p.kind == "match":
            p.kind = "changed" if changed_words(p.hunks) >= MIN_CHANGED_WORDS else "minor"

    # 4. unmatched: new / dropped, unless the text lives on in another risk
    def leftovers(items, sh, other, other_sh, used):
        out = []
        for k, r in enumerate(items):
            if k in used:
                continue
            best = max(range(len(other)), key=lambda j: overlap(sh[k], other_sh[j]), default=None)
            ov = overlap(sh[k], other_sh[best]) if best is not None else 0.0
            out.append({**r, "overlap_elsewhere": round(ov, 3),
                        "reorganized_with": other[best]["heading"] if best is not None and ov >= REORG_OVERLAP else ""})
        return out

    new = leftovers(current, cur_sh, prior, pri_sh, used_c)
    dropped = leftovers(prior, pri_sh, current, cur_sh, used_p)
    return MatchResult(pairs, new, dropped)


def resolve_possible(result: MatchResult, pair_id: str) -> None:
    """The judge said a 'possible' pair is two different risks: split it into new + dropped."""
    for p in list(result.pairs):
        if p.id == pair_id and p.kind == "possible":
            result.pairs.remove(p)
            result.new.append({**p.current, "overlap_elsewhere": p.overlap, "reorganized_with": ""})
            result.dropped.append({**p.prior, "overlap_elsewhere": p.overlap, "reorganized_with": ""})
