"""Risk-factor changes, year over year (milestone M2).

For each company: what risks are new in this year's 10-K, which were dropped, and
which changed in a way that matters, each with verbatim quotes from both years.

Two levels of parallel fan-out with LangGraph's Send:

    START --Send per ticker--> company subgraph (all tickers at once) --> combine --> END

    company subgraph:
    load (2 10-Ks, split Item 1A; no LLM) -> match (embeddings + text overlap; no LLM)
        --Send per batch of changed pairs--> judge (small model, in parallel) --> render (code)

Each `judge` call returns its verdicts into a list with an `operator.add` reducer, so
the parallel branches merge without overwriting each other. That reducer is what
makes map-reduce in LangGraph work.

Faithfulness rules, checked in code by `render` (and scored by evals/risk_faithfulness.py):
- a changed risk is reported only with a quote from each year, and each quote must
  appear verbatim in that year's Item 1A;
- a new or dropped risk is quoted by its own heading, and is flagged if most of its
  text still appears in the other year's Item 1A.

    python -m graphs.risk_changes NVDA
    python -m graphs.risk_changes NVDA AAPL MSFT AMZN GOOGL --out reports/
    python -m graphs.risk_changes NVDA --no-llm      # matching only, zero model calls
"""
from __future__ import annotations

import argparse
import operator
from pathlib import Path
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from common.llm import get_llm, structured
from core.risk_factors import risk_sections_two_years
from core.tickers import lookup
from tools.risk_matching import Pair, contains_quote, match_risks, overlap, shingles

BATCH_CHARS = 7000  # diff text per judge call: ~1.8K tokens in, well under Groq's 8K tokens/minute
BATCH_PAIRS = 5
MAX_HUNKS_SHOWN = 4  # the biggest changes in a pair; the rest are counted, not shown


# ----------------------------------------------------------------- state ----
class CompanyState(TypedDict, total=False):
    ticker: str
    no_llm: bool
    company: dict
    current: dict  # RiskSection.to_dict() for this year's 10-K
    prior: dict
    pairs: list[dict]
    new: list[dict]
    dropped: list[dict]
    batch: list[dict]  # only inside a judge branch
    verdicts: Annotated[list[dict], operator.add]
    reports: Annotated[list[dict], operator.add]  # one per company; shared with the parent graph


class CompanyOutput(TypedDict):
    # Only this key flows back to the parent graph. Without it, every parallel company
    # branch would also write `ticker`, `no_llm`... and LangGraph would reject the clash.
    reports: Annotated[list[dict], operator.add]


class WatchlistState(TypedDict, total=False):
    tickers: list[str]
    no_llm: bool
    reports: Annotated[list[dict], operator.add]
    summary_md: str


# --------------------------------------------------------- model schemas ----
class PairVerdict(BaseModel):
    pair_id: str = Field(description="The pair id, e.g. P3")
    same_risk: bool = Field(description="True if both versions describe the same underlying risk")
    material: bool = Field(description="True if the change matters to an investor: a new exposure, event, "
                           "market, regulation or number; a wider or narrower scope; a removed safeguard. "
                           "False for rewording, reordering, updated dates or style")
    change_type: Literal["added", "removed", "revised"] = Field(
        description="added: new wording with no counterpart last year; removed: wording deleted this year; "
                    "revised: a passage rewritten")
    summary: str = Field(description="One sentence: what changed, saying only what the quotes show. Empty if not material")
    quote_now: str = Field(description="Exact words copied from the NOW text that show the change. Empty if removed")
    quote_before: str = Field(description="Exact words copied from the BEFORE text that were replaced or deleted. "
                              "Empty if added")


class JudgeBatch(BaseModel):
    verdicts: list[PairVerdict]


JUDGE_SYSTEM = """You compare two versions of risk factors from a company's 10-K filings: last year (BEFORE)
and this year (NOW). For each pair you get the headings and the passages that changed.
For every pair, return one verdict:
- same_risk: do both versions describe the same underlying risk?
- material: would an investor care about the change? New exposures, events, markets, laws, customers,
  numbers or a wider/narrower scope are material. Rewording, reordering, updated dates are not.
- change_type: "added" (new wording, nothing it replaced), "removed" (wording deleted, nothing replaced it),
  or "revised" (a passage rewritten: before and after say the same kind of thing differently).
- summary: one plain sentence saying what changed. Say only what your quotes show; no other facts or numbers.
- quote_now: copy words EXACTLY from a NOW passage that shows the change (empty if removed).
- quote_before: copy words EXACTLY from the BEFORE passage that was replaced or deleted (empty if added).
  Never pick an unrelated BEFORE sentence just to fill the field.
Quotes are 5 to 40 words, never paraphrased. Use only the text given. Return a verdict for EVERY pair id."""


# ----------------------------------------------------------------- nodes ----
def load(state: CompanyState) -> dict:
    co = lookup(state["ticker"])
    cur, pri = risk_sections_two_years(co.cik)
    return {"company": {"ticker": co.ticker, "cik": co.cik, "name": co.name},
            "current": cur.to_dict(), "prior": pri.to_dict()}


def _pair_to_dict(p: Pair) -> dict:
    return {"id": p.id, "kind": p.kind, "similarity": p.similarity, "overlap": p.overlap,
            "current": {k: p.current[k] for k in ("id", "heading", "category", "text")},
            "prior": {k: p.prior[k] for k in ("id", "heading", "category", "text")},
            "hunks": [h.__dict__ for h in p.hunks]}


def match(state: CompanyState) -> dict:
    from common.embeddings import get_embeddings

    res = match_risks(state["current"]["risks"], state["prior"]["risks"], get_embeddings())
    return {"pairs": [_pair_to_dict(p) for p in res.pairs], "new": res.new, "dropped": res.dropped}


def pair_prompt(p: dict) -> str:
    lines = [f"### {p['id']}", f"BEFORE heading: {p['prior']['heading']}", f"NOW heading: {p['current']['heading']}"]
    hunks = sorted(p["hunks"], key=lambda h: -h["changed_words"])[:MAX_HUNKS_SHOWN]
    for k, h in enumerate(hunks, 1):
        lines += [f"Change {k} BEFORE: {h['before'] or '(nothing)'}", f"Change {k} NOW: {h['after'] or '(nothing)'}"]
    if len(p["hunks"]) > len(hunks):
        lines.append(f"({len(p['hunks']) - len(hunks)} more changes not shown)")
    return "\n".join(lines)


def batches(pairs: list[dict]) -> list[list[dict]]:
    out, cur, size = [], [], 0
    for p in pairs:
        n = len(pair_prompt(p))
        if cur and (size + n > BATCH_CHARS or len(cur) >= BATCH_PAIRS):
            out.append(cur)
            cur, size = [], 0
        cur.append(p)
        size += n
    return out + ([cur] if cur else [])


def fan_out_judges(state: CompanyState):
    """Map step: one judge branch per batch of changed pairs, all running at once."""
    todo = [p for p in state["pairs"] if p["kind"] in ("changed", "possible")]
    if state.get("no_llm") or not todo:
        return "render"
    return [Send("judge", {"batch": b, "company": state["company"]}) for b in batches(todo)]


def judge(state: CompanyState) -> dict:
    co = state["company"]
    judge_llm = structured(get_llm(small=True), JudgeBatch)

    def ask(pairs: list[dict]) -> list[dict]:
        prompt = f"Company: {co['name']} ({co['ticker']})\n\n" + "\n\n".join(pair_prompt(p) for p in pairs)
        out = judge_llm.invoke([SystemMessage(JUDGE_SYSTEM), HumanMessage(prompt)])
        return [v.model_dump() for v in out.verdicts]

    try:
        verdicts = ask(state["batch"])
        got = {v["pair_id"] for v in verdicts}
        missing = [p for p in state["batch"] if p["id"] not in got]
        if missing:  # the small model sometimes skips a pair in a batch: ask once more for just those
            verdicts += ask(missing)
        return {"verdicts": verdicts}
    except Exception as e:  # one failed batch should not sink the whole company
        return {"verdicts": [{"pair_id": p["id"], "error": str(e)[:200]} for p in state["batch"]]}


def check_evidence(v: dict, now_text: str, before_text: str, now_full: str, before_full: str) -> dict:
    """The faithfulness rule for one claimed change, checked in code.

    revised: a quote from each year, each found word for word in that year's version of the risk.
    added:   a quote from this year that is NOT anywhere in last year's Item 1A.
    removed: a quote from last year that is NOT anywhere in this year's Item 1A.
    """
    kind, qn, qb = v["change_type"], v["quote_now"].strip(), v["quote_before"].strip()
    if kind == "added":
        qb = ""  # a "before" quote for pure additions is filler; never show it
    if kind == "removed":
        qn = ""
    problems = []
    now_ok = contains_quote(now_text, qn) if qn else kind == "removed"
    before_ok = contains_quote(before_text, qb) if qb else kind == "added"
    if not now_ok:
        problems.append(f"this year's quote missing or not found verbatim: {qn[:100]!r}")
    if not before_ok:
        problems.append(f"last year's quote missing or not found verbatim: {qb[:100]!r}")
    if kind == "added" and qn and contains_quote(before_full, qn):
        problems.append(f"'added' text was already in last year's 10-K: {qn[:100]!r}")
    if kind == "removed" and qb and contains_quote(now_full, qb):
        problems.append(f"'removed' text is still in this year's 10-K: {qb[:100]!r}")
    if kind == "revised" and qn and qn.lower() == qb.lower():
        problems.append("'revised' quotes are identical")
    return {"quote_now": qn, "quote_before": qb, "quote_now_ok": now_ok, "quote_before_ok": before_ok,
            "problems": problems}


def classify(state: CompanyState) -> dict:
    """Combine matches and verdicts into the final lists, and check every quote."""
    cur_full, pri_full = state["current"]["full_text"], state["prior"]["full_text"]
    verdicts = {v["pair_id"]: v for v in state.get("verdicts", [])}
    changed, minor, unjudged, issues = [], 0, [], []
    new, dropped = list(state["new"]), list(state["dropped"])
    for p in state["pairs"]:
        if p["kind"] in ("identical", "minor"):
            minor += p["kind"] == "minor"
            continue
        v = verdicts.get(p["id"])
        if v is None or "error" in v:
            unjudged.append({**p, "error": (v or {}).get("error", "skipped (--no-llm)" if state.get("no_llm")
                                                         else "no verdict returned")})
            continue
        if not v["same_risk"]:
            if p["kind"] == "possible":  # two different risks after all
                new.append({**p["current"], "overlap_elsewhere": p["overlap"], "reorganized_with": ""})
                dropped.append({**p["prior"], "overlap_elsewhere": p["overlap"], "reorganized_with": ""})
                continue
        if not v["material"]:
            minor += 1
            continue
        entry = {"pair_id": p["id"], "heading_now": p["current"]["heading"], "heading_before": p["prior"]["heading"],
                 "category": p["current"]["category"], "summary": v["summary"], "change_type": v["change_type"],
                 **check_evidence(v, p["current"]["text"], p["prior"]["text"], cur_full, pri_full)}
        issues += [f"{p['id']}: {problem}" for problem in entry["problems"]]
        changed.append(entry)

    def lifecycle(items: list[dict], other_full: str) -> list[dict]:
        other_sh = shingles(other_full)
        out = []
        for r in items:
            still = overlap(shingles(r["text"]), other_sh)
            out.append({"id": r["id"], "heading": r["heading"], "category": r["category"],
                        "reorganized_with": r.get("reorganized_with", ""), "text_in_other_year": round(still, 3),
                        "heading_in_other_year": contains_quote(other_full, r["heading"])})
        return out

    new_l, dropped_l = lifecycle(new, pri_full), lifecycle(dropped, cur_full)
    return {"changed": changed, "minor": minor, "unjudged": unjudged, "issues": issues,
            "new": [r for r in new_l if not r["reorganized_with"]],
            "dropped": [r for r in dropped_l if not r["reorganized_with"]],
            "reorganized": [r for r in new_l + dropped_l if r["reorganized_with"]]}


def render(state: CompanyState) -> dict:
    co, cur, pri = state["company"], state["current"], state["prior"]
    issues = [f"this year's split: {n}" for n in cur["notes"] if cur["method"] != "emphasis"]
    issues += [f"last year's split: {n}" for n in pri["notes"] if pri["method"] != "emphasis"]
    res = classify(state)
    issues += res["issues"]
    for r in res["new"]:
        if r["text_in_other_year"] >= 0.5 or r["heading_in_other_year"]:
            issues.append(f"'new' risk {r['id']} still mostly in last year's text: {r['heading'][:80]!r}")
    for r in res["dropped"]:
        if r["text_in_other_year"] >= 0.5 or r["heading_in_other_year"]:
            issues.append(f"'dropped' risk {r['id']} still mostly in this year's text: {r['heading'][:80]!r}")

    identical = sum(p["kind"] == "identical" for p in state["pairs"])
    f_cur, f_pri = cur["filing"], pri["filing"]
    md = [
        f"# {co['name']} ({co['ticker']}): risk factor changes",
        "",
        f"This year: 10-K filed {f_cur['filing_date']} (period {f_cur['report_date']}), {len(cur['risks'])} risk factors: {cur['url']}",
        f"Last year: 10-K filed {f_pri['filing_date']} (period {f_pri['report_date']}), {len(pri['risks'])} risk factors: {pri['url']}",
        "",
        f"**{len(res['new'])} new, {len(res['dropped'])} dropped, {len(res['changed'])} materially changed**, "
        f"{len(res['reorganized'])} reorganized, {res['minor']} with minor edits, {identical} identical"
        + (f", {len(res['unjudged'])} not judged" if res["unjudged"] else "") + ".",
        "",
        "## New this year",
    ]
    md += [f"- **{r['heading']}**" + (f" _({r['category']})_" if r["category"] else "") for r in res["new"]] or ["- none"]
    md += ["", "## Dropped since last year"]
    md += [f"- **{r['heading']}**" + (f" _({r['category']})_" if r["category"] else "") for r in res["dropped"]] or ["- none"]
    md += ["", "## Materially changed"]
    for c in res["changed"]:
        flag = " ⚠ see Unverified" if c["problems"] else ""
        now = f"\"{c['quote_now']}\"" if c["quote_now"] else "(removed; not in this year's 10-K)"
        before = f"\"{c['quote_before']}\"" if c["quote_before"] else "(new wording; not in last year's 10-K)"
        md += [f"- **{c['heading_now']}** ({c['change_type']}){flag}", f"  - {c['summary']}",
               f"  - Now: {now}", f"  - Before: {before}"]
    if not res["changed"]:
        md.append("- none" if not state.get("no_llm") else "- (not judged: run without --no-llm)")
    if res["reorganized"]:
        md += ["", "## Reorganized (split, merged or moved; text largely carried over)"]
        md += [f"- {r['heading'][:160]}  <-> {r['reorganized_with'][:160]}" for r in res["reorganized"]]
    if res["unjudged"]:
        md += ["", "## Not judged"]
        md += [f"- {p['current']['heading'][:160]} ({p.get('error', 'no LLM')})" for p in res["unjudged"]]
    if issues:
        md += ["", "## Unverified", *[f"- {i}" for i in issues]]
    report = {"ticker": co["ticker"], "name": co["name"], "current_filing": f_cur, "prior_filing": f_pri,
              "counts": {"current": len(cur["risks"]), "prior": len(pri["risks"]), "identical": identical,
                         "minor": res["minor"], "changed": len(res["changed"]), "new": len(res["new"]),
                         "dropped": len(res["dropped"]), "reorganized": len(res["reorganized"]),
                         "unjudged": len(res["unjudged"])},
              **{k: res[k] for k in ("changed", "new", "dropped", "reorganized")},
              "issues": issues, "report_md": "\n".join(md)}
    return {"reports": [report]}


# ----------------------------------------------------------------- graph ----
def build_company_graph():
    g = StateGraph(CompanyState, output_schema=CompanyOutput)
    g.add_node("load", load)
    g.add_node("match", match)
    g.add_node("judge", judge)
    g.add_node("render", render)
    g.add_edge(START, "load")
    g.add_edge("load", "match")
    g.add_conditional_edges("match", fan_out_judges, ["judge", "render"])
    g.add_edge("judge", "render")  # render waits for every judge branch (reduce step)
    g.add_edge("render", END)
    return g.compile()


company_graph = build_company_graph()


def fan_out_companies(state: WatchlistState) -> list[Send]:
    return [Send("company", {"ticker": t.upper(), "no_llm": state.get("no_llm", False)}) for t in state["tickers"]]


def combine(state: WatchlistState) -> dict:
    order = {t.upper(): i for i, t in enumerate(state["tickers"])}
    reports = sorted(state["reports"], key=lambda r: order.get(r["ticker"], 99))
    rows = ["| Company | Risks now / before | New | Dropped | Changed | Reorganized | Unverified |",
            "|---|---|---|---|---|---|---|"]
    for r in reports:
        c = r["counts"]
        rows.append(f"| {r['ticker']} | {c['current']} / {c['prior']} | {c['new']} | {c['dropped']} | "
                    f"{c['changed']} | {c['reorganized']} | {len(r['issues'])} |")
    summary = "\n".join(["# Risk factor changes, year over year", "", *rows, "",
                         *[r["report_md"] + "\n" for r in reports]])
    return {"summary_md": summary}


def build_graph(checkpointer=None):
    g = StateGraph(WatchlistState)
    g.add_node("company", company_graph)  # a compiled graph used as a node = subgraph
    g.add_node("combine", combine)
    g.add_conditional_edges(START, fan_out_companies, ["company"])
    g.add_edge("company", "combine")
    g.add_edge("combine", END)
    return g.compile(checkpointer=checkpointer)


graph = build_graph()  # for langgraph dev / Studio: input {"tickers": ["NVDA", "AAPL"]}


def run(tickers: list[str], no_llm: bool = False) -> dict:
    from langchain_core.callbacks import get_usage_metadata_callback

    with get_usage_metadata_callback() as usage:
        out = graph.invoke({"tickers": tickers, "no_llm": no_llm}, {"max_concurrency": 4})
    out["usage"] = {m: {k: u.get(k, 0) for k in ("input_tokens", "output_tokens", "total_tokens")}
                    for m, u in usage.usage_metadata.items()}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Year-over-year risk factor changes from 10-K Item 1A.")
    ap.add_argument("tickers", nargs="+")
    ap.add_argument("--out", help="folder to write one markdown report per company plus summary.md")
    ap.add_argument("--no-llm", action="store_true", help="split and match only; no model calls")
    args = ap.parse_args(argv)
    out = run(args.tickers, args.no_llm)
    print(out["summary_md"])
    for model, u in out["usage"].items():
        print(f"tokens used on {model}: {u['total_tokens']:,} ({u['input_tokens']:,} in, {u['output_tokens']:,} out)")
    if args.out:
        d = Path(args.out)
        d.mkdir(parents=True, exist_ok=True)
        for r in out["reports"]:
            (d / f"{r['ticker']}.md").write_text(r["report_md"])
        (d / "summary.md").write_text(out["summary_md"])
        print(f"\nreports written to {d}/")
    return 1 if any(r["issues"] for r in out["reports"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
