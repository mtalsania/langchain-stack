"""Company research subgraph (milestone M1).

    START -> fetch_numbers -> plan -> retrieve -> write -> render -> END
             (XBRL, no LLM)   (small)  (RAG)     (strong)  (code)

Two rules keep the brief trustworthy:
1. The model never types a number. It writes placeholders like {{revenue:2025}} or
   {{growth:revenue:2025}}, and `render` fills them in from XBRL with a citation.
   Any number the model types anyway is flagged in `issues`.
2. Every qualitative claim cites a retrieved 10-K passage as [S1], [S2]...
   Citations to passages that were never retrieved are flagged too.

Later milestones reuse this graph as a step: M3 runs it once per company in parallel,
and M4 adds a verification + human review loop after it.

    python -m graphs.company_research NVDA
    python -m graphs.company_research AAPL --focus "services growth and China risk"
"""
from __future__ import annotations

import argparse
import re
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from common.llm import get_llm
from tools.filing_search import search
from tools.financials import LABELS, archive_url, company_facts, facts_prompt_table, format_value


# ----------------------------------------------------------------- state ----
class ResearchState(TypedDict, total=False):
    ticker: str
    focus: str  # optional angle, e.g. "data center demand"
    years: int
    snapshot: dict  # company + XBRL facts (tools.financials.company_facts)
    queries: list[str]
    sources: list[dict]  # retrieved 10-K passages, id S1..Sn
    draft: dict  # CompanyBrief as dict, with placeholders
    brief_md: str  # final rendered markdown
    issues: list[str]  # anything the renderer could not verify


# --------------------------------------------------------- model schemas ----
class ResearchPlan(BaseModel):
    queries: list[str] = Field(
        description="3 to 4 short search queries for the 10-K text: what drove results, "
        "segments/products, strategy and priorities, biggest risks",
        min_length=1,
        max_length=4,
    )


class CompanyBrief(BaseModel):
    overview: str = Field(description="2-3 sentences: what the company does and how it makes money, citing [S#]")
    financial_story: list[str] = Field(
        description="3-5 bullets on the numbers. Use ONLY placeholders like {{revenue:2025}}, "
        "{{gross_margin:2025}} or {{growth:revenue:2025}} for figures; never type a number"
    )
    drivers: list[str] = Field(description="2-4 bullets on what drove results, each citing [S#]")
    risks: list[str] = Field(description="2-4 most material risks, each citing [S#]")
    priorities: list[str] = Field(description="1-3 management priorities or strategic bets, each citing [S#]")


# ----------------------------------------------------------------- nodes ----
def fetch_numbers(state: ResearchState) -> dict:
    return {"snapshot": company_facts(state["ticker"], state.get("years", 3)), "issues": []}


def plan(state: ResearchState) -> dict:
    snap = state["snapshot"]
    focus = state.get("focus") or "a general research brief"
    planner = get_llm(small=True).with_structured_output(ResearchPlan)
    result = planner.invoke(
        f"You are planning research on {snap['name']} ({snap['ticker']}) from its latest 10-K. "
        f"Focus: {focus}. Write 3-4 search queries for the filing text."
    )
    return {"queries": result.queries[:4]}


def retrieve(state: ResearchState) -> dict:
    seen, sources = set(), []
    for q in state["queries"]:
        for d in search(state["ticker"], q, k=3):
            key = (d.metadata["item"], d.metadata["chunk"])
            if key in seen:
                continue
            seen.add(key)
            sources.append({"id": f"S{len(sources) + 1}", "text": d.page_content, **d.metadata})
            if len(sources) >= 10:  # stay inside the free-tier token budget
                return {"sources": sources}
    return {"sources": sources}


WRITER_SYSTEM = """You write concise, factual company research briefs from SEC 10-K filings.
Rules:
- Figures: write ONLY the placeholders from the FIGURES list, exactly as shown, e.g. {{revenue:2025}}.
  For year-over-year change write {{growth:METRIC:YEAR}}. Never type a number, percentage or dollar amount.
- Every claim about drivers, risks or priorities cites a passage like [S3]. Use only the passage ids given.
- If the passages don't support a point, leave it out. No outside knowledge."""


def write(state: ResearchState) -> dict:
    snap = state["snapshot"]
    passages = "\n\n".join(f"[{s['id']}] (Item {s['item']} {s['section']})\n{s['text']}" for s in state["sources"])
    prompt = (
        f"Company: {snap['name']} ({snap['ticker']}), fiscal years {', '.join(map(str, snap['years']))}.\n"
        f"Focus: {state.get('focus') or 'general research brief'}\n\n"
        f"FIGURES (placeholder = value):\n{facts_prompt_table(snap)}\n\n"
        f"PASSAGES:\n{passages}"
    )
    writer = get_llm().with_structured_output(CompanyBrief)
    brief = writer.invoke([SystemMessage(WRITER_SYSTEM), HumanMessage(prompt)])
    return {"draft": brief.model_dump()}


# ------------------------------------------------------------- rendering ----
PLACEHOLDER = re.compile(r"\{\{\s*(growth:)?([a-z_]+)\s*:\s*(?:FY)?(\d{4})\s*\}\}")
CITATION = re.compile(r"\[(S\d+)\]")
# Numbers the model typed itself: money, percentages, or scaled amounts. Years alone are fine.
TYPED_NUMBER = re.compile(r"\$\s?\d|\d(?:[\d,]*\.?\d*)\s?%|\d[\d,.]*\s?(?:billion|million|thousand|bn)\b", re.I)


def fill_placeholders(text: str, snap: dict, cited: dict, issues: list[str]) -> str:
    facts = snap["facts"]

    def cite(accns: list[str]) -> str:
        tags = []
        for a in accns:
            if a not in cited:
                cited[a] = f"F{len(cited) + 1}"
            tags.append(cited[a])
        return "[" + ",".join(tags) + "]"

    def sub(m: re.Match) -> str:
        growth, name, fy = bool(m.group(1)), m.group(2), m.group(3)
        cur = facts.get(name, {}).get(fy)
        if not growth:
            if not cur:
                issues.append(f"no XBRL value for {name} FY{fy}")
                return f"[missing {name} FY{fy}]"
            return f"{format_value(cur['value'], cur['unit'])} {cite(cur['accessions'])}"
        prev = facts.get(name, {}).get(str(int(fy) - 1))
        if not (cur and prev) or not prev["value"]:
            issues.append(f"cannot compute growth of {name} for FY{fy}")
            return f"[missing growth {name} FY{fy}]"
        if cur["unit"] == "ratio":  # margin change is in percentage points
            return f"{(cur['value'] - prev['value']) * 100:+.1f} pts {cite(cur['accessions'])}"
        return f"{(cur['value'] / prev['value'] - 1) * 100:+.1f}% {cite(cur['accessions'])}"

    return PLACEHOLDER.sub(sub, text)


def check_text(text: str, source_ids: set[str], issues: list[str]) -> None:
    bare = PLACEHOLDER.sub("", text)
    for m in TYPED_NUMBER.finditer(bare):
        issues.append(f"model typed a number itself: {bare[max(0, m.start() - 30):m.end() + 10]!r}")
    for sid in CITATION.findall(text):
        if sid not in source_ids:
            issues.append(f"citation {sid} does not match any retrieved passage")


def render(state: ResearchState) -> dict:
    snap, draft = state["snapshot"], state["draft"]
    issues = list(state.get("issues", []))
    source_ids = {s["id"] for s in state["sources"]}
    cited_filings: dict[str, str] = {}

    def fix(text: str) -> str:
        check_text(text, source_ids, issues)
        return fill_placeholders(text, snap, cited_filings, issues)

    def bullets(items: list[str]) -> str:
        return "\n".join(f"- {fix(t)}" for t in items)

    years = snap["years"]
    table_rows = ["| Metric | " + " | ".join(f"FY{y}" for y in years) + " |", "|---" * (len(years) + 1) + "|"]
    for name, by_year in snap["facts"].items():
        cells = [format_value(by_year[str(y)]["value"], by_year[str(y)]["unit"]) if str(y) in by_year else "n/a"
                 for y in years]
        table_rows.append(f"| {LABELS[name]} | " + " | ".join(cells) + " |")

    body = [
        f"# {snap['name']} ({snap['ticker']})",
        "",
        fix(draft["overview"]),
        "",
        "## Financial story",
        bullets(draft["financial_story"]),
        "",
        "## What drove results",
        bullets(draft["drivers"]),
        "",
        "## Key risks",
        bullets(draft["risks"]),
        "",
        "## Management priorities",
        bullets(draft["priorities"]),
        "",
        "## Numbers (SEC XBRL, as filed)",
        *table_rows,
        "",
        "## Sources",
    ]
    for accn, tag in cited_filings.items():
        body.append(f"- [{tag}] 10-K XBRL, accession {accn}: {archive_url(snap['cik'], accn)}")
    used = set(CITATION.findall(" ".join([draft["overview"], *draft["drivers"], *draft["risks"], *draft["priorities"]])))
    for s in state["sources"]:
        if s["id"] in used:
            body.append(f"- [{s['id']}] 10-K Item {s['item']} ({s['section']}), filed {s['filing_date']}: {s['url']}")
    if issues:
        body += ["", "## Unverified", *[f"- {i}" for i in issues]]
    return {"brief_md": "\n".join(body), "issues": issues}


# ----------------------------------------------------------------- graph ----
def build_graph(checkpointer=None):
    g = StateGraph(ResearchState)
    for name, fn in [("fetch_numbers", fetch_numbers), ("plan", plan), ("retrieve", retrieve),
                     ("write", write), ("render", render)]:
        g.add_node(name, fn)
    g.add_edge(START, "fetch_numbers")
    g.add_edge("fetch_numbers", "plan")
    g.add_edge("plan", "retrieve")
    g.add_edge("retrieve", "write")
    g.add_edge("write", "render")
    g.add_edge("render", END)
    return g.compile(checkpointer=checkpointer)


graph = build_graph()  # for langgraph dev / Studio


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Research one company from its 10-K and XBRL data.")
    ap.add_argument("ticker")
    ap.add_argument("--focus", default="")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args(argv)
    out = graph.invoke({"ticker": args.ticker.upper(), "focus": args.focus, "years": args.years})
    print(out["brief_md"])
    return 1 if out["issues"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
