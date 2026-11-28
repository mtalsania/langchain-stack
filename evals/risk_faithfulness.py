"""M2 eval: is every reported risk change faithful to the filings?

No hand labels needed: each claim carries its own evidence, and the evidence can be
checked against the source. Evaluators re-read both 10-Ks from the SEC (cached on
disk) rather than trusting the graph's own state.

Code evaluators (exact):
- quotes_verbatim:      each "changed" claim's quotes appear word for word in the right year's Item 1A
- evidence_fits_change: revised = a quote from each year, and they differ; added = this year's quote is
                        absent from last year's Item 1A; removed = last year's quote is absent from this year's
- new_is_new:           a "new" risk's heading is absent from last year's Item 1A, and under half its text is there
- dropped_is_dropped:   the same check the other way round
LLM evaluator (with --judge, small model, one call per company):
- summary_supported:    the one-line summary says only what the two quotes show

    python -m evals.risk_faithfulness --local                 # run the graph, score, print a table
    python -m evals.risk_faithfulness                         # same, as a LangSmith dataset + experiment
    python -m evals.risk_faithfulness --judge --out reports/  # add the LLM judge; save the 5 reports

Pass bar: every code evaluator >= 0.9 on average across the 5 companies. The M2 "done"
bar is still your spot-check of the reports; this eval catches unfaithful output first.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from core.filings import Filing
from core.risk_factors import risk_section
from tools.risk_matching import contains_quote, overlap, shingles

load_dotenv()  # LANGSMITH_API_KEY and friends, before the LangSmith client is created

TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]
DATASET = "risk-factor-changes"
PASS = 0.9
CODE_KEYS = ["quotes_verbatim", "evidence_fits_change", "new_is_new", "dropped_is_dropped"]


def target(inputs: dict) -> dict:
    from graphs.risk_changes import company_graph

    return company_graph.invoke({"ticker": inputs["ticker"]}, {"max_concurrency": 2})["reports"][0]


def _section_text(filing: dict) -> str:
    return risk_section(Filing(**filing)).full_text


def _ratio(ok: int, total: int) -> float:
    return ok / total if total else 1.0  # nothing claimed = nothing unfaithful


# ------------------------------------------------------------ evaluators ----
def quotes_verbatim(outputs: dict) -> dict:
    now, before = _section_text(outputs["current_filing"]), _section_text(outputs["prior_filing"])
    bad = [c["heading_now"][:60] for c in outputs["changed"]
           if (c["quote_now"] and not contains_quote(now, c["quote_now"]))
           or (c["quote_before"] and not contains_quote(before, c["quote_before"]))
           or not (c["quote_now"] or c["quote_before"])]
    n = len(outputs["changed"])
    return {"key": "quotes_verbatim", "score": _ratio(n - len(bad), n), "comment": "; ".join(bad)[:500]}


def evidence_fits_change(outputs: dict) -> dict:
    now, before = _section_text(outputs["current_filing"]), _section_text(outputs["prior_filing"])
    bad = []
    for c in outputs["changed"]:
        qn, qb, kind = c["quote_now"], c["quote_before"], c.get("change_type", "revised")
        ok = {"revised": bool(qn and qb) and qn.strip().lower() != qb.strip().lower(),
              "added": bool(qn) and not contains_quote(before, qn),
              "removed": bool(qb) and not contains_quote(now, qb)}.get(kind, False)
        if not ok:
            bad.append(f"{kind}: {c['heading_now'][:60]}")
    n = len(outputs["changed"])
    return {"key": "evidence_fits_change", "score": _ratio(n - len(bad), n), "comment": "; ".join(bad)[:500]}


def _absent(items: list[dict], other_text: str, key: str) -> dict:
    other_sh = shingles(other_text)
    bad = []
    for r in items:
        # The reports carry headings only; the full text comes from that year's section.
        still = overlap(shingles(r.get("text", r["heading"])), other_sh)
        if contains_quote(other_text, r["heading"]) or still >= 0.5:
            bad.append(r["heading"][:60])
    return {"key": key, "score": _ratio(len(items) - len(bad), len(items)), "comment": "; ".join(bad)[:500]}


def _with_text(items: list[dict], filing: dict) -> list[dict]:
    by_heading = {r.heading: r.text for r in risk_section(Filing(**filing)).risks}
    return [{**r, "text": by_heading.get(r["heading"], r["heading"])} for r in items]


def new_is_new(outputs: dict) -> dict:
    items = _with_text(outputs["new"], outputs["current_filing"])
    return _absent(items, _section_text(outputs["prior_filing"]), "new_is_new")


def dropped_is_dropped(outputs: dict) -> dict:
    items = _with_text(outputs["dropped"], outputs["prior_filing"])
    return _absent(items, _section_text(outputs["current_filing"]), "dropped_is_dropped")


class Support(BaseModel):
    index: int
    supported: bool = Field(description="True only if the summary states nothing the quotes don't show")


class SupportBatch(BaseModel):
    results: list[Support]


def summary_supported(outputs: dict) -> dict:
    from common.llm import get_llm, structured

    claims = outputs["changed"]
    if not claims:
        return {"key": "summary_supported", "score": 1.0}
    prompt = ("For each item, is the SUMMARY fully supported by the two quotes from a company's 10-K "
              "(BEFORE = last year, NOW = this year)? Unsupported = it states a fact, number or direction "
              "the quotes don't show.\n\n" + "\n\n".join(
                  f"[{i}] SUMMARY: {c['summary']}\nBEFORE: {c['quote_before'] or '(nothing: new wording)'}\n"
                  f"NOW: {c['quote_now'] or '(nothing: wording removed)'}"
                  for i, c in enumerate(claims)))
    out = structured(get_llm(small=True), SupportBatch).invoke(prompt)
    ok = {r.index for r in out.results if r.supported}
    bad = [claims[i]["heading_now"][:60] for i in range(len(claims)) if i not in ok]
    return {"key": "summary_supported", "score": _ratio(len(claims) - len(bad), len(claims)),
            "comment": "; ".join(bad)[:500]}


def evaluators(judge: bool) -> list:
    return [quotes_verbatim, evidence_fits_change, new_is_new, dropped_is_dropped] + ([summary_supported] if judge else [])


# ------------------------------------------------------------------ main ----
def _print_row(ticker: str, scores: dict, report: dict | None) -> None:
    c = (report or {}).get("counts", {})
    cells = "  ".join(f"{k}={scores[k]:.2f}" for k in scores)
    print(f"{ticker:6} new={c.get('new', '?')} dropped={c.get('dropped', '?')} changed={c.get('changed', '?')}  {cells}")


def _verdict(table: dict[str, dict]) -> bool:
    means = {k: sum(s[k] for s in table.values()) / len(table) for k in next(iter(table.values()))}
    print("\naverage: " + "  ".join(f"{k}={v:.2f}" for k, v in means.items()))
    ok = all(means[k] >= PASS for k in CODE_KEYS if k in means)
    print(f"{'PASS' if ok else 'FAIL'} (bar: every code evaluator >= {PASS})")
    return ok


def _save(reports: list[dict], out_dir: str | None) -> None:
    if not out_dir:
        return
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    for r in reports:
        (d / f"{r['ticker']}.md").write_text(r["report_md"])
    print(f"reports written to {d}/")


def run_local(tickers: list[str], judge: bool, out_dir: str | None) -> bool:
    table, reports = {}, []
    for t in tickers:
        try:
            rep = target({"ticker": t})
            reports.append(rep)
            table[t] = {r["key"]: r["score"] for r in (ev(rep) for ev in evaluators(judge))}
        except Exception as e:  # a crashed company scores zero and we keep going
            print(f"{t}: failed: {str(e)[:200]}")
            rep, table[t] = None, {k: 0.0 for k in CODE_KEYS}
        _print_row(t, table[t], rep)
    _save(reports, out_dir)
    return _verdict(table)


def run_langsmith(tickers: list[str], judge: bool, out_dir: str | None) -> bool:
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
        old = list(client.list_examples(dataset_id=ds.id))
        if old:
            client.delete_examples([e.id for e in old])
    else:
        ds = client.create_dataset(DATASET, description="Tickers for the year-over-year Item 1A change report")
    client.create_examples(dataset_id=ds.id, examples=[{"inputs": {"ticker": t}} for t in tickers])
    # LangSmith evaluators receive (outputs) when the function asks only for outputs.
    res = client.evaluate(target, data=DATASET, evaluators=evaluators(judge),
                          experiment_prefix=f"m2-{os.getenv('SMALL_MODEL', 'small')}".replace("/", "-"),
                          max_concurrency=1)  # Groq free tier: one company at a time
    print(f"Experiment: {res.experiment_name}")
    table, reports = {}, []
    for row in res:
        t = row["example"].inputs["ticker"]
        out = row["run"].outputs or {}
        if out.get("ticker"):
            reports.append(out)
        table[t] = {r.key: (r.score or 0.0) for r in row["evaluation_results"]["results"]}
        _print_row(t, table[t], out)
    _save(reports, out_dir)
    return _verdict(table)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", default=TICKERS)
    ap.add_argument("--local", action="store_true", help="score locally without LangSmith")
    ap.add_argument("--judge", action="store_true", help="also run the LLM summary_supported check")
    ap.add_argument("--out", help="folder to save each company's markdown report")
    args = ap.parse_args(argv)
    tickers = [t.upper() for t in args.tickers]
    ok = (run_local if args.local else run_langsmith)(tickers, args.judge, args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
