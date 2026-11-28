"""M1 eval: can the research agent get exact numbers right?

The SEC's XBRL data is the answer key, so the dataset builds itself: no hand labels.
For each company and metric we ask "What was X's R&D expense in fiscal 2025?", let
an agent answer with its tools, and compare to the XBRL value.

Evaluators (plain code, no LLM judge needed when ground truth is exact):
- numeric_match: answer within 0.5% of the XBRL value
- cites_filing:  answer names the 10-K accession the value was filed in

    python -m evals.xbrl_qa --local         # print a score table, no LangSmith needed
    python -m evals.xbrl_qa                 # create/refresh the LangSmith dataset and run an experiment

Done-when for M1: numeric_match >= 0.90 on the ~20 questions (a company missing a
metric in XBRL simply has one question fewer).
Free-tier cost: ~2 small-model calls per question, ~40 calls and ~60K tokens on Groq's
small model, which has its own daily quota.
"""
from __future__ import annotations

import argparse
import os
import re

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from common.llm import get_llm, resilient_middleware
from tools.financials import LABELS, company_facts, get_metric, list_fiscal_years

TICKERS = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"]
QA_METRICS = ["revenue", "net_income", "rd", "operating_income"]
DATASET = "xbrl-numeric-qa"
TOLERANCE = 0.005


def build_examples(tickers=TICKERS, metrics=QA_METRICS) -> list[dict]:
    """One question per company x metric, for the latest fiscal year, answered from XBRL."""
    examples = []
    for t in tickers:
        snap = company_facts(t, years=1)
        fy = str(snap["years"][-1])
        for m in metrics:
            d = snap["facts"].get(m, {}).get(fy)
            if not d:
                continue
            examples.append(
                {
                    "inputs": {"question": f"What was {snap['name']}'s ({t}) {LABELS[m].lower()} for fiscal year {fy}?"},
                    "outputs": {"value": d["value"], "unit": d["unit"], "accessions": d["accessions"],
                                "ticker": t, "metric": m, "fiscal_year": int(fy)},
                }
            )
    return examples


class NumericAnswer(BaseModel):
    value: float | None = Field(description="The number in base units: dollars (not millions), or a 0-1 ratio")
    unit: str = Field(description="USD, ratio, or USD/shares")
    accession: str = Field(description="10-K accession number the value came from, e.g. 0001045810-25-000023")


SYSTEM = """You answer questions about public companies' reported financials.
Always call get_metric to get the exact value; never estimate. Use list_fiscal_years if unsure
which years exist. Then answer in ONE plain sentence with the full number in base units
(e.g. $120,067,000,000, not $120B) and the accession, e.g.
"NVIDIA's revenue for fiscal year 2026 was $215,938,000,000 (accession 0001045810-26-000021)."
"""


def make_agent():
    # No response_format on purpose. Structured output inside a tool-calling agent forces
    # tool_choice=required on Groq, and gpt-oss often answers in plain text instead, which
    # Groq rejects with 400 'tool_use_failed'. The model answers in text; parse_answer reads it.
    return create_agent(
        model=get_llm(small=True),
        tools=[get_metric, list_fiscal_years],
        system_prompt=SYSTEM,
        # gpt-oss-20b on Groq sometimes returns a malformed tool call ("output_parse_failed").
        # Retry it twice, then fall back to the large model for that step.
        middleware=resilient_middleware(),
    )


ACCESSION = re.compile(r"\b\d{10}-\d{2}-\d{6}\b")
NUMBER = re.compile(r"(-?\$?\s?-?\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|bn|[TBMK]\b)?", re.I)
SCALE = {"trillion": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9, "million": 1e6, "m": 1e6,
         "thousand": 1e3, "k": 1e3}


def parse_answer(text: str) -> NumericAnswer:
    """Read the value and accession out of the agent's sentence. Plain code, no extra model call.

    The accession is removed first so its digits are not mistaken for the value, and
    4-digit years are skipped. The largest remaining number is the answer.
    """
    accn = ACCESSION.search(text)
    rest = ACCESSION.sub(" ", text)
    best = None
    for m in NUMBER.finditer(rest):
        raw, scale = m.group(1).replace("$", "").replace(",", "").replace(" ", ""), (m.group(2) or "").lower()
        try:
            v = float(raw)
        except ValueError:
            continue
        if not scale and re.fullmatch(r"(19|20)\d\d", raw):
            continue  # a fiscal year, not the answer
        v *= SCALE.get(scale, 1)
        if best is None or abs(v) > abs(best):
            best = v
    return NumericAnswer(value=best, unit="USD", accession=accn.group(0) if accn else "")


def target(inputs: dict, agent=None) -> dict:
    agent = agent or make_agent()
    result = agent.invoke({"messages": [("user", inputs["question"])]})
    final = result["messages"][-1].content
    if isinstance(final, list):  # some providers return content blocks
        final = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in final)
    return {**parse_answer(final).model_dump(), "answer_text": final}


# ------------------------------------------------------------ evaluators ----
def numeric_match(outputs: dict, reference_outputs: dict) -> dict:
    ref, got = reference_outputs["value"], outputs.get("value")
    ok = got is not None and abs(got - ref) <= TOLERANCE * abs(ref)
    return {"key": "numeric_match", "score": ok, "comment": f"got {got}, expected {ref}"}


def cites_filing(outputs: dict, reference_outputs: dict) -> dict:
    ok = outputs.get("accession") in reference_outputs["accessions"]
    return {"key": "cites_filing", "score": ok}


# ------------------------------------------------------------------ main ----
def run_local(examples: list[dict]) -> float:
    agent = make_agent()
    hits = 0
    for ex in examples:
        try:
            out = target(ex["inputs"], agent)
        except Exception as e:  # a failed call counts as wrong, and we keep going
            out = {"value": None, "accession": None, "error": str(e)[:80]}
        n, c = numeric_match(out, ex["outputs"]), cites_filing(out, ex["outputs"])
        hits += n["score"]
        mark = "PASS" if n["score"] else "FAIL"
        print(f"{mark}  cite={'y' if c['score'] else 'n'}  {ex['inputs']['question']}  ({n['comment']})")
    acc = hits / len(examples)
    print(f"\nnumeric_match: {hits}/{len(examples)} = {acc:.0%}  (M1 target: 90%)")
    return acc


def run_langsmith(examples: list[dict]) -> float:
    from langsmith import Client

    client = Client()
    if client.has_dataset(dataset_name=DATASET):
        ds = client.read_dataset(dataset_name=DATASET)
        old = list(client.list_examples(dataset_id=ds.id))
        if old:  # regenerate from current XBRL so new fiscal years flow in
            client.delete_examples([e.id for e in old])
    else:
        ds = client.create_dataset(DATASET, description="Numeric questions answered from SEC XBRL (auto-generated)")
    client.create_examples(dataset_id=ds.id, examples=examples)
    agent = make_agent()
    res = client.evaluate(
        lambda inputs: target(inputs, agent),
        data=DATASET,
        evaluators=[numeric_match, cites_filing],
        experiment_prefix=f"m1-{os.getenv('SMALL_MODEL', 'small')}".replace("/", "-"),
        max_concurrency=1,  # Groq free tier: one question at a time
    )
    print(f"Experiment: {res.experiment_name}")
    return summarize(res)


def summarize(results) -> float:
    """Print the same score table as --local, from a LangSmith ExperimentResults."""
    hits = total = cites = 0
    for row in results:
        scores = {r.key: r.score for r in row["evaluation_results"]["results"]}
        out = row["run"].outputs or {}
        ok = bool(scores.get("numeric_match"))
        hits, total, cites = hits + ok, total + 1, cites + bool(scores.get("cites_filing"))
        if not ok:
            print(f"FAIL  {row['example'].inputs['question']}  -> {out.get('answer_text') or out.get('error') or out}")
    acc = hits / total if total else 0.0
    print(f"\nnumeric_match: {hits}/{total} = {acc:.0%}   cites_filing: {cites}/{total}   (M1 target: 90%)")
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--local", action="store_true", help="score locally without LangSmith")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (saves free-tier quota)")
    args = ap.parse_args(argv)
    examples = build_examples()
    if args.limit:
        examples = examples[: args.limit]
    print(f"{len(examples)} questions generated from XBRL")
    if args.local:
        return 0 if run_local(examples) >= 0.9 else 1
    return 0 if run_langsmith(examples) >= 0.9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
