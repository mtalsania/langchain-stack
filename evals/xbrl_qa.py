"""M1 eval: can the research agent get exact numbers right?

The SEC's XBRL data is the answer key, so the dataset builds itself: no hand labels.
For each company and metric we ask "What was X's R&D expense in fiscal 2025?", let
an agent answer with its tools, and compare to the XBRL value.

Evaluators (plain code, no LLM judge needed when ground truth is exact):
- numeric_match: answer within 0.5% of the XBRL value
- cites_filing:  answer names the 10-K accession the value was filed in

    python -m evals.xbrl_qa --local         # print a score table, no LangSmith needed
    python -m evals.xbrl_qa                 # create/refresh the LangSmith dataset and run an experiment

Done-when for M1: numeric_match >= 0.90 on the 20 questions.
Free-tier cost: ~2 small-model calls per question, ~40 calls and ~60K tokens on Groq's
small model, which has its own daily quota.
"""
from __future__ import annotations

import argparse
import os

from langchain.agents import create_agent
from pydantic import BaseModel, Field

from common.llm import get_llm
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
    value: float = Field(description="The number in base units: dollars (not millions), or a 0-1 ratio")
    unit: str = Field(description="USD, ratio, or USD/shares")
    accession: str = Field(description="10-K accession number the value came from, e.g. 0001045810-25-000023")


SYSTEM = """You answer questions about public companies' reported financials.
Always call get_metric to get the exact value; never estimate. Use list_fiscal_years if unsure
which years exist. Report the value exactly as returned, in base units, with its accession."""


def make_agent():
    return create_agent(
        model=get_llm(small=True),
        tools=[get_metric, list_fiscal_years],
        system_prompt=SYSTEM,
        response_format=NumericAnswer,
    )


def target(inputs: dict, agent=None) -> dict:
    agent = agent or make_agent()
    result = agent.invoke({"messages": [("user", inputs["question"])]})
    ans: NumericAnswer = result["structured_response"]
    return ans.model_dump()


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


def run_langsmith(examples: list[dict]) -> None:
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
    run_langsmith(examples)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
