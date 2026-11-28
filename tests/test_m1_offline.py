"""Offline tests for M1 (tools, research graph, eval plumbing). No network, no LLM.

Financial inputs reuse the Apple fixture from test_core_offline (Apple's reported
FY2022-2024 10-K figures). The model is replaced by a stub that returns a fixed
structured answer, so these tests check the wiring and the number-safety rules,
not the model's writing. Live runs: `python -m graphs.company_research NVDA` and
`python -m evals.xbrl_qa --local`.
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

import graphs.company_research as cr
import tools.financials as fin
from core.filings import Filing
from core.tickers import Company
from evals import xbrl_qa
from tests.test_core_offline import M, apple_facts
from tools.filing_search import chunk_sections

APPLE = Company(ticker="AAPL", cik=320193, name="Apple Inc.")


@pytest.fixture(autouse=True)
def apple_data(monkeypatch):
    fin._table.cache_clear()
    monkeypatch.setattr(fin, "lookup", lambda t: APPLE)
    monkeypatch.setattr(fin, "load_company_facts", lambda cik: apple_facts())
    yield
    fin._table.cache_clear()


# -- tools.financials ------------------------------------------------------

def test_company_facts_snapshot_keeps_accessions():
    snap = fin.company_facts("AAPL", years=2)
    assert snap["years"] == [2023, 2024]
    rev = snap["facts"]["revenue"]["2024"]
    assert rev["value"] == 391_035 * M and rev["accessions"] == ["accn-fy2024"]
    assert snap["facts"]["gross_margin"]["2024"]["unit"] == "ratio"


def test_get_metric_tool_returns_value_and_filing_url():
    out = fin.get_metric.invoke({"ticker": "aapl", "metric": "rd", "fiscal_year": 2024})
    assert out["value"] == 31_370 * M and out["display"] == "$31.37B"
    assert out["filing_urls"][0].endswith("/320193/accnfy2024/")


def test_get_metric_tool_explains_bad_input():
    assert "valid" in fin.get_metric.invoke({"ticker": "AAPL", "metric": "ebitda", "fiscal_year": 2024})
    assert fin.get_metric.invoke({"ticker": "AAPL", "metric": "rd", "fiscal_year": 1999})["available_years"]


# -- rendering rules -------------------------------------------------------

def test_placeholders_become_cited_xbrl_values():
    snap, cited, issues = fin.company_facts("AAPL"), {}, []
    text = cr.fill_placeholders("Revenue was {{revenue:2024}} ({{growth:revenue:2024}}).", snap, cited, issues)
    assert text == "Revenue was $391.04B [F1] (+2.0% [F1])."
    assert cited == {"accn-fy2024": "F1"} and issues == []


def test_margin_growth_is_in_points_and_missing_values_are_flagged():
    snap, issues = fin.company_facts("AAPL"), []
    assert "pts" in cr.fill_placeholders("{{growth:gross_margin:2024}}", snap, {}, issues)
    assert "[missing" in cr.fill_placeholders("{{revenue:2019}}", snap, {}, issues)
    assert issues == ["no XBRL value for revenue FY2019"]


def test_typed_numbers_and_unknown_citations_are_flagged_but_years_are_not():
    issues = []
    cr.check_text("In fiscal 2024 sales rose 12% to $5 billion [S9], see {{revenue:2024}} [S1].", {"S1"}, issues)
    assert len([i for i in issues if "typed a number" in i]) == 2
    assert any("S9" in i for i in issues)
    issues = []
    cr.check_text("Since 2019 the company has grown {{growth:revenue:2024}} [S1].", {"S1"}, issues)
    assert issues == []


# -- graph wiring ----------------------------------------------------------

class StubLLM:
    """Stands in for a chat model: with_structured_output returns a fixed object."""

    def __init__(self, obj):
        self.obj = obj

    def with_structured_output(self, schema, **kw):
        return RunnableLambda(lambda _: self.obj)


def test_research_graph_end_to_end_with_stub_model(monkeypatch):
    docs = [
        Document(page_content="iPhone demand drove revenue.", metadata={
            "item": "7", "section": "MD&A", "chunk": 0, "accession": "a", "filing_date": "2024-11-01",
            "report_date": "2024-09-28", "url": "https://www.sec.gov/x"}),
        Document(page_content="Supply chain concentration in China.", metadata={
            "item": "1A", "section": "Risk Factors", "chunk": 3, "accession": "a", "filing_date": "2024-11-01",
            "report_date": "2024-09-28", "url": "https://www.sec.gov/x"}),
    ]
    monkeypatch.setattr(cr, "search", lambda t, q, k=3: docs)
    brief = cr.CompanyBrief(
        overview="Apple sells devices and services [S1].",
        financial_story=["Revenue reached {{revenue:2024}}, {{growth:revenue:2024}} year over year."],
        drivers=["iPhone demand [S1]."],
        risks=["Manufacturing concentrated in China [S2]."],
        priorities=["Services growth [S1]."],
    )

    def fake_llm(small=False, **kw):
        return StubLLM(cr.ResearchPlan(queries=["drivers", "risks"]) if small else brief)

    monkeypatch.setattr(cr, "get_llm", fake_llm)
    out = cr.build_graph().invoke({"ticker": "AAPL", "years": 2})
    md = out["brief_md"]
    assert out["issues"] == []
    assert "$391.04B [F1]" in md and "+2.0%" in md
    assert "[S2] 10-K Item 1A (Risk Factors)" in md
    assert "accession accn-fy2024" in md
    assert "{{" not in md


def test_chunks_remember_their_item():
    f = Filing(cik=1, form="10-K", accession="0001-24-1", filing_date="2024-11-01", report_date="2024-09-28",
               primary_document="x.htm")
    docs = chunk_sections(f, {"1A": "Risk. " * 400, "7": "Sales grew.", "1": ""})
    assert {d.metadata["item"] for d in docs} == {"1A", "7"}
    assert all(len(d.page_content) <= 1000 for d in docs)
    # fake embeddings are enough to prove the index round-trips
    from langchain_core.vectorstores import InMemoryVectorStore
    store = InMemoryVectorStore(embedding=DeterministicFakeEmbedding(size=16))
    store.add_documents(docs)
    assert store.similarity_search("risk", k=1, filter=lambda d: d.metadata["item"] == "7")[0].metadata["item"] == "7"


# -- eval plumbing ---------------------------------------------------------

def test_eval_examples_come_from_xbrl():
    exs = xbrl_qa.build_examples(tickers=["AAPL"], metrics=["revenue", "rd", "net_income"])
    assert [e["outputs"]["metric"] for e in exs] == ["revenue", "rd"]  # fixture has no net income
    assert exs[0]["outputs"]["value"] == 391_035 * M
    assert "fiscal year 2024" in exs[0]["inputs"]["question"]


def test_numeric_match_tolerance_and_citation_check():
    ref = {"value": 100.0, "accessions": ["a1"]}
    assert xbrl_qa.numeric_match({"value": 100.4}, ref)["score"]
    assert not xbrl_qa.numeric_match({"value": 101.0}, ref)["score"]
    assert not xbrl_qa.numeric_match({"value": None}, ref)["score"]
    assert xbrl_qa.cites_filing({"accession": "a1"}, ref)["score"]


def test_bullets_that_are_only_numbers_are_flagged(monkeypatch):
    snap = fin.company_facts("AAPL")
    draft = cr.CompanyBrief(overview="x [S1].", financial_story=["{{revenue:2024}}", "Revenue reached {{revenue:2024}} this year."],
                            drivers=[], risks=[], priorities=[]).model_dump()
    out = cr.render({"snapshot": snap, "draft": draft, "sources": [{"id": "S1", "item": "7", "section": "MD&A", "filing_date": "2024-11-01", "url": "u"}], "issues": []})
    assert out["issues"] == ["financial story bullet has no explanation: '{{revenue:2024}}'"]
