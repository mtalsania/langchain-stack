"""Exact financial numbers from SEC XBRL, as plain data and as LangChain tools.

The rule for the whole project: **the model never types a number it read somewhere.**
Numbers come from here, carry the 10-K accession they were filed in, and are
rendered into text by code (see graphs/company_research.py).
"""
from __future__ import annotations

from functools import lru_cache

from langchain_core.tools import tool

from core.metrics import DERIVED, METRICS, METRIC_NAMES, Value, all_metrics, latest_years
from core.tickers import lookup
from core.xbrl import load_company_facts

LABELS = {m.name: m.label for m in METRICS} | {d.name: d.label for d in DERIVED}

# Metrics shown in briefs and offered to the model. The others (tag variants,
# debt components) stay available to code but would only confuse the model.
BRIEF_METRICS = [
    "revenue", "gross_profit", "gross_margin", "operating_income", "operating_margin",
    "net_income", "net_margin", "rd", "rd_pct_revenue", "sga", "operating_cash_flow",
    "capex", "free_cash_flow", "cash", "total_debt", "eps_diluted",
]


def archive_url(cik: int, accn: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{cik}/{accn.replace('-', '')}/"


def value_to_dict(v: Value) -> dict:
    return {
        "value": v.value,
        "unit": v.unit,
        "fiscal_year": v.fiscal_year,
        "source": v.source,
        "accessions": v.accessions,
        "period_end": v.period_end,
    }


def format_value(value: float, unit: str) -> str:
    if unit == "ratio":
        return f"{value * 100:.1f}%"
    if unit == "USD/shares":
        return f"${value:.2f}"
    if unit == "shares":
        return f"{value / 1e6:,.0f}M shares"
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f}B"
    return f"${value / 1e6:,.0f}M"


@lru_cache(maxsize=32)
def _table(ticker: str):
    co = lookup(ticker)
    cf = load_company_facts(co.cik)
    return co, cf, all_metrics(cf)


def company_facts(ticker: str, years: int = 3) -> dict:
    """JSON-friendly snapshot: company, fiscal years, and {metric: {year: value dict}}."""
    co, cf, table = _table(ticker.upper())
    fys = latest_years(table, years)
    facts = {
        name: {str(fy): value_to_dict(table[name][fy]) for fy in fys if fy in table.get(name, {})}
        for name in BRIEF_METRICS
    }
    return {
        "ticker": co.ticker,
        "name": cf.entity_name or co.name,
        "cik": co.cik,
        "years": fys,
        "facts": {k: v for k, v in facts.items() if v},
    }


def facts_prompt_table(snapshot: dict) -> str:
    """Compact table for the model: placeholder keys plus human-readable values.

    The model sees the values so it can reason about them, but must write the
    placeholder key, not the number, into its draft.
    """
    lines = []
    for name, by_year in snapshot["facts"].items():
        cells = [f"{{{{{name}:{fy}}}}} = {format_value(d['value'], d['unit'])}" for fy, d in by_year.items()]
        lines.append(f"{LABELS[name]}: " + "; ".join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------- tools ----
@tool
def get_metric(ticker: str, metric: str, fiscal_year: int) -> dict:
    """Exact value of one financial metric from the company's 10-K XBRL data.

    metric is one of: revenue, gross_profit, gross_margin, operating_income,
    operating_margin, net_income, net_margin, rd (R&D expense), rd_pct_revenue,
    sga, operating_cash_flow, capex, free_cash_flow, cash, total_debt, eps_diluted.
    fiscal_year uses the company's own fiscal-year label (e.g. NVIDIA FY2025 ended Jan 2025).
    Returns value (USD, ratio 0-1, or USD/share), unit, and the 10-K accession number to cite.
    """
    if metric not in METRIC_NAMES:
        return {"error": f"unknown metric {metric!r}", "valid": BRIEF_METRICS}
    co, _, table = _table(ticker.upper())
    col = table.get(metric, {})
    if fiscal_year not in col:
        return {"error": f"no {metric} for FY{fiscal_year}", "available_years": sorted(col)}
    d = value_to_dict(col[fiscal_year])
    d["display"] = format_value(d["value"], d["unit"])
    d["filing_urls"] = [archive_url(co.cik, a) for a in d["accessions"]]
    return d


@tool
def list_fiscal_years(ticker: str) -> list[int]:
    """Fiscal years with 10-K data for this ticker, oldest first."""
    _, _, table = _table(ticker.upper())
    return latest_years(table, 50)
