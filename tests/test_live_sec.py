"""End-to-end checks against the live SEC APIs.

    pytest --live            # needs SEC_USER_AGENT in .env; about 20 requests, then cached

The expected figures are the totals printed in each company's 10-K income
statement, so a pass means our XBRL picking matches the filed numbers exactly.
"""
from __future__ import annotations

import pytest

from core.filings import latest_10k, tenk_sections
from core.metrics import all_metrics, latest_years
from core.sec_client import default_client
from core.tickers import lookup
from core.xbrl import load_company_facts

pytestmark = pytest.mark.live
M = 1_000_000

# (ticker, fiscal year, revenue, gross profit, R&D), in USD, from each 10-K.
EXPECTED = [
    ("AAPL", 2024, 391_035 * M, 180_683 * M, 31_370 * M),  # 10-K for FY ended 2024-09-28
    ("AAPL", 2023, 383_285 * M, 169_148 * M, 29_915 * M),
    ("NVDA", 2025, 130_497 * M, 97_858 * M, 12_914 * M),   # 10-K for FY ended 2025-01-26
    ("MSFT", 2024, 245_122 * M, 171_008 * M, 29_510 * M),  # 10-K for FY ended 2024-06-30
]


@pytest.fixture(scope="module")
def tables():
    cache = {}

    def get(ticker):
        if ticker not in cache:
            cache[ticker] = all_metrics(load_company_facts(lookup(ticker).cik))
        return cache[ticker]

    return get


def test_ticker_lookup():
    assert lookup("AAPL").cik == 320193
    assert lookup("NVDA").cik == 1045810
    assert lookup("msft").cik == 789019
    assert lookup("BRK.B").cik == 1067983


@pytest.mark.parametrize("ticker,fy,revenue,gross_profit,rd", EXPECTED)
def test_exact_revenue_gross_margin_rd(tables, ticker, fy, revenue, gross_profit, rd):
    t = tables(ticker)
    assert t["revenue"][fy].value == revenue
    assert t["gross_profit"][fy].value == gross_profit
    assert t["gross_margin"][fy].value == pytest.approx(gross_profit / revenue)
    assert t["rd"][fy].value == rd
    assert t["revenue"][fy].accn  # every number is citable


@pytest.mark.parametrize("ticker", ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "META", "KO", "XOM", "WMT", "INTC"])
def test_three_recent_years_of_core_metrics(tables, ticker):
    t = tables(ticker)
    years = latest_years(t, 3)
    assert len(years) == 3 and years[-1] >= 2024
    for fy in years:
        for name in ("revenue", "net_income", "operating_cash_flow", "eps_diluted"):
            assert fy in t[name], f"{ticker} FY{fy} missing {name}"
        if fy in t["gross_margin"]:
            assert 0 < t["gross_margin"][fy].value < 1


def test_latest_10k_sections_for_apple():
    cik = lookup("AAPL").cik
    filing, secs = tenk_sections(cik)
    assert filing.form == "10-K" and filing == latest_10k(cik)
    assert secs["1A"].lower().startswith("item 1a") and len(secs["1A"].split()) > 5_000
    assert secs["1"].lower().startswith("item 1") and len(secs["1"].split()) > 1_000
    assert secs["7"].lower().startswith("item 7") and len(secs["7"].split()) > 2_000
    assert "risk factors" not in secs["1"].split("\n", 1)[0].lower()


@pytest.mark.parametrize("ticker", ["NVDA", "MSFT", "KO"])
def test_risk_factors_found_for_other_layouts(ticker):
    _, secs = tenk_sections(lookup(ticker).cik, items=("1A", "7"))
    assert len(secs["1A"].split()) > 3_000, f"{ticker} Item 1A too short"
    assert len(secs["7"].split()) > 2_000, f"{ticker} Item 7 too short"


def test_rate_limit_respected():
    assert default_client().limiter.max_calls <= 10


# -- M2: risk factors ---------------------------------------------------------
# Pinned filings, so the expected values don't move when a company files its next 10-K.
# Counts and headings were read from these filings (CI run on PR #32, 2026-11-28).
from core.filings import Filing  # noqa: E402
from core.risk_factors import risk_section  # noqa: E402

TENKS = {
    "NVDA": Filing(1045810, "10-K", "0001045810-26-000021", "2026-02-25", "2026-01-25", "nvda-20260125.htm"),
    "NVDA-prior": Filing(1045810, "10-K", "0001045810-25-000023", "2025-02-26", "2025-01-26", "nvda-20250126.htm"),
    "AAPL": Filing(320193, "10-K", "0000320193-26-000026", "2026-11-03", "2026-09-26", "aapl-20260926.htm"),
    "AAPL-prior": Filing(320193, "10-K", "0000320193-25-000079", "2025-10-31", "2025-09-27", "aapl-20250927.htm"),
    "MSFT": Filing(789019, "10-K", "0001193125-26-323660", "2026-07-29", "2026-06-30", "msft-20260630.htm"),
    "AMZN": Filing(1018724, "10-K", "0001018724-26-000004", "2026-02-06", "2025-12-31", "amzn-20251231.htm"),
    "GOOGL": Filing(1652044, "10-K", "0001652044-26-000018", "2026-02-05", "2025-12-31", "goog-20251231.htm"),
    "GOOGL-prior": Filing(1652044, "10-K", "0001652044-25-000014", "2025-02-05", "2024-12-31", "goog-20241231.htm"),
}


@pytest.mark.parametrize("key,count,first_heading,first_category", [
    ("NVDA", 24, "Failure to meet the evolving needs of our industry and markets", "Risks Related to Our Industry and Markets"),
    ("AAPL", 28, "The Company’s operations and performance depend significantly on global and regional economic",
     "Macroeconomic and Industry Risks"),
    ("MSFT", 24, "We face intense competition across all markets for our products and services",
     "STRATEGIC AND COMPETITIVE RISKS"),
    ("AMZN", 24, "We Face Intense Competition", "Business and Industry Risks"),
    ("GOOGL", 28, "We generate a significant portion of our revenues from advertising", "Risks Specific to our Company"),
])
def test_item_1a_splits_into_risk_factors(key, count, first_heading, first_category):
    sec = risk_section(TENKS[key])
    assert sec.method == "emphasis"
    assert len(sec.risks) == count
    assert sec.risks[0].heading.startswith(first_heading) and sec.risks[0].category == first_category
    assert all(len(r.body.split()) >= 50 for r in sec.risks)  # no heading-only fragments
    assert not any(r.heading.lower().startswith(("risks related", "table of contents")) for r in sec.risks)


@pytest.mark.parametrize("key,new_heading", [
    ("NVDA", "Commercial arrangements expose us to counterparty risks."),
    ("AAPL", "Issues related to artificial intelligence may result in reputational, competitive and financial harm"),
    ("GOOGL", "Disruptions in our ability to access future financing or manage our indebtedness"),
])
def test_new_risk_found_year_over_year(key, new_heading):
    pytest.importorskip("fastembed")
    from common.embeddings import get_embeddings
    from tools.risk_matching import match_risks

    cur, pri = risk_section(TENKS[key]).to_dict()["risks"], risk_section(TENKS[f"{key}-prior"]).to_dict()["risks"]
    res = match_risks(cur, pri, get_embeddings())
    assert any(r["heading"].startswith(new_heading) for r in res.new)
    assert len(res.new) <= 2 and len(res.dropped) <= 2  # most risks carry over
