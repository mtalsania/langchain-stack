"""Offline tests for core/: logic only, no network.

The inputs below are tiny dicts in the exact shape the SEC APIs return. Dollar
figures are Apple's and NVIDIA's reported 10-K numbers; accession numbers are
placeholders ("accn-...") because these tests only check which fact is picked.
Real end-to-end checks against live SEC data live in test_live_sec.py.
"""
from __future__ import annotations

from datetime import date

import pytest

from core.filings import extract_section, html_to_text, parse_submissions
from core.metrics import all_metrics, latest_years
from core.sec_client import RateLimiter, SECClient, SECConfigError, user_agent_from_env
from core.tickers import normalize_ticker, parse_ticker_file
from core.xbrl import CompanyFacts, merge_facts

M = 1_000_000


def fact(val, end, start=None, fy=2024, fp="FY", form="10-K", filed="2024-11-01", accn="accn-fy2024"):
    row = {"val": val, "end": end, "fy": fy, "fp": fp, "form": form, "filed": filed, "accn": accn}
    if start:
        row["start"] = start
    return row


def companyfacts(us_gaap: dict, cik=320193, name="Apple Inc."):
    return {"cik": cik, "entityName": name, "facts": {"us-gaap": us_gaap}}


# Apple 10-K for FY2024 (filed 2024-11) and FY2023 (filed 2023-11): each shows three years.
FY24 = dict(fy=2024, filed="2024-11-01", accn="accn-fy2024")
FY23 = dict(fy=2023, filed="2023-11-03", accn="accn-fy2023")
P24 = ("2024-09-28", "2023-10-01")
P23 = ("2023-09-30", "2022-10-02")
P22 = ("2022-09-24", "2021-09-26")
P21 = ("2021-09-25", "2020-09-27")


def apple_facts(include_gross_profit=True):
    rev = [
        fact(391_035 * M, *P24, **FY24),
        fact(383_285 * M, *P23, **FY24),
        fact(394_328 * M, *P22, **FY24),
        fact(383_285 * M, *P23, **FY23),
        fact(394_328 * M, *P22, **FY23),
        fact(365_817 * M, *P21, **FY23),
        # Q4 FY2024 (3 months) also appears in the 10-K with fp=FY; it must be ignored.
        fact(94_930 * M, "2024-09-28", "2024-06-30", **FY24),
        # 10-Q facts must be ignored.
        fact(85_777 * M, "2024-06-29", "2024-03-31", fy=2024, fp="Q3", form="10-Q", filed="2024-08-02", accn="accn-q3"),
    ]
    cost = [fact(210_352 * M, *P24, **FY24), fact(214_137 * M, *P23, **FY24)]
    gp = [fact(180_683 * M, *P24, **FY24), fact(169_148 * M, *P23, **FY24)]
    rd = [fact(31_370 * M, *P24, **FY24), fact(29_915 * M, *P23, **FY24), fact(26_251 * M, *P22, **FY24)]
    cash = [
        fact(29_943 * M, "2024-09-28", **FY24),
        fact(29_965 * M, "2023-09-30", **FY24),
        fact(25_565 * M, "2024-06-29", fy=2024, fp="Q3", form="10-Q", filed="2024-08-02", accn="accn-q3"),
    ]
    g = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": rev}},
        "CostOfGoodsAndServicesSold": {"units": {"USD": cost}},
        "ResearchAndDevelopmentExpense": {"units": {"USD": rd}},
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": cash}},
    }
    if include_gross_profit:
        g["GrossProfit"] = {"units": {"USD": gp}}
    return CompanyFacts(companyfacts(g))


# -- xbrl -----------------------------------------------------------------

def test_values_are_labelled_by_their_own_fiscal_year_not_the_filings():
    rev = apple_facts().annual("RevenueFromContractWithCustomerExcludingAssessedTax")
    assert sorted(rev) == [2021, 2022, 2023, 2024]
    assert rev[2024].value == 391_035 * M
    assert rev[2022].value == 394_328 * M
    assert rev[2021].value == 365_817 * M


def test_quarterly_and_10q_facts_are_dropped():
    rev = apple_facts().annual("RevenueFromContractWithCustomerExcludingAssessedTax")
    assert all(v.days > 330 for v in rev.values())
    assert 94_930 * M not in {v.value for v in rev.values()}


def test_latest_filing_wins_and_keeps_its_accession():
    rev = apple_facts().annual("RevenueFromContractWithCustomerExcludingAssessedTax")
    assert rev[2023].accn == "accn-fy2024"  # reported in both 10-Ks; the later one wins
    assert rev[2021].accn == "accn-fy2023"


def test_balance_sheet_values_only_at_fiscal_year_end():
    cash = apple_facts().annual("CashAndCashEquivalentsAtCarryingValue")
    assert {fy: f.value for fy, f in cash.items()} == {2024: 29_943 * M, 2023: 29_965 * M}


def test_fiscal_year_label_uses_company_calendar():
    # NVIDIA's fiscal 2025 ended 2025-01-26 and its 10-K says fy=2025.
    nv = CompanyFacts(companyfacts({
        "Revenues": {"units": {"USD": [
            fact(130_497 * M, "2025-01-26", "2024-01-29", fy=2025, filed="2025-02-26", accn="accn-nv25"),
            fact(60_922 * M, "2024-01-28", "2023-01-30", fy=2025, filed="2025-02-26", accn="accn-nv25"),
        ]}}
    }, cik=1045810, name="NVIDIA CORP"))
    assert nv.fiscal_year_ends[date(2025, 1, 26)] == 2025
    assert nv.annual("Revenues")[2025].value == 130_497 * M
    assert nv.annual("Revenues")[2024].value == 60_922 * M


# -- metrics --------------------------------------------------------------

def test_core_metrics_for_apple_fy2024():
    t = all_metrics(apple_facts())
    assert t["revenue"][2024].value == 391_035 * M
    assert t["gross_profit"][2024].value == 180_683 * M
    assert t["gross_margin"][2024].value == pytest.approx(180_683 / 391_035)
    assert t["rd"][2024].value == 31_370 * M
    assert t["rd_pct_revenue"][2024].value == pytest.approx(31_370 / 391_035)
    assert t["gross_margin"][2024].accessions == ["accn-fy2024"]
    assert latest_years(t, 3) == [2022, 2023, 2024]


def test_gross_profit_falls_back_to_revenue_minus_cost():
    t = all_metrics(apple_facts(include_gross_profit=False))
    gp = t["gross_profit"][2024]
    assert gp.source == "derived"
    assert gp.value == (391_035 - 210_352) * M


def test_revenue_tag_variants_merge_across_years():
    # Apple tagged FY2017 revenue as SalesRevenueNet, later years under the ASC 606 tag.
    cf = CompanyFacts(companyfacts({
        "SalesRevenueNet": {"units": {"USD": [
            fact(229_234 * M, "2017-09-30", "2016-09-25", fy=2017, filed="2017-11-03", accn="accn-fy2017")]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [
            fact(391_035 * M, *P24, **FY24)]}},
    }))
    rev = all_metrics(cf)["revenue"]
    assert rev[2017].source == "SalesRevenueNet"
    assert rev[2024].source == "RevenueFromContractWithCustomerExcludingAssessedTax"


def test_revenue_takes_total_when_company_tags_total_and_subset():
    cf = CompanyFacts(companyfacts({
        "Revenues": {"units": {"USD": [fact(100 * M, *P24, **FY24)]}},
        "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": [fact(90 * M, *P24, **FY24)]}},
    }))
    assert all_metrics(cf)["revenue"][2024].value == 100 * M


# -- tickers --------------------------------------------------------------

def test_ticker_file_parsing_and_share_class_normalization():
    idx = parse_ticker_file({
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC"},
    })
    assert idx["AAPL"].cik10 == "0000320193"
    assert idx[normalize_ticker("brk.b")].cik == 1067983


# -- sec client -----------------------------------------------------------

def test_user_agent_must_contain_an_email(monkeypatch):
    monkeypatch.setattr("core.sec_client.load_dotenv", lambda: None)
    monkeypatch.setenv("SEC_USER_AGENT", "no email here")
    with pytest.raises(SECConfigError):
        user_agent_from_env()
    monkeypatch.setenv("SEC_USER_AGENT", "Your Name your.email@example.com")  # the .env.example placeholder
    with pytest.raises(SECConfigError):
        user_agent_from_env()
    monkeypatch.setenv("SEC_USER_AGENT", "Test Runner tests@sec-client.test")
    assert user_agent_from_env() == "Test Runner tests@sec-client.test"


def test_rate_limiter_never_exceeds_10_per_second():
    now = [0.0]
    calls = []
    lim = RateLimiter(10, 1.0, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(35):
        lim.wait()
        calls.append(now[0])
    for i in range(len(calls) - 10):
        assert calls[i + 10] - calls[i] >= 1.0 - 1e-9
    assert calls[-1] >= 3.0


class FakeResp:
    def __init__(self, status, content=b"{}", headers=None):
        self.status_code, self.content, self.headers = status, content, headers or {}
        self.text = content.decode()


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.responses.pop(0)


def test_client_sends_user_agent_caches_and_retries(tmp_path, monkeypatch):
    monkeypatch.setattr("core.sec_client.time.sleep", lambda s: None)
    s = FakeSession([FakeResp(429, b"slow down"), FakeResp(200, b'{"ok": 1}')])
    c = SECClient(user_agent="Test Runner tests@sec-client.test", cache_dir=tmp_path, session=s)
    assert s.headers["User-Agent"] == "Test Runner tests@sec-client.test"
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": 1}
    assert c.requests_made == 2
    assert c.get_json("https://data.sec.gov/x.json") == {"ok": 1}  # from disk
    assert c.requests_made == 2


# -- filings --------------------------------------------------------------

TENK_HTML = """
<html><body>
<div style="display:none"><ix:header>hidden xbrl metadata</ix:header></div>
<table>
 <tr><td>Item 1.</td><td>Business</td><td>1</td></tr>
 <tr><td>Item 1A.</td><td>Risk Factors</td><td>5</td></tr>
 <tr><td>Item 1B.</td><td>Unresolved Staff Comments</td><td>17</td></tr>
 <tr><td>Item 7.</td><td>Management's Discussion and Analysis</td><td>20</td></tr>
 <tr><td>Item 7A.</td><td>Quantitative and Qualitative Disclosures About Market Risk</td><td>27</td></tr>
 <tr><td>Item 8.</td><td>Financial Statements</td><td>28</td></tr>
</table>
<div><span>Item 1.</span><span>&#160;&#160;&#160;Business</span></div>
<p>The Company designs, manufactures and markets smartphones.</p>
<p>See Item 1A of this Form 10-K for risks.</p>
<div><span style="font-weight:700">Item 1A.&#160;&#160;&#160;&#160;Risk Factors</span></div>
<p>The Company's business can be affected by many risks.</p>
<p>Global economic conditions could materially adversely affect the Company.</p>
<div>Item 1B. Unresolved Staff Comments</div><p>None.</p>
<div>ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS</div>
<p>Net sales increased during the year.</p>
<div>Item 7A. Quantitative and Qualitative Disclosures About Market Risk</div><p>Interest rate risk.</p>
<div>Item 8. Financial Statements</div><p>Statements follow.</p>
</body></html>
"""


def test_html_to_text_keeps_headings_on_one_line_and_drops_ix_header():
    text = html_to_text(TENK_HTML)
    assert "hidden xbrl metadata" not in text
    assert "Item 1A. Risk Factors" in text.split("\n")


def test_section_extractor_skips_table_of_contents():
    text = html_to_text(TENK_HTML)
    risk = extract_section(text, "1A")
    assert risk.startswith("Item 1A. Risk Factors")
    assert "Global economic conditions" in risk
    assert "Unresolved" not in risk
    biz = extract_section(text, "1")
    assert "smartphones" in biz and "See Item 1A" in biz and "Risk Factors" not in biz
    mdna = extract_section(text, "7")
    assert "Net sales increased" in mdna and "Interest rate risk" not in mdna
    assert extract_section(text, "9A") == ""


def test_parse_submissions_newest_first():
    data = {"cik": "320193", "filings": {"recent": {
        "accessionNumber": ["accn-b", "accn-a"], "form": ["10-Q", "10-K"],
        "filingDate": ["2025-01-31", "2024-11-01"], "reportDate": ["2024-12-28", "2024-09-28"],
        "primaryDocument": ["q.htm", "k.htm"]}}}
    fs = parse_submissions(data)
    assert [f.form for f in fs] == ["10-Q", "10-K"]
    assert fs[1].url == "https://www.sec.gov/Archives/edgar/data/320193/accna/k.htm"


def test_successor_cik_inherits_predecessor_history():
    # Shape of the 2026 Exxon reorganization: new CIK has only 10-Qs, history sits under the old one.
    new = companyfacts({"Revenues": {"units": {"USD": [
        fact(113_164 * M, "2026-09-30", "2026-07-01", fy=2026, fp="Q3", form="10-Q", filed="2026-11-02", accn="accn-new-q3")]}}},
        cik=2115436, name="ExxonMobil Holdings Corporation")
    old = companyfacts({"Revenues": {"units": {"USD": [
        fact(100 * M, "2025-12-31", "2025-01-01", fy=2025, filed="2026-02-18", accn="accn-old-fy25"),
        fact(113_164 * M, "2026-09-30", "2026-07-01", fy=2026, fp="Q3", form="10-Q", filed="2026-11-02", accn="accn-new-q3")]}}},
        cik=34088, name="EXXON MOBIL CORP")
    assert CompanyFacts(new).annual("Revenues") == {}
    merged = CompanyFacts(merge_facts(new, old))
    assert merged.cik == 2115436
    assert merged.annual("Revenues")[2025].accn == "accn-old-fy25"
    assert len(merged.series("Revenues", "USD")) == 2  # the shared 10-Q fact is not duplicated
