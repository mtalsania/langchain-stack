"""Offline tests for M2 (risk factor split, matching, change graph). No network, no LLM.

The HTML below copies the *layout* of real 10-K risk sections (styled spans inside
divs, as Workiva and other filing tools emit them), with short placeholder text.
These tests check the parsing and the faithfulness rules; the live tests in
test_live_sec.py check the splitter against real filings.
"""
from __future__ import annotations

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda

import graphs.risk_changes as rc
from core.filings import Filing, html_to_text, section_bounds, strip_marks
from core.risk_factors import RiskFactor, RiskSection, split_risks
from tools.risk_matching import contains_quote, diff_hunks, match_risks

B = 'style="font-weight:700"'
I = 'style="font-style:italic"'
BODY = "This paragraph explains the risk in detail and how it could affect results of operations."


def risk_html(headings_and_bodies, heading_style=I, category_style=B, extra=""):
    parts = ['<html><body><div><span>Table of Contents</span></div>',
             '<div><span>Item 1A. Risk Factors</span> 12</div>',
             '<div><span>Item 7. Management\'s Discussion</span> 30</div>',
             f'<div><span {B}>Item 1A. Risk Factors</span></div>',
             '<div><span>The following risks could harm our business.</span></div>', extra]
    for kind, text in headings_and_bodies:
        style = heading_style if kind == "h" else category_style if kind == "c" else ""
        parts.append(f"<div><span {style}>{text}</span></div>" if style else f"<div><span>{text}</span></div>")
    parts.append('<div><span>Item 1B. Unresolved Staff Comments</span></div><div>None.</div></body></html>')
    return "".join(parts)


def section_of(html):
    marked = html_to_text(html, mark_emphasis=True).split("\n")
    a, b = section_bounds([strip_marks(x) for x in marked], "1A")
    return "\n".join(marked[a:b])


LAYOUT = [
    ("c", "Risks Related to Our Industry"),
    ("h", "Competition could adversely impact our market share and financial results."),
    ("p", BODY), ("p", "A second paragraph about competitors."),
    ("h", "Our business depends on our ability to"), ("h", "anticipate technology trends."),  # heading on 2 lines
    ("p", BODY),
    ("c", "General Risk Factors"),
    ("h", "Adverse economic conditions may harm our business."),
    ("p", BODY),
]


# -- splitting -------------------------------------------------------------

def test_marked_text_strips_back_to_plain_text():
    html = risk_html(LAYOUT)
    assert strip_marks(html_to_text(html, mark_emphasis=True)) == html_to_text(html)


def test_split_finds_headings_categories_and_bodies():
    risks, _ = split_risks(section_of(risk_html(LAYOUT)))
    assert [r.heading for r in risks] == [
        "Competition could adversely impact our market share and financial results.",
        "Our business depends on our ability to anticipate technology trends.",
        "Adverse economic conditions may harm our business.",
    ]
    assert [r.category for r in risks] == ["Risks Related to Our Industry"] * 2 + ["General Risk Factors"]
    assert "second paragraph" in risks[0].body and "Table of Contents" not in risks[0].body
    assert "following risks could harm" not in " ".join(r.text for r in risks)  # intro is not a risk


def test_split_when_categories_and_headings_share_a_style():
    risks, _ = split_risks(section_of(risk_html(LAYOUT, heading_style=B, category_style=B)))
    assert len(risks) == 3 and risks[2].category == "General Risk Factors"


def test_short_title_case_headings_are_still_risks():
    # Amazon-style: short headings, categories in a different style.
    layout = [("c", "Business and Industry Risks"), ("h", "We Face Intense Competition"), ("p", BODY),
              ("h", "Our International Operations Expose Us to a Number of Risks"), ("p", BODY),
              ("h", "Government Regulation Is Evolving"), ("p", BODY)]
    risks, _ = split_risks(section_of(risk_html(layout)))
    assert [r.heading for r in risks][0] == "We Face Intense Competition" and len(risks) == 3


def test_run_in_headings_and_summary_block():
    summary = (f'<div><span {B}>Summary of Risk Factors</span></div>'
               '<div><span>Our business is subject to the following risks.</span></div>'
               f'<div><span {B}>Risks Related to Our Industry</span></div>'
               '<div><span>Competition could hurt us.</span></div>')
    run_in = [("raw", f'<span {I}>Changes in tax laws could affect our effective tax rate significantly.</span>'
                      f'<span> We are subject to taxes in many countries. {BODY}</span>'),
              ("raw", f'<span {I}>Cyber attacks could disrupt our operations and harm our reputation.</span>'
                      f'<span> {BODY}</span>')]
    html = risk_html([], extra=summary + "".join(f"<div>{x}</div>" for _, x in run_in))
    risks, _ = split_risks(section_of(html))
    assert [r.heading for r in risks] == ["Changes in tax laws could affect our effective tax rate significantly.",
                                          "Cyber attacks could disrupt our operations and harm our reputation."]
    assert risks[0].body.startswith("We are subject to taxes")
    assert all("Competition could hurt us" not in r.text for r in risks)


# -- quotes and diffs --------------------------------------------------------

def test_contains_quote_ignores_spacing_case_curly_quotes_and_ellipses():
    text = "We rely on third‑party  foundries. Our suppliers’ capacity is limited, and costs may rise."
    assert contains_quote(text, "our suppliers' capacity is limited")
    assert contains_quote(text, "We rely on ... costs may rise.")
    assert not contains_quote(text, "capacity is unlimited")
    assert not contains_quote(text, "")


def test_diff_hunks_are_whole_sentences_from_each_year():
    before = "We sell chips worldwide. Export rules may limit sales to China. Demand can change quickly."
    after = "We sell chips worldwide. Export rules have limited sales to China and the Middle East. Demand can change quickly."
    (h,) = diff_hunks(before, after)
    assert h.before == "Export rules may limit sales to China."
    assert h.after == "Export rules have limited sales to China and the Middle East."
    assert contains_quote(before, h.before) and contains_quote(after, h.after)


def test_numbers_and_dates_only_changes_are_not_counted():
    assert diff_hunks("In fiscal 2024 we had 29,600 employees.", "In fiscal 2025 we had 36,000 employees.") == []


# -- matching ----------------------------------------------------------------

def risk(rid, heading, body, category="General"):
    return {"id": rid, "heading": heading, "body": body, "category": category, "text": f"{heading}\n{body}"}


LONG = ("Our products are manufactured by third parties and any disruption in supply, including capacity limits, "
        "natural disasters or geopolitical events, could delay shipments, raise costs and reduce our revenue and margins.")
PRIOR = [risk("R1", "We depend on third-party manufacturers.", LONG),
         risk("R2", "Currency swings affect results.", "Most of our revenue is earned outside the United States and a "
              "stronger dollar reduces reported revenue and profit in every market where we operate today."),
         risk("R3", "A pandemic could disrupt operations.", "Health crises may close offices, factories and stores, and "
              "reduce demand for our products for an extended period of time across many regions.")]
CURRENT = [risk("R1", "We depend on third-party manufacturers.", LONG + " New export rules now restrict shipments of our "
                "most advanced products to several countries, which has already reduced our sales."),
           risk("R2", "Currency swings affect results.", PRIOR[1]["body"]),
           risk("R3", "Regulation of artificial intelligence may increase our costs.", "New laws on artificial intelligence "
                "in the European Union and elsewhere may require audits, disclosures and changes to our products.")]


def test_match_risks_classifies_identical_changed_new_dropped():
    res = match_risks(CURRENT, PRIOR, DeterministicFakeEmbedding(size=32))
    kinds = {(p.current["id"], p.prior["id"]): p.kind for p in res.pairs}
    assert kinds == {("R2", "R2"): "identical", ("R1", "R1"): "changed"}
    assert [r["id"] for r in res.new] == ["R3"] and [r["id"] for r in res.dropped] == ["R3"]
    changed = next(p for p in res.pairs if p.kind == "changed")
    assert "export rules" in changed.hunks[0].after.lower()


# -- graph -------------------------------------------------------------------

def section(risks, accn, date):
    f = Filing(cik=1, form="10-K", accession=accn, filing_date=date, report_date=date, primary_document="k.htm")
    rs = [RiskFactor(r["id"], r["heading"], r["body"], r["category"]) for r in risks]
    return RiskSection(f, rs, "\n".join(r.text for r in rs))


@pytest.fixture
def fake_company(monkeypatch):
    from core.tickers import Company

    monkeypatch.setattr(rc, "lookup", lambda t: Company(t, 1, "Example Corp"))
    monkeypatch.setattr(rc, "risk_sections_two_years",
                        lambda cik: (section(CURRENT, "accn-now", "2026-02-01"), section(PRIOR, "accn-before", "2025-02-01")))
    import common.embeddings

    monkeypatch.setattr(common.embeddings, "get_embeddings", lambda: DeterministicFakeEmbedding(size=32))


def stub_judge(quote_now, quote_before):
    def answer(_messages):
        return rc.JudgeBatch(verdicts=[rc.PairVerdict(
            pair_id="P2", same_risk=True, material=True, summary="Adds that export rules already cut sales.",
            quote_now=quote_now, quote_before=quote_before)])
    return RunnableLambda(answer)


def test_graph_fans_out_and_keeps_only_verified_quotes(fake_company, monkeypatch):
    monkeypatch.setattr(rc, "structured", lambda llm, schema: stub_judge(
        "New export rules now restrict shipments of our most advanced products", "could delay shipments, raise costs"))
    monkeypatch.setattr(rc, "get_llm", lambda **kw: None)
    out = rc.graph.invoke({"tickers": ["EXM", "EXN"]})
    assert [r["ticker"] for r in out["reports"]] and len(out["reports"]) == 2  # one branch per company
    r = out["reports"][0]
    assert r["counts"] == {"current": 3, "prior": 3, "identical": 1, "minor": 0, "changed": 1, "new": 1,
                           "dropped": 1, "reorganized": 0, "unjudged": 0}
    assert r["changed"][0]["quote_now_ok"] and r["changed"][0]["quote_before_ok"]
    assert r["issues"] == []
    assert "Regulation of artificial intelligence" in r["report_md"] and "| EXM |" in out["summary_md"]


def test_graph_flags_a_paraphrased_quote(fake_company, monkeypatch):
    monkeypatch.setattr(rc, "structured", lambda llm, schema: stub_judge(
        "export controls now block our best chips", "could delay shipments, raise costs"))
    monkeypatch.setattr(rc, "get_llm", lambda **kw: None)
    r = rc.graph.invoke({"tickers": ["EXM"]})["reports"][0]
    assert not r["changed"][0]["quote_now_ok"]
    assert any("not found verbatim" in i for i in r["issues"])


def test_no_llm_mode_makes_no_model_calls(fake_company, monkeypatch):
    monkeypatch.setattr(rc, "get_llm", lambda **kw: (_ for _ in ()).throw(AssertionError("model called")))
    r = rc.graph.invoke({"tickers": ["EXM"], "no_llm": True})["reports"][0]
    assert r["counts"]["unjudged"] == 1 and r["counts"]["new"] == 1


def test_page_furniture_is_dropped_and_page_breaks_rejoined():
    layout = [("h", "Competition could adversely impact our market share and financial results."),
              ("p", "Our competitors include large companies that may have greater resources and"),
              ("p", "29"), ("p", "Table of Contents"), ("p", "Example Corp | 2026 Form 10-K | 12"),
              ("p", "longer operating histories than we do."),
              ("h", "Adverse economic conditions may harm our business."), ("p", BODY)]
    risks, _ = split_risks(section_of(risk_html(layout)))
    assert risks[0].body == ("Our competitors include large companies that may have greater resources and "
                             "longer operating histories than we do.")


def test_sub_headings_inside_a_risk_stay_in_its_body():
    # Microsoft-style: short italic labels inside a risk, followed by body text.
    layout = [("c", "STRATEGIC RISKS"),
              ("h", "We face intense competition across all markets for our products and services."),
              ("raw", f'<span {B}>Business model competition</span>'), ("p", BODY),
              ("h", "Our cloud strategy requires substantial investments."), ("p", BODY)]
    html = risk_html([], heading_style=I, extra="".join(
        f"<div><span {B}>{t}</span></div>" if k == "c" else f"<div><span {I}>{t}</span></div>" if k == "h"
        else f"<div>{t}</div>" if k == "raw" else f"<div><span>{t}</span></div>" for k, t in layout))
    risks, _ = split_risks(section_of(html))
    assert [r.heading[:20] for r in risks] == ["We face intense comp", "Our cloud strategy r"]
    assert risks[0].body.startswith("Business model competition") and risks[0].category == "STRATEGIC RISKS"


def test_summary_with_heading_styled_categories_is_skipped():
    # NVIDIA-style: the summary's category labels use the same style as the real headings.
    layout = [("h", "Risk Factors Summary"), ("p", "Our business faces these risks:"),
              ("h", "Risks Related to Our Industry and Markets"), ("p", "Failure to meet the evolving needs of our industry."),
              ("h", "Risks Related to Our Industry and Markets"),
              ("h", "Failure to meet the evolving needs of our industry may adversely impact our results."), ("p", BODY)]
    risks, _ = split_risks(section_of(risk_html(layout, heading_style=B)))
    assert [r.heading for r in risks] == ["Failure to meet the evolving needs of our industry may adversely impact our results."]
    assert risks[0].category == "Risks Related to Our Industry and Markets"
