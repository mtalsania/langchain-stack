"""Print a 3-year table of core metrics for a ticker, straight from SEC XBRL data.

    python -m core.demo NVDA
    python -m core.demo AAPL --years 5
    python -m core.demo MSFT --sections     # also cut Items 1, 1A, 7 from the latest 10-K

No LLM involved: every number is the value the company tagged in its 10-K,
with the accession number of the filing it came from.
"""
from __future__ import annotations

import argparse
import sys

from core.filings import SECTION_NAMES, tenk_sections
from core.metrics import DERIVED, METRICS, Value, all_metrics, latest_years
from core.sec_client import SECConfigError, default_client
from core.tickers import lookup
from core.xbrl import load_company_facts

SHOWN = [
    "revenue", "gross_profit", "gross_margin", "rd", "rd_pct_revenue", "sga", "operating_income",
    "operating_margin", "net_income", "net_margin", "eps_diluted", "diluted_shares",
    "operating_cash_flow", "capex", "free_cash_flow", "cash", "total_debt",
]
LABELS = {m.name: m.label for m in METRICS} | {d.name: d.label for d in DERIVED}


def fmt(v: Value | None) -> str:
    if v is None:
        return "n/a"
    if v.unit == "ratio":
        return f"{v.value * 100:.1f}%"
    if v.unit == "USD/shares":
        return f"${v.value:.2f}"
    if v.unit == "shares":
        return f"{v.value / 1e6:,.0f}M sh"
    return f"${v.value / 1e6:,.0f}M"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ticker")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--sections", action="store_true", help="extract Items 1, 1A, 7 from the latest 10-K")
    args = ap.parse_args(argv)

    try:
        client = default_client()
    except SECConfigError as e:
        print(e, file=sys.stderr)
        return 2

    co = lookup(args.ticker)
    cf = load_company_facts(co.cik)
    table = all_metrics(cf)
    years = latest_years(table, args.years)

    print(f"\n{cf.entity_name} ({co.ticker}, CIK {co.cik}) — fiscal years {', '.join(map(str, years))}")
    print("Source: SEC XBRL companyfacts, annual 10-K values (latest filing wins)\n")
    w = max(len(LABELS[n]) for n in SHOWN) + 2
    print("".ljust(w) + "".join(f"FY{fy}".rjust(16) for fy in years) + "   tag")
    for name in SHOWN:
        col = table.get(name, {})
        src = next((col[fy].source for fy in reversed(years) if fy in col), "")
        print(LABELS[name].ljust(w) + "".join(fmt(col.get(fy)).rjust(16) for fy in years) + f"   {src}")

    print("\nFilings cited:")
    rev = table.get("revenue", {})
    cited = sorted({a for col in table.values() for fy in years if fy in col for a in col[fy].accessions})
    for a in cited:
        print(f"  {a}  https://www.sec.gov/Archives/edgar/data/{co.cik}/{a.replace('-', '')}/")
    if rev:
        last = rev[years[-1]]
        print(f"\nFY{years[-1]} revenue {fmt(last)} = tag {last.source}, period ending {last.period_end}, accession {last.accn}")

    if args.sections:
        filing, secs = tenk_sections(co.cik)
        print(f"\nLatest 10-K: filed {filing.filing_date}, period {filing.report_date}\n  {filing.url}")
        for item, text in secs.items():
            first = text.split("\n", 1)[0] if text else "(not found)"
            print(f"  Item {item} ({SECTION_NAMES.get(item, '')}): {len(text.split()):,} words  | {first[:70]}")

    print(f"\nSEC requests made: {client.requests_made} (the rest came from the local cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
