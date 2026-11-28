"""Show how a company's XBRL facts are tagged, to debug why a metric is missing.

    python -m scripts.diagnose_facts XOM
"""
import sys
from collections import Counter

from core.tickers import lookup
from core.xbrl import load_company_facts

ticker = sys.argv[1] if len(sys.argv) > 1 else "XOM"
co = lookup(ticker)
cf = load_company_facts(co.cik)
gaap = cf.data.get("facts", {})
print(co, "| entity:", cf.entity_name, "| taxonomies:", {k: len(v) for k, v in gaap.items()})
for tag in ("Revenues", "NetIncomeLoss", "EarningsPerShareDiluted"):
    for tax, body in gaap.items():
        if tag in body:
            for unit, rows in body[tag]["units"].items():
                print(f"\n{tax}:{tag} [{unit}] {len(rows)} facts")
                print("  form/fp:", Counter((r.get("form"), r.get("fp")) for r in rows).most_common(8))
                for r in rows[-4:]:
                    print("  ", r)
print("\nfiscal_year_ends:", sorted(cf.fiscal_year_ends.items())[-5:])
