"""Print how Item 1A was split and how risks matched, for tuning thresholds on live data.

    python -m scripts.risk_diagnostics NVDA AAPL
"""
from __future__ import annotations

import sys

import numpy as np

from common.embeddings import get_embeddings
from core.risk_factors import risk_sections_two_years
from core.tickers import lookup
from tools.risk_matching import embed_texts, match_risks, overlap, shingles


def main(tickers: list[str]) -> None:
    for t in tickers:
        cur, pri = risk_sections_two_years(lookup(t).cik)
        print(f"\n===== {t}: now {len(cur.risks)} risks ({cur.method}, {cur.filing.filing_date}), "
              f"before {len(pri.risks)} ({pri.method}, {pri.filing.filing_date})")
        for n in cur.notes[:5] + pri.notes[:5]:
            print("  note:", n)
        cats = []
        for r in cur.risks:
            if r.category not in cats:
                cats.append(r.category)
        print("  categories:", cats)
        for r in cur.risks[:4] + cur.risks[-2:]:
            print(f"  {r.id}: {r.heading[:120]!r} [{len(r.body.split())} words]")
        words = sorted(len(r.body.split()) for r in cur.risks)
        print(f"  body words min/median/max: {words[0] if words else 0}/{words[len(words)//2] if words else 0}/{words[-1] if words else 0}")
        cd, pd = [r for r in (x.to_dict()["risks"] for x in (cur,))][0], pri.to_dict()["risks"]
        emb = get_embeddings()
        sim = embed_texts(cd, emb) @ embed_texts(pd, emb).T
        csh, psh = [shingles(r["text"]) for r in cd], [shingles(r["text"]) for r in pd]
        best = [(float(sim[i].max()), max(overlap(csh[i], s) for s in psh)) for i in range(len(cd))]
        print("  best sim per current risk (sorted):", [round(b[0], 2) for b in sorted(best)][:25])
        print("  best overlap per current risk (sorted):", [round(b[1], 2) for b in sorted(best, key=lambda x: x[1])][:25])
        low = [(cd[i]["heading"][:70], round(best[i][0], 3), round(best[i][1], 2)) for i in range(len(cd)) if best[i][1] < 0.5]
        for h, s, o in low:
            j = int(np.argmax(sim[[k for k in range(len(cd)) if cd[k]["heading"][:70] == h][0]]))
            print(f"    low-overlap: {h!r} sim={s} ov={o} best-prior={pd[j]['heading'][:70]!r}")
        res = match_risks(cd, pd, emb)
        kinds = {}
        for p in res.pairs:
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        print(f"  match: {kinds} new={len(res.new)} dropped={len(res.dropped)}")
        for p in res.pairs:
            if p.kind == "possible":
                print(f"    possible: {p.current['heading'][:60]!r} ~ {p.prior['heading'][:60]!r} sim={p.similarity} ov={p.overlap}")
        for r in res.new:
            print(f"    new: {r['heading'][:100]!r} reorg={bool(r['reorganized_with'])} ov={r['overlap_elsewhere']}")
        for r in res.dropped:
            print(f"    dropped: {r['heading'][:100]!r} reorg={bool(r['reorganized_with'])} ov={r['overlap_elsewhere']}")
        changed = [p for p in res.pairs if p.kind == "changed"]
        cw = sorted(sum(h.changed_words for h in p.hunks) for p in changed)
        print(f"  changed pairs: {len(changed)}, changed words per pair: {cw}")
        if changed:
            p = changed[len(changed) // 2]
            print(f"  sample hunk for {p.current['heading'][:60]!r}:")
            for h in p.hunks[:2]:
                print(f"    BEFORE: {h.before[:300]!r}\n    NOW:    {h.after[:300]!r}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL"])
