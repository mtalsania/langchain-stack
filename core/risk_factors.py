"""Split a 10-K's Item 1A (Risk Factors) into individual risk factors. No LLM.

Every 10-K lists its risks the same way: a styled heading that states the risk in
one sentence (bold, italic or both), then plain paragraphs explaining it. Groups of
risks sit under short category headings such as "Risks Related to Our Industry".

So we keep the text styling (core.filings.html_to_text(mark_emphasis=True)) and:
1. find the style used most often for whole emphasized lines in Item 1A: that is the
   risk-heading style (headings outnumber categories by about 5 to 1);
2. a line in that style starts a new risk; consecutive heading lines are one heading;
3. a short emphasized line in another style, or a short one without a full stop
   right before a heading, is a category;
4. a paragraph that opens with a long run in the heading style and continues in plain
   text is a "run-in" heading: the run is the heading, the rest is body;
5. the "Summary of risk factors" block some companies put first is skipped.

    python -m core.risk_factors NVDA          # list this year's risk headings
    python -m core.risk_factors NVDA --prior  # last year's
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from core.filings import (EMPH_CLOSE, EMPH_OPEN, STYLE_BASE, Filing, html_to_text, recent_10ks, section_bounds,
                          strip_marks)
from core.sec_client import FOREVER, SECClient, default_client

MIN_RISKS = 5  # fewer headings than this means the layout was not understood
CATEGORY_MAX_WORDS = 15
RUN_IN_MIN_CHARS = 40
_SUMMARY = re.compile(r"\bsummary\b|^overview$", re.I)
_RUN = re.compile(f"{EMPH_OPEN}(.)(.*?){EMPH_CLOSE}", re.S)


@dataclass
class RiskFactor:
    id: str  # R1, R2... in filing order
    heading: str
    body: str
    category: str = ""

    @property
    def text(self) -> str:
        return f"{self.heading}\n{self.body}".strip()


@dataclass
class RiskSection:
    filing: Filing
    risks: list[RiskFactor]
    full_text: str  # Item 1A as plain text, for verifying quotes
    method: str = "emphasis"  # "emphasis" or "failed"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"filing": asdict(self.filing), "url": self.filing.url, "method": self.method, "notes": self.notes,
                "full_text": self.full_text,
                "risks": [{**asdict(r), "text": r.text} for r in self.risks]}


@dataclass
class _Line:
    text: str  # without markers
    style: int | None  # style bits if the whole line is emphasized in one style
    lead: str = ""  # emphasized opening run (for run-in headings)
    lead_style: int | None = None
    rest: str = ""

    @property
    def words(self) -> int:
        return len(self.text.split())


def _parse_line(marked: str) -> _Line:
    clean = strip_marks(marked).strip()
    runs = list(_RUN.finditer(marked))
    if not runs:
        return _Line(clean, None)
    outside = _RUN.sub("", marked)
    styles = {ord(m.group(1)) - STYLE_BASE for m in runs if m.group(2).strip()}
    if not outside.strip() and styles:
        # Whole line emphasized. Mixed styles (e.g. a bold word inside an italic heading): take the longest run's.
        longest = max(runs, key=lambda m: len(m.group(2)))
        return _Line(clean, ord(longest.group(1)) - STYLE_BASE)
    # Emphasized opening run(s), then plain text.
    first = runs[0]
    if marked[: first.start()].strip():
        return _Line(clean, None)
    lead_parts, style, pos = [], ord(first.group(1)) - STYLE_BASE, 0
    for m in runs:
        if marked[pos: m.start()].strip() or ord(m.group(1)) - STYLE_BASE != style:
            break
        lead_parts.append(marked[pos: m.start()] + m.group(2))
        pos = m.end()
    lead = "".join(lead_parts).strip()
    rest = strip_marks(marked[pos:]).strip()
    return _Line(clean, None, lead=lead, lead_style=style, rest=rest)


def heading_style(lines: list[_Line]) -> int | None:
    # A line that reads as a sentence (ends with a full stop) counts double: risk headings
    # are statements, categories are labels.
    counts: Counter = Counter()
    for ln in lines:
        if ln.style and 3 <= ln.words <= 80:
            counts[ln.style] += 2 if ln.text.endswith(".") else 1
        elif ln.lead_style and len(ln.lead) >= RUN_IN_MIN_CHARS:
            counts[ln.lead_style] += 2
    return counts.most_common(1)[0][0] if counts else None


# Page furniture that print-formatted 10-Ks repeat on every page.
_ARTIFACT = re.compile(r"^(\d{1,3}|page \d+|table of contents|part i{1,3}|item 1a\.?(\s*risk factors)?|"
                       r".{0,80}\|\s*(20\d\d\s*)?form 10-k\s*\|\s*\d{1,3})$", re.I)
_CATEGORY = re.compile(r"^(risks?\b|general risk)", re.I)
_SHORT_CATEGORY = re.compile(r"\brisks( factors)?$", re.I)  # "Operating Risks", "Legal and Regulatory Risks"
_CONTINUES = re.compile(r"\b(to|and|or|of|the|a|an|in|on|for|our|with|by|that|its)$", re.I)


def _is_label(ln: _Line) -> bool:
    return ln.style is not None and ln.words <= CATEGORY_MAX_WORDS and not ln.text.endswith(".")


def _is_category(ln: _Line, hstyle: int, next_ln: _Line | None) -> bool:
    """A short emphasized label that introduces a group of risks.

    Short emphasized lines that are followed by body text are sub-headings inside a
    risk (Microsoft uses these) and stay in that risk's body.
    """
    if not _is_label(ln):
        return False
    # "Operating Risks" is a label; "We Are Subject to Payments-Related Risks" (Amazon) is a risk.
    if _CATEGORY.search(ln.text) or (_SHORT_CATEGORY.search(ln.text) and ln.words <= 6
                                     and not re.match(r"(we|our|the company)\b", ln.text, re.I)):
        return True
    if next_ln is None or next_ln.style != hstyle:
        return False
    # The next line is heading-styled: a category, unless this line is the first half of a wrapped heading.
    return ln.style != hstyle or (not next_ln.text[:1].islower() and not _CONTINUES.search(ln.text))


def _is_heading_like(ln: _Line) -> bool:
    return ln.text.endswith(".") or ln.words > CATEGORY_MAX_WORDS


def _append_body(body: list[str], text: str) -> None:
    # A paragraph cut by a page break continues in lower case: glue it back together.
    if body and text[:1].islower() and not re.search(r"[.:;!?\"\u201d)]$", body[-1]):
        body[-1] += " " + text
    else:
        body.append(text)


def split_risks(marked_section: str) -> tuple[list[RiskFactor], list[str]]:
    """Risk factors from Item 1A text marked with emphasis. Returns (risks, notes)."""
    raw = [ln for ln in marked_section.split("\n") if strip_marks(ln).strip()]
    lines = [_parse_line(ln) for ln in raw]
    if lines and re.match(r"^\s*item\s*1a", lines[0].text, re.I):
        lines = lines[1:]  # the "Item 1A. Risk Factors" heading itself
    lines = [ln for ln in lines if not _ARTIFACT.match(ln.text)]
    hstyle = heading_style(lines)
    if hstyle is None:
        return [], ["no emphasized headings found in Item 1A"]

    risks: list[dict] = []
    category, in_summary, notes = "", False, []
    cur: dict | None = None

    def start(heading: str, body: str = "") -> dict:
        r = {"heading": heading, "body": [body] if body else [], "category": category}
        risks.append(r)
        return r

    for i, ln in enumerate(lines):
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if _is_label(ln) and _SUMMARY.search(ln.text):
            in_summary, cur, category = True, None, ln.text
            continue
        if in_summary:
            # A summary lists category labels and short bullets. It ends at the first real
            # heading: a heading-styled sentence, or a heading followed by a full paragraph.
            real = (ln.style == hstyle and (_is_heading_like(ln) or (nxt is not None and nxt.style is None
                                                                     and nxt.words > 40))) \
                or (ln.lead_style == hstyle and len(ln.lead) >= RUN_IN_MIN_CHARS)
            if not real:
                if _is_label(ln):
                    category = ln.text
                continue
            in_summary = False
        if _is_category(ln, hstyle, nxt):
            category, cur = ln.text, None
            continue
        if ln.style == hstyle:
            if cur is not None and not cur["body"]:  # heading continues on the next line
                cur["heading"] += " " + ln.text
            else:
                cur = start(ln.text)
            continue
        if ln.lead_style == hstyle and len(ln.lead) >= RUN_IN_MIN_CHARS:
            cur = start(ln.lead, ln.rest)
            continue
        if cur is not None:
            _append_body(cur["body"], ln.text)

    out = []
    for r in risks:
        body = "\n".join(r["body"]).strip()
        if not body:
            notes.append(f"heading with no body dropped: {r['heading'][:80]!r}")
            continue
        out.append(RiskFactor(id=f"R{len(out) + 1}", heading=r["heading"].strip(), body=body, category=r["category"]))
    return out, notes


def risk_section(filing: Filing, client: SECClient | None = None) -> RiskSection:
    """Item 1A of one 10-K, split into risk factors."""
    client = client or default_client()
    marked = html_to_text(client.get_text(filing.url, ttl=FOREVER), mark_emphasis=True)
    marked_lines = marked.split("\n")
    clean_lines = [strip_marks(ln) for ln in marked_lines]
    bounds = section_bounds(clean_lines, "1A")
    if bounds is None:
        return RiskSection(filing, [], "", method="failed", notes=["Item 1A not found"])
    a, b = bounds
    risks, notes = split_risks("\n".join(marked_lines[a:b]))
    method = "emphasis" if len(risks) >= MIN_RISKS else "failed"
    if method == "failed":
        notes.append(f"only {len(risks)} risk headings found; layout not understood")
    return RiskSection(filing, risks, "\n".join(clean_lines[a:b]).strip(), method, notes)


def risk_sections_two_years(cik: int, client: SECClient | None = None) -> tuple[RiskSection, RiskSection]:
    """(this year's, last year's) Item 1A, from the two most recent 10-Ks."""
    current, prior = recent_10ks(cik, 2, client)
    return risk_section(current, client), risk_section(prior, client)


def main(argv: list[str] | None = None) -> int:
    from core.tickers import lookup

    ap = argparse.ArgumentParser(description="List the risk factor headings in a 10-K's Item 1A.")
    ap.add_argument("ticker")
    ap.add_argument("--prior", action="store_true", help="use last year's 10-K")
    args = ap.parse_args(argv)
    filings = recent_10ks(lookup(args.ticker).cik, 2)
    sec = risk_section(filings[1] if args.prior else filings[0])
    print(f"{args.ticker.upper()} 10-K filed {sec.filing.filing_date} ({sec.filing.accession}): "
          f"{len(sec.risks)} risk factors, method={sec.method}\n{sec.filing.url}\n")
    cat = None
    for r in sec.risks:
        if r.category != cat:
            cat = r.category
            print(f"\n## {cat or '(no category)'}")
        print(f"{r.id:>4}  {r.heading[:150]}  ({len(r.body.split())} words)")
    for n in sec.notes:
        print(f"note: {n}")
    return 0 if sec.method == "emphasis" else 1


if __name__ == "__main__":
    raise SystemExit(main())
