"""Find a company's 10-K filings and cut out Item 1, 1A and 7 as plain text.

Filing list: https://data.sec.gov/submissions/CIK##########.json
Documents:   https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<file>

The hard part is that "Item 1A. Risk Factors" appears twice in every 10-K: once
in the table of contents and once as the real heading. We find every heading
for every item, and for the item we want keep the occurrence whose section
(up to the next item's heading) is longest. Table-of-contents entries are a
line long, so the real section always wins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from core.sec_client import DAY, FOREVER, SECClient, default_client

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accn_nodash}/{doc}"


@dataclass(frozen=True)
class Filing:
    cik: int
    form: str
    accession: str
    filing_date: str
    report_date: str
    primary_document: str

    @property
    def url(self) -> str:
        return ARCHIVE_URL.format(cik=self.cik, accn_nodash=self.accession.replace("-", ""), doc=self.primary_document)

    @property
    def index_url(self) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{self.cik}/{self.accession.replace('-', '')}/"


def parse_submissions(data: dict) -> list[Filing]:
    """Recent filings, newest first (the SEC's own order)."""
    cik = int(data["cik"])
    r = data["filings"]["recent"]
    return [
        Filing(
            cik=cik,
            form=r["form"][i],
            accession=r["accessionNumber"][i],
            filing_date=r["filingDate"][i],
            report_date=r["reportDate"][i],
            primary_document=r["primaryDocument"][i],
        )
        for i in range(len(r["accessionNumber"]))
    ]


def list_filings(cik: int, forms: tuple[str, ...] = ("10-K",), client: SECClient | None = None) -> list[Filing]:
    data = (client or default_client()).get_json(SUBMISSIONS_URL.format(cik=cik), ttl=DAY)
    return [f for f in parse_submissions(data) if f.form in forms]


def latest_10k(cik: int, client: SECClient | None = None) -> Filing:
    filings = list_filings(cik, ("10-K",), client)
    if not filings:
        raise LookupError(f"No 10-K among the recent filings of CIK {cik}")
    return filings[0]


# -- HTML to text -----------------------------------------------------------

_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]


def html_to_text(html: str) -> str:
    """Plain text with one line per block element, so headings land on their own line."""
    # Inline-XBRL 10-Ks start with an XML declaration; they still parse correctly as HTML.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    # Inline XBRL keeps its hidden metadata in <ix:header>; it is not part of the text.
    for t in soup.find_all(re.compile(r"^ix:header$", re.I)):
        t.decompose()
    for t in soup.find_all(_BLOCK_TAGS):
        t.insert_after("\n")
    for t in soup.find_all(["td", "th"]):
        t.insert_after(" ")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("​", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


# -- section extraction -----------------------------------------------------

# 10-K items in order. A section ends where any later item's heading starts.
ITEM_ORDER = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A", "8", "9", "9A", "9B", "9C",
              "10", "11", "12", "13", "14", "15", "16"]

SECTION_NAMES = {"1": "Business", "1A": "Risk Factors", "7": "Management's Discussion and Analysis"}

_HEADING_MAX_LEN = 150  # a heading line is short; body text mentioning "Item 1A" is not


def _heading_re(item: str) -> re.Pattern:
    # "Item 1A.", "ITEM 1A:", "Item 1A —", "Item 1A Risk Factors"; never matches 1 inside 1A or 10.
    return re.compile(rf"^\s*item\s*{re.escape(item)}(?![0-9a-z])", re.I)


def _heading_positions(lines: list[str]) -> dict[str, list[int]]:
    pos: dict[str, list[int]] = {it: [] for it in ITEM_ORDER}
    pats = {it: _heading_re(it) for it in ITEM_ORDER}
    for i, ln in enumerate(lines):
        if not ln or len(ln) > _HEADING_MAX_LEN or not ln.lower().lstrip().startswith("item"):
            continue
        for it in ITEM_ORDER:
            if pats[it].match(ln):
                pos[it].append(i)
                break
    return pos


def extract_section(text: str, item: str) -> str:
    """Return the text of one 10-K item (heading included), or "" if not found."""
    item = item.upper()
    lines = text.split("\n")
    pos = _heading_positions(lines)
    later = ITEM_ORDER[ITEM_ORDER.index(item) + 1:]
    later_starts = sorted(p for it in later for p in pos[it])

    best = (0, None, None)
    for start in pos[item]:
        end = next((p for p in later_starts if p > start), len(lines))
        size = sum(len(ln) for ln in lines[start:end])
        if size > best[0]:
            best = (size, start, end)
    if best[1] is None:
        return ""
    return "\n".join(lines[best[1]:best[2]]).strip()


def extract_sections(text: str, items: tuple[str, ...] = ("1", "1A", "7")) -> dict[str, str]:
    return {it: extract_section(text, it) for it in items}


def tenk_sections(cik: int, items: tuple[str, ...] = ("1", "1A", "7"), filing: Filing | None = None,
                  client: SECClient | None = None) -> tuple[Filing, dict[str, str]]:
    """Latest 10-K (or the one given) with the requested items as plain text."""
    client = client or default_client()
    filing = filing or latest_10k(cik, client)
    html = client.get_text(filing.url, ttl=FOREVER)
    return filing, extract_sections(html_to_text(html), items)
