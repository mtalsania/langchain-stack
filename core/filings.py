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


def recent_10ks(cik: int, n: int = 2, client: SECClient | None = None) -> list[Filing]:
    """The n most recent 10-Ks, newest first (amendments, 10-K/A, are left out)."""
    filings = list_filings(cik, ("10-K",), client)[:n]
    if len(filings) < n:
        raise LookupError(f"CIK {cik} has {len(filings)} 10-K(s) among its recent filings; need {n}")
    return filings


def latest_10k(cik: int, client: SECClient | None = None) -> Filing:
    filings = list_filings(cik, ("10-K",), client)
    if not filings:
        raise LookupError(f"No 10-K among the recent filings of CIK {cik}")
    return filings[0]


# -- HTML to text -----------------------------------------------------------

_BLOCK_TAGS = ["p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"]


# Emphasis markers (only with mark_emphasis=True). Risk factor headings are styled
# (bold, italic, underlined) and their bodies are not, so core.risk_factors needs to
# know which text was emphasized. A marked run looks like  \x01<style char>text\x02,
# where the style char is chr(STYLE_BASE + bits) with bold=1, italic=2, underline=4.
EMPH_OPEN, EMPH_CLOSE, STYLE_BASE = "\x01", "\x02", 0xE000
BOLD, ITALIC, UNDERLINE = 1, 2, 4

_WEIGHT = re.compile(r"font-weight\s*:\s*(bold|bolder|[6-9]00)", re.I)
_WEIGHT_NORMAL = re.compile(r"font-weight\s*:\s*(normal|lighter|[1-5]00)", re.I)
_ITALIC = re.compile(r"font-style\s*:\s*(italic|oblique)", re.I)
_ITALIC_NORMAL = re.compile(r"font-style\s*:\s*normal", re.I)
_UNDER = re.compile(r"text-decoration[^:;]*:\s*[^;]*underline", re.I)


def _style_bits(node, stop) -> int:
    """Bold/italic/underline of a text node: the nearest ancestor that sets a property decides."""
    weight = italic = under = None
    el = node.parent
    while el is not None and el is not stop:
        name = (el.name or "").lower()
        style = el.get("style", "") if hasattr(el, "get") else ""
        if weight is None:
            if name in ("b", "strong") or _WEIGHT.search(style):
                weight = True
            elif _WEIGHT_NORMAL.search(style):
                weight = False
        if italic is None:
            if name in ("i", "em") or _ITALIC.search(style):
                italic = True
            elif _ITALIC_NORMAL.search(style):
                italic = False
        if under is None and (name == "u" or _UNDER.search(style)):
            under = True
        el = el.parent
    return (BOLD if weight else 0) | (ITALIC if italic else 0) | (UNDERLINE if under else 0)


def strip_marks(text: str) -> str:
    return re.sub(f"{EMPH_OPEN}.|{EMPH_CLOSE}", "", text, flags=re.S)


def html_to_text(html: str, mark_emphasis: bool = False) -> str:
    """Plain text with one line per block element, so headings land on their own line.

    mark_emphasis=True wraps bold/italic/underlined runs in markers (see EMPH_OPEN);
    strip_marks() of that output equals the unmarked text, line for line.
    """
    # Inline-XBRL 10-Ks start with an XML declaration; they still parse correctly as HTML.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "head"]):
        t.decompose()
    # Inline XBRL keeps its hidden metadata in <ix:header>; it is not part of the text.
    for t in soup.find_all(re.compile(r"^ix:header$", re.I)):
        t.decompose()
    if mark_emphasis:
        for s in list(soup.find_all(string=True)):
            if not s.strip():
                continue
            bits = _style_bits(s, None)
            if bits:
                s.replace_with(f"{EMPH_OPEN}{chr(STYLE_BASE + bits)}{s}{EMPH_CLOSE}")
    for t in soup.find_all(_BLOCK_TAGS):
        t.insert_after("\n")
    for t in soup.find_all(["td", "th"]):
        t.insert_after(" ")
    text = soup.get_text()
    text = text.replace("\xa0", " ").replace("\u200b", "")
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if strip_marks(ln).strip() or (out and strip_marks(out[-1]).strip()):
            out.append(ln if strip_marks(ln).strip() else "")
    while out and not out[-1]:
        out.pop()
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


def section_bounds(lines: list[str], item: str) -> tuple[int, int] | None:
    """(first line, end line) of one 10-K item in a list of lines, or None if not found."""
    item = item.upper()
    pos = _heading_positions(lines)
    later = ITEM_ORDER[ITEM_ORDER.index(item) + 1:]
    later_starts = sorted(p for it in later for p in pos[it])

    best = (0, None, None)
    for start in pos[item]:
        end = next((p for p in later_starts if p > start), len(lines))
        size = sum(len(ln) for ln in lines[start:end])
        if size > best[0]:
            best = (size, start, end)
    return None if best[1] is None else (best[1], best[2])


def extract_section(text: str, item: str) -> str:
    """Return the text of one 10-K item (heading included), or "" if not found."""
    lines = text.split("\n")
    bounds = section_bounds(lines, item)
    if bounds is None:
        return ""
    return "\n".join(lines[bounds[0]:bounds[1]]).strip()


def extract_sections(text: str, items: tuple[str, ...] = ("1", "1A", "7")) -> dict[str, str]:
    return {it: extract_section(text, it) for it in items}


def tenk_sections(cik: int, items: tuple[str, ...] = ("1", "1A", "7"), filing: Filing | None = None,
                  client: SECClient | None = None) -> tuple[Filing, dict[str, str]]:
    """Latest 10-K (or the one given) with the requested items as plain text."""
    client = client or default_client()
    filing = filing or latest_10k(cik, client)
    html = client.get_text(filing.url, ttl=FOREVER)
    return filing, extract_sections(html_to_text(html), items)
