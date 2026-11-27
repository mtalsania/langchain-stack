"""Ticker -> CIK lookup from the SEC's official company_tickers.json.

https://www.sec.gov/files/company_tickers.json maps every listed ticker to its
CIK (the SEC's company id). It is cached on disk and refreshed daily.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.sec_client import DAY, SECClient, default_client

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


@dataclass(frozen=True)
class Company:
    ticker: str
    cik: int
    name: str

    @property
    def cik10(self) -> str:
        """Zero-padded 10-digit CIK, the form data.sec.gov URLs use."""
        return f"{self.cik:010d}"


class TickerNotFound(KeyError):
    pass


def normalize_ticker(ticker: str) -> str:
    # The SEC file uses dashes for share classes: BRK-B, not BRK.B or BRK/B.
    return ticker.strip().upper().replace(".", "-").replace("/", "-")


def parse_ticker_file(data: dict) -> dict[str, Company]:
    """The file is {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}."""
    out: dict[str, Company] = {}
    for row in data.values():
        t = normalize_ticker(row["ticker"])
        # A few tickers appear twice; the first row is the SEC's primary listing.
        out.setdefault(t, Company(ticker=t, cik=int(row["cik_str"]), name=row["title"]))
    return out


_index: dict[str, Company] | None = None


def load_index(client: SECClient | None = None) -> dict[str, Company]:
    global _index
    if _index is None or client is not None:
        index = parse_ticker_file((client or default_client()).get_json(TICKERS_URL, ttl=DAY))
        if client is None:
            _index = index
        return index
    return _index


def lookup(ticker: str, client: SECClient | None = None) -> Company:
    t = normalize_ticker(ticker)
    index = load_index(client)
    if t not in index:
        raise TickerNotFound(f"{ticker!r} is not in the SEC ticker file ({TICKERS_URL})")
    return index[t]


def ticker_to_cik(ticker: str, client: SECClient | None = None) -> int:
    return lookup(ticker, client).cik
