"""Load a company's XBRL facts from the SEC and pick one clean value per fiscal year.

Source: https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json, every
number the company has tagged in its filings. Each fact looks like:

    {"start": "2023-10-01", "end": "2024-09-28", "val": 391035000000,
     "accn": "0000320193-24-000123", "fy": 2024, "fp": "FY", "form": "10-K",
     "filed": "2024-11-01", "frame": "CY2024"}

Three traps this module handles:
1. `fy` is the fiscal year of the *filing*, not of the number. A FY2024 10-K
   also restates FY2023 and FY2022, all tagged fy=2024. We label each value by
   its own period end instead (see `fiscal_year_ends`).
2. 10-K facts include quarterly values (e.g. Q4) too. For flows (revenue, R&D)
   we keep only ~12-month periods.
3. The same period is reported in up to three 10-Ks. We keep the most recently
   filed value, so restatements win, and keep its accession number for citation.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

from core.sec_client import DAY, SECClient, default_client

COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A", "20-F", "20-F/A", "40-F", "40-F/A"}
YEAR_DAYS = (330, 400)  # 52/53-week fiscal years fall inside this


@dataclass(frozen=True)
class Fact:
    taxonomy: str
    tag: str
    unit: str
    value: float
    end: date
    start: date | None
    fy: int | None
    fp: str | None
    form: str
    filed: date
    accn: str

    @property
    def is_duration(self) -> bool:
        return self.start is not None

    @property
    def days(self) -> int | None:
        return (self.end - self.start).days + 1 if self.start else None

    def filing_url(self, cik: int) -> str:
        return f"https://www.sec.gov/Archives/edgar/data/{cik}/{self.accn.replace('-', '')}/"


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _parse(taxonomy: str, tag: str, unit: str, row: dict) -> Fact:
    return Fact(
        taxonomy=taxonomy,
        tag=tag,
        unit=unit,
        value=row["val"],
        end=_d(row["end"]),
        start=_d(row.get("start")),
        fy=row.get("fy"),
        fp=row.get("fp"),
        form=row.get("form", ""),
        filed=_d(row["filed"]),
        accn=row["accn"],
    )


def _is_annual_filing(f: Fact) -> bool:
    return f.form in ANNUAL_FORMS and f.fp == "FY"


def _is_full_year(f: Fact) -> bool:
    return f.days is not None and YEAR_DAYS[0] <= f.days <= YEAR_DAYS[1]


class CompanyFacts:
    def __init__(self, data: dict):
        self.data = data
        self.cik: int = int(data["cik"])
        self.entity_name: str = data.get("entityName", "")
        self._fy_ends: dict[date, int] | None = None

    # -- raw access --------------------------------------------------------
    def has(self, tag: str, taxonomy: str = "us-gaap") -> bool:
        return tag in self.data.get("facts", {}).get(taxonomy, {})

    def units(self, tag: str, taxonomy: str = "us-gaap") -> list[str]:
        return list(self.data.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {}))

    def series(self, tag: str, unit: str, taxonomy: str = "us-gaap") -> list[Fact]:
        rows = self.data.get("facts", {}).get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])
        return [_parse(taxonomy, tag, unit, r) for r in rows]

    def _all_facts(self, taxonomy: str = "us-gaap") -> Iterable[Fact]:
        for tag, body in self.data.get("facts", {}).get(taxonomy, {}).items():
            for unit, rows in body.get("units", {}).items():
                for r in rows:
                    yield _parse(taxonomy, tag, unit, r)

    # -- fiscal calendar ---------------------------------------------------
    @property
    def fiscal_year_ends(self) -> dict[date, int]:
        """Map each fiscal-year end date to the company's own fiscal-year label.

        For every annual filing, its reporting period ends on the latest end
        date among its full-year facts, and the filing's `fy` names that year.
        So NVIDIA's year ending 2025-01-26 is labelled 2025, Apple's ending
        2024-09-28 is 2024: the labels the companies use themselves.
        """
        if self._fy_ends is None:
            latest_end: dict[str, Fact] = {}
            for f in self._all_facts():
                if _is_annual_filing(f) and _is_full_year(f) and f.fy:
                    cur = latest_end.get(f.accn)
                    if cur is None or f.end > cur.end:
                        latest_end[f.accn] = f
            ends: dict[date, int] = {}
            # Oldest filing first, so a later filing (e.g. an amendment) can correct a label.
            for f in sorted(latest_end.values(), key=lambda x: x.filed):
                ends[f.end] = f.fy
            self._fy_ends = ends
        return self._fy_ends

    def fiscal_year_of(self, end: date) -> int:
        ends = self.fiscal_year_ends
        if end in ends:
            return ends[end]
        # A 52/53-week year can end a few days off a known end date.
        for known, fy in ends.items():
            if abs((known - end).days) <= 7:
                return fy
        return end.year

    # -- annual values -----------------------------------------------------
    def annual(self, tag: str, unit: str = "USD", taxonomy: str = "us-gaap") -> dict[int, Fact]:
        """One value per fiscal year for `tag`, most recently filed wins.

        Duration facts (income and cash-flow items) must cover ~12 months.
        Instant facts (balance-sheet items) must sit on a fiscal-year end.
        """
        facts = [f for f in self.series(tag, unit, taxonomy) if _is_annual_filing(f)]
        if not facts:
            return {}
        if any(f.is_duration for f in facts):
            facts = [f for f in facts if _is_full_year(f)]
        else:
            ends = self.fiscal_year_ends
            facts = [f for f in facts if f.end in ends]

        best: dict[date, Fact] = {}
        for f in facts:
            cur = best.get(f.end)
            if cur is None or (f.filed, f.accn) > (cur.filed, cur.accn):
                best[f.end] = f
        out: dict[int, Fact] = {}
        for end in sorted(best):
            out[self.fiscal_year_of(end)] = best[end]
        return out


def load_company_facts(cik: int, client: SECClient | None = None) -> CompanyFacts:
    data = (client or default_client()).get_json(COMPANYFACTS_URL.format(cik=cik), ttl=DAY)
    return CompanyFacts(data)
