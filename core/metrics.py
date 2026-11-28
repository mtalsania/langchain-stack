"""~15 core metrics, each mapped to the XBRL tags companies actually use.

Companies name the same number differently, and change names over time (Apple
moved from SalesRevenueNet to RevenueFromContractWithCustomerExcludingAssessedTax
when ASC 606 arrived in 2018). So each metric lists tag variants, and the value
for each fiscal year comes from the best variant that has that year.

Every value keeps the tag and the 10-K accession it came from, so a later LLM
step can cite it and an eval can check it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.xbrl import CompanyFacts, Fact


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    unit: str  # XBRL unit: USD, USD/shares, shares
    tags: tuple[str, ...]
    # "first": take the first tag in `tags` that has the year (most specific wins).
    # "max": take the largest value across tags. Used for revenue, where a
    #        company may tag both a total and a subset (e.g. contract revenue only).
    policy: str = "first"


@dataclass(frozen=True)
class Derived:
    name: str
    label: str
    unit: str  # "USD", "ratio"
    inputs: tuple[str, ...]
    fn: Callable[..., float | None]


@dataclass
class Value:
    value: float
    unit: str
    fiscal_year: int
    source: str  # tag name, or "derived"
    accn: str | None = None  # 10-K accession number, for citation
    period_end: str | None = None
    inputs: dict[str, "Value"] = field(default_factory=dict)  # for derived values

    @property
    def accessions(self) -> list[str]:
        if self.accn:
            return [self.accn]
        seen: list[str] = []
        for v in self.inputs.values():
            for a in v.accessions:
                if a not in seen:
                    seen.append(a)
        return seen


METRICS: tuple[Metric, ...] = (
    Metric(
        "revenue",
        "Revenue",
        "USD",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
            "SalesRevenueServicesNet",
        ),
        policy="max",
    ),
    Metric(
        "cost_of_revenue",
        "Cost of revenue",
        "USD",
        ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"),
    ),
    Metric("gross_profit_reported", "Gross profit (as tagged)", "USD", ("GrossProfit",)),
    Metric(
        "operating_income",
        "Operating income",
        "USD",
        ("OperatingIncomeLoss",),
    ),
    Metric(
        "net_income",
        "Net income",
        "USD",
        ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"),
    ),
    Metric(
        "rd",
        "R&D expense",
        "USD",
        (
            "ResearchAndDevelopmentExpense",
            "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
            "ResearchAndDevelopmentExpenseSoftwareExcludingAcquiredInProcessCost",
        ),
    ),
    Metric(
        "sga",
        "SG&A expense",
        "USD",
        ("SellingGeneralAndAdministrativeExpense",),
    ),
    Metric(
        "operating_cash_flow",
        "Operating cash flow",
        "USD",
        (
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ),
    ),
    Metric(
        "capex",
        "Capital expenditures",
        "USD",
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForCapitalImprovements",
        ),
    ),
    Metric(
        "cash",
        "Cash and equivalents",
        "USD",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
            "Cash",
        ),
    ),
    Metric(
        "long_term_debt",
        "Long-term debt (incl. current portion)",
        "USD",
        ("LongTermDebt", "LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities"),
    ),
    Metric(
        "long_term_debt_noncurrent",
        "Long-term debt, noncurrent",
        "USD",
        ("LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations"),
    ),
    Metric(
        "long_term_debt_current",
        "Long-term debt, current portion",
        "USD",
        ("LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent"),
    ),
    Metric(
        "diluted_shares",
        "Diluted weighted-average shares",
        "shares",
        ("WeightedAverageNumberOfDilutedSharesOutstanding",),
    ),
    Metric("eps_diluted", "Diluted EPS", "USD/shares", ("EarningsPerShareDiluted",)),
)


def _sub(a, b):
    return a - b


def _ratio(a, b):
    return a / b if b else None


def _debt(total, noncurrent, current):
    if total is not None:
        return total
    if noncurrent is None:
        return None
    return noncurrent + (current or 0)


DERIVED: tuple[Derived, ...] = (
    # Gross profit: the tagged value if the company reports one, else revenue minus cost.
    Derived(
        "gross_profit",
        "Gross profit",
        "USD",
        ("gross_profit_reported", "revenue", "cost_of_revenue"),
        lambda gp, rev, cost: gp if gp is not None else (rev - cost if rev is not None and cost is not None else None),
    ),
    Derived("gross_margin", "Gross margin", "ratio", ("gross_profit", "revenue"), _ratio),
    Derived("operating_margin", "Operating margin", "ratio", ("operating_income", "revenue"), _ratio),
    Derived("net_margin", "Net margin", "ratio", ("net_income", "revenue"), _ratio),
    Derived("rd_pct_revenue", "R&D % of revenue", "ratio", ("rd", "revenue"), _ratio),
    Derived("free_cash_flow", "Free cash flow", "USD", ("operating_cash_flow", "capex"), _sub),
    Derived(
        "total_debt",
        "Total long-term debt",
        "USD",
        ("long_term_debt", "long_term_debt_noncurrent", "long_term_debt_current"),
        _debt,
    ),
)

# Derived functions that accept None inputs (they pick among alternatives).
_NONE_OK = {"gross_profit", "total_debt"}

METRIC_NAMES = [m.name for m in METRICS] + [d.name for d in DERIVED]


def _value_from_fact(f: Fact, fy: int) -> Value:
    return Value(value=f.value, unit=f.unit, fiscal_year=fy, source=f.tag, accn=f.accn, period_end=f.end.isoformat())


def metric_by_year(cf: CompanyFacts, metric: Metric) -> dict[int, Value]:
    per_tag = [(tag, cf.annual(tag, metric.unit)) for tag in metric.tags if cf.has(tag)]
    years = sorted({fy for _, by_year in per_tag for fy in by_year})
    out: dict[int, Value] = {}
    for fy in years:
        candidates = [by_year[fy] for _, by_year in per_tag if fy in by_year]
        if metric.policy == "max":
            chosen = max(candidates, key=lambda f: f.value)
        else:
            chosen = candidates[0]
        out[fy] = _value_from_fact(chosen, fy)
    return out


def all_metrics(cf: CompanyFacts) -> dict[str, dict[int, Value]]:
    """Every metric for every fiscal year the company has reported in a 10-K."""
    table: dict[str, dict[int, Value]] = {m.name: metric_by_year(cf, m) for m in METRICS}
    for d in DERIVED:
        years = sorted({fy for name in d.inputs for fy in table.get(name, {})})
        col: dict[int, Value] = {}
        for fy in years:
            ins = {name: table[name].get(fy) for name in d.inputs}
            if d.name not in _NONE_OK and any(v is None for v in ins.values()):
                continue
            # A tagged gross profit passes straight through, keeping its own citation.
            if d.name == "gross_profit" and ins["gross_profit_reported"] is not None:
                col[fy] = ins["gross_profit_reported"]
                continue
            if d.name == "total_debt" and ins["long_term_debt"] is not None:
                col[fy] = ins["long_term_debt"]
                continue
            result = d.fn(*(v.value if v is not None else None for v in ins.values()))
            if result is None:
                continue
            col[fy] = Value(
                value=result,
                unit=d.unit,
                fiscal_year=fy,
                source="derived",
                period_end=next((v.period_end for v in ins.values() if v is not None), None),
                inputs={k: v for k, v in ins.items() if v is not None},
            )
        table[d.name] = col
    return table


def latest_years(table: dict[str, dict[int, Value]], n: int = 3, anchor: str = "revenue") -> list[int]:
    years = sorted(table.get(anchor, {}))
    if not years:
        years = sorted({fy for col in table.values() for fy in col})
    return years[-n:]
