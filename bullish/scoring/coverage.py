"""Data-coverage grades: can the database support a judgement on this symbol?

Reads the bearish ``coverage`` census plus the reference row and grades six
dimensions A / B / C / insufficient, naming what is missing and the bearish
command that would fill it.

The census is aggregated once per symbol (:class:`CoverageIndex`); grading
a symbol is then a handful of scalar rules, so the whole universe grades in
well under a second rather than scanning the census per symbol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

Grade = Literal["A", "B", "C", "insufficient"]
ORDER: dict[str, int] = {"A": 3, "B": 2, "C": 1, "insufficient": 0}

STATEMENT_TABLES = ("financialmetrics", "balancesheet", "cashflow")
QUARTERLY_TABLES = (
    "quarterlyfinancialmetrics",
    "quarterlybalancesheet",
    "quarterlycashflow",
)
COUNT_TABLES = (
    "pricetarget",
    "earningsdate",
    "analystrecommendation",
    "holding13fsummary",
    "insidertransaction",
    "filing",
)


class DimensionGrade(BaseModel):
    dimension: str
    grade: Grade
    detail: str
    missing: list[str]
    fill_with: str | None = None


class CoverageReport(BaseModel):
    symbol: str
    overall: Grade
    dimensions: list[DimensionGrade]

    def grade_of(self, dimension: str) -> Grade:
        return next(d.grade for d in self.dimensions if d.dimension == dimension)


@dataclass(frozen=True)
class SymbolCoverage:
    """What the census says about one symbol, reduced to the numbers the rules need."""

    years: float = 0.0
    quarters: float = 0.0
    sources: tuple[str, ...] = ()
    has_prices: bool = False
    price_years: float = 0.0
    price_last: pd.Timestamp | None = None
    targets: float = 0.0
    earnings: float = 0.0
    recs: float = 0.0
    holdings: float = 0.0
    insiders: float = 0.0
    filings: float = 0.0


NO_COVERAGE = SymbolCoverage()


class CoverageIndex:
    """Per-symbol :class:`SymbolCoverage` aggregated once from the long census frame."""

    def __init__(self, coverage: pd.DataFrame) -> None:
        self._rows: dict[str, SymbolCoverage] = {}
        if coverage.empty:
            return
        frame = coverage.assign(
            rows=pd.to_numeric(coverage["rows"], errors="coerce").fillna(0.0),
            first=pd.to_datetime(coverage["first_date"]),
            last=pd.to_datetime(coverage["last_date"]),
        )
        by_table = frame.groupby(["symbol", "table_name"], sort=False)
        rows_max = by_table["rows"].max().unstack("table_name")  # noqa: PD010
        rows_sum = by_table["rows"].sum().unstack("table_name")  # noqa: PD010
        prices = frame[frame["table_name"] == "price"].groupby("symbol")
        price_first = prices["first"].min()
        price_last = prices["last"].max()
        sources: dict[str, tuple[str, ...]] = {
            str(symbol): tuple(sorted(set(group.astype(str))))
            for symbol, group in frame[
                frame["table_name"] == "financialmetrics"
            ].groupby("symbol")["source"]
        }
        years = _max_over(rows_max, STATEMENT_TABLES).to_dict()
        quarters = _max_over(rows_max, QUARTERLY_TABLES).to_dict()
        counts = {t: _column(rows_sum, t).to_dict() for t in COUNT_TABLES}
        first_by_symbol = price_first.to_dict()
        last_by_symbol = price_last.to_dict()
        for symbol in rows_max.index:
            key = str(symbol)
            last = last_by_symbol.get(symbol)
            first = first_by_symbol.get(symbol)
            self._rows[key] = SymbolCoverage(
                years=float(years[symbol]),
                quarters=float(quarters[symbol]),
                sources=sources.get(key, ()),
                has_prices=symbol in last_by_symbol,
                price_years=(
                    float((last - first).days / 365.25)
                    if last is not None and first is not None
                    else 0.0
                ),
                price_last=last,
                targets=float(counts["pricetarget"][symbol]),
                earnings=float(counts["earningsdate"][symbol]),
                recs=float(counts["analystrecommendation"][symbol]),
                holdings=float(counts["holding13fsummary"][symbol]),
                insiders=float(counts["insidertransaction"][symbol]),
                filings=float(counts["filing"][symbol]),
            )

    def get(self, symbol: str) -> SymbolCoverage:
        return self._rows.get(symbol, NO_COVERAGE)


def _column(wide: pd.DataFrame, table: str) -> pd.Series:
    return (
        wide[table].fillna(0.0) if table in wide else pd.Series(0.0, index=wide.index)
    )


def _max_over(wide: pd.DataFrame, tables: tuple[str, ...]) -> pd.Series:
    present = [t for t in tables if t in wide]
    if not present:
        return pd.Series(0.0, index=wide.index)
    return wide[present].fillna(0.0).max(axis=1)


def _grade_statements(
    c: SymbolCoverage, symbol: str, country: str | None
) -> DimensionGrade:
    sec = "SEC" in c.sources
    missing = []
    if c.years < 4:
        missing.append(f"only {int(c.years)} fiscal years of statements (need 4)")
    if c.quarters < 8:
        missing.append(f"only {int(c.quarters)} quarters of statements (need 8)")
    if country == "USA" and not sec:
        missing.append("statements not from SEC XBRL")
    if c.years == 0 and c.quarters == 0:
        grade: Grade = "insufficient"
    elif not missing:
        grade = "A"
    elif c.years >= 2 and c.quarters >= 4:
        grade = "B"
    else:
        grade = "C"
    return DimensionGrade(
        dimension="statements",
        grade=grade,
        detail=f"{int(c.years)} years, {int(c.quarters)} quarters, sources {list(c.sources) or 'none'}",
        missing=missing,
        fill_with=(
            f"bearish financials <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_prices(
    c: SymbolCoverage, symbol: str, as_of: pd.Timestamp
) -> DimensionGrade:
    last = c.price_last
    stale_days = (as_of - last).days if last is not None else None
    missing = []
    if c.price_years < 3:
        missing.append(f"{c.price_years:.1f} years of prices (need 3)")
    if stale_days is None or stale_days > 5:
        missing.append(
            "last price older than 5 days" if stale_days is not None else "no prices"
        )
    if not c.has_prices:
        grade: Grade = "insufficient"
    elif not missing:
        grade = "A"
    elif c.price_years >= 1:
        grade = "B"
    else:
        grade = "C"
    return DimensionGrade(
        dimension="prices",
        grade=grade,
        detail=f"{c.price_years:.1f} years, last {last.date() if last is not None else 'none'}",
        missing=missing,
        fill_with=(
            f"bearish prices <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_expectations(c: SymbolCoverage, symbol: str) -> DimensionGrade:
    missing = []
    if c.targets < 1:
        missing.append("no analyst price targets")
    if c.earnings < 4:
        missing.append(f"{int(c.earnings)} earnings rows (need 4)")
    if c.recs < 1:
        missing.append("no recommendation trend")
    grade: Grade = (
        "insufficient"
        if c.targets + c.earnings + c.recs == 0
        else ("A" if not missing else ("B" if c.targets >= 1 else "C"))
    )
    return DimensionGrade(
        dimension="expectations",
        grade=grade,
        detail=f"{int(c.targets)} target snapshots, {int(c.earnings)} earnings rows, {int(c.recs)} recommendation rows",
        missing=missing,
        fill_with=(
            f"bearish financials <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_ownership(
    c: SymbolCoverage, symbol: str, country: str | None
) -> DimensionGrade:
    if country != "USA":
        return DimensionGrade(
            dimension="ownership",
            grade="B",
            detail="SEC data only covers US filers",
            missing=[],
            fill_with=None,
        )
    missing = []
    if c.holdings < 2:
        missing.append("fewer than two 13F quarters")
    if c.insiders < 1:
        missing.append("no insider transactions")
    if c.filings < 1:
        missing.append("no EDGAR filings")
    grade: Grade = (
        "insufficient"
        if c.holdings + c.insiders + c.filings == 0
        else ("A" if not missing else "B")
    )
    return DimensionGrade(
        dimension="ownership",
        grade=grade,
        detail=f"{int(c.holdings)} 13F quarters, {int(c.insiders)} insider rows, {int(c.filings)} filings",
        missing=missing,
        fill_with=(
            f"bearish holdings <db>; bearish insiders <db> US --filters {symbol}; bearish filings <db> "
            f"US --filters {symbol}"
            if missing
            else None
        ),
    )


def _grade_reference(
    universe_row: Mapping[Any, Any] | pd.Series | None,
) -> DimensionGrade:
    if universe_row is None:
        return DimensionGrade(
            dimension="reference",
            grade="insufficient",
            detail="symbol not in the equity table",
            missing=["equity row"],
            fill_with="bearish tickers <db> <country>",
        )
    required = (
        "sector",
        "industry",
        "country",
        "currency",
        "market_capitalization",
        "shares_outstanding",
    )
    missing = [name for name in required if pd.isna(universe_row.get(name))]
    grade: Grade = "A" if not missing else ("B" if len(missing) <= 2 else "C")
    return DimensionGrade(
        dimension="reference",
        grade=grade,
        detail=(
            "sector, industry and country resolved"
            if not missing
            else f"missing {missing}"
        ),
        missing=[f"{m} unresolved" for m in missing],
        fill_with=(
            "bearish tickers <db> <country>; bearish link <db>" if missing else None
        ),
    )


def _report(
    symbol: str,
    c: SymbolCoverage,
    universe_row: Mapping[Any, Any] | pd.Series | None,
    as_of: pd.Timestamp,
) -> CoverageReport:
    country = None if universe_row is None else universe_row.get("country")
    dimensions = [
        _grade_statements(c, symbol, country),
        _grade_prices(c, symbol, as_of),
        _grade_expectations(c, symbol),
        _grade_ownership(c, symbol, country),
        _grade_reference(universe_row),
    ]
    core = [
        d.grade
        for d in dimensions
        if d.dimension in {"statements", "prices", "reference"}
    ]
    overall: Grade = min(core, key=lambda g: ORDER[g])
    return CoverageReport(symbol=symbol, overall=overall, dimensions=dimensions)


def grade_symbol(
    symbol: str,
    coverage: pd.DataFrame | CoverageIndex,
    universe_row: Mapping[Any, Any] | pd.Series | None,
    as_of: pd.Timestamp,
) -> CoverageReport:
    """Full report for one symbol; pass a :class:`CoverageIndex` to grade many symbols cheaply."""
    index = coverage if isinstance(coverage, CoverageIndex) else CoverageIndex(coverage)
    return _report(symbol, index.get(symbol), universe_row, as_of)


def grade_universe(
    coverage: pd.DataFrame | CoverageIndex, universe: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """One row per symbol with the overall grade and each dimension's grade."""
    index = coverage if isinstance(coverage, CoverageIndex) else CoverageIndex(coverage)
    rows = {}
    for symbol, universe_row in universe.to_dict("index").items():
        report = _report(str(symbol), index.get(str(symbol)), universe_row, as_of)
        rows[symbol] = {"overall": report.overall} | {
            d.dimension: d.grade for d in report.dimensions
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "symbol"
    return frame
