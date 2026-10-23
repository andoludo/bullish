"""Data-coverage grades: can the database support a judgement on this symbol?

Reads the bearish ``coverage`` census plus the reference row and grades six
dimensions A / B / C / insufficient, naming what is missing and the bearish
command that would fill it.
"""

from __future__ import annotations

from typing import Literal

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


def _rows(coverage: pd.DataFrame, symbol: str, table: str) -> pd.DataFrame:
    return coverage[(coverage["symbol"] == symbol) & (coverage["table_name"] == table)]


def _span_years(rows: pd.DataFrame) -> float:
    if rows.empty:
        return 0.0
    first = pd.to_datetime(rows["first_date"]).min()
    last = pd.to_datetime(rows["last_date"]).max()
    return float((last - first).days / 365.25)


def _grade_statements(
    coverage: pd.DataFrame, symbol: str, country: str | None
) -> DimensionGrade:
    years = max(
        (
            _rows(coverage, symbol, t)["rows"].max()
            if not _rows(coverage, symbol, t).empty
            else 0
        )
        for t in STATEMENT_TABLES
    )
    quarters = max(
        (
            _rows(coverage, symbol, t)["rows"].max()
            if not _rows(coverage, symbol, t).empty
            else 0
        )
        for t in QUARTERLY_TABLES
    )
    sources = set(_rows(coverage, symbol, "financialmetrics")["source"])
    sec = "SEC" in sources
    missing = []
    if years < 4:
        missing.append(f"only {int(years)} fiscal years of statements (need 4)")
    if quarters < 8:
        missing.append(f"only {int(quarters)} quarters of statements (need 8)")
    if country == "USA" and not sec:
        missing.append("statements not from SEC XBRL")
    if years == 0 and quarters == 0:
        grade: Grade = "insufficient"
    elif not missing:
        grade = "A"
    elif years >= 2 and quarters >= 4:
        grade = "B"
    else:
        grade = "C"
    return DimensionGrade(
        dimension="statements",
        grade=grade,
        detail=f"{int(years)} years, {int(quarters)} quarters, sources {sorted(sources) or 'none'}",
        missing=missing,
        fill_with=(
            f"bearish financials <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_prices(
    coverage: pd.DataFrame, symbol: str, as_of: pd.Timestamp
) -> DimensionGrade:
    rows = _rows(coverage, symbol, "price")
    years = _span_years(rows)
    last = pd.to_datetime(rows["last_date"]).max() if not rows.empty else None
    stale_days = (as_of - last).days if last is not None else None
    missing = []
    if years < 3:
        missing.append(f"{years:.1f} years of prices (need 3)")
    if stale_days is None or stale_days > 5:
        missing.append(
            "last price older than 5 days" if stale_days is not None else "no prices"
        )
    if rows.empty:
        grade: Grade = "insufficient"
    elif not missing:
        grade = "A"
    elif years >= 1:
        grade = "B"
    else:
        grade = "C"
    return DimensionGrade(
        dimension="prices",
        grade=grade,
        detail=f"{years:.1f} years, last {last.date() if last is not None else 'none'}",
        missing=missing,
        fill_with=(
            f"bearish prices <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_expectations(coverage: pd.DataFrame, symbol: str) -> DimensionGrade:
    targets = _rows(coverage, symbol, "pricetarget")["rows"].sum()
    earnings = _rows(coverage, symbol, "earningsdate")["rows"].sum()
    recs = _rows(coverage, symbol, "analystrecommendation")["rows"].sum()
    missing = []
    if targets < 1:
        missing.append("no analyst price targets")
    if earnings < 4:
        missing.append(f"{int(earnings)} earnings rows (need 4)")
    if recs < 1:
        missing.append("no recommendation trend")
    grade: Grade = (
        "insufficient"
        if targets + earnings + recs == 0
        else ("A" if not missing else ("B" if targets >= 1 else "C"))
    )
    return DimensionGrade(
        dimension="expectations",
        grade=grade,
        detail=f"{int(targets)} target snapshots, {int(earnings)} earnings rows, {int(recs)} recommendation rows",
        missing=missing,
        fill_with=(
            f"bearish financials <db> <country> --filters {symbol}" if missing else None
        ),
    )


def _grade_ownership(
    coverage: pd.DataFrame, symbol: str, country: str | None
) -> DimensionGrade:
    holdings = _rows(coverage, symbol, "holding13fsummary")["rows"].sum()
    insiders = _rows(coverage, symbol, "insidertransaction")["rows"].sum()
    filings = _rows(coverage, symbol, "filing")["rows"].sum()
    if country != "USA":
        return DimensionGrade(
            dimension="ownership",
            grade="B",
            detail="SEC data only covers US filers",
            missing=[],
            fill_with=None,
        )
    missing = []
    if holdings < 2:
        missing.append("fewer than two 13F quarters")
    if insiders < 1:
        missing.append("no insider transactions")
    if filings < 1:
        missing.append("no EDGAR filings")
    grade: Grade = (
        "insufficient"
        if holdings + insiders + filings == 0
        else ("A" if not missing else "B")
    )
    return DimensionGrade(
        dimension="ownership",
        grade=grade,
        detail=f"{int(holdings)} 13F quarters, {int(insiders)} insider rows, {int(filings)} filings",
        missing=missing,
        fill_with=(
            f"bearish holdings <db>; bearish insiders <db> US --filters {symbol}; bearish filings <db> "
            f"US --filters {symbol}"
            if missing
            else None
        ),
    )


def _grade_reference(universe_row: pd.Series | None) -> DimensionGrade:
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


def grade_symbol(
    symbol: str,
    coverage: pd.DataFrame,
    universe_row: pd.Series | None,
    as_of: pd.Timestamp,
) -> CoverageReport:
    country = None if universe_row is None else universe_row.get("country")
    dimensions = [
        _grade_statements(coverage, symbol, country),
        _grade_prices(coverage, symbol, as_of),
        _grade_expectations(coverage, symbol),
        _grade_ownership(coverage, symbol, country),
        _grade_reference(universe_row),
    ]
    core = [
        d.grade
        for d in dimensions
        if d.dimension in {"statements", "prices", "reference"}
    ]
    overall: Grade = min(core, key=lambda g: ORDER[g])
    return CoverageReport(symbol=symbol, overall=overall, dimensions=dimensions)


def grade_universe(
    coverage: pd.DataFrame, universe: pd.DataFrame, as_of: pd.Timestamp
) -> pd.DataFrame:
    """One row per symbol with the overall grade and each dimension's grade."""
    rows = {}
    for symbol in universe.index:
        report = grade_symbol(symbol, coverage, universe.loc[symbol], as_of)
        rows[symbol] = {"overall": report.overall} | {
            d.dimension: d.grade for d in report.dimensions
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "symbol"
    return frame
