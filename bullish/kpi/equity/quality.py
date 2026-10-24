"""Profitability, earnings quality, balance-sheet and capital-discipline KPIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.equity._statements import (
    annual,
    growth,
    last_valid,
    latest_balance,
    nth_last,
    safe_div,
    ttm,
)
from bullish.kpi.registry import declare_many

BLOCK = "quality"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "gross_margin",
            "A",
            "ratio",
            "Gross profit over revenue, trailing twelve months",
            True,
        ),
        (
            "operating_margin",
            "A",
            "ratio",
            "Operating income over revenue, trailing twelve months",
            True,
        ),
        (
            "net_margin",
            "B",
            "ratio",
            "Net income over revenue, trailing twelve months",
            True,
        ),
        (
            "fcf_margin",
            "A",
            "ratio",
            "Free cash flow over revenue, trailing twelve months",
            True,
        ),
        (
            "gross_margin_trend",
            "B",
            "ratio",
            "Trailing gross margin minus the gross margin of the prior twelve months",
            True,
        ),
        (
            "operating_margin_trend",
            "B",
            "ratio",
            "Trailing operating margin minus the prior twelve months",
            True,
        ),
        ("roe", "A", "ratio", "Trailing net income over shareholder equity", True),
        ("roa", "A", "ratio", "Trailing net income over total assets", True),
        (
            "roic",
            "A",
            "ratio",
            "After-tax operating income over equity plus debt minus cash",
            True,
        ),
        (
            "fcf_conversion",
            "A",
            "ratio",
            "Free cash flow over net income, trailing twelve months",
            True,
        ),
        (
            "accruals_ratio",
            "A",
            "ratio",
            "(Net income minus operating cash flow) over total assets (Sloan accruals)",
            False,
        ),
        (
            "net_debt_to_ebitda",
            "A",
            "ratio",
            "Debt minus cash over trailing EBITDA (negative = net cash)",
            False,
        ),
        (
            "interest_coverage",
            "A",
            "ratio",
            "Operating income over interest expense, trailing twelve months",
            True,
        ),
        (
            "current_ratio",
            "B",
            "ratio",
            "Current assets over current liabilities",
            True,
        ),
        (
            "cash_to_debt",
            "B",
            "ratio",
            "Cash and short-term investments over total debt",
            True,
        ),
        (
            "altman_z",
            "A",
            "score",
            "Altman Z-score (public manufacturer formula)",
            True,
        ),
        (
            "share_count_change_1y",
            "A",
            "ratio",
            "Diluted share count over one year earlier minus one (dilution)",
            False,
        ),
        (
            "sbc_to_revenue",
            "B",
            "ratio",
            "Share-based compensation over revenue, trailing twelve months",
            False,
        ),
        (
            "capex_to_revenue",
            "C",
            "ratio",
            "Capital expenditure over revenue, trailing twelve months",
            None,
        ),
        (
            "rd_to_revenue",
            "B",
            "ratio",
            "Research and development over revenue, trailing twelve months",
            None,
        ),
        (
            "piotroski_f",
            "A",
            "score",
            "Piotroski F-score, 0-9, from the last two fiscal years",
            True,
        ),
        (
            "mohanram_g",
            "A",
            "score",
            "Mohanram G-score adapted to 7 signals (no advertising data), sector-relative",
            True,
        ),
        (
            "profitable_years_4",
            "B",
            "count",
            "Fiscal years with positive net income among the last four",
            True,
        ),
        (
            "fcf_positive_years_4",
            "B",
            "count",
            "Fiscal years with positive free cash flow among the last four",
            True,
        ),
        (
            "dividend_payout_fcf",
            "B",
            "ratio",
            "Dividends paid over free cash flow, trailing twelve months",
            False,
        ),
        (
            "dividend_growth_3y",
            "B",
            "ratio",
            "Compound growth of dividends paid over three fiscal years",
            True,
        ),
        (
            "statement_years",
            "C",
            "count",
            "Fiscal years of annual statements available",
            True,
        ),
        (
            "statement_quarters",
            "C",
            "count",
            "Quarters of quarterly statements available",
            True,
        ),
    ],
)
MANUFACTURER_Z = (1.2, 1.4, 3.3, 0.6, 1.0)
PERFORMANCE_WINDOW_YEARS = 4


def _piotroski(frame: pd.DataFrame) -> pd.Series:
    def _score(rows: pd.DataFrame) -> float:
        if len(rows) < 2:
            return np.nan
        now, before = rows.iloc[-1], rows.iloc[-2]
        assets_now, assets_before = now.get("total_assets"), before.get("total_assets")
        if not assets_now or not assets_before:
            return np.nan
        roa_now, roa_before = (
            now.get("net_income", np.nan) / assets_now,
            before.get("net_income", np.nan) / assets_before,
        )
        cfo = now.get("operating_cash_flow", np.nan)
        signals = [
            roa_now > 0,
            cfo > 0,
            roa_now > roa_before,
            cfo / assets_now > roa_now,
            (now.get("long_term_debt", 0) or 0) / assets_now
            < (before.get("long_term_debt", 0) or 0) / assets_before,
            _ratio(now, "total_current_assets", "total_current_liabilities")
            > _ratio(before, "total_current_assets", "total_current_liabilities"),
            (now.get("common_stock_shares_outstanding") or 0)
            <= (before.get("common_stock_shares_outstanding") or 0),
            _ratio(now, "gross_profit", "total_revenue")
            > _ratio(before, "gross_profit", "total_revenue"),
            now.get("total_revenue", np.nan) / assets_now
            > before.get("total_revenue", np.nan) / assets_before,
        ]
        return float(sum(bool(s) for s in signals if not pd.isna(s)))

    if frame.empty:
        return pd.Series(dtype=float)
    scores = {symbol: _score(rows) for symbol, rows in frame.groupby("symbol")}
    return pd.Series(scores, dtype=float).rename_axis("symbol")


def _ratio(row: pd.Series, numerator: str, denominator: str) -> float:
    value = row.get(denominator)
    if not value or pd.isna(value):
        return np.nan
    return float(row.get(numerator, np.nan)) / float(value)


def _mohanram(frame: pd.DataFrame, sector: pd.Series) -> pd.Series:
    """Seven binary signals, each judged against the sector median (advertising is not available)."""
    if frame.empty:
        return pd.Series(dtype=float)
    grouped = frame.groupby("symbol")
    assets = grouped["total_assets"].last().replace(0.0, np.nan)
    per_symbol = pd.DataFrame(
        {
            "roa": grouped["net_income"].last() / assets,
            "cfo_assets": grouped["operating_cash_flow"].last() / assets,
            "accrual_ok": grouped["operating_cash_flow"].last()
            > grouped["net_income"].last(),
            "roa_var": (
                grouped["net_income"].apply(
                    lambda s: s.tail(PERFORMANCE_WINDOW_YEARS).std()
                )
                / assets
            ),
            "growth_var": grouped["total_revenue"].apply(
                lambda s: s.tail(PERFORMANCE_WINDOW_YEARS).pct_change().std()
            ),
            "rd_assets": (
                grouped["research_and_development"].last() / assets
                if "research_and_development" in frame
                else np.nan
            ),
            "capex_assets": grouped["capital_expenditure"].last().abs() / assets,
        }
    )
    peers = sector.reindex(per_symbol.index).fillna("_all")
    score = pd.Series(0.0, index=per_symbol.index)
    available = pd.Series(0, index=per_symbol.index)
    for column, higher in (
        ("roa", True),
        ("cfo_assets", True),
        ("roa_var", False),
        ("growth_var", False),
        ("rd_assets", True),
        ("capex_assets", True),
    ):
        values = per_symbol[column]
        median = values.groupby(peers).transform("median")
        passed = (values > median) if higher else (values < median)
        score += passed.astype(float).where(values.notna(), 0.0)
        available += values.notna().astype(int)
    score += per_symbol["accrual_ok"].astype(float)
    available += 1
    return score.where(available >= 4)


def _count_positive(frame: pd.DataFrame, column: str, years: int = 4) -> pd.Series:
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    return frame.groupby("symbol")[column].agg(
        lambda s: float((s.dropna().tail(years) > 0).sum())
    )


def compute(ctx: DataContext) -> pd.DataFrame:  # noqa: PLR0915
    index = ctx.universe.index
    out = pd.DataFrame(index=index, columns=NAMES, dtype=float)
    annual_frame = annual(ctx)
    if annual_frame.empty and ctx.metrics_quarterly.empty:
        return out

    revenue = ttm(ctx, "total_revenue")
    revenue_prev = ttm(ctx, "total_revenue", 4)
    net_income = ttm(ctx, "net_income")
    operating_income = ttm(ctx, "operating_income")
    cfo = ttm(ctx, "operating_cash_flow")
    capex = ttm(ctx, "capital_expenditure").abs()
    fcf = cfo - capex
    gross = ttm(ctx, "gross_profit")
    out["gross_margin"] = safe_div(gross, revenue)
    out["operating_margin"] = safe_div(operating_income, revenue)
    out["net_margin"] = safe_div(net_income, revenue)
    out["fcf_margin"] = safe_div(fcf, revenue)
    out["gross_margin_trend"] = out["gross_margin"] - safe_div(
        ttm(ctx, "gross_profit", 4), revenue_prev
    )
    out["operating_margin_trend"] = out["operating_margin"] - safe_div(
        ttm(ctx, "operating_income", 4), revenue_prev
    )

    equity = latest_balance(ctx, "total_shareholder_equity")
    assets = latest_balance(ctx, "total_assets")
    debt = latest_balance(ctx, "short_long_term_debt_total").fillna(
        latest_balance(ctx, "long_term_debt").fillna(0.0)
        + latest_balance(ctx, "current_debt").fillna(0.0)
    )
    cash = latest_balance(ctx, "cash_and_short_term_investments").fillna(
        latest_balance(ctx, "cash_and_cash_equivalents_at_carrying_value")
    )
    out["roe"] = safe_div(net_income, equity.where(equity > 0))
    out["roa"] = safe_div(net_income, assets)
    tax_rate = (
        last_valid(ctx.metrics_annual, "tax_rate")
        .reindex(index)
        .fillna(0.21)
        .clip(0, 0.5)
    )
    invested = (equity + debt.fillna(0.0) - cash.fillna(0.0)).where(lambda s: s > 0)
    out["roic"] = safe_div(operating_income * (1 - tax_rate), invested)
    out["fcf_conversion"] = safe_div(fcf, net_income.where(net_income > 0))
    out["accruals_ratio"] = safe_div(net_income - cfo, assets)
    ebitda = ttm(ctx, "ebitda").fillna(
        operating_income + ttm(ctx, "depreciation_amortization_depletion").fillna(0.0)
    )
    out["net_debt_to_ebitda"] = safe_div(
        debt.fillna(0.0) - cash.fillna(0.0), ebitda.where(ebitda > 0)
    )
    out["interest_coverage"] = safe_div(
        operating_income, ttm(ctx, "interest_expense").abs()
    ).clip(upper=100)
    out["current_ratio"] = safe_div(
        latest_balance(ctx, "total_current_assets"),
        latest_balance(ctx, "total_current_liabilities"),
    )
    out["cash_to_debt"] = safe_div(cash, debt.where(debt > 0))

    liabilities = latest_balance(ctx, "total_liabilities")
    working_capital = latest_balance(ctx, "total_current_assets") - latest_balance(
        ctx, "total_current_liabilities"
    )
    retained = latest_balance(ctx, "retained_earnings")
    market_cap = ctx.universe["market_capitalization"].reindex(index)
    a, b, c, d, e = MANUFACTURER_Z
    out["altman_z"] = (
        a * safe_div(working_capital, assets)
        + b * safe_div(retained, assets)
        + c * safe_div(operating_income, assets)
        + d * safe_div(market_cap, liabilities)
        + e * safe_div(revenue, assets)
    )

    shares_now = ttm(ctx, "diluted_average_shares") / 4
    shares_prev = ttm(ctx, "diluted_average_shares", 4) / 4
    fallback_now = nth_last(annual_frame, "common_stock_shares_outstanding", 0).reindex(
        index
    )
    fallback_prev = nth_last(
        annual_frame, "common_stock_shares_outstanding", 1
    ).reindex(index)
    out["share_count_change_1y"] = growth(shares_now, shares_prev).fillna(
        growth(fallback_now, fallback_prev)
    )
    out["sbc_to_revenue"] = safe_div(ttm(ctx, "stock_based_compensation"), revenue)
    out["capex_to_revenue"] = safe_div(capex, revenue)
    out["rd_to_revenue"] = safe_div(ttm(ctx, "research_and_development"), revenue)
    out["piotroski_f"] = _piotroski(annual_frame).reindex(index)
    out["mohanram_g"] = _mohanram(annual_frame, ctx.universe["sector"]).reindex(index)
    out["profitable_years_4"] = _count_positive(annual_frame, "net_income").reindex(
        index
    )
    if not annual_frame.empty and "operating_cash_flow" in annual_frame:
        annual_capex: pd.Series | float = (
            annual_frame["capital_expenditure"].abs()
            if "capital_expenditure" in annual_frame
            else 0.0
        )
        annual_fcf = annual_frame.assign(
            _fcf=annual_frame["operating_cash_flow"] - annual_capex
        )
        out["fcf_positive_years_4"] = _count_positive(annual_fcf, "_fcf").reindex(index)
    dividends = ttm(ctx, "cash_dividends_paid").abs()
    out["dividend_payout_fcf"] = safe_div(dividends, fcf.where(fcf > 0))
    div_now = nth_last(annual_frame, "cash_dividends_paid", 0).abs().reindex(index)
    div_3 = nth_last(annual_frame, "cash_dividends_paid", 3).abs().reindex(index)
    out["dividend_growth_3y"] = (div_now / div_3.replace(0.0, np.nan)).where(
        div_3 > 0
    ) ** (1 / 3) - 1
    out["statement_years"] = (
        ctx.metrics_annual.groupby("symbol").size().reindex(index)
        if not ctx.metrics_annual.empty
        else 0.0
    )
    out["statement_quarters"] = (
        ctx.metrics_quarterly.groupby("symbol").size().reindex(index)
        if not ctx.metrics_quarterly.empty
        else 0.0
    )
    return out.replace([np.inf, -np.inf], np.nan)
