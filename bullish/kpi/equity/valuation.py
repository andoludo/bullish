"""Valuation KPIs: multiples on trailing fundamentals, yields, and the multiple against its own history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.equity._statements import annual, latest_balance, safe_div, ttm
from bullish.kpi.registry import declare_many

BLOCK = "valuation"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "market_cap_base",
            "C",
            "currency",
            "Market capitalisation in the base currency",
            None,
        ),
        (
            "pe_trailing",
            "A",
            "ratio",
            "Price over trailing diluted EPS (provider value when statements are missing)",
            False,
        ),
        ("pe_forward", "B", "ratio", "Price over the provider's forward EPS", False),
        ("peg", "B", "ratio", "Trailing PEG ratio from the provider", False),
        ("ev_to_ebitda", "A", "ratio", "Enterprise value over trailing EBITDA", False),
        ("ev_to_sales", "B", "ratio", "Enterprise value over trailing revenue", False),
        (
            "price_to_book",
            "A",
            "ratio",
            "Market capitalisation over shareholder equity",
            False,
        ),
        (
            "price_to_fcf",
            "A",
            "ratio",
            "Market capitalisation over trailing free cash flow",
            False,
        ),
        (
            "fcf_yield",
            "A",
            "ratio",
            "Trailing free cash flow over market capitalisation",
            True,
        ),
        (
            "earnings_yield",
            "A",
            "ratio",
            "Trailing net income over market capitalisation",
            True,
        ),
        (
            "earnings_yield_minus_10y",
            "B",
            "pct",
            "Earnings yield minus the domestic 10-year yield, percentage points",
            True,
        ),
        (
            "dividend_yield",
            "B",
            "ratio",
            "Dividends paid over the last year over the last close",
            True,
        ),
        (
            "pe_vs_5y_z",
            "B",
            "z",
            "Trailing P/E against its own five-year monthly history, in standard deviations",
            False,
        ),
        (
            "pe_vs_5y_pct",
            "B",
            "pct",
            "Percentile of the trailing P/E inside its own five-year history",
            False,
        ),
    ],
)
MONTH = 21


def _pe_history(ctx: DataContext, eps_history: pd.DataFrame) -> pd.DataFrame:
    """Monthly trailing P/E per symbol from the close panel and the annual diluted EPS stepped forward."""
    close = ctx.close.ffill()
    if close.empty or eps_history.empty:
        return pd.DataFrame()
    monthly = close.iloc[::-MONTH][::-1]
    eps = eps_history.pivot_table(
        index="date", columns="symbol", values="diluted_eps", aggfunc="last"
    )
    eps = eps.reindex(columns=monthly.columns)
    eps_stepped = (
        eps.reindex(eps.index.union(monthly.index))
        .sort_index()
        .ffill()
        .reindex(monthly.index)
    )
    return (monthly / eps_stepped.where(eps_stepped > 0)).replace(
        [np.inf, -np.inf], np.nan
    )


def compute(ctx: DataContext) -> pd.DataFrame:
    index = ctx.universe.index
    out = pd.DataFrame(index=index, columns=NAMES, dtype=float)
    universe = ctx.universe
    price = ctx.last_price.reindex(index)
    market_cap = universe["market_capitalization"].astype(float)
    out["market_cap_base"] = ctx.to_base(market_cap, universe["currency"])

    eps_ttm = ttm(ctx, "diluted_eps")
    pe = safe_div(price, eps_ttm.where(eps_ttm > 0))
    out["pe_trailing"] = pe.fillna(universe["trailing_price_to_earnings"].astype(float))
    out["pe_forward"] = universe["forward_price_to_earnings"].astype(float)
    out["peg"] = universe["trailing_peg_ratio"].astype(float)

    debt = latest_balance(ctx, "short_long_term_debt_total").fillna(
        latest_balance(ctx, "long_term_debt").fillna(0.0)
        + latest_balance(ctx, "current_debt").fillna(0.0)
    )
    cash = latest_balance(ctx, "cash_and_short_term_investments").fillna(
        latest_balance(ctx, "cash_and_cash_equivalents_at_carrying_value")
    )
    enterprise_value = market_cap + debt.fillna(0.0) - cash.fillna(0.0)
    ebitda = ttm(ctx, "ebitda").fillna(
        ttm(ctx, "operating_income")
        + ttm(ctx, "depreciation_amortization_depletion").fillna(0.0)
    )
    revenue = ttm(ctx, "total_revenue")
    out["ev_to_ebitda"] = safe_div(enterprise_value, ebitda.where(ebitda > 0))
    out["ev_to_sales"] = safe_div(enterprise_value, revenue.where(revenue > 0))
    equity = latest_balance(ctx, "total_shareholder_equity")
    out["price_to_book"] = safe_div(market_cap, equity.where(equity > 0)).fillna(
        universe["price_to_book"].astype(float)
    )
    fcf = ttm(ctx, "operating_cash_flow") - ttm(ctx, "capital_expenditure").abs()
    net_income = ttm(ctx, "net_income")
    out["price_to_fcf"] = safe_div(market_cap, fcf.where(fcf > 0))
    out["fcf_yield"] = safe_div(fcf, market_cap)
    out["earnings_yield"] = safe_div(net_income, market_cap)

    y10 = pd.Series(
        {
            symbol: _domestic_10y(ctx, country)
            for symbol, country in universe["country"].items()
        },
        dtype=float,
    )
    out["earnings_yield_minus_10y"] = out["earnings_yield"] * 100 - y10.reindex(index)

    dividends = ctx.dividends.reindex(columns=index).fillna(0.0)
    if len(dividends):
        out["dividend_yield"] = safe_div(dividends.iloc[-252:].sum(), price).fillna(
            universe["dividend_yield"].astype(float)
        )
    else:
        out["dividend_yield"] = universe["dividend_yield"].astype(float)

    annual_frame = annual(ctx)
    history = _pe_history(
        ctx,
        (
            annual_frame[["symbol", "date", "diluted_eps"]]
            if "diluted_eps" in annual_frame
            else pd.DataFrame()
        ),
    )
    if not history.empty:
        current = out["pe_trailing"]
        mean, std = history.mean(), history.std()
        out["pe_vs_5y_z"] = (
            (current - mean.reindex(index)) / std.reindex(index).replace(0.0, np.nan)
        ).where(history.count().reindex(index) >= 12)
        rank = (
            history.lt(current, axis=1).sum() / history.count().replace(0, np.nan)
        ) * 100
        out["pe_vs_5y_pct"] = rank.reindex(index).where(
            history.count().reindex(index) >= 12
        )
    return out.replace([np.inf, -np.inf], np.nan)


def _domestic_10y(ctx: DataContext, country: str | None) -> float:
    for entity in (country, "USA"):
        if not entity:
            continue
        series = ctx.macro_series("Y10", entity)
        if len(series):
            return float(series.iloc[-1])
    return np.nan
