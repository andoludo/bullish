"""How each equity is exposed to commodities, rates, the dollar and its listing currency."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext, returns_over
from bullish.kpi.registry import declare_many

BLOCK = "macro_exposure"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "commodity_tailwind_3m",
            "C",
            "score",
            "Sum over the sector's commodity sensitivities of sign x strength x 3-month commodity return",
            True,
        ),
        (
            "rate_beta_1y",
            "C",
            "ratio",
            "Beta of weekly returns to weekly changes of the domestic 10-year yield",
            None,
        ),
        (
            "usd_beta_1y",
            "C",
            "ratio",
            "Beta of weekly returns to weekly changes of the trade-weighted dollar",
            None,
        ),
        (
            "fx_momentum_3m",
            "C",
            "ratio",
            "Listing currency against the base currency over 3 months (positive = stronger)",
            True,
        ),
    ],
)
STRENGTH = {"high": 1.0, "medium": 0.6, "low": 0.3}
QUARTER = 63
WEEK = 5


def _commodity_returns(ctx: DataContext) -> pd.Series:
    close = ctx.commodity_close
    commodities = ctx.commodities
    if close.empty or commodities.empty:
        return pd.Series(dtype=float)
    returns = returns_over(close, QUARTER)
    by_symbol = commodities.dropna(subset=["yfinance_symbol"]).set_index(
        "yfinance_symbol"
    )["code"]
    return pd.Series(
        {code: returns.get(symbol, np.nan) for symbol, code in by_symbol.items()},
        dtype=float,
    ).dropna()


def _tailwind(ctx: DataContext) -> pd.Series:
    returns = _commodity_returns(ctx)
    sensitivities = ctx.sector_sensitivities
    if returns.empty or sensitivities.empty:
        return pd.Series(np.nan, index=ctx.universe.index)
    sensitivities = sensitivities.assign(
        _weight=sensitivities["direction"]
        * sensitivities["strength"].map(STRENGTH).fillna(0.6),
        _ret=sensitivities["commodity"].map(returns),
    ).dropna(subset=["_ret"])
    by_group = (
        (sensitivities["_weight"] * sensitivities["_ret"])
        .groupby(sensitivities["sector"])
        .sum()
    )
    universe = ctx.universe
    industry_score = universe["industry"].map(by_group)
    sector_score = universe["sector"].map(by_group)
    return industry_score.fillna(sector_score).astype(float)


def _weekly_beta(ctx: DataContext, driver: pd.Series) -> pd.Series:
    close = ctx.close.ffill()
    if close.empty or driver.empty:
        return pd.Series(np.nan, index=ctx.universe.index)
    weekly = close.iloc[::-WEEK][::-1].iloc[-52:]
    returns = weekly.pct_change().dropna(how="all")
    driver_weekly = driver.reindex(returns.index, method="ffill").diff()
    aligned = returns.loc[driver_weekly.notna()]
    x = driver_weekly.loc[aligned.index]
    if len(x) < 20 or float(x.var()) == 0:
        return pd.Series(np.nan, index=ctx.universe.index)
    cov = aligned.sub(aligned.mean()).mul(x - x.mean(), axis=0).sum() / (len(x) - 1)
    return (cov / x.var()).reindex(ctx.universe.index)


def _fx_momentum(ctx: DataContext) -> pd.Series:
    fx = ctx.fx
    base = ctx.settings.base_currency.upper()
    if fx.empty:
        return pd.Series(np.nan, index=ctx.universe.index)
    base_per_usd = fx[base] if base in fx.columns else pd.Series(1.0, index=fx.index)
    out = {}
    for symbol, currency in ctx.universe["currency"].items():
        code = (currency or "").upper()
        if code == base:
            out[symbol] = 0.0
            continue
        ccy_per_usd = (
            fx[code]
            if code in fx.columns
            else (pd.Series(1.0, index=fx.index) if code == "USD" else None)
        )
        if ccy_per_usd is None:
            continue
        cross = (
            base_per_usd / ccy_per_usd
        ).dropna()  # base units per one listing-currency unit
        if len(cross) > QUARTER:
            out[symbol] = float(cross.iloc[-1] / cross.iloc[-1 - QUARTER] - 1)
    return pd.Series(out, dtype=float).reindex(ctx.universe.index)


def compute(ctx: DataContext) -> pd.DataFrame:
    out = pd.DataFrame(index=ctx.universe.index, columns=NAMES, dtype=float)
    out["commodity_tailwind_3m"] = _tailwind(ctx)
    countries = ctx.universe["country"].fillna("USA")
    for name, concept in (("rate_beta_1y", "Y10"), ("usd_beta_1y", "DXY")):
        values = pd.Series(np.nan, index=ctx.universe.index)
        for country in countries.unique():
            entity = "USA" if concept == "DXY" else country
            driver = ctx.macro_series(concept, entity)
            if driver.empty and concept == "Y10":
                driver = ctx.macro_series(concept, "USA")
            members = countries[countries == country].index
            beta = _weekly_beta(ctx, driver)
            values.loc[members] = beta.reindex(members)
        out[name] = values
    out["fx_momentum_3m"] = _fx_momentum(ctx)
    return out.replace([np.inf, -np.inf], np.nan)
