"""Market-regime KPIs per region: trend and breadth of the universe, volatility, credit, rates, growth
/ inflation, factors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.group.aggregate import MONTH, QUARTER, YEAR, cap_weighted_index
from bullish.kpi.registry import declare_many

BLOCK = "regime"
ALL = "ALL"
NAMES = declare_many(
    "market",
    BLOCK,
    [
        ("members", "C", "count", "Equities in the region's universe", None),
        (
            "trend_vs_sma200",
            "B",
            "ratio",
            "Cap-weighted index over its 200-day average minus one",
            True,
        ),
        (
            "sma50_vs_sma200",
            "B",
            "ratio",
            "50-day over 200-day average of the index minus one",
            True,
        ),
        ("ret_1m", "B", "ratio", "Cap-weighted index return, 1 month", True),
        ("ret_3m", "B", "ratio", "Cap-weighted index return, 3 months", True),
        ("ret_12m", "B", "ratio", "Cap-weighted index return, 12 months", True),
        (
            "breadth_above_200",
            "B",
            "ratio",
            "Share of equities above their 200-day average",
            True,
        ),
        (
            "breadth_above_50",
            "B",
            "ratio",
            "Share of equities above their 50-day average",
            True,
        ),
        (
            "pct_positive_3m",
            "B",
            "ratio",
            "Share of equities with a positive 3-month return",
            True,
        ),
        (
            "new_highs_minus_lows",
            "B",
            "ratio",
            "Share near a 52-week high minus share near a 52-week low",
            True,
        ),
        (
            "realized_vol_20d",
            "B",
            "ratio",
            "Annualised volatility of the index, 20 days",
            False,
        ),
        (
            "equal_vs_cap_weight_3m",
            "B",
            "ratio",
            "Equal-weight minus cap-weight 3-month return (breadth of leadership)",
            True,
        ),
        (
            "small_vs_large_3m",
            "B",
            "ratio",
            "Bottom-tercile minus top-tercile market cap 3-month return",
            None,
        ),
        (
            "cyclicals_vs_defensives_3m",
            "B",
            "ratio",
            "Cyclical sectors minus defensive sectors, 3-month return",
            True,
        ),
        (
            "copper_vs_gold_3m",
            "B",
            "ratio",
            "Copper over gold, 3-month change (growth appetite)",
            True,
        ),
        ("vix", "B", "index", "CBOE volatility index, latest", False),
        ("vix_pct_1y", "B", "pct", "Percentile of the VIX inside the last year", False),
        ("hy_spread", "A", "pp", "High-yield OAS, latest", False),
        (
            "hy_spread_change_3m",
            "A",
            "pp",
            "Change of the high-yield OAS over 3 months",
            False,
        ),
        ("ig_spread", "A", "pp", "Investment-grade OAS, latest", False),
        (
            "nfci",
            "B",
            "index",
            "Chicago Fed financial conditions index (positive = tight)",
            False,
        ),
        ("curve_10_2", "B", "pp", "10-year minus 2-year yield", True),
        ("real_y10", "B", "pct", "10-year inflation-indexed yield", False),
        (
            "policy_rate_change_6m",
            "B",
            "pp",
            "Change of the policy rate over 6 months (negative = cutting)",
            False,
        ),
        (
            "cli_change_3m",
            "B",
            "index",
            "Change of the OECD leading indicator over 3 months",
            True,
        ),
        ("pmi", "B", "index", "Manufacturing PMI, latest", True),
        ("cpi_yoy", "B", "pct", "Headline inflation, latest", False),
        (
            "cpi_change_3m",
            "B",
            "pp",
            "Change of headline inflation over 3 months",
            False,
        ),
        (
            "growth_inflation_quadrant",
            "C",
            "code",
            "1 growth up / inflation down, 2 up / up, 3 down / up, 4 down / down",
            None,
        ),
        ("recession_prob", "C", "pct", "Smoothed recession probability", False),
        ("ff_mom_3m", "A", "pct", "Momentum factor return, last 3 months", None),
        ("ff_hml_3m", "A", "pct", "Value factor return, last 3 months", None),
        ("ff_smb_3m", "A", "pct", "Size factor return, last 3 months", None),
        ("ff_rmw_3m", "A", "pct", "Profitability factor return, last 3 months", None),
    ],
)
CYCLICALS = {
    "Information Technology",
    "Consumer Discretionary",
    "Industrials",
    "Materials",
    "Financials",
}
DEFENSIVES = {"Consumer Staples", "Health Care", "Utilities"}
MACRO_ENTITY = {ALL: "USA"}


def _change(series: pd.Series, months: int) -> float:
    if len(series) <= months:
        return np.nan
    return float(series.iloc[-1] - series.iloc[-1 - months])


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else np.nan


def _sum_last(series: pd.Series, months: int) -> float:
    return float(series.iloc[-months:].sum()) if len(series) >= months else np.nan


def _macro(ctx: DataContext, entity: str) -> dict[str, float]:
    s = lambda concept: ctx.macro_series(concept, entity)  # noqa: E731
    us = lambda concept: ctx.macro_series(concept, "USA")  # noqa: E731
    vix = us("VIX")
    cli, cpi = s("CLI"), s("CPI_YOY")
    growth_up = _change(cli, 3) > 0 if len(cli) > 3 else None
    inflation_up = _change(cpi, 3) > 0 if len(cpi) > 3 else None
    quadrant = np.nan
    if growth_up is not None and inflation_up is not None:
        quadrant = float(
            {(True, False): 1, (True, True): 2, (False, True): 3, (False, False): 4}[
                (growth_up, inflation_up)
            ]
        )
    return {
        "vix": _latest(vix),
        "vix_pct_1y": (
            float((vix.iloc[-12:] < vix.iloc[-1]).mean() * 100)
            if len(vix) >= 12
            else np.nan
        ),
        "hy_spread": _latest(us("HY_SPREAD")),
        "hy_spread_change_3m": _change(us("HY_SPREAD"), 3),
        "ig_spread": _latest(us("IG_SPREAD")),
        "nfci": _latest(us("NFCI")),
        "curve_10_2": (
            _latest(s("CURVE_10_2"))
            if len(s("CURVE_10_2"))
            else _latest(s("Y10")) - _latest(s("Y2"))
        ),
        "real_y10": _latest(s("REAL_Y10")),
        "policy_rate_change_6m": _change(s("POLICY_RATE"), 6),
        "cli_change_3m": _change(cli, 3),
        "pmi": _latest(s("PMI_MFG")),
        "cpi_yoy": _latest(cpi),
        "cpi_change_3m": _change(cpi, 3),
        "growth_inflation_quadrant": quadrant,
        "recession_prob": _latest(s("RECESSION_PROB")),
        "ff_mom_3m": _sum_last(us("FF_MOM"), 3),
        "ff_hml_3m": _sum_last(us("FF_HML"), 3),
        "ff_smb_3m": _sum_last(us("FF_SMB"), 3),
        "ff_rmw_3m": _sum_last(us("FF_RMW"), 3),
    }


def _return(index: pd.Series, days: int) -> float:
    level = (1 + index).cumprod()
    return (
        float(level.iloc[-1] / level.iloc[-1 - days] - 1)
        if len(level) > days
        else np.nan
    )


def _copper_vs_gold(ctx: DataContext) -> float:
    commodities = ctx.commodities
    close = ctx.commodity_close
    if close.empty or commodities.empty:
        return np.nan
    symbols = commodities.set_index("code")["yfinance_symbol"]
    copper, gold = symbols.get("COPPER"), symbols.get("GOLD")
    if not copper or not gold or copper not in close or gold not in close:
        return np.nan
    ratio = (close[copper] / close[gold]).ffill().dropna()
    return (
        float(ratio.iloc[-1] / ratio.iloc[-1 - QUARTER] - 1)
        if len(ratio) > QUARTER
        else np.nan
    )


def _region_row(
    ctx: DataContext, equity: pd.DataFrame, members: list[str], entity: str
) -> dict[str, float]:
    close = ctx.close.reindex(columns=members).ffill()
    rows = equity.reindex(members)
    index = cap_weighted_index(close, rows["market_cap_base"])
    equal = close.pct_change().mean(axis=1).fillna(0.0)
    level = (1 + index).cumprod()
    sma200 = level.rolling(200).mean().iloc[-1]
    sma50 = level.rolling(50).mean().iloc[-1]
    window = close.iloc[-YEAR:]
    last = window.iloc[-1]
    caps = rows["market_cap_base"]
    terciles = (
        pd.qcut(caps.rank(method="first"), 3, labels=False)
        if caps.notna().sum() >= 6
        else None
    )
    small_large = np.nan
    if terciles is not None:
        small = rows.loc[terciles == 0, "ret_3m"].mean()
        large = rows.loc[terciles == 2, "ret_3m"].mean()
        small_large = float(small - large)
    sectors = ctx.universe["sector"].reindex(members)
    cyc = rows.loc[sectors.isin(CYCLICALS), "ret_3m"].mean()
    dfn = rows.loc[sectors.isin(DEFENSIVES), "ret_3m"].mean()
    return {
        "members": float(len(members)),
        "trend_vs_sma200": (
            float(level.iloc[-1] / sma200 - 1) if not np.isnan(sma200) else np.nan
        ),
        "sma50_vs_sma200": (
            float(sma50 / sma200 - 1) if not np.isnan(sma200) else np.nan
        ),
        "ret_1m": _return(index, MONTH),
        "ret_3m": _return(index, QUARTER),
        "ret_12m": _return(index, YEAR),
        "breadth_above_200": float((rows["price_vs_sma200"] > 0).mean()),
        "breadth_above_50": float((rows["price_vs_sma50"] > 0).mean()),
        "pct_positive_3m": float((rows["ret_3m"] > 0).mean()),
        "new_highs_minus_lows": float(
            (last >= window.max() * 0.95).mean() - (last <= window.min() * 1.05).mean()
        ),
        "realized_vol_20d": float(index.iloc[-20:].std() * np.sqrt(YEAR)),
        "equal_vs_cap_weight_3m": _return(equal, QUARTER) - _return(index, QUARTER),
        "small_vs_large_3m": small_large,
        "cyclicals_vs_defensives_3m": (
            float(cyc - dfn) if not (np.isnan(cyc) or np.isnan(dfn)) else np.nan
        ),
        "copper_vs_gold_3m": _copper_vs_gold(ctx),
    } | _macro(ctx, MACRO_ENTITY.get(entity, entity))


def compute(ctx: DataContext, equity: pd.DataFrame) -> pd.DataFrame:
    """One row for ``ALL`` and one per country with at least five priced equities."""
    rows: dict[str, dict[str, float]] = {}
    priced = [
        s
        for s in ctx.universe.index
        if s in ctx.close.columns and ctx.close[s].notna().any()
    ]
    if not priced:
        return pd.DataFrame(columns=NAMES)
    rows[ALL] = _region_row(ctx, equity, priced, ALL)
    for country, members in (
        ctx.universe["country"].dropna().groupby(ctx.universe["country"]).groups.items()
    ):
        symbols = [s for s in members if s in priced]
        if len(symbols) >= 5:
            rows[str(country)] = _region_row(ctx, equity, symbols, str(country))
    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=NAMES)
    frame.index.name = "market"
    return frame.astype(float)
