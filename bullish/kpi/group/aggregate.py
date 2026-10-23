"""Group-level KPIs (sector, industry, country) from the equity KPI frame and the close panel.

Two kinds of measures:

* **aggregations** of the per-equity KPIs (median, cap-weighted mean, share
  positive, sum), declared once in :data:`AGGREGATIONS`;
* **group index** measures: a cap-weighted index of the members' prices gives
  the group's own return, its relative strength against the universe and the
  RRG-style quadrant.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.registry import EntityType, Tier, declare_many

GroupType = Literal["sector", "industry", "country"]
GROUP_TYPES: tuple[GroupType, ...] = ("sector", "industry", "country")
MONTH, QUARTER, YEAR = 21, 63, 252
Aggregation = Literal["median", "cap_weighted", "share_positive", "sum", "mean"]

# (group kpi, equity kpi, aggregation, tier, unit, description, higher_is_better)
AGGREGATIONS: list[tuple[str, str, Aggregation, Tier, str, str, bool | None]] = [
    (
        "median_ret_3m",
        "ret_3m",
        "median",
        "A",
        "ratio",
        "Median 3-month return of the members",
        True,
    ),
    (
        "median_mom_12_1",
        "mom_12_1",
        "median",
        "A",
        "ratio",
        "Median 12-1 momentum of the members",
        True,
    ),
    (
        "breadth_above_200",
        "price_vs_sma200",
        "share_positive",
        "B",
        "ratio",
        "Share of members above their 200-day average",
        True,
    ),
    (
        "breadth_above_50",
        "price_vs_sma50",
        "share_positive",
        "B",
        "ratio",
        "Share of members above their 50-day average",
        True,
    ),
    (
        "pct_positive_3m",
        "ret_3m",
        "share_positive",
        "B",
        "ratio",
        "Share of members with a positive 3-month return",
        True,
    ),
    (
        "pct_golden_cross",
        "sma50_vs_sma200",
        "share_positive",
        "B",
        "ratio",
        "Share of members whose 50-day average is above the 200-day",
        True,
    ),
    (
        "median_revenue_growth",
        "revenue_ttm_growth",
        "median",
        "A",
        "ratio",
        "Median trailing revenue growth of the members",
        True,
    ),
    (
        "median_revenue_acceleration",
        "revenue_acceleration",
        "median",
        "B",
        "ratio",
        "Median revenue acceleration of the members",
        True,
    ),
    (
        "median_eps_growth",
        "eps_yoy_last_q",
        "median",
        "A",
        "ratio",
        "Median last-quarter EPS growth of the members",
        True,
    ),
    (
        "beat_rate",
        "eps_beat_rate_4q",
        "mean",
        "A",
        "ratio",
        "Average share of beats over the last four reports",
        True,
    ),
    (
        "revision_ratio",
        "target_mean_change_3m",
        "share_positive",
        "A",
        "ratio",
        "Share of members whose mean target rose over 3 months",
        True,
    ),
    (
        "median_target_upside",
        "target_upside",
        "median",
        "B",
        "ratio",
        "Median analyst upside of the members",
        True,
    ),
    (
        "median_margin_trend",
        "operating_margin_trend",
        "median",
        "B",
        "ratio",
        "Median change of the operating margin over a year",
        True,
    ),
    (
        "median_pe",
        "pe_trailing",
        "median",
        "B",
        "ratio",
        "Median trailing P/E of the members",
        False,
    ),
    (
        "median_ev_to_ebitda",
        "ev_to_ebitda",
        "median",
        "B",
        "ratio",
        "Median EV/EBITDA of the members",
        False,
    ),
    (
        "median_fcf_yield",
        "fcf_yield",
        "median",
        "A",
        "ratio",
        "Median free-cash-flow yield of the members",
        True,
    ),
    (
        "median_pe_vs_5y_z",
        "pe_vs_5y_z",
        "median",
        "B",
        "z",
        "Median P/E against own history of the members",
        False,
    ),
    (
        "inst_holders_change_sum",
        "inst_holders_change",
        "sum",
        "B",
        "count",
        "Net new 13F holders summed over the members",
        True,
    ),
    (
        "insider_buyers_sum",
        "insider_buyers_3m",
        "sum",
        "A",
        "count",
        "Distinct insider buyers summed over the members, 3 months",
        True,
    ),
    (
        "insider_sellers_sum",
        "insider_sellers_3m",
        "sum",
        "B",
        "count",
        "Distinct insider sellers summed over the members, 3 months",
        False,
    ),
    (
        "median_short_interest",
        "short_interest_pct_float",
        "median",
        "B",
        "ratio",
        "Median short interest as a share of float",
        False,
    ),
    (
        "commodity_tailwind_3m",
        "commodity_tailwind_3m",
        "median",
        "C",
        "score",
        "Median commodity tailwind of the members",
        True,
    ),
    (
        "rate_beta_1y",
        "rate_beta_1y",
        "median",
        "C",
        "ratio",
        "Median beta of the members to the 10-year yield",
        None,
    ),
    (
        "usd_beta_1y",
        "usd_beta_1y",
        "median",
        "C",
        "ratio",
        "Median beta of the members to the dollar",
        None,
    ),
    (
        "median_quality_f",
        "piotroski_f",
        "median",
        "A",
        "score",
        "Median Piotroski F-score of the members",
        True,
    ),
    (
        "market_cap_base_total",
        "market_cap_base",
        "sum",
        "C",
        "currency",
        "Total market capitalisation of the members, base currency",
        None,
    ),
]
INDEX_KPIS: list[tuple[str, Tier, str, str, bool | None]] = [
    ("members", "C", "count", "Members with a price", True),
    ("ret_1m", "A", "ratio", "Cap-weighted index return, 1 month", True),
    ("ret_3m", "A", "ratio", "Cap-weighted index return, 3 months", True),
    ("ret_6m", "A", "ratio", "Cap-weighted index return, 6 months", True),
    ("ret_12m", "A", "ratio", "Cap-weighted index return, 12 months", True),
    ("mom_12_1", "A", "ratio", "Cap-weighted index 12-1 momentum", True),
    (
        "rs_vs_market_3m",
        "A",
        "ratio",
        "Index 3-month return minus the universe cap-weighted return",
        True,
    ),
    (
        "rs_vs_market_12m",
        "A",
        "ratio",
        "Index 12-month return minus the universe cap-weighted return",
        True,
    ),
    (
        "rs_ratio",
        "B",
        "index",
        "Relative strength ratio vs the universe, 100 = in line",
        True,
    ),
    (
        "rs_momentum",
        "B",
        "index",
        "Rate of change of the relative strength ratio, 100 = flat",
        True,
    ),
    (
        "rrg_quadrant",
        "B",
        "code",
        "1 Leading, 2 Weakening, 3 Lagging, 4 Improving",
        None,
    ),
    ("vol_60d", "B", "ratio", "Annualised volatility of the index, 60 days", False),
    (
        "dispersion_3m",
        "B",
        "ratio",
        "Cross-sectional dispersion of the members' 3-month returns",
        None,
    ),
    ("drawdown_52w", "B", "ratio", "Index level over its 52-week high minus one", True),
    (
        "new_highs_minus_lows",
        "B",
        "ratio",
        "Share of members within 5 % of a 52-week high minus the share near a 52-week low",
        True,
    ),
]
NAMES_BY_TYPE: dict[str, list[str]] = {}
for _group_type in GROUP_TYPES:
    _names = declare_many(_group_type, "group_index", INDEX_KPIS)
    _names += declare_many(
        _group_type,
        "aggregate",
        [(n, t, u, d, h) for n, _, _, t, u, d, h in AGGREGATIONS],
    )
    NAMES_BY_TYPE[_group_type] = _names

NEAR = 0.05


def cap_weighted_index(close: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Daily cap-weighted return series of ``close`` columns with static ``weights`` (NaN-aware)."""
    returns = close.ffill().pct_change()
    w = weights.reindex(returns.columns).fillna(0.0).clip(lower=0.0)
    if float(w.sum()) == 0:
        w = pd.Series(1.0, index=returns.columns)
    available = returns.notna().mul(w, axis=1)
    weighted = returns.fillna(0.0).mul(w, axis=1).sum(axis=1) / available.sum(
        axis=1
    ).replace(0.0, np.nan)
    return weighted.fillna(0.0)


def _return(index: pd.Series, days: int, skip: int = 0) -> float:
    level = (1 + index).cumprod()
    if len(level) <= days:
        return np.nan
    end = level.iloc[-1 - skip] if skip else level.iloc[-1]
    return float(end / level.iloc[-1 - days] - 1)


def rrg(index: pd.Series, market: pd.Series) -> tuple[float, float, float]:
    """RS-ratio, RS-momentum and quadrant code of ``index`` against ``market``."""
    rs = (1 + index).cumprod() / (1 + market).cumprod()
    ratio = 100 * rs / rs.rolling(QUARTER).mean()
    momentum = 100 * ratio / ratio.shift(MONTH)
    r, m = float(ratio.iloc[-1]), float(momentum.iloc[-1])
    if np.isnan(r) or np.isnan(m):
        return r, m, np.nan
    quadrant = {(True, True): 1, (True, False): 2, (False, False): 3, (False, True): 4}[
        (r >= 100, m >= 100)
    ]
    return r, m, float(quadrant)


def _index_measures(
    ctx: DataContext, members: list[str], market: pd.Series, equity: pd.DataFrame
) -> dict[str, float]:
    close = ctx.close.reindex(columns=members)
    weights = equity["market_cap_base"].reindex(members)
    index = cap_weighted_index(close, weights)
    ratio, momentum, quadrant = rrg(index, market)
    level = (1 + index).cumprod()
    window = close.ffill().iloc[-YEAR:]
    last = window.iloc[-1]
    near_high = (last >= window.max() * (1 - NEAR)).mean()
    near_low = (last <= window.min() * (1 + NEAR)).mean()
    return {
        "members": float(close.notna().any().sum()),
        "ret_1m": _return(index, MONTH),
        "ret_3m": _return(index, QUARTER),
        "ret_6m": _return(index, 2 * QUARTER),
        "ret_12m": _return(index, YEAR),
        "mom_12_1": _return(index, YEAR - MONTH, skip=MONTH),
        "rs_vs_market_3m": _return(index, QUARTER) - _return(market, QUARTER),
        "rs_vs_market_12m": _return(index, YEAR) - _return(market, YEAR),
        "rs_ratio": ratio,
        "rs_momentum": momentum,
        "rrg_quadrant": quadrant,
        "vol_60d": float(index.iloc[-60:].std() * np.sqrt(YEAR)),
        "dispersion_3m": float(equity["ret_3m"].reindex(members).std()),
        "drawdown_52w": float(level.iloc[-1] / level.iloc[-YEAR:].max() - 1),
        "new_highs_minus_lows": float(near_high - near_low),
    }


def _aggregate(equity: pd.DataFrame, members: list[str]) -> dict[str, float]:
    rows = equity.reindex(members)
    weights = rows["market_cap_base"].fillna(0.0)
    out: dict[str, float] = {}
    for name, source, how, *_ in AGGREGATIONS:
        values = rows[source]
        if how == "median":
            out[name] = float(values.median())
        elif how == "mean":
            out[name] = float(values.mean())
        elif how == "share_positive":
            out[name] = (
                float((values > 0).sum() / values.notna().sum())
                if values.notna().any()
                else np.nan
            )
        elif how == "sum":
            out[name] = float(values.sum()) if values.notna().any() else np.nan
        else:
            mask = values.notna() & (weights > 0)
            out[name] = (
                float((values[mask] * weights[mask]).sum() / weights[mask].sum())
                if mask.any()
                else np.nan
            )
    return out


def market_index(ctx: DataContext, equity: pd.DataFrame) -> pd.Series:
    return cap_weighted_index(ctx.close, equity["market_cap_base"])


def compute_group(
    ctx: DataContext, equity: pd.DataFrame, group_type: GroupType, market: pd.Series
) -> pd.DataFrame:
    """One row per group value of ``group_type`` with index measures and aggregations."""
    groups = ctx.universe[group_type].dropna()
    rows: dict[str, dict[str, float]] = {}
    for group, members in groups.groupby(groups).groups.items():
        symbols = [s for s in members if s in ctx.close.columns]
        if not symbols:
            continue
        rows[str(group)] = _index_measures(ctx, symbols, market, equity) | _aggregate(
            equity, symbols
        )
    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(
        columns=NAMES_BY_TYPE[group_type]
    )
    frame.index.name = group_type
    return frame.astype(float)


def entity_types() -> tuple[EntityType, ...]:
    return GROUP_TYPES
