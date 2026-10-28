"""ETF vehicle KPIs: cost, size and flows, liquidity, tracking, concentration, risk-adjusted return."""

from __future__ import annotations

import datetime
import json

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext, returns_over
from bullish.kpi.registry import declare_many
from bullish.utils.typed import as_float

BLOCK = "vehicle"
MONTH, QUARTER, YEAR = 21, 63, 252
NAMES = declare_many(
    "etf",
    BLOCK,
    [
        ("expense_ratio", "A", "ratio", "Annual report expense ratio", False),
        (
            "aum",
            "B",
            "currency",
            "Total net assets, latest, in the base currency",
            True,
        ),
        (
            "aum_change_3m",
            "B",
            "ratio",
            "Total net assets over their value three months earlier minus one (flows + performance)",
            True,
        ),
        (
            "dollar_volume_60d",
            "B",
            "currency",
            "Average daily traded value over 60 days, base currency",
            True,
        ),
        (
            "premium_to_nav",
            "B",
            "ratio",
            "Last close over the reported NAV minus one",
            None,
        ),
        ("top10_weight", "B", "ratio", "Weight of the reported top holdings", False),
        (
            "holdings_hhi",
            "B",
            "ratio",
            "Herfindahl index of the reported top holdings (1 = one holding)",
            False,
        ),
        (
            "n_exposures",
            "C",
            "count",
            "Sector, country and commodity exposures derived for the fund",
            None,
        ),
        ("ret_3m", "A", "ratio", "Return over 3 months", True),
        ("ret_1y", "A", "ratio", "Return over 1 year", True),
        ("mom_12_1", "A", "ratio", "12-month return skipping the last month", True),
        ("vol_1y", "B", "ratio", "Annualised volatility, 1 year", False),
        (
            "max_drawdown_1y",
            "B",
            "ratio",
            "Worst peak-to-trough loss over 1 year",
            True,
        ),
        ("sharpe_1y", "B", "ratio", "Annualised return over volatility, 1 year", True),
        (
            "dist_52w_high",
            "B",
            "ratio",
            "Last close over the 52-week high minus one",
            True,
        ),
        (
            "tracking_difference_1y",
            "A",
            "ratio",
            "Fund 1-year return minus the best-correlated index return",
            True,
        ),
        (
            "tracking_error_1y",
            "A",
            "ratio",
            "Annualised volatility of the daily return difference to that index",
            False,
        ),
        (
            "benchmark_correlation",
            "C",
            "ratio",
            "Correlation of daily returns with the best-correlated index",
            None,
        ),
        (
            "beta_vs_benchmark",
            "C",
            "ratio",
            "Beta of daily returns to that index",
            None,
        ),
        (
            "dividend_yield",
            "B",
            "ratio",
            "Trailing dividend yield reported by the provider",
            True,
        ),
        ("age_years", "C", "years", "Years since inception", True),
        ("price_history_days", "C", "days", "Trading days of price history", True),
    ],
)


def _benchmark_stats(
    fund: pd.Series, indices: pd.DataFrame
) -> tuple[float, float, float, float]:
    """Tracking difference, tracking error, correlation and beta against the best-correlated index."""
    if indices.empty:
        return np.nan, np.nan, np.nan, np.nan
    fund_returns = fund.ffill().pct_change().iloc[-YEAR:]
    index_returns = indices.ffill().pct_change().reindex(fund_returns.index)
    correlations = index_returns.corrwith(fund_returns)
    if correlations.dropna().empty:
        return np.nan, np.nan, np.nan, np.nan
    best = correlations.idxmax()
    benchmark = index_returns[best]
    aligned = pd.concat([fund_returns, benchmark], axis=1).dropna()
    if len(aligned) < QUARTER:
        return np.nan, np.nan, float(correlations[best]), np.nan
    fund_col = aligned.iloc[:, 0].astype(float)
    index_col = aligned.iloc[:, 1].astype(float)
    diff = fund_col - index_col
    fund_total = as_float((1 + fund_col).prod()) - 1
    index_total = as_float((1 + index_col).prod()) - 1
    index_var = as_float(index_col.var())
    beta = as_float(aligned.cov().iloc[0, 1]) / index_var if index_var else np.nan
    return (
        fund_total - index_total,
        float(diff.std() * np.sqrt(YEAR)),
        float(correlations[best]),
        beta,
    )


def _aum_change(snapshots: pd.DataFrame, symbol: str, as_of: pd.Timestamp) -> float:
    rows = (
        snapshots[(snapshots["symbol"] == symbol)]
        .dropna(subset=["total_net_assets"])
        .sort_values("as_of")
    )
    if rows.empty:
        return np.nan
    latest = float(rows["total_net_assets"].iloc[-1])
    earlier = rows[rows["as_of"] <= as_of - pd.Timedelta(days=80)]
    if earlier.empty or not earlier["total_net_assets"].iloc[-1]:
        return np.nan
    return latest / float(earlier["total_net_assets"].iloc[-1]) - 1


def _holdings_concentration(holdings: object) -> tuple[float, float]:
    if isinstance(holdings, str):
        try:
            holdings = json.loads(holdings)
        except ValueError:
            return np.nan, np.nan
    if not isinstance(holdings, dict) or not holdings:
        return np.nan, np.nan
    weights = pd.Series(
        {k: float(v) for k, v in holdings.items() if v is not None}, dtype=float
    )
    weights = weights / 100 if weights.max() > 1.5 else weights
    return float(weights.sum()), float((weights**2).sum())


def compute(ctx: DataContext) -> pd.DataFrame:
    etfs = ctx.etf_universe
    out = pd.DataFrame(index=etfs.index, columns=NAMES, dtype=float)
    if etfs.empty:
        out.index.name = "etf"
        return out
    close = ctx.etf_close.ffill()
    volume = ctx.etf_volume
    currencies = etfs["currency"]
    out["expense_ratio"] = etfs["annual_report_expense_ratio"].astype(float)
    out["aum"] = ctx.to_base(etfs["total_net_assets"].astype(float), currencies)
    snapshots = ctx.etf_snapshots
    if not snapshots.empty:
        out["aum_change_3m"] = pd.Series(
            {s: _aum_change(snapshots, s, ctx.as_of_ts) for s in etfs.index},
            dtype=float,
        )
    if not close.empty:
        last = close.iloc[-1]
        dollar = (close * volume.reindex(columns=close.columns)).iloc[-60:].mean()
        out["dollar_volume_60d"] = ctx.to_base(
            dollar, currencies.reindex(dollar.index)
        ).reindex(out.index)
        out["premium_to_nav"] = (
            last / etfs["nav_price"].astype(float).replace(0.0, np.nan) - 1
        ).reindex(out.index)
        out["ret_3m"] = returns_over(close, QUARTER).reindex(out.index)
        out["ret_1y"] = returns_over(close, YEAR).reindex(out.index)
        if len(close) > YEAR:
            out["mom_12_1"] = (
                close.iloc[-1 - MONTH] / close.iloc[-1 - YEAR] - 1
            ).reindex(out.index)
        returns = close.pct_change().iloc[-YEAR:]
        out["vol_1y"] = (returns.std() * np.sqrt(YEAR)).reindex(out.index)
        window = close.iloc[-YEAR:]
        out["max_drawdown_1y"] = (window / window.cummax() - 1).min().reindex(out.index)
        out["sharpe_1y"] = (
            returns.mean() / returns.std().replace(0.0, np.nan) * np.sqrt(YEAR)
        ).reindex(out.index)
        out["dist_52w_high"] = (last / window.max() - 1).reindex(out.index)
        out["price_history_days"] = close.notna().sum().reindex(out.index)
        indices = ctx.index_close
        stats = {
            symbol: _benchmark_stats(close[symbol], indices) for symbol in close.columns
        }
        stats_frame = pd.DataFrame.from_dict(
            stats, orient="index", columns=["td", "te", "corr", "beta"]
        )
        out["tracking_difference_1y"] = stats_frame["td"].reindex(out.index)
        out["tracking_error_1y"] = stats_frame["te"].reindex(out.index)
        out["benchmark_correlation"] = stats_frame["corr"].reindex(out.index)
        out["beta_vs_benchmark"] = stats_frame["beta"].reindex(out.index)
    concentration = pd.DataFrame(
        [_holdings_concentration(h) for h in etfs["holding_percent"]],
        index=etfs.index,
        columns=["top", "hhi"],
    )
    out["top10_weight"] = concentration["top"]
    out["holdings_hhi"] = concentration["hhi"]
    exposures = ctx.etf_exposures
    if not exposures.empty:
        out["n_exposures"] = (
            exposures.groupby("symbol").size().reindex(out.index).fillna(0.0)
        )
    out["dividend_yield"] = (
        etfs["trailing_annual_dividend_yield"]
        .astype(float)
        .fillna(etfs["yield_"].astype(float))
    )
    raw_inception = pd.to_numeric(etfs["fund_inception_date"], errors="coerce")
    inception = pd.to_datetime(raw_inception, unit="s", errors="coerce")
    out["age_years"] = (
        (pd.Timestamp(datetime.date.today()) - inception).dt.days / 365.25
    ).reindex(out.index)
    out.index.name = "etf"
    return out.replace([np.inf, -np.inf], np.nan).astype(float)
