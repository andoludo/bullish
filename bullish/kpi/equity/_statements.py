"""Helpers shared by the statement-based KPI blocks (growth, quality, valuation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext

QUARTERS_PER_YEAR = 4


def merge_statements(
    metrics: pd.DataFrame, balance: pd.DataFrame, cash_flow: pd.DataFrame
) -> pd.DataFrame:
    """One row per (symbol, date) with the columns of all three statements."""
    frames = [
        f.drop(columns=["source", "filed_date"], errors="ignore")
        for f in (metrics, balance, cash_flow)
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["symbol", "date"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(
            frame, on=["symbol", "date"], how="outer", suffixes=("", "_dup")
        )
        for column in [c for c in merged.columns if c.endswith("_dup")]:
            base = column[: -len("_dup")]
            merged[base] = merged[base].fillna(merged[column])
            merged = merged.drop(columns=[column])
    return merged.sort_values(["symbol", "date"]).reset_index(drop=True)


def annual(ctx: DataContext) -> pd.DataFrame:
    return merge_statements(
        ctx.metrics_annual, ctx.balance_annual, ctx.cash_flow_annual
    )


def quarterly(ctx: DataContext) -> pd.DataFrame:
    return merge_statements(
        ctx.metrics_quarterly, ctx.balance_quarterly, ctx.cash_flow_quarterly
    )


def nth_last(frame: pd.DataFrame, column: str, n: int = 0) -> pd.Series:
    """Value ``n`` rows before the last row of each symbol (``n=0`` is the latest)."""
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    series = frame.groupby("symbol")[column].nth(-1 - n)
    return (
        series.set_axis(frame.loc[series.index, "symbol"])
        if "symbol" in frame
        else series
    )


def last_valid(frame: pd.DataFrame, column: str) -> pd.Series:
    """Latest non-null value per symbol."""
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    valid = frame.dropna(subset=[column])
    return valid.groupby("symbol")[column].last()


def rolling_sum(
    frame: pd.DataFrame, column: str, periods: int = QUARTERS_PER_YEAR, offset: int = 0
) -> pd.Series:
    """Sum of the ``periods`` rows ending ``offset`` rows before the last one, per symbol; NaN when short."""
    if frame.empty or column not in frame:
        return pd.Series(dtype=float)

    def _window(values: pd.Series) -> float:
        window = (
            values.iloc[len(values) - offset - periods : len(values) - offset]
            if offset
            else values.iloc[-periods:]
        )
        if len(window) < periods or window.isna().any():
            return np.nan
        return float(window.sum())

    return frame.groupby("symbol")[column].agg(_window)


def ttm(ctx: DataContext, column: str, offset_quarters: int = 0) -> pd.Series:
    """Trailing-twelve-month flow from quarterly rows, annual value as fallback."""
    quarterly_frame = quarterly(ctx)
    from_quarters = rolling_sum(
        quarterly_frame, column, QUARTERS_PER_YEAR, offset_quarters
    )
    annual_frame = annual(ctx)
    from_annual = nth_last(annual_frame, column, offset_quarters // QUARTERS_PER_YEAR)
    quarters_aligned = from_quarters.reindex(ctx.universe.index).astype(float)
    return quarters_aligned.fillna(
        from_annual.reindex(ctx.universe.index).astype(float)
    )


def latest_balance(ctx: DataContext, column: str) -> pd.Series:
    """Most recent balance-sheet value: quarterly when present, else annual."""
    from_quarters = last_valid(ctx.balance_quarterly, column)
    from_annual = last_valid(ctx.balance_annual, column)
    quarters_aligned = from_quarters.reindex(ctx.universe.index).astype(float)
    return quarters_aligned.fillna(
        from_annual.reindex(ctx.universe.index).astype(float)
    )


def free_cash_flow(frame: pd.DataFrame) -> pd.Series:
    if frame.empty or "operating_cash_flow" not in frame:
        return pd.Series(np.nan, index=frame.index)
    capex = (
        frame["capital_expenditure"].abs() if "capital_expenditure" in frame else 0.0
    )
    return frame["operating_cash_flow"] - capex


def safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Growth rate that keeps its sign meaningful when the base is negative."""
    previous = previous.replace(0.0, np.nan)
    return ((current - previous) / previous.abs()).replace([np.inf, -np.inf], np.nan)
