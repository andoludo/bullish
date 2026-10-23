"""Rank backtest: does a KPI or composite sort future returns?

For every stored KPI run date, the forward return of each equity over the
horizon is taken from the price panel, then the Spearman rank correlation
(information coefficient) between the KPI value and the forward return is
computed, plus the spread between the top and bottom quantile. Runs are
point-in-time by construction, so the test is honest to what was known.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.database.crud import BullishDb
from bullish.kpi import engine
from bullish.kpi.registry import REGISTRY

logger = logging.getLogger(__name__)
TRADING_DAYS_PER_MONTH = 21


class RankBacktest(BaseModel):
    name: str
    entity_type: str = "equity"
    horizon_months: int
    dates: int = Field(description="Run dates with a full forward window")
    observations: int
    mean_ic: float | None = Field(
        description="Average Spearman rank correlation with the forward return"
    )
    ic_t_stat: float | None
    hit_rate: float | None = Field(description="Share of dates with a positive IC")
    top_minus_bottom: float | None = Field(
        description="Average forward return of the top quantile minus the bottom quantile"
    )
    quantiles: int
    direction_applied: bool = Field(
        description="True when a lower-is-better KPI was sign-flipped before testing"
    )


def forward_returns(
    db: BullishDb,
    settings: BullishSettings,
    dates: list[datetime.date],
    horizon_months: int,
) -> pd.DataFrame:
    """``run_date x symbol`` forward returns over the horizon, NaN when the window is not over yet."""
    if not dates:
        return pd.DataFrame()
    end = max(dates) + datetime.timedelta(days=31 * horizon_months + 10)
    ctx = DataContext(
        db=db, settings=settings.model_copy(update={"lookback_years": 20}), as_of=end
    )
    close = ctx.close.ffill()
    rows = {}
    days = horizon_months * TRADING_DAYS_PER_MONTH
    for date in dates:
        position = close.index.searchsorted(pd.Timestamp(date), side="right") - 1
        if position < 0 or position + days >= len(close):
            continue
        start, stop = close.iloc[position], close.iloc[position + days]
        rows[date] = stop / start - 1
    return pd.DataFrame.from_dict(rows, orient="index")


def _spearman(x: pd.Series, y: pd.Series) -> float:
    joined = pd.concat([x, y], axis=1).dropna()
    if len(joined) < 5:
        return np.nan
    return float(joined.iloc[:, 0].rank().corr(joined.iloc[:, 1].rank()))


def _quantile_spread(values: pd.Series, returns: pd.Series, quantiles: int) -> float:
    joined = pd.concat([values, returns], axis=1).dropna()
    if len(joined) < quantiles * 2:
        return np.nan
    bucket = pd.qcut(joined.iloc[:, 0].rank(method="first"), quantiles, labels=False)
    top = joined.iloc[:, 1][bucket == quantiles - 1].mean()
    bottom = joined.iloc[:, 1][bucket == 0].mean()
    return float(top - bottom)


def backtest_values(  # noqa: PLR0913
    name: str,
    values_by_date: dict[datetime.date, pd.Series],
    forward: pd.DataFrame,
    horizon_months: int,
    quantiles: int = 5,
    lower_is_better: bool = False,
) -> RankBacktest:
    ics, spreads, observations = [], [], 0
    for date, values in values_by_date.items():
        if date not in forward.index:
            continue
        signal = -values if lower_is_better else values
        returns = forward.loc[date].reindex(signal.index)
        ic = _spearman(signal, returns)
        if np.isnan(ic):
            continue
        ics.append(ic)
        spreads.append(_quantile_spread(signal, returns, quantiles))
        observations += int(pd.concat([signal, returns], axis=1).dropna().shape[0])
    ic_array = np.array(ics)
    t_stat = (
        float(ic_array.mean() / ic_array.std(ddof=1) * np.sqrt(len(ic_array)))
        if len(ic_array) > 1 and ic_array.std(ddof=1) > 0
        else None
    )
    return RankBacktest(
        name=name,
        horizon_months=horizon_months,
        dates=len(ics),
        observations=observations,
        mean_ic=float(ic_array.mean()) if len(ics) else None,
        ic_t_stat=t_stat,
        hit_rate=float((ic_array > 0).mean()) if len(ics) else None,
        top_minus_bottom=(
            float(np.nanmean(spreads))
            if spreads and not np.all(np.isnan(spreads))
            else None
        ),
        quantiles=quantiles,
        direction_applied=lower_is_better,
    )


def backtest_kpi(
    db: BullishDb,
    settings: BullishSettings,
    name: str,
    horizon_months: int = 3,
    quantiles: int = 5,
) -> RankBacktest:
    """Rank IC of one stored equity KPI across every run date."""
    definition = REGISTRY.get("equity", name)
    dates = db.kpi_dates("equity")
    forward = forward_returns(db, settings, dates, horizon_months)
    values = {
        date: db.read_kpi_frame("equity", date)[name]
        for date in dates
        if name in db.read_kpi_frame("equity", date)
    }
    return backtest_values(
        name,
        values,
        forward,
        horizon_months,
        quantiles,
        lower_is_better=definition.higher_is_better is False,
    )


def backtest_score(
    db: BullishDb,
    settings: BullishSettings,
    composite: str,
    horizon_months: int = 3,
    quantiles: int = 5,
) -> RankBacktest:
    """Rank IC of a stored equity composite across every run date (eligible names only)."""
    dates = db.kpi_dates("equity")
    forward = forward_returns(db, settings, dates, horizon_months)
    values = {}
    for date in dates:
        scores = db.read_scores("equity", composite, date)
        if not scores.empty and scores.attrs["as_of"] == date:
            values[date] = scores[scores["eligible"]]["value"]
    return backtest_values(composite, values, forward, horizon_months, quantiles)


def build_history(  # noqa: PLR0913
    db: BullishDb,
    settings: BullishSettings,
    start: datetime.date,
    end: datetime.date,
    step_months: int = 1,
    countries: list[str] | None = None,
) -> list[datetime.date]:
    """Run the KPI engine on a grid of past dates so that series and backtests have history."""
    dates = []
    current = pd.Timestamp(start)
    while current <= pd.Timestamp(end):
        as_of = current.date()
        engine.run(db, settings, as_of=as_of, countries=countries)
        dates.append(as_of)
        current = current + pd.DateOffset(months=step_months)
    return dates
