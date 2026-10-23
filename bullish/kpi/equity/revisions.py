"""Expectations KPIs: analyst targets and recommendations through time, earnings surprises, event timing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.registry import declare_many

BLOCK = "revisions"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "analysts",
            "C",
            "count",
            "Number of analysts behind the latest price target",
            True,
        ),
        (
            "target_upside",
            "B",
            "ratio",
            "Mean analyst price target over the last close minus one",
            True,
        ),
        (
            "target_mean_change_1m",
            "A",
            "ratio",
            "Mean price target over its value one month earlier minus one",
            True,
        ),
        (
            "target_mean_change_3m",
            "A",
            "ratio",
            "Mean price target over its value three months earlier minus one",
            True,
        ),
        (
            "target_dispersion",
            "C",
            "ratio",
            "(High minus low target) over the mean target",
            False,
        ),
        (
            "recommendation_mean",
            "B",
            "score",
            "Provider recommendation mean, 1 = strong buy ... 5 = strong sell",
            False,
        ),
        (
            "recommendation_change_3m",
            "A",
            "score",
            "Recommendation mean minus its value three months earlier",
            False,
        ),
        (
            "buy_ratio",
            "B",
            "ratio",
            "(Strong buy + buy) over all recommendations, current period",
            True,
        ),
        (
            "buy_ratio_change_3m",
            "A",
            "ratio",
            "buy_ratio minus the same ratio three periods earlier",
            True,
        ),
        (
            "eps_surprise_last",
            "A",
            "ratio",
            "Reported EPS over the estimate minus one, latest report",
            True,
        ),
        (
            "eps_surprise_mean_4q",
            "A",
            "ratio",
            "Mean EPS surprise of the last four reports",
            True,
        ),
        (
            "eps_beat_rate_4q",
            "A",
            "ratio",
            "Share of the last four reports that beat the estimate",
            True,
        ),
        (
            "revenue_surprise_last",
            "B",
            "ratio",
            "Reported revenue over the estimate minus one, latest report",
            True,
        ),
        (
            "days_since_earnings",
            "C",
            "days",
            "Days since the last earnings report",
            None,
        ),
        (
            "days_to_next_earnings",
            "C",
            "days",
            "Days until the next scheduled earnings report",
            None,
        ),
        (
            "return_since_earnings",
            "B",
            "ratio",
            "Price change since the last earnings report",
            None,
        ),
    ],
)
SURPRISE_WINDOW = 4


def _value_before(
    frame: pd.DataFrame, column: str, as_of: pd.Timestamp, days: int
) -> pd.Series:
    cutoff = as_of - pd.Timedelta(days=days)
    earlier = frame[frame["date"] <= cutoff]
    return (
        earlier.groupby("symbol")[column].last()
        if len(earlier)
        else pd.Series(dtype=float)
    )


def _surprises(earnings: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    past = earnings[
        (earnings["date"] <= as_of)
        & earnings["eps_reported"].notna()
        & earnings["eps_estimate"].notna()
    ]
    past = past.sort_values(["symbol", "date"])
    surprise = (past["eps_reported"] - past["eps_estimate"]) / past[
        "eps_estimate"
    ].abs().replace(0.0, np.nan)
    past = past.assign(_surprise=surprise)
    grouped = past.groupby("symbol")["_surprise"]
    return pd.DataFrame(
        {
            "eps_surprise_last": grouped.last(),
            "eps_surprise_mean_4q": grouped.apply(
                lambda s: s.tail(SURPRISE_WINDOW).mean()
            ),
            "eps_beat_rate_4q": grouped.apply(
                lambda s: float((s.tail(SURPRISE_WINDOW) > 0).mean())
            ),
        }
    )


def compute(ctx: DataContext) -> pd.DataFrame:  # noqa: C901, PLR0915
    index = ctx.universe.index
    out = pd.DataFrame(index=index, columns=NAMES, dtype=float)
    as_of = ctx.as_of_ts
    price = ctx.last_price.reindex(index)

    targets = ctx.price_targets
    if not targets.empty:
        targets = targets.sort_values(["symbol", "date"])
        latest = targets.groupby("symbol").last()
        out["analysts"] = latest["number_of_analysts"].reindex(index)
        out["target_upside"] = (
            latest["target_mean"].reindex(index) / price - 1
        ).replace([np.inf, -np.inf], np.nan)
        for name, days in (
            ("target_mean_change_1m", 30),
            ("target_mean_change_3m", 91),
        ):
            before = _value_before(targets, "target_mean", as_of, days).reindex(index)
            out[name] = (
                latest["target_mean"].reindex(index) / before.replace(0.0, np.nan) - 1
            ).replace([np.inf, -np.inf], np.nan)
        spread = latest["target_high"] - latest["target_low"]
        out["target_dispersion"] = (
            spread / latest["target_mean"].replace(0.0, np.nan)
        ).reindex(index)
        out["recommendation_mean"] = latest["recommendation_mean"].reindex(index)
        before = _value_before(targets, "recommendation_mean", as_of, 91).reindex(index)
        out["recommendation_change_3m"] = (
            latest["recommendation_mean"].reindex(index) - before
        )

    recs = ctx.recommendations
    if not recs.empty:
        recs = recs.sort_values(["symbol", "date"])
        latest_date = recs.groupby("symbol")["date"].transform("max")
        current = recs[recs["date"] == latest_date]
        counts = current.set_index(["symbol", "period"])[
            ["strong_buy", "buy", "hold", "sell", "strong_sell"]
        ]
        total = counts.sum(axis=1).replace(0.0, np.nan)
        buy_ratio = (counts["strong_buy"] + counts["buy"]) / total
        now = (
            buy_ratio.xs("0m", level="period")
            if "0m" in buy_ratio.index.get_level_values("period")
            else pd.Series(dtype=float)
        )
        periods = buy_ratio.index.get_level_values("period")
        earlier = (
            buy_ratio.xs("-3m", level="period")
            if "-3m" in periods
            else pd.Series(dtype=float)
        )
        out["buy_ratio"] = now.reindex(index)
        out["buy_ratio_change_3m"] = (now - earlier.reindex(now.index)).reindex(index)

    earnings = ctx.earnings
    if not earnings.empty:
        surprises = _surprises(earnings, as_of)
        for column in surprises.columns:
            out[column] = surprises[column].reindex(index)
        past = earnings[earnings["date"] <= as_of].sort_values(["symbol", "date"])
        if "revenue_reported" in past and "revenue_estimate" in past:
            last_rows = (
                past.dropna(subset=["revenue_reported", "revenue_estimate"])
                .groupby("symbol")
                .last()
            )
            out["revenue_surprise_last"] = (
                last_rows["revenue_reported"]
                / last_rows["revenue_estimate"].replace(0.0, np.nan)
                - 1
            ).reindex(index)
        last_date = past.groupby("symbol")["date"].max()
        out["days_since_earnings"] = (as_of - last_date).dt.days.reindex(index)
        future = earnings[earnings["date"] > as_of].groupby("symbol")["date"].min()
        out["days_to_next_earnings"] = (future - as_of).dt.days.reindex(index)
        close = ctx.close.ffill()
        if not close.empty:
            since = {}
            for symbol, day in last_date.items():
                if symbol not in close.columns:
                    continue
                history = close[symbol].loc[:day]
                if len(history) and not np.isnan(history.iloc[-1]):
                    since[symbol] = close[symbol].iloc[-1] / history.iloc[-1] - 1
            out["return_since_earnings"] = pd.Series(since, dtype=float).reindex(index)
    return out.replace([np.inf, -np.inf], np.nan)
