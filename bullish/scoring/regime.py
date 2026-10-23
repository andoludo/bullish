"""Rule-based market-regime label from the market KPI row (too few rows to rank cross-sectionally)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

RegimeLabel = Literal["risk_on", "neutral", "risk_off", "unknown"]
QUADRANTS = {
    1: "growth up, inflation down (recovery)",
    2: "growth up, inflation up (overheat)",
    3: "growth down, inflation up (stagflation)",
    4: "growth down, inflation down (slowdown)",
}


class RegimeSignal(BaseModel):
    name: str
    value: float | None
    vote: int
    detail: str


class MarketRegime(BaseModel):
    entity: str
    label: RegimeLabel
    score: float | None
    quadrant: str | None
    factor_leadership: str | None
    signals: list[RegimeSignal]
    summary: str


def _value(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    return (
        None
        if value is None or (isinstance(value, float) and np.isnan(value))
        else float(value)
    )


def _vote(
    row: pd.Series, name: str, positive_when: str, threshold: float, detail: str
) -> RegimeSignal:
    value = _value(row, name)
    vote = 0
    if value is not None:
        vote = (
            1
            if (
                (value > threshold) if positive_when == "above" else (value < threshold)
            )
            else -1
        )
    return RegimeSignal(name=name, value=value, vote=vote, detail=detail)


def classify(entity: str, row: pd.Series) -> MarketRegime:
    signals = [
        _vote(row, "trend_vs_sma200", "above", 0.0, "Index above its 200-day average"),
        _vote(row, "sma50_vs_sma200", "above", 0.0, "50-day average above the 200-day"),
        _vote(
            row,
            "breadth_above_200",
            "above",
            0.5,
            "Most stocks above their 200-day average",
        ),
        _vote(row, "new_highs_minus_lows", "above", 0.0, "More new highs than lows"),
        _vote(
            row,
            "equal_vs_cap_weight_3m",
            "above",
            0.0,
            "Equal weight beating cap weight (broad leadership)",
        ),
        _vote(
            row, "hy_spread_change_3m", "below", 0.0, "High-yield spreads tightening"
        ),
        _vote(row, "vix_pct_1y", "below", 60.0, "Volatility not in its top range"),
        _vote(row, "nfci", "below", 0.0, "Financial conditions loose"),
        _vote(row, "cli_change_3m", "above", 0.0, "Leading indicator rising"),
        _vote(row, "pmi", "above", 50.0, "Manufacturing expanding"),
        _vote(row, "curve_10_2", "above", 0.0, "Yield curve not inverted"),
        _vote(
            row,
            "cyclicals_vs_defensives_3m",
            "above",
            0.0,
            "Cyclicals leading defensives",
        ),
    ]
    known = [s.vote for s in signals if s.vote != 0]
    score = round(100 * (sum(known) / len(known) + 1) / 2, 1) if known else None
    label: RegimeLabel = "unknown"
    if score is not None:
        label = "risk_on" if score >= 65 else ("risk_off" if score <= 35 else "neutral")
    quadrant_code = _value(row, "growth_inflation_quadrant")
    quadrant = QUADRANTS.get(int(quadrant_code)) if quadrant_code is not None else None
    factors = {
        name: _value(row, f"ff_{name}_3m") for name in ("mom", "hml", "smb", "rmw")
    }
    known_factors = {k: v for k, v in factors.items() if v is not None}
    leadership = None
    if known_factors:
        best = max(known_factors, key=lambda k: known_factors[k])
        leadership = {
            "mom": "momentum",
            "hml": "value",
            "smb": "small caps",
            "rmw": "profitability",
        }[best]
    summary = f"{entity}: {label.replace('_', ' ')} ({score if score is not None else 'n/a'}/100)"
    if quadrant:
        summary += f"; {quadrant}"
    if leadership:
        summary += f"; {leadership} leading over three months"
    return MarketRegime(
        entity=entity,
        label=label,
        score=score,
        quadrant=quadrant,
        factor_leadership=leadership,
        signals=signals,
        summary=summary,
    )
