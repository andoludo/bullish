"""TA-Lib signals of :mod:`bullish.analysis.indicators` as KPIs, one row per symbol.

Each indicator signal becomes a KPI: value signals keep their value, date
signals (``GOLDEN_CROSS``, ``MACD_12_26_9_BULLISH_CROSSOVER``, ...) become
``<signal>_days_since``, the trading-day age of the last occurrence (NaN when
it never fired in the loaded history).
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from bullish.analysis.indicators import Indicators, indicators_factory
from bullish.data.panels import DataContext
from bullish.kpi.registry import DeclarationRow, declare_many

logger = logging.getLogger(__name__)
BLOCK = "technical"
BULLISH_DATES = (
    "BULLISH",
    "GOLDEN",
    "LONG",
    "UPTREND",
    "CROSSES_ABOVE",
    "BREAKOUT",
    "CDL",
    "CROSS_UP",
    "OVERSOLD",
)


def _rows() -> list[DeclarationRow]:
    rows: list[DeclarationRow] = []
    seen: set[str] = set()
    for indicator in indicators_factory():
        for signal in indicator.signals:
            base = signal.name.lower()
            if base in seen:  # the PRICE indicator declares one signal twice
                continue
            seen.add(base)
            description = (
                signal.description or f"{indicator.description}: {signal.name}"
            )
            if signal.is_date():
                bullish = any(token in signal.name for token in BULLISH_DATES)
                rows.append(
                    (
                        f"{base}_days_since",
                        "B",
                        "days",
                        (
                            f"Trading days since the last {signal.name} ({description}); recent is good"
                            if bullish
                            else f"Trading days since the last {signal.name} ({description}); recent is bad"
                        ),
                        not bullish,
                    )
                )
            else:
                rows.append((base, "B", "value", description, None))
    return rows


NAMES = declare_many("equity", BLOCK, _rows())


def _ohlcv(ctx: DataContext, symbol: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "open": ctx.open[symbol],
            "high": ctx.high[symbol],
            "low": ctx.low[symbol],
            "close": ctx.close[symbol],
            "volume": ctx.volume[symbol],
        }
    ).dropna(subset=["close"])
    frame.index = pd.to_datetime(frame.index, utc=True)
    return frame


def _signals(frame: pd.DataFrame, as_of: datetime.date) -> dict[str, float]:
    results = Indicators().compute(frame)
    trading_days = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    out: dict[str, float] = {}
    for name, value in results.items():
        if isinstance(value, datetime.date):
            position = trading_days.searchsorted(pd.Timestamp(value), side="right")
            out[f"{name}_days_since"] = float(len(trading_days) - position)
        elif value is None:
            continue
        else:
            out[name] = float(value)
    return out


def compute(ctx: DataContext) -> pd.DataFrame:
    out = pd.DataFrame(index=ctx.universe.index, columns=NAMES, dtype=float)
    if ctx.close.empty:
        return out
    for symbol in ctx.universe.index:
        if symbol not in ctx.close.columns:
            continue
        frame = _ohlcv(ctx, symbol)
        if len(frame) < 60:
            continue
        try:
            values = _signals(frame, ctx.as_of)
        except Exception as e:  # one broken history must not stop the run
            logger.warning(f"technical signals failed for {symbol}: {e}")
            continue
        for name, value in values.items():
            if name in out.columns:
                out.loc[symbol, name] = value
    return out.replace([np.inf, -np.inf], np.nan)
