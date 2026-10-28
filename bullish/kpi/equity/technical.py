"""TA-Lib signals of :mod:`bullish.analysis.indicators` as KPIs, one row per symbol.

Each indicator signal becomes a KPI: value signals keep their value, date
signals (``GOLDEN_CROSS``, ``MACD_12_26_9_BULLISH_CROSSOVER``, ...) become
``<signal>_days_since``, the trading-day age of the last occurrence (NaN when
it never fired in the loaded history).

The signals are inherently per series (TA-Lib works on one array at a
time), so the block ships the OHLCV arrays of a chunk of symbols to worker
processes (``BullishSettings.workers``) and collects one dict per symbol.
"""

from __future__ import annotations

import datetime
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any
import numpy as np
import pandas as pd

from bullish.analysis.indicators import Indicators, indicators_factory
from bullish.data.panels import DataContext
from bullish.kpi.registry import DeclarationRow, declare_many

logger = logging.getLogger(__name__)
BLOCK = "technical"
MIN_ROWS = 60
COLUMNS = ("open", "high", "low", "close", "volume")
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
type Chunk = tuple[list[str], np.ndarray[Any, Any], dict[str, np.ndarray[Any, Any]]]


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


def _signals(indicators: Indicators, frame: pd.DataFrame) -> dict[str, float]:
    results = indicators.compute(frame)
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


def compute_chunk(chunk: Chunk) -> dict[str, dict[str, float]]:
    """Signals of every symbol in ``chunk``: ``(symbols, utc dates, {column: dates x symbols})``."""
    symbols, dates, arrays = chunk
    indicators = Indicators()
    index = pd.DatetimeIndex(dates, tz="UTC")
    out: dict[str, dict[str, float]] = {}
    for position, symbol in enumerate(symbols):
        frame = pd.DataFrame(
            {column: arrays[column][:, position] for column in COLUMNS}, index=index
        ).dropna(subset=["close"])
        if len(frame) < MIN_ROWS:
            continue
        try:
            out[symbol] = _signals(indicators, frame)
        except Exception as e:  # one broken history must not stop the run
            logger.warning(f"technical signals failed for {symbol}: {e}")
    return out


def _chunks(ctx: DataContext, symbols: list[str], n_chunks: int) -> list[Chunk]:
    close = ctx.close
    panels = {
        "open": ctx.open,
        "high": ctx.high,
        "low": ctx.low,
        "close": close,
        "volume": ctx.volume,
    }
    dates = pd.DatetimeIndex(close.index).tz_localize(None).to_numpy()
    size = max(1, -(-len(symbols) // n_chunks))
    chunks: list[Chunk] = []
    for start in range(0, len(symbols), size):
        part = symbols[start : start + size]
        arrays = {
            column: panel.reindex(index=close.index, columns=part).to_numpy(dtype=float)
            for column, panel in panels.items()
        }
        chunks.append((part, dates, arrays))
    return chunks


def _workers(ctx: DataContext, n_symbols: int) -> int:
    configured = ctx.settings.workers or max(1, (os.cpu_count() or 2) - 1)
    return max(1, min(configured, n_symbols // 50 or 1))


def compute(ctx: DataContext) -> pd.DataFrame:
    out = pd.DataFrame(index=ctx.universe.index, columns=NAMES, dtype=float)
    if ctx.close.empty:
        return out
    symbols = [s for s in ctx.universe.index if s in ctx.close.columns]
    if not symbols:
        return out
    workers = _workers(ctx, len(symbols))
    chunks = _chunks(ctx, symbols, workers * 4 if workers > 1 else 1)
    rows: dict[str, dict[str, float]] = {}
    if workers == 1:
        for chunk in chunks:
            rows.update(compute_chunk(chunk))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(compute_chunk, chunks):
                rows.update(result)
    if not rows:
        return out
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame = frame.reindex(index=out.index, columns=NAMES).astype(float)
    return frame.replace([np.inf, -np.inf], np.nan)
