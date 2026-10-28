"""Performance guardrail: a full run on a synthetic 500-symbol database must stay fast and lean.

The database is written with raw SQL (no pydantic per row) so the fixture
itself takes a few seconds. The bounds are loose enough for a CI runner
and tight enough to fail on a quadratic loop or a universe-wide panel.
"""

from __future__ import annotations

import datetime
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from bearish.database.crud import BearishDb

from bullish.config import BullishSettings
from bullish.database.crud import BullishDb
from bullish.kpi import engine
from bullish.kpi.profile import peak_rss_mb

SYMBOLS = 500
DAYS = 756  # three years of trading days
AS_OF = datetime.date(2025, 6, 30)
MAX_SECONDS = 90.0
MAX_PEAK_RSS_MB = 1500.0
SECTORS = ("Technology", "Health Care", "Financials", "Energy", "Industrials")


def _build(path: Path) -> None:
    BearishDb(database_path=path)  # creates the bearish schema
    rng = np.random.default_rng(7)
    dates = [
        d.date().isoformat()
        for d in __import__("pandas").bdate_range(end=AS_OF, periods=DAYS)
    ]
    today = AS_OF.isoformat()
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA synchronous = OFF")
        equities = []
        prices = []
        for i in range(SYMBOLS):
            symbol = f"B{i:04d}"
            close = 20.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, DAYS)))
            volume = rng.integers(100_000, 2_000_000, DAYS).astype(float)
            equities.append(
                (
                    symbol,
                    "Yfinance",
                    today,
                    today,
                    f"Bench {i}",
                    symbol,
                    "USD",
                    "United states",
                    SECTORS[i % len(SECTORS)],
                    "Benchmark industry",
                    float(close[-1] * 1e8),
                    1e8,
                )
            )
            prices.extend(
                (
                    symbol,
                    "Yfinance",
                    f"{day} 00:00:00.000000",
                    today,
                    c,
                    c * 1.01,
                    c * 0.99,
                    c,
                    v,
                    0.0,
                    0.0,
                )
                for day, c, v in zip(
                    dates, close.tolist(), volume.tolist(), strict=True
                )
            )
        conn.executemany(
            "INSERT INTO equity (symbol, source, date, created_at, name, base_symbol, currency, country, "
            "sector, industry, market_capitalization, shares_outstanding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            equities,
        )
        conn.executemany(
            "INSERT INTO price (symbol, source, date, created_at, open, high, low, close, volume, "
            "dividends, stock_splits) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            prices,
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(scope="module")
def benchmark_db(tmp_path_factory: pytest.TempPathFactory) -> BullishDb:
    path = tmp_path_factory.mktemp("bench") / "bench.db"
    _build(path)
    return BullishDb(database_path=path)


def test_full_run_stays_within_time_and_memory_bounds(benchmark_db: BullishDb) -> None:
    settings = BullishSettings(
        database_path=benchmark_db.database_path,
        min_market_cap=0.0,
        min_dollar_volume=0.0,
        lookback_years=3,
    )
    rss_before = peak_rss_mb()
    start = time.perf_counter()
    result = engine.run(benchmark_db, settings, as_of=AS_OF)
    elapsed = time.perf_counter() - start
    peak = peak_rss_mb()
    table = result.timer.table()
    assert result.run is not None, table
    assert result.run.entities == SYMBOLS, table
    assert result.equity.notna().sum().sum() > SYMBOLS * 20, table
    assert elapsed < MAX_SECONDS, f"{elapsed:.1f}s\n{table}"
    assert (
        peak < MAX_PEAK_RSS_MB
    ), f"peak rss {peak:.0f} MB (before run {rss_before:.0f})\n{table}"
    technical = next(t for t in result.timer.timings if t.phase == "equity.technical")
    assert (
        technical.seconds / SYMBOLS < 0.05
    ), table  # 50 ms per symbol is the old single-process cost
