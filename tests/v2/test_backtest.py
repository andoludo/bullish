import datetime

import pandas as pd
import pytest

from bullish.backtest.rank import (
    backtest_kpi,
    backtest_score,
    backtest_values,
    build_history,
    forward_returns,
)
from bullish.config import BullishSettings
from bullish.database.crud import BullishDb
from tests.v2.synthetic import AS_OF


def test_backtest_values_detects_a_perfect_signal() -> None:
    dates = [datetime.date(2025, 1, 31), datetime.date(2025, 2, 28)]
    symbols = list("ABCDEFGHIJ")
    forward = pd.DataFrame(
        {s: [0.01 * i, 0.02 * i] for i, s in enumerate(symbols)}, index=dates
    )
    values = {d: pd.Series(range(10), index=symbols, dtype=float) for d in dates}
    result = backtest_values("perfect", values, forward, horizon_months=1)
    assert result.mean_ic == pytest.approx(1.0)
    assert result.hit_rate == 1.0
    assert result.top_minus_bottom is not None and result.top_minus_bottom > 0
    inverted = backtest_values(
        "perfect", values, forward, horizon_months=1, lower_is_better=True
    )
    assert inverted.mean_ic == pytest.approx(-1.0)


def test_history_and_kpi_backtest_on_the_synthetic_database(
    bullish_db_v2: BullishDb, settings: BullishSettings
) -> None:
    start = AS_OF - datetime.timedelta(days=300)
    dates = build_history(
        bullish_db_v2,
        settings,
        start,
        AS_OF - datetime.timedelta(days=120),
        step_months=2,
    )
    assert len(dates) >= 3
    forward = forward_returns(bullish_db_v2, settings, dates, horizon_months=3)
    assert set(forward.index) == set(dates)
    assert forward.shape[1] == 12
    result = backtest_kpi(bullish_db_v2, settings, "mom_12_1", horizon_months=3)
    assert result.dates >= 3
    assert result.mean_ic is not None and -1 <= result.mean_ic <= 1
    lower = backtest_kpi(bullish_db_v2, settings, "vol_60d", horizon_months=3)
    assert lower.direction_applied
    score = backtest_score(
        bullish_db_v2, settings, "quality_compounder", horizon_months=3
    )
    assert score.dates >= 3
