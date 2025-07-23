import datetime
import random
from datetime import date, timedelta
from pydantic import BaseModel

import pandas as pd

pd.options.plotting.backend = "plotly"
from bullish.analysis.backtest import run_backtest, BackTestConfig, run_tests, BackTests
from bullish.analysis.functions import compute_rsi, cross_value
import pandas_ta as ta  # type: ignore
import vectorbt as vbt

from bullish.analysis.predefined_filters import NamedFilterQuery
from bullish.database.crud import BullishDb

import plotly.graph_objects as go


def test_backtesting(bullish_db_with_signal_series: BullishDb):

    config = BackTestConfig(start=date(2024, 3, 10))
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
    )

    back_tests = run_tests(bullish_db_with_signal_series, filtred_query, config)
    fig = back_tests.to_figure()
    assert fig


def test_run_a_backtest(bullish_db_with_signal_series: BullishDb):

    config = BackTestConfig(start=date(2024, 3, 10))
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
    )
    test = run_backtest(bullish_db_with_signal_series, filtred_query, config)
    assert test


def test_backtesting_query(bullish_db_with_signal_series: BullishDb):
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        cash_flow=["positive_free_cash_flow"],
        properties=["operating_cash_flow_is_higher_than_net_income"],
        price_per_earning_ratio=[10, 500],
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
        macd_12_26_9_bullish_crossover=[
            date.today() - timedelta(days=5),
            date.today(),
        ],
        golden_cross=[
            date.today() - timedelta(days=5000),
            date.today(),
        ],
        market_capitalization=[5e8, 1e12],
        order_by_desc="momentum",
        country=[
            "Germany",
            "United states",
            "France",
            "United kingdom",
            "Canada",
            "Japan",
        ],
    )
    start_date = date(2024, 3, 11)
    symbols = filtred_query.get_backtesting_symbols(
        bullish_db_with_signal_series, start_date
    )
    assert symbols
