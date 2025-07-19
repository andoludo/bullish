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

def test_backtesting(bullish_db_with_signal_series:BullishDb):


    config = BackTestConfig(start=date(2022, 3, 10))
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
    )

    df = run_tests(bullish_db_with_signal_series, filtred_query, config)
    fig = go.Figure()
    for name, data in df.items():
        fig.add_trace(go.Scatter(x=df.index, y=data, mode='lines+markers'))


    fig.update_layout(title='Company Metrics Over Time', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def test_run_a_backtest(bullish_db_with_signal_series:BullishDb):


    config = BackTestConfig(start=date(2024, 3, 10))
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
    )
    test = run_backtest(bullish_db_with_signal_series, filtred_query, config)
    back_tests = BackTests(tests=[test])
    back_tests.show()
    df = back_tests.to_dataframe()
    fig = go.Figure()
    for name, data in df.items():
        fig.add_trace(go.Scatter(x=df.index, y=data, mode='lines+markers'))


    fig.update_layout(title='Company Metrics Over Time', xaxis_title='Date', yaxis_title='Value')
    fig.show()

def test_backtesting_query(bullish_db_with_signal_series:BullishDb):
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
        country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
    )
    start_date = date(2024, 3, 11)
    symbols = filtred_query.get_backtesting_symbols(bullish_db_with_signal_series,start_date)
    assert symbols

def test_simple_backtesting(data_aapl: pd.DataFrame) -> None:
    data_aapl = data_aapl[-600:]
    data_rsi = compute_rsi(data_aapl)
    entry_signal = ta.cross(
        series_a=data_rsi.RSI, series_b=pd.Series(30, index=data_aapl.index), above=True
    )
    exit_signal = pd.Series(False, index=data_aapl.index)

    # Track positions and add exit signal when gain >= 10%
    position_open = False
    entry_price = 0.0

    for i in range(1, len(data_aapl.close)):
        if entry_signal.iloc[i] and not position_open:
            entry_price = data_aapl.close.iloc[i]
            position_open = True
        elif position_open:
            ret = (data_aapl.close.iloc[i] - entry_price) / data_aapl.close.iloc[i]
            if ret >= 0.10:
                exit_signal.iloc[i] = True
                position_open = False

    # Backtest using vectorbt
    pf = vbt.Portfolio.from_signals(
        data_aapl.close, entries=entry_signal, exits=exit_signal, freq="1D"
    )

    # Stats
    print(pf.stats())

    # Plot
    pf.plot().show()
