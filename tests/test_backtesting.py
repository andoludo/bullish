import datetime
import json
from datetime import date, timedelta

import pandas as pd
import pytest

from bullish.analysis.analysis import compute_financials_series

pd.options.plotting.backend = "plotly"
from bullish.analysis.backtest import (
    run_backtest,
    BackTestConfig,
    run_tests,
    run_many_tests,
    BacktestResult,
    BacktestResults,
)
import pandas_ta as ta  # type: ignore

from bullish.analysis.predefined_filters import NamedFilterQuery, predefined_filters
from bullish.database.crud import BullishDb

from bearish.models.base import (
    Ticker,
)
from bearish.models.financials.base import Financials, FinancialsWithDate


@pytest.mark.skip(reason="This is too slow")
def test_backtesting(bullish_db_with_signal_series: BullishDb):

    config = BackTestConfig(start=date(2024, 3, 10))
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        market_capitalization=[1e9, 1e12],
        rsi_bullish_crossover_30=[
            date.today() - datetime.timedelta(days=5),
            date.today(),
        ],
    )

    back_tests = run_tests(bullish_db_with_signal_series, filtred_query, config)
    fig = back_tests.to_figure()
    assert fig
    backtest_result = back_tests.to_backtest_result()
    bullish_db_with_signal_series.write_many_backtest_results([backtest_result])
    saved_results = bullish_db_with_signal_series.read_many_backtest_results()
    assert len(saved_results) > 0

@pytest.mark.skip(reason="This is too slow")
def test_many_tests(bullish_db_with_signal_series: BullishDb) -> None:
    config = BackTestConfig(start=date(2024, 3, 10), iterations=5)
    run_many_tests(bullish_db_with_signal_series, predefined_filters(), config)
    saved_results = bullish_db_with_signal_series.read_many_backtest_results()
    assert len(saved_results) > 0


def test_backtesting_loading_dataframe():
    data_raw_1 = """{"mean": {"1710115200000": 1000.0, 
    "1714435200000": 1025.5407666248, 
    "1715212800000": 1050.2865344562, 
    "1717977600000": 1065.1072691344, 
    "1727395200000": 1075.3811300243, 
    "1729209600000": 1085.1426725404, "1739404800000": 1099.2062856429, "1741219200000": 1118.2653986664, "1743552000000": 1129.9772769427, "1745971200000": 1162.9726830494, "1748995200000": 1179.5157532809, "1750377600000": 1100.1322764747}, "upper": {"1710115200000": 1000.0, "1714435200000": 1082.6516569953, "1715212800000": 1119.1585478137, "1717977600000": 1128.1819601227, "1727395200000": 1128.6157068158, "1729209600000": 1123.4932717995, "1739404800000": 1134.8749774608, "1741219200000": 1144.4784540335, "1743552000000": 1170.5890883304, "1745971200000": 1271.3043385787, "1748995200000": 1290.5279990392, "1750377600000": 1285.6284639903}, "lower": {"1710115200000": 1000.0, "1714435200000": 968.4298762543, "1715212800000": 981.4145210988, "1717977600000": 1002.0325781462, "1727395200000": 1022.1465532327, "1729209600000": 1046.7920732814, "1739404800000": 1063.537593825, "1741219200000": 1092.0523432993, "1743552000000": 1089.365465555, "1745971200000": 1054.64102752, "1748995200000": 1068.5035075227, "1750377600000": 914.6360889591}, "median": {"1710115200000": 1000.0, "1714435200000": 1000.0, "1715212800000": 1000.0, "1717977600000": 1074.103673391, "1727395200000": 1074.103673391, "1729209600000": 1074.103673391, "1739404800000": 1121.6873699615, "1741219200000": 1123.7288391573, "1743552000000": 1123.7288391573, "1745971200000": 1123.7288391573, "1748995200000": 1123.7288391573, "1750377600000": 1040.0530129803}}"""
    data_raw_2 = """{"mean": {"1712620800000": 1000.0, "1746144000000": 1043.2957491626, "1750377600000": 1043.3378374495}, "upper": {"1712620800000": 1000.0, "1746144000000": 1043.2957491626, "1750377600000": 1047.4344891748}, "lower": {"1712620800000": 1000.0, "1746144000000": 1043.2957491626, "1750377600000": 1039.2411857243}, "median": {"1712620800000": 1000.0, "1746144000000": 1043.2957491626, "1750377600000": 1043.3378374495}}"""
    data_1 = json.loads(data_raw_1)
    data_2 = json.loads(data_raw_2)
    results = [
        BacktestResult(start=date.today(), name="test_1", data=data_1),
        BacktestResult(start=date.today(), name="test_2", data=data_2),
    ]
    backtest_results = BacktestResults(results=results)
    fig = backtest_results.figure()
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
        golden_cross=[
            date.today() - timedelta(days=5000),
            date.today(),
        ],
        market_capitalization=[5e9, 1e12],
        order_by_desc="momentum",
        country=[
            "Germany",
            "United states",
            "France",
            "United kingdom",
            "Canada",
            "Japan",
            "Belgium",
        ],
    )
    start_date = date(2024, 3, 11)
    symbols = filtred_query.get_backtesting_symbols(
        bullish_db_with_signal_series, start_date
    )
    assert symbols


def test_backtesting_query_fundamentals(
    bullish_db_with_signal_series: BullishDb,
) -> None:
    filtred_query = NamedFilterQuery(
        name="Momentum Growth Good Fundamentals (RSI 30)",
        cash_flow=["positive_free_cash_flow"],
        income=[
            "positive_operating_income",
            "growing_operating_income",
            "positive_net_income",
            "growing_net_income",
        ],
        properties=["operating_cash_flow_is_higher_than_net_income"],
        country=[
            "Belgium",
        ],
    )
    start_date = date(2024, 12, 30)
    symbols = filtred_query.get_backtesting_symbols(
        bullish_db_with_signal_series, start_date
    )
    assert symbols


def test_financial_series(bullish_db_with_signal_series: BullishDb) -> None:
    ticker = Ticker(symbol="ACKB.BR")
    financials = Financials.from_ticker(bullish_db_with_signal_series, ticker)
    series = compute_financials_series(financials, ticker)
    assert series
