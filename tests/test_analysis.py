import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import pytest

from strategy.func import intersection, rsi
from strategy.model import TickerAnalysis, Backtest
from strategy.plot import plot_strategy
from strategy.strategy import (
    MACD,
    MovingAverage5To20,
    MovingAverage5To10,
    MovingAverage4To18,
    ExponentialMovingAverage5To10,
)


@pytest.fixture
def ticker() -> dict[str, Any]:
    path_file = Path(__file__).parent / "data" / "tickers.json"
    return json.loads(path_file.read_text())[0]


@pytest.fixture
def ticker_data(ticker: dict[str, Any]) -> pd.DataFrame:
    ticker = TickerAnalysis(**ticker)
    return ticker.get_price()


def test_backtest_ticker(ticker_data: pd.DataFrame):
    backtest = Backtest(
        strategies=[
            MACD(),
            MovingAverage5To20(),
            MovingAverage5To10(),
            MovingAverage4To18(),
            ExponentialMovingAverage5To10(),
        ],
        price=ticker_data,
    )
    data = backtest.play(time_span=1500, plot=False)
    assert not data.empty
    assert {input.name for input in backtest._inputs}.issubset(set(data.columns))




def test_plot(ticker_data):
    backtest = Backtest(
        strategies=[MACD(), MovingAverage5To20()],
        price=ticker_data,
    )
    ticker_data = backtest.add_inputs(ticker_data)
    figure_json = plot_strategy(ticker_data, backtest.strategies, show=False)
    assert figure_json



def test_rsi(ticker_data):
    rsi_value = rsi(ticker_data)
    assert not rsi_value.empty


def test_backtest(ticker_data):
    backtest = Backtest(
        strategies=[
            MACD(),
            MovingAverage5To20(),
            MovingAverage5To10(),
            MovingAverage4To18(),
            ExponentialMovingAverage5To10(),
        ],
        price=ticker_data,
    )
    data = backtest.play(time_span=1500, plot=False)
    assert not data.empty


def test_intersection_dataframe():
    index = pd.date_range("2024-01-01", "2024-01-10")
    first_parameter = [12, 13, 14, 15, 15.1, 14.5, 14.2, 13.5, 12, 11]
    second_parameter = [13, 13.1, 13.5, 14, 15, 15.1, 14, 13, 13, 13.5]
    data = pd.DataFrame(
        {"first_parameter": first_parameter, "second_parameter": second_parameter},
        index=index,
    )
    data_ = intersection(data.first_parameter, data.second_parameter, "intersection")
    data = pd.concat([data, data_], axis=1)
    fig = go.Figure()
    for x in [
        "first_parameter",
        "second_parameter",
    ]:
        fig.add_trace(go.Line(x=data.index, y=data[x]))
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["intersection_intersection"],
            mode="markers",
            marker={"size": 10, "color": data.sign_intersection},
        )
    )
    fig.show()
    assert fig.to_json()
