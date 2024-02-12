import json
from enum import Enum
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import List

import pandas as pd
import sys
# sys.path.append("/home/aan/Documents/bearish")
from bearish.scrapers.main import Scraper, DataSource, Country
from bearish.scrapers.model import Ticker
from pydantic import ConfigDict
from tinydb import TinyDB, Query

from patterns.candlestick import CandleStick
from strategy.func import intersection
from strategy.inputs import BaseInput
from strategy.plot import plot_strategy
from strategy.strategy import MACD, MovingAverage5To20, BaseStrategy


class TickerAnalysis(Ticker):
    def growing(self):
        if not self.fundamental.income_statement.net_income:
            return False
        data = pd.DataFrame.from_dict(
            self.fundamental.income_statement.net_income.yearly, orient="index"
        )
        data.index = pd.to_datetime(data.index)
        data = (data[0] / max(data[0])).sort_index()
        growing = data[-1] == 1

        if self.fundamental.income_statement.net_income.quarterly:
            data = pd.DataFrame.from_dict(
                self.fundamental.income_statement.net_income.quarterly, orient="index"
            )
            data.index = pd.to_datetime(data.index)
            data = (data[0] / max(data[0])).sort_index()
            growing = growing and (data[-1] == 1)
        return growing

    def get_price(self):
        if not self.historical.price:
            return
        data = pd.DataFrame.from_dict(self.historical.model_dump())
        data.index = pd.to_datetime(data.index)
        return data.sort_index()


import plotly.graph_objects as go


def gap(data: pd.DataFrame):
    return ((data.high[0] - data.low[1]) < 0) or ((data.low[0] - data.high[1]) > 0)

def test_select_stocsk():
    for file in Path("/home/aan/Documents/bullish/tests/results").glob("*"):
        data = pd.read_parquet(file)
        # data = data[-90:]
        selection_data = data[-14:]
        v = selection_data["sign_movingaverage5to20"].drop(selection_data[selection_data["sign_movingaverage5to20"]==0].index).dropna().values
        if len(v) >0 and v[-1] == 1:
            plot_strategy(data, [MACD(), MovingAverage5To20()], name=file.stem)

        a = 12

def test_follow_up():
    scraper = Scraper(source=DataSource.investing, country=Country.belgium, bearish_path=Path("/home/aan/Documents/bullish/follow_up"))
    scraper.scrape(skip_existing=False, symbols=["ACKB"])
    db_json = scraper.create_db_json()
    f = Path("/home/aan/Documents/bullish/follow_up/data/db_json.json")
    f.touch(exist_ok=True)
    with f.open(mode="w") as p:
        json.dump(db_json, p, indent=4)
    tiny_path = Path("/home/aan/Documents/bullish/follow_up/data/tiny_db_json.json")
    db = TinyDB(tiny_path)
    db.insert_multiple(json.loads(f.read_text()))
def test_load_data():
    tiny_path = Path("/home/aan/Documents/bullish/data/tiny_db_json.json")
    db = TinyDB(tiny_path)
    # path = Path("/home/aan/Documents/stocks/data/db_json.json")
    # db.insert_multiple(json.loads(path.read_text()))
    equity = Query()
    results = db.search(
        (equity.fundamental.ratios.price_earning_ratio > 5)
        & (equity.fundamental.ratios.price_earning_ratio < 15)
    )
    ts = [TickerAnalysis(**rt) for rt in results]

    for t in ts:
        df = t.get_price()
        backtest = Backtest(
            strategies=[MACD(), MovingAverage5To20()],
            price=df,
        )
        data = backtest.play()
        data.to_parquet(f"/home/aan/Documents/bullish/tests/results/{t.symbol}.pqt")



from pydantic import BaseModel, computed_field


class Status(Enum):
    buy: int = 1
    sell: int = 2
    unknown: int = 0


class Point(CandleStick):
    price: float
    high: float
    low: float
    volume: float
    support: float
    resistance: float
    moving_average_200: float
    moving_average_20: float
    moving_average_5: float


import numpy as np


class Backtest(BaseModel):
    strategies: list[BaseStrategy]
    price: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @computed_field
    @cached_property
    def _inputs(self) -> List[BaseInput]:
        return list(
            {input for strategy in self.strategies for input in strategy.inputs}
        )

    def play(self):
        for i in range(len(self.price.index)):
            data = self.price[: i + 1]
            for input in self._inputs:
                data = input.compute(data)
            for strategy in self.strategies:
                strategy.assess(data)
        for strategy in self.strategies:
            data = pd.concat([strategy._dataframe, data], axis=1)
            data = pd.concat([data, strategy.performance()], axis=1)
        plot_strategy(data, self.strategies)
        return data


def test_plot():
    data = pd.read_parquet("/home/aan/Documents/bullish/tests/results.pqt")
    backtest = Backtest(
        strategies=[MACD(), MovingAverage5To20()],
        price=data,
    )
    plot_strategy(data, backtest.strategies)

    # fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3])




def test_backtest():
    df = pd.read_csv("/home/aan/Documents/bullish/tests/data.csv")
    index = pd.DatetimeIndex(df[df.columns[0]])
    index.name = None
    df = df.drop(df.columns[0], axis=1)
    df.index = index
    backtest = Backtest(
        strategies=[MACD(), MovingAverage5To20()],
        price=df,
    )
    backtest.play()


def test_intersection_dataframe():
    index = pd.date_range("2024-01-01", "2024-01-10")
    first_parameter = [12, 13, 14, 15, 15.1, 14.5, 14.2, 13.5, 12, 11]
    second_parameter = [13, 13.1, 13.5, 14, 15, 15.1, 14, 13, 13, 13.5]
    data = pd.DataFrame(
        {"first_parameter": first_parameter, "second_parameter": second_parameter},
        index=index,
    )
    data_ = intersection(data.first_parameter, data.second_parameter)
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
            y=data["intersection"],
            mode="markers",
            marker={"size": 10, "color": data.sign},
        )
    )
    fig.show()


def test_intersection():
    data_up = pd.DataFrame.from_dict(
        {"moving_average_5": [0, 0, 0, 0], "moving_average_20": [-1, -0.5, 0.5, 2]}
    )
    res_up = np.diff(np.sign(data_up.moving_average_5 - data_up.moving_average_20))
    pd.DataFrame(index=data_up.index, columns=data_up.columns).combine_first(
        data_up.moving_average_5.iloc[np.argwhere(res_up).flatten()]
    )
    pd.DataFrame(
        data_up.moving_average_5.iloc[np.argwhere(res_up).flatten()],
        index=data_up.index,
    )

    # data_down = pd.DataFrame.from_dict({"moving_average_5":[0,0,0,0], "moving_average_20":[1,0.5, -0.5, -2]})
    # res_down = np.diff(np.sign(data_down.moving_average_5 - data_down.moving_average_20))
    # a = 12
    #
    # data.iloc[np.argwhere(res_up).flatten()]
    # data.iloc[np.argwhere(np.diff(np.sign(data.moving_average_5 - data.moving_average_20))).flatten()]


def test_strategy():
    path = Path("/home/aan/Documents/bullish/data/db_json.json")
    tiny_path = Path("/home/aan/Documents/bullish/data/tiny_db_json.json")
    db = TinyDB(tiny_path)
    # db.insert_multiple(json.loads(path.read_text()))
    equity = Query()
    results = db.search(
        (equity.fundamental.ratios.price_earning_ratio > 5)
        & (equity.fundamental.ratios.price_earning_ratio < 15)
    )
    # # results = db.search((equity.symbol == "PROX"))
    ts = [TickerAnalysis(**rt) for rt in results]
    for t in ts:
        df = t.get_price()
        backtest = Backtest(
            strategies=[Strategy()],
            price=df,
        )
        backtest.play()



