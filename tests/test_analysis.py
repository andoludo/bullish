import abc
import json
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
import os

sys.path.append("/home/aan/Documents/stocks")
from scrapers.model import Ticker

from numba import jit
from pydantic import BaseModel, ConfigDict


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
from numpy.lib.stride_tricks import sliding_window_view


class CandleStick(BaseModel):
    price: float
    open: float
    high: float
    low: float
    trading_range: float

    @classmethod
    def from_dataframe(cls, data):
        data["trading_range"] = abs(data["high"] - data["low"])
        return [cls(**data_) for data_ in data.to_dict(orient="records")]

    def is_bullish(self):
        return self.price > self.open

    def is_bearish(self):
        return self.open > self.price

    def is_a_doji(self):
        return abs(self.open - self.price) < self.trading_range * 0.1

    def bullish_gap(self, other: "CandleStick"):
        return (self.high - other.low) < 0

    def bearish_gap(self, other: "CandleStick"):
        return (self.low - other.high) > 0

    def increase_close(self, other: "CandleStick"):
        return self.price < other.price

    def decrease_close(self, other: "CandleStick"):
        return self.price > other.price

    def close_smaller_than_open(self, other: "CandleStick"):
        return self.price < other.open

    def close_greater_than_open(self, other: "CandleStick"):
        return self.price > other.open

    def close_smaller_than_close(self, other: "CandleStick"):
        return self.price <= other.price

    def close_greater_than_close(self, other: "CandleStick"):
        return self.price > other.price

    def high_superior_than(self, others: list["CandleStick"]):
        return all([self.high >= other.high for other in others])

    def low_inferior_than(self, others: list["CandleStick"]):
        return all([self.low >= other.low for other in others])

    def high_inferior_than_low(self, other: "CandleStick"):
        return self.high <= other.low

    def low_superior_than_high(self, other: "CandleStick"):
        return self.low >= other.high

    def low_inferior_than_high(self, other: "CandleStick"):
        return self.low <= other.high

    def low_inferior_than_low(self, other: "CandleStick"):
        return self.low <= other.low

    def high_inferior_than_high(self, other: "CandleStick"):
        return self.high <= other.high

    def high_superior_than_low(self, other: "CandleStick"):
        return self.high <= other.low

    def breaking_high(self, other: "CandleStick"):
        return self.price > other.high

    def breaking_low(self, other: "CandleStick"):
        return self.price < other.low

    def same_low(self, other: "CandleStick"):
        return self.low == other.low

    def same_high(self, other: "CandleStick"):
        return self.high == other.high

    def low_equal_close(self):
        return self.price == self.low

    def low_equal_open(self):
        return self.open == self.low

    def high_equal_open(self):
        return self.open == self.high

    def high_equal_close(self):
        return self.price == self.high

    def open_equal_close(self):
        return self.price == self.open

    def embedded_in(self, other: "CandleStick"):
        if abs(other.open - other.price) > abs(self.open - self.price):
            if self.is_bearish() and other.is_bearish():
                return other.open > self.open and other.price < self.price
            elif self.is_bearish() and other.is_bullish():
                return other.price > self.open and other.open < self.price
            elif self.is_bullish() and other.is_bullish():
                return other.price > self.price and other.open < self.open
            else:
                return other.open > self.price and other.price < self.open
        else:
            return False


class Pattern(abc.ABC, BaseModel):
    name: str
    window: int
    data: pd.DataFrame
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abc.abstractmethod
    def logic_bullish(self, data: pd.DataFrame):
        ...

    @abc.abstractmethod
    def logic_bearish(self, data: pd.DataFrame):
        ...

    def compute(self) -> pd.DataFrame:
        bullish_patterns = []
        bearish_patterns = []
        sliding_windows = sliding_window_view(self.data, self.window, axis=0)
        for sliding_window in sliding_windows:
            data = pd.DataFrame(sliding_window.T, columns=self.data.columns)
            sticks = CandleStick.from_dataframe(data)
            bullish_patterns.append(self.logic_bullish(sticks))
            bearish_patterns.append(self.logic_bearish(sticks))
        for pattern_name, patterns in {
            "bullish": bullish_patterns,
            "bearish": bearish_patterns,
        }.items():
            patterns = pd.DataFrame(
                list(
                    reversed(
                        list(zip_longest(reversed(self.data.index), reversed(patterns)))
                    )
                )
            )
            patterns.index = patterns[0]
            name = f"{self.name}_{pattern_name}"
            patterns[name] = patterns[1]
            self.data[name] = patterns[1]
        return self.data


class ThreeCandles(Pattern):
    name: str = "three_candles"
    window: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def logic_bullish(self, sticks: list[CandleStick]):
        if all(
            [
                stick_0.increase_close(stick_1)
                for stick_0, stick_1 in zip(
                    sticks[: len(sticks) - 2], sticks[1 : len(sticks) - 1]
                )
            ]
        ) and all([stick.is_bullish() for stick in sticks]):
            return sticks[1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if all(
            [
                stick_0.decrease_close(stick_1)
                for stick_0, stick_1 in zip(
                    sticks[: len(sticks) - 2], sticks[1 : len(sticks) - 1]
                )
            ]
        ) and all([stick.is_bearish() for stick in sticks]):
            return sticks[1].price


class Quintuplets(ThreeCandles):
    name: str = "quintuplets"
    window: int = 5


class Tasuki(Pattern):
    name: str = "tasuki"
    window: int = 3
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            all([stick.is_bullish() for stick in sticks[:1]])
            and sticks[2].is_bearish()
            and sticks[0].close_smaller_than_open(sticks[1])
            and sticks[0].close_smaller_than_close(sticks[2])
        ):
            return sticks[1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            all([stick.is_bearish() for stick in sticks[:1]])
            and sticks[2].is_bullish()
            and sticks[0].close_greater_than_open(sticks[1])
            and sticks[0].close_greater_than_close(sticks[2])
        ):
            return sticks[1].price


class Hikkake(Pattern):
    name: str = "hikkake"
    window: int = 5
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[2].is_bearish()
            and sticks[3].is_bullish()
            and sticks[4].is_bullish()
            and sticks[1].embedded_in(sticks[0])
            and sticks[1].high_superior_than(sticks[2:3])
            and sticks[4].breaking_high(sticks[3])
        ):
            return sticks[1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[2].is_bullish()
            and sticks[3].is_bearish()
            and sticks[4].is_bearish()
            and sticks[1].embedded_in(sticks[0])
            and sticks[1].low_inferior_than(sticks[2:3])
            and sticks[4].breaking_low(sticks[3])
        ):
            return sticks[1].price


class Bottle(Pattern):
    name: str = "bottle"
    window: int = 2

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bullish()
            and sticks[0].close_greater_than_open(sticks[1])
            and sticks[1].low_equal_open()
        ):
            return sticks[1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bearish()
            and sticks[0].close_smaller_than_open(sticks[1])
            and sticks[1].high_equal_open()
        ):
            return sticks[1].price


class SlingShot(Pattern):
    name: str = "slingshot"
    window: int = 4

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[3].is_bullish()
            and sticks[1].low_superior_than_high(sticks[0])
            and sticks[3].low_inferior_than_high(sticks[0])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[3].is_bearish()
            and sticks[1].high_inferior_than_low(sticks[0])
            and sticks[3].high_inferior_than_low(sticks[0])
        ):
            return sticks[-1].price


class H(Pattern):
    name: str = "h"
    window: int = 3

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_a_doji()
            and sticks[2].is_bullish()
            and sticks[1].low_inferior_than_low(sticks[2])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_a_doji()
            and sticks[2].is_bearish()
            and sticks[2].high_inferior_than_high(sticks[1])
        ):
            return sticks[-1].price


class Doji(Pattern):
    name: str = "doji"
    window: int = 3

    def logic_bullish(self, sticks: list[CandleStick]):
        if sticks[0].is_bearish() and sticks[1].is_a_doji() and sticks[2].is_bullish():
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if sticks[0].is_bullish() and sticks[1].is_a_doji() and sticks[2].is_bearish():
            return sticks[-1].price


class Harami(Pattern):
    name: str = "harami"
    window: int = 2

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[1].embedded_in(sticks[0])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[1].embedded_in(sticks[0])
        ):
            return sticks[-1].price


class OnNeck(Pattern):
    name: str = "one_neck"
    window: int = 2

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[1].high_equal_close()
            and sticks[1].close_smaller_than_close(sticks[0])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[1].low_equal_close()
            and sticks[1].close_smaller_than_close(sticks[0])
        ):
            return sticks[-1].price


class Tweezers(Pattern):
    name: str = "tweezers"
    window: int = 3

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bearish()
            and sticks[2].is_bullish()
            and sticks[1].same_low(sticks[2])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bullish()
            and sticks[2].is_bearish()
            and sticks[1].same_high(sticks[2])
        ):
            return sticks[-1].price


class Sandwich(Pattern):
    name: str = "sandwich"
    window: int = 3

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[2].is_bearish()
            and sticks[1].embedded_in(sticks[0])
            and sticks[1].embedded_in(sticks[2])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[2].is_bullish()
            and sticks[1].embedded_in(sticks[0])
            and sticks[1].embedded_in(sticks[2])
        ):
            return sticks[-1].price


class Hammer(Pattern):
    name: str = "hammer"
    window: int = 3

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[2].is_bullish()
            and sticks[1].high_equal_close()
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[2].is_bearish()
            and sticks[1].low_equal_close()
        ):
            return sticks[-1].price


class Piercing(Pattern):
    name: str = "piercing"
    window: int = 2

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[0].close_greater_than_open(sticks[0])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[1].close_greater_than_open(sticks[0])
        ):
            return sticks[-1].price


class Engulfing(Pattern):
    name: str = "engulfing"
    window: int = 2

    def logic_bullish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bearish()
            and sticks[1].is_bullish()
            and sticks[0].embedded_in(sticks[1])
        ):
            return sticks[-1].price

    def logic_bearish(self, sticks: list[CandleStick]):
        if (
            sticks[0].is_bullish()
            and sticks[1].is_bearish()
            and sticks[1].embedded_in(sticks[0])
        ):
            return sticks[-1].price


def gap(data: pd.DataFrame):
    return ((data.high[0] - data.low[1]) < 0) or ((data.low[0] - data.high[1]) > 0)


import numpy as np


def support_line(data: pd.DataFrame):
    data_lowest = data.where(data.low == data.low.min()).dropna()
    lowest_index = data_lowest.index
    data_from_lowest = data.loc[lowest_index[0] :]
    data_to_max = data_from_lowest.where(
        data_from_lowest.high == data_from_lowest.high.max()
    ).dropna()
    base_data = data_from_lowest.loc[lowest_index[0] : data_to_max.index[0]]
    ys = base_data.low.values
    xs = range(len(base_data.index))
    xs_ = xs[1:]
    ys_ = ys[1:]
    slopes = [(y - ys[0]) / (x - xs[0]) for x, y in zip(xs_, ys_)]
    if not slopes:
        return

    min_slope = min(slopes)
    min_slope_index = slopes.index(min_slope)
    intercept = ys_[min_slope_index] - min_slope * xs_[min_slope_index]
    length = 4*len(data_from_lowest) if len(data_from_lowest) <4 else 6*len(data_from_lowest)
    return min_slope,intercept, pd.DataFrame(
        [min_slope * x + intercept for x in range(length)],
        index=pd.date_range(start=lowest_index[0], end=lowest_index[0] + pd.Timedelta(days=length-1)),
        columns=["support"],
    )


def test_load_data():
    path = Path("/home/aan/Documents/bullish/data/db_json.json")
    tiny_path = Path("/home/aan/Documents/bullish/data/tiny_db_json.json")
    db = TinyDB(tiny_path)
    # db.insert_multiple(json.loads(path.read_text()))
    equity = Query()
    results = db.search(
        (equity.fundamental.ratios.price_earning_ratio > 5)
        & (equity.fundamental.ratios.price_earning_ratio < 15)
        & (equity.fundamental.valuation.market_cap > 1 * 10 ** 9)
    )
    # results = db.search((equity.symbol == "PROX"))
    ts = [TickerAnalysis(**rt) for rt in results]
    filtered_data = []
    data = {}

    sectors = {
        "Basic Materials",
        "Consumer Cyclicals",
        "Consumer Non-Cyclicals",
        "Energy",
        "Financials",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Utilities",
    }
    lines  =set()
    for t in ts:
        df = t.get_price()[806:]
        # fig = go.Figure(
        # )
        fig = go.Figure(
            data=go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["price"],
            )
        )
        for i in range(len(df.index)):
            res = support_line(df[:i+1])
            if res is None:
                continue
            (min_slope, intercept, support) = res
            if (min_slope, intercept) not in lines:
                lines.add((min_slope, intercept))
            else:
                continue
            if support is not None:
                fig.add_trace(go.Scatter(x=support.index, y=support["support"], mode="lines", marker=dict(
                    color="black")))

        # support = pd.concat([df_lowest, new_df.iloc[[min_index+1]]])
        patterns = [
            ThreeCandles(data=df),
            Tasuki(data=df),
            Hikkake(data=df),
            Bottle(data=df),
            Quintuplets(data=df),
            SlingShot(data=df),
            H(data=df),
            Doji(data=df),
            Harami(data=df),
            OnNeck(data=df),
            Tweezers(data=df),
            Engulfing(data=df),
            Piercing(data=df),
            Hammer(data=df),
            Sandwich(data=df),
        ]
        for pattern in patterns:
            df = pattern.compute()
        a = 12

        # fig = go.Figure(
        #     data=go.Candlestick(
        #         x=df.index,
        #         open=df["open"],
        #         high=df["high"],
        #         low=df["low"],
        #         close=df["price"],
        #     )
        # )

        # support = support_line(df)
        # fig.add_trace(go.Line(x=support.index, y=support["support"]))
        # fig.add_scatter(x=df.index, y=df.price)
        for x in [50, 100, 200]:
            df_ma = df.rolling(window=x).mean()
            fig.add_trace(go.Line(x=df_ma.index, y=df_ma["price"]))
        for pattern in patterns:
            fig.add_scatter(
                x=df[[f"{pattern.name}_bearish"]].index,
                y=df[f"{pattern.name}_bearish"],
                name=f"{pattern.name}-bearish",
                mode="markers",
                marker=dict(
                    color="black"
                    if pattern.name in ["doji", "harami", "one_neck", "tweezers","engulfing", "piercing", "hammer","sandwich"]
                    else "red",  # Set color to red
                    size=8,  # Set marker size
                    symbol="triangle-down",  # Set marker symbol to square
                ),
            )
            fig.add_scatter(
                x=df[[f"{pattern.name}_bullish"]].index,
                y=df[f"{pattern.name}_bullish"],
                name=f"{pattern.name}-bullish",
                mode="markers",
                marker=dict(
                    color="black"
                    if pattern.name in ["doji", "harami", "one_neck", "tweezers","engulfing", "piercing", "hammer","sandwich"]
                    else "green",  # Set color to red
                    size=8,  # Set marker size
                    symbol="triangle-up",  # Set marker symbol to square
                ),
            )
        fig.show()


def resample(df):
    df_resample = pd.DataFrame()
    resampling = "M"
    df_resample["high"] = (
        df[["high"]].groupby(pd.Grouper(freq=resampling)).max().apply(max, axis=1)
    )
    df_resample["low"] = (
        df[["low"]].groupby(pd.Grouper(freq=resampling)).min().apply(min, axis=1)
    )
    df_resample["open"] = (
        df[["open"]].groupby(pd.Grouper(freq=resampling)).apply(lambda x: x["open"][0])
    )
    df_resample["price"] = (
        df[["price"]]
        .groupby(pd.Grouper(freq=resampling))
        .apply(lambda x: x["price"][0])
    )
    return df_resample
