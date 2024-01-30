import abc
import json
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
from tinydb import TinyDB, Query
import os

from trend.lines import support, resistance

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
        df = t.get_price()
        # fig = go.Figure(
        # )
        df = df.loc[(df.index > pd.Timestamp.now() - pd.Timedelta(days=90)) & (df.index <= pd.Timestamp.now())]
        if df.empty:
            continue
        fig = go.Figure(
            data=go.Candlestick(
                name=t.symbol,
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["price"],
            )
        )
        for i in range(len(df.index)):
            res = support(df[:i+1])
            if res is None:
                continue
            (min_slope, intercept, support_) = res
            if (min_slope, intercept) not in lines:
                lines.add((min_slope, intercept))
            else:
                continue
            if support_ is not None:
                fig.add_trace(go.Scatter(x=support_.index, y=support_["Support"], mode="lines", marker=dict(
                    color="black")))

        for i in range(len(df.index)):
            res = resistance(df[:i+1])
            if res is None:
                continue
            (min_slope, intercept, support_) = res
            if (min_slope, intercept) not in lines:
                lines.add((min_slope, intercept))
            else:
                continue
            if support_ is not None:
                fig.add_trace(go.Scatter(x=support_.index, y=support_["Resistance"], mode="lines", marker=dict(
                    color="red")))

        # support = pd.concat([df_lowest, new_df.iloc[[min_index+1]]])
        # patterns = [
        #     ThreeCandles(data=df),
        #     Tasuki(data=df),
        #     Hikkake(data=df),
        #     Bottle(data=df),
        #     Quintuplets(data=df),
        #     SlingShot(data=df),
        #     H(data=df),
        #     Doji(data=df),
        #     Harami(data=df),
        #     OnNeck(data=df),
        #     Tweezers(data=df),
        #     Engulfing(data=df),
        #     Piercing(data=df),
        #     Hammer(data=df),
        #     Sandwich(data=df),
        # ]
        # for pattern in patterns:
        #     df = pattern.compute()
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
        for x in [20, 50,200]:
            df_ma = df.rolling(window=x).mean()
            fig.add_trace(go.Line(x=df_ma.index, y=df_ma["price"]))
        # for pattern in patterns:
        #     fig.add_scatter(
        #         x=df[[f"{pattern.name}_bearish"]].index,
        #         y=df[f"{pattern.name}_bearish"],
        #         name=f"{pattern.name}-bearish",
        #         mode="markers",
        #         marker=dict(
        #             color="black"
        #             if pattern.name in ["doji", "harami", "one_neck", "tweezers","engulfing", "piercing", "hammer","sandwich"]
        #             else "red",  # Set color to red
        #             size=8,  # Set marker size
        #             symbol="triangle-down",  # Set marker symbol to square
        #         ),
        #     )
        #     fig.add_scatter(
        #         x=df[[f"{pattern.name}_bullish"]].index,
        #         y=df[f"{pattern.name}_bullish"],
        #         name=f"{pattern.name}-bullish",
        #         mode="markers",
        #         marker=dict(
        #             color="black"
        #             if pattern.name in ["doji", "harami", "one_neck", "tweezers","engulfing", "piercing", "hammer","sandwich"]
        #             else "green",  # Set color to red
        #             size=8,  # Set marker size
        #             symbol="triangle-up",  # Set marker symbol to square
        #         ),
        #     )
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
