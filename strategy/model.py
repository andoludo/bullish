from enum import Enum
from functools import cached_property
from typing import List

import pandas as pd
from bearish.scrapers.model import Ticker
from pydantic import BaseModel, ConfigDict, computed_field

from patterns.candlestick import CandleStick
from strategy.inputs import BaseInput
from strategy.plot import plot_strategy
from strategy.strategy import BaseStrategy


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


class Status(Enum):
    buy: int = 1
    sell: int = 2
    unknown: int = 0


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

    def add_inputs(self, data:pd.DataFrame)->pd.DataFrame:
        for input in self._inputs:
            data = input.compute(data)
        return data
    def play(self, name: str = "test", time_span: int = 0, plot: bool = True) -> None:
        for i in range(len(self.price.index[time_span:])):
            data = self.price[time_span : time_span + i + 1]
            data = self.add_inputs(data)
            for strategy in self.strategies:
                strategy.assess(data)
        for strategy in self.strategies:
            data = pd.concat([strategy._dataframe, data], axis=1)
            data = pd.concat([data, strategy.performance()], axis=1)
        if plot:
            plot_strategy(data, self.strategies, name=name)
        return data
