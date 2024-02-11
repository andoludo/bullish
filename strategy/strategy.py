import abc
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, computed_field


from strategy.inputs import (
    BaseInput,
    Resistance,
    SupportLine,
    BaseMovingAverage,
    Macd,
    MacdSignal,
)
from strategy.func import intersection, difference


class BaseStrategy(BaseModel):
    window: int
    inputs: List[BaseInput]
    _dataframe: pd.DataFrame = PrivateAttr(default=pd.DataFrame())

    @computed_field
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    def assess(self, data: pd.DataFrame):
        ...

    def performance(self):
        return difference(self._dataframe, self.name)


class MovingAverage5To20(BaseStrategy):
    window: int = 5
    inputs: List[BaseInput] = Field(
        default=[
            Resistance(),
            SupportLine(),
            BaseMovingAverage(window=200),
            BaseMovingAverage(window=20),
            BaseMovingAverage(window=5),
        ]
    )

    def assess(self, data: pd.DataFrame):
        data = data[-self.window :]
        data = intersection(data.moving_average_5, data.moving_average_20, self.name)
        self._dataframe = self._dataframe.combine_first(data)


class MACD(BaseStrategy):
    window: int = 5
    inputs: List[BaseInput] = Field(
        default=[
            Macd(),
            MacdSignal(),
        ]
    )

    def assess(self, data: pd.DataFrame):
        data_window = data[-self.window :]
        data_intersection = intersection(data_window.macd, data_window.macd_signal, self.name)
        data_intersection[f"intersection_{self.name}"] = pd.DataFrame(
            data_window.price.loc[data_intersection.dropna().index],
            index=data_window.index,
        )
        self._dataframe = self._dataframe.combine_first(data_intersection)
