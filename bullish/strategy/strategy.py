import abc
from typing import List

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, computed_field

from bullish.strategy.func import difference, intersection
from bullish.strategy.inputs import (
    RSI,
    BaseInput,
    ExponentialMovingAverage,
    Macd,
    MacdSignal,
    MovingAverage,
    Resistance,
    Support,
)


class BaseStrategy(BaseModel):
    window: int
    inputs: List[BaseInput]
    _dataframe: pd.DataFrame = PrivateAttr(default=pd.DataFrame())

    @computed_field  # type: ignore
    @property
    def name(self) -> str:
        return self.__class__.__name__.lower()

    @abc.abstractmethod
    def assess(self, data: pd.DataFrame) -> None:
        ...

    def performance(self) -> pd.DataFrame:
        return difference(self._dataframe, self.name)


class MovingAverageBaseStrategy(BaseStrategy):
    window: int = 5

    @abc.abstractmethod
    def _function(self, data: pd.DataFrame) -> pd.DataFrame:
        ...

    def assess(self, data: pd.DataFrame) -> None:
        data = data[-self.window :]
        data_intersection = self._function(data)
        data_intersection[f"buy_{self.name}"] = data_intersection[
            f"intersection_{self.name}"
        ].where(data_intersection[f"sign_{self.name}"] == 1)
        data_intersection[f"sell_{self.name}"] = data_intersection[
            f"intersection_{self.name}"
        ].where(data_intersection[f"sign_{self.name}"] == -1)
        self._dataframe = self._dataframe.combine_first(data_intersection)


class MovingAverage5To20(MovingAverageBaseStrategy):

    inputs: List[BaseInput] = Field(
        default=[
            Resistance(),
            Support(),
            RSI(),
            MovingAverage(window=20),
            MovingAverage(window=5),
            MovingAverage(window=200),
        ]
    )

    def _function(self, data: pd.DataFrame) -> pd.DataFrame:
        return intersection(data.movingaverage_5, data.movingaverage_20, self.name)


class ExponentialMovingAverage5To10(MovingAverageBaseStrategy):

    inputs: List[BaseInput] = Field(
        default=[
            ExponentialMovingAverage(window=10),
            ExponentialMovingAverage(window=5),
        ]
    )

    def _function(self, data: pd.DataFrame) -> pd.DataFrame:
        return intersection(
            data.exponentialmovingaverage_5, data.exponentialmovingaverage_10, self.name
        )


class MovingAverage5To10(MovingAverageBaseStrategy):
    inputs: List[BaseInput] = Field(
        default=[
            MovingAverage(window=10),
            MovingAverage(window=5),
        ]
    )

    def _function(self, data: pd.DataFrame) -> pd.DataFrame:
        return intersection(data.movingaverage_5, data.movingaverage_10, self.name)


class MovingAverage4To18(MovingAverageBaseStrategy):
    inputs: List[BaseInput] = Field(
        default=[
            MovingAverage(window=18),
            MovingAverage(window=4),
        ]
    )

    def _function(self, data: pd.DataFrame) -> pd.DataFrame:
        return intersection(data.movingaverage_4, data.movingaverage_18, self.name)


class MACD(BaseStrategy):
    window: int = 5
    inputs: List[BaseInput] = Field(
        default=[
            Macd(),
            MacdSignal(),
        ]
    )

    def assess(self, data: pd.DataFrame) -> None:
        data_window = data[-self.window :]
        data_intersection = intersection(
            data_window.macd, data_window.macdsignal, self.name
        )
        data_intersection[f"intersection_{self.name}"] = pd.DataFrame(
            data_window.price.loc[data_intersection.dropna().index],
            index=data_window.index,
        )
        data_intersection[f"buy_{self.name}"] = data_intersection[
            f"intersection_{self.name}"
        ].where(data_intersection[f"sign_{self.name}"] == 1)
        data_intersection[f"sell_{self.name}"] = data_intersection[
            f"intersection_{self.name}"
        ].where(data_intersection[f"sign_{self.name}"] == -1)
        self._dataframe = self._dataframe.combine_first(data_intersection)
