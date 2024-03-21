import abc
from typing import Callable, Optional

import pandas as pd
from pydantic import BaseModel, computed_field, Field, PrivateAttr


class BaseInput(BaseModel):
    window: Optional[int] = None
    def __hash__(self):
        return hash(self.name)

    @computed_field
    def name(self) -> str:
        if not self.window:
            return f"{self.__class__.__name__.lower()}"

        return f"{self.__class__.__name__.lower()}_{self.window}"

    @abc.abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        ...


class MovingAverage(BaseInput):
    window: int

    def compute(self, data: pd.DataFrame):
        data[self.name] = data.price.rolling(window=self.window).mean()
        return data


class ExponentialMovingAverage(BaseInput):
    window: int


    def compute(self, data: pd.DataFrame):
        data[self.name] = data.price.ewm(span=self.window, adjust=False).mean()
        return data


class Macd(BaseInput):
    window: int = None
    def compute(self, data: pd.DataFrame):
        data = ExponentialMovingAverage(window=12).compute(data)
        data = ExponentialMovingAverage(window=26).compute(data)
        data[self.name] = (
            data.exponentialmovingaverage_12 - data.exponentialmovingaverage_26
        )
        return data


class MacdSignal(BaseInput):
    window: int = None
    def compute(self, data: pd.DataFrame):
        data = ExponentialMovingAverage(window=12).compute(data)
        data = ExponentialMovingAverage(window=26).compute(data)
        macd = data.exponentialmovingaverage_12 - data.exponentialmovingaverage_26
        data[self.name] = macd.ewm(span=9, adjust=False).mean()
        return data


class RSI(BaseInput):
    window: int = 14

    def compute(self, data: pd.DataFrame):
        close_diff = data.price.diff()
        up_days = (
            close_diff.where(close_diff > 0)
            .rolling(window=self.window, min_periods=1)
            .mean()
        )
        down_days = (
            -close_diff.where(close_diff < 0)
            .rolling(window=self.window, min_periods=1)
            .mean()
        )

        rs = up_days / down_days
        rsi = 100 - (100 / (1 + rs))
        data[self.name] = rsi
        return data


class Support(BaseInput):
    extreme: Callable = Field(default=lambda data: data.low == data.low.min())
    opposite: Callable = Field(default=lambda data: data.high == data.high.max())
    extract_value: Callable = Field(default=lambda data: data.low)
    slope: Callable = Field(default=min)
    _previous_data: pd.DataFrame = PrivateAttr(default=pd.DataFrame())

    def _to_timestamp(self, data: pd.DataFrame) -> pd:
        return pd.DatetimeIndex(list(data.index)).astype("int64") // 10 ** 9

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        data_extreme = self.extract_value(data.where(self.extreme(data))).dropna()
        extreme_index = data_extreme.index
        data_from_extreme = data.loc[extreme_index[0] :]
        data_to_opposite = self.extract_value(
            data_from_extreme.where(self.opposite(data_from_extreme))
        ).dropna()
        base_data = data_from_extreme.loc[extreme_index[0] : data_to_opposite.index[0]]
        ys = self.extract_value(base_data).values
        xs = self._to_timestamp(base_data)
        xs_ = xs[1:]
        ys_ = ys[1:]
        slopes = [(y - ys[0]) / (x - xs[0]) for x, y in zip(xs_, ys_)]
        support = pd.DataFrame(columns=[self.name])
        if slopes:
            _slope = self.slope(slopes)
            min_slope_index = slopes.index(_slope)
            intercept = ys_[min_slope_index] - _slope * xs_[min_slope_index]
            support = pd.DataFrame(
                [_slope * x + intercept for x in xs],
                index=pd.DatetimeIndex(list(base_data.index)),
                columns=[self.name],
            )

        self._previous_data = self._previous_data.combine_first(support)
        data[self.name] = self._previous_data[self.name]
        return data


class Resistance(Support):
    extreme: Callable = Field(default=lambda data: data.high == data.high.max())
    opposite: Callable = Field(default=lambda data: data.low == data.low.min())
    extract_value: Callable = Field(default=lambda data: data.high)
    slope: Callable = Field(default=max)
