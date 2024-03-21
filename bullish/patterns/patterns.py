import abc
from itertools import zip_longest

import pandas as pd

from bullish.patterns.candlestick import CandleStick
from pydantic import ConfigDict, BaseModel
from numpy.lib.stride_tricks import sliding_window_view


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
