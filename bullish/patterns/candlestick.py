from typing import List

import pandas as pd
from pydantic import BaseModel


class CandleStick(BaseModel):  # noqa : PLR0904
    price: float
    open: float
    high: float
    low: float
    trading_range: float

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame) -> List["CandleStick"]:

        data["trading_range"] = abs(data["high"] - data["low"])
        data.columns = data.columns.astype(str)
        return [cls(**data_) for data_ in data.to_dict(orient="records")]  # type: ignore

    def is_bullish(self) -> bool:
        return self.price > self.open

    def is_bearish(self) -> bool:
        return self.open > self.price

    def is_a_doji(self) -> bool:
        return abs(self.open - self.price) < self.trading_range * 0.1

    def bullish_gap(self, other: "CandleStick") -> bool:
        return (self.high - other.low) < 0

    def bearish_gap(self, other: "CandleStick") -> bool:
        return (self.low - other.high) > 0

    def increase_close(self, other: "CandleStick") -> bool:
        return self.price < other.price

    def decrease_close(self, other: "CandleStick") -> bool:
        return self.price > other.price

    def close_smaller_than_open(self, other: "CandleStick") -> bool:
        return self.price < other.open

    def close_greater_than_open(self, other: "CandleStick") -> bool:
        return self.price > other.open

    def close_smaller_than_close(self, other: "CandleStick") -> bool:
        return self.price <= other.price

    def close_greater_than_close(self, other: "CandleStick") -> bool:
        return self.price > other.price

    def high_superior_than(self, others: list["CandleStick"]) -> bool:
        return all(self.high >= other.high for other in others)

    def low_inferior_than(self, others: list["CandleStick"]) -> bool:
        return all(self.low >= other.low for other in others)

    def high_inferior_than_low(self, other: "CandleStick") -> bool:
        return self.high <= other.low

    def low_superior_than_high(self, other: "CandleStick") -> bool:
        return self.low >= other.high

    def low_inferior_than_high(self, other: "CandleStick") -> bool:
        return self.low <= other.high

    def low_inferior_than_low(self, other: "CandleStick") -> bool:
        return self.low <= other.low

    def high_inferior_than_high(self, other: "CandleStick") -> bool:
        return self.high <= other.high

    def high_superior_than_low(self, other: "CandleStick") -> bool:
        return self.high <= other.low

    def breaking_high(self, other: "CandleStick") -> bool:
        return self.price > other.high

    def breaking_low(self, other: "CandleStick") -> bool:
        return self.price < other.low

    def same_low(self, other: "CandleStick") -> bool:
        return self.low == other.low

    def same_high(self, other: "CandleStick") -> bool:
        return self.high == other.high

    def low_equal_close(self) -> bool:
        return self.price == self.low

    def low_equal_open(self) -> bool:
        return self.open == self.low

    def high_equal_open(self) -> bool:
        return self.open == self.high

    def high_equal_close(self) -> bool:
        return self.price == self.high

    def open_equal_close(self) -> bool:
        return self.price == self.open

    def embedded_in(self, other: "CandleStick") -> bool:
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
