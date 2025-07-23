import logging
from datetime import date
from typing import Optional, Callable, cast

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore

from pydantic import BaseModel

logger = logging.getLogger(__name__)
try:
    import talib
except Exception:
    logger.warning("Talib is not installed, skipping analysis")


def cross_simple(
    series_a: pd.Series, series_b: pd.Series, above: bool = True
) -> pd.Series:
    crossing = ta.cross(series_a=series_a, series_b=series_b, above=above)
    return crossing  # type: ignore


def cross(
    series_a: pd.Series, series_b: pd.Series, above: bool = True
) -> Optional[date]:
    crossing = cross_simple(series_a=series_a, series_b=series_b, above=above)
    if not crossing[crossing == 1].index.empty:
        return crossing[crossing == 1].last_valid_index().date()  # type: ignore
    return None


def cross_value(series: pd.Series, number: int, above: bool = True) -> Optional[date]:
    return cross(series, pd.Series(number, index=series.index), above=above)


def cross_value_series(
    series_a: pd.Series, number: int, above: bool = True
) -> pd.Series:
    crossing = cross_simple(
        series_a, pd.Series(number, index=series_a.index), above=above
    )
    return crossing


def compute_adx(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ADX_14"] = talib.ADX(data.high, data.low, close=data.close)  # type: ignore
    results["MINUS_DI"] = talib.MINUS_DI(data.high, data.low, data.close)  # type: ignore
    results["PLUS_DI"] = talib.PLUS_DI(data.high, data.low, data.close)  # type: ignore
    return results


def compute_pandas_ta_adx(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    adx = ta.adx(data.high, data.low, data.close, length=14)
    results["ADX_14"] = adx["ADX_14"]
    results["MINUS_DI"] = adx["DMN_14"]
    results["PLUS_DI"] = adx["DMP_14"]
    return results


def compute_macd(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    (
        results["MACD_12_26_9"],
        results["MACD_12_26_9_SIGNAL"],
        results["MACD_12_26_9_HIST"],
    ) = talib.MACD(
        data.close  # type: ignore
    )
    return results


def compute_pandas_ta_macd(data: pd.DataFrame) -> pd.DataFrame:

    macd = ta.macd(data.close, fast=12, slow=26, signal=9)
    results = pd.DataFrame(index=macd.index)
    results["MACD_12_26_9"] = macd["MACD_12_26_9"]
    results["MACD_12_26_9_SIGNAL"] = macd["MACDs_12_26_9"]
    results["MACD_12_26_9_HIST"] = macd["MACDh_12_26_9"]
    return results


def compute_rsi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["RSI"] = talib.RSI(data.close)  # type: ignore
    results["CLOSE"] = data.close
    return results


def compute_pandas_ta_rsi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["RSI"] = ta.rsi(data.close, length=14)
    results["CLOSE"] = data.close
    return results


def compute_stoch(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["SLOW_K"], results["SLOW_D"] = talib.STOCH(data.high, data.low, data.close)  # type: ignore
    return results


def compute_pandas_ta_stoch(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    stoch = ta.stoch(data.high, data.low, data.close, k=5, d=3, smooth_k=3)
    results["SLOW_K"] = stoch["STOCHk_5_3_3"]
    results["SLOW_D"] = stoch["STOCHd_5_3_3"]
    return results


def compute_mfi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["MFI"] = talib.MFI(data.high, data.low, data.close, data.volume)  # type: ignore
    return results


def compute_pandas_ta_mfi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["MFI"] = ta.mfi(data.high, data.low, data.close, data.volume, length=14)
    return results


def compute_roc(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ROC_7"] = talib.ROC(data.close, timeperiod=7)  # type: ignore
    results["ROC_1"] = talib.ROC(data.close, timeperiod=1)  # type: ignore
    results["ROC_30"] = talib.ROC(data.close, timeperiod=30)  # type: ignore
    mom = talib.MOM(data.close, timeperiod=252)  # type: ignore
    results["MOM"] = mom.shift(21)  # type: ignore

    return results


def compute_pandas_ta_roc(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ROC_7"] = ta.roc(data.close, length=7)
    results["ROC_1"] = ta.roc(data.close, length=1)
    results["ROC_30"] = ta.roc(data.close, length=30)
    return results


def compute_sma(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["SMA_50"] = talib.SMA(data.close, timeperiod=50)  # type: ignore
    results["SMA_200"] = talib.SMA(data.close, timeperiod=200)  # type: ignore
    results["CLOSE"] = data.close
    return results


def compute_pandas_ta_sma(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["SMA_50"] = ta.sma(data.close, length=50)
    results["SMA_200"] = ta.sma(data.close, length=200)
    results["CLOSE"] = data.close

    return results


def compute_adosc(data: pd.DataFrame) -> pd.DataFrame:
    data_ = data.copy()
    results = pd.DataFrame(index=data.index)
    results["ADOSC"] = talib.ADOSC(data.high, data.low, data.close, data.volume)  # type: ignore
    data_["ADOSC"] = results["ADOSC"]
    data_["HIGHEST_20"] = data_.close.rolling(window=20).max()
    results["ADOSC_SIGNAL"] = (data_.close > data_["HIGHEST_20"].shift(1)) & (
        data_["ADOSC"] > 0
    )
    return results


def compute_pandas_ta_adosc(data: pd.DataFrame) -> pd.DataFrame:
    data_ = data.copy()
    results = pd.DataFrame(index=data.index)
    results["ADOSC"] = ta.adosc(data.high, data.low, data.close, data.volume)
    data_["ADOSC"] = results["ADOSC"]
    data_["HIGHEST_20"] = data_.close.rolling(window=20).max()
    results["ADOSC_SIGNAL"] = (data_.close > data_["HIGHEST_20"].shift(1)) & (
        data_["ADOSC"] > 0
    )
    return results


def compute_ad(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["AD"] = talib.AD(data.high, data.low, data.close, data.volume)  # type: ignore
    return results


def compute_pandas_ta_ad(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["AD"] = ta.ad(data.high, data.low, data.close, data.volume)
    return results


def compute_obv(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["OBV"] = talib.OBV(data.close, data.volume)  # type: ignore
    return results


def compute_pandas_ta_obv(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["OBV"] = ta.obv(data.close, data.volume)
    return results


def compute_atr(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ATR"] = talib.ATR(data.high, data.low, data.close)  # type: ignore
    return results


def compute_pandas_ta_atr(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ATR"] = ta.atr(data.high, data.low, data.close, length=14)
    return results


def compute_natr(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["NATR"] = talib.NATR(data.high, data.low, data.close)  # type: ignore
    return results


def compute_pandas_ta_natr(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["NATR"] = ta.natr(data.high, data.low, data.close, length=14)
    return results


def compute_trange(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["TRANGE"] = talib.TRANGE(data.high, data.low, data.close)  # type: ignore
    return results


def compute_pandas_ta_trange(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["TRANGE"] = ta.true_range(data.high, data.low, data.close)
    return results


def compute_patterns(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["CDLMORNINGSTAR"] = talib.CDLMORNINGSTAR(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDL3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLABANDONEDBABY"] = talib.CDLABANDONEDBABY(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLTASUKIGAP"] = talib.CDLTASUKIGAP(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLPIERCING"] = talib.CDLPIERCING(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLENGULFING"] = talib.CDLENGULFING(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    return results


def perc(data: pd.Series) -> float:
    if len(data) < 2 or data.iloc[0] == 0:
        return np.nan
    return cast(float, ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100)


def compute_price(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["200_DAY_HIGH"] = data.close.rolling(window=200).max()
    results["200_DAY_LOW"] = data.close.rolling(window=200).min()
    results["20_DAY_HIGH"] = data.close.rolling(window=20).max()
    results["20_DAY_LOW"] = data.close.rolling(window=20).min()
    results["LAST_PRICE"] = data.close
    results["WEEKLY_GROWTH"] = data.close.resample("W").transform(perc)  # type: ignore
    results["MONTHLY_GROWTH"] = data.close.resample("ME").transform(perc)  # type: ignore
    results["YEARLY_GROWTH"] = data.close.resample("YE").transform(perc)  # type: ignore
    return results


def find_last_true_run_start(series: pd.Series) -> Optional[date]:
    if not series.iloc[-1]:
        return None
    arr = series.to_numpy()
    change_points = np.flatnonzero(np.r_[True, arr[1:] != arr[:-1]])
    run_starts = change_points
    true_runs = run_starts[arr[run_starts]]
    last_true_run_start = true_runs[-1]
    return series.index[last_true_run_start].date()  # type: ignore


def sma_50_above_sma_200(data: pd.DataFrame) -> Optional[date]:
    date_1 = find_last_true_run_start(data.SMA_50 > data.SMA_200)
    return date_1


def price_above_sma50(data: pd.DataFrame) -> Optional[date]:
    date_1 = find_last_true_run_start(data.SMA_50 < data.CLOSE)
    return date_1


class IndicatorFunction(BaseModel):
    expected_columns: list[str]
    functions: list[Callable[[pd.DataFrame], pd.DataFrame]]

    def call(self, data: pd.DataFrame) -> pd.DataFrame:
        data_ = None
        for function in self.functions:
            try:
                data_ = function(data)
                break
            except Exception as e:
                logger.error(f"Fail to compute function {function.__name__}: {e}")
        if data_ is None:
            raise ValueError(
                f"No data returned from indicator functions with expected columns {self.expected_columns}."
            )
        if not set(self.expected_columns).issubset(set(data_.columns)):
            raise ValueError(
                f"Expected columns {self.expected_columns} not found in data columns {data_.columns.tolist()}"
            )
        return data_


ADX = IndicatorFunction(
    expected_columns=["ADX_14", "MINUS_DI", "PLUS_DI"],
    functions=[compute_adx, compute_pandas_ta_adx],
)
MACD = IndicatorFunction(
    expected_columns=["MACD_12_26_9", "MACD_12_26_9_SIGNAL", "MACD_12_26_9_HIST"],
    functions=[compute_macd, compute_pandas_ta_macd],
)
RSI = IndicatorFunction(
    expected_columns=["RSI"], functions=[compute_rsi, compute_pandas_ta_rsi]
)
STOCH = IndicatorFunction(
    expected_columns=["SLOW_K", "SLOW_D"],
    functions=[compute_stoch, compute_pandas_ta_stoch],
)
MFI = IndicatorFunction(
    expected_columns=["MFI"], functions=[compute_mfi, compute_pandas_ta_mfi]
)
ROC = IndicatorFunction(
    expected_columns=["ROC_7", "ROC_1", "ROC_30", "MOM"],
    functions=[compute_roc, compute_pandas_ta_roc],
)
CANDLESTOCK_PATTERNS = IndicatorFunction(
    expected_columns=[
        "CDLMORNINGSTAR",
        "CDL3LINESTRIKE",
        "CDL3WHITESOLDIERS",
        "CDLABANDONEDBABY",
        "CDLTASUKIGAP",
        "CDLPIERCING",
        "CDLENGULFING",
    ],
    functions=[compute_patterns],
)

SMA = IndicatorFunction(
    expected_columns=["SMA_50", "SMA_200", "CLOSE"],
    functions=[compute_sma, compute_pandas_ta_sma],
)

ADOSC = IndicatorFunction(
    expected_columns=["ADOSC", "ADOSC_SIGNAL"],
    functions=[compute_adosc, compute_pandas_ta_adosc],
)

AD = IndicatorFunction(
    expected_columns=["AD"],
    functions=[compute_ad, compute_pandas_ta_ad],
)
OBV = IndicatorFunction(
    expected_columns=["OBV"],
    functions=[compute_obv, compute_pandas_ta_obv],
)
ATR = IndicatorFunction(
    expected_columns=["ATR"],
    functions=[compute_atr, compute_pandas_ta_atr],
)
NATR = IndicatorFunction(
    expected_columns=["NATR"],
    functions=[compute_natr, compute_pandas_ta_natr],
)
TRANGE = IndicatorFunction(
    expected_columns=["TRANGE"],
    functions=[compute_trange, compute_pandas_ta_trange],
)
PRICE = IndicatorFunction(
    expected_columns=[
        "200_DAY_HIGH",
        "200_DAY_LOW",
        "20_DAY_HIGH",
        "20_DAY_LOW",
        "LAST_PRICE",
        "WEEKLY_GROWTH",
        "MONTHLY_GROWTH",
        "YEARLY_GROWTH",
    ],
    functions=[compute_price],
)


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    indicators = [ADX, MACD, RSI, STOCH, SMA, ADOSC, AD, OBV, ATR, NATR, TRANGE]
    expected_columns = [c for i in indicators for c in i.expected_columns]
    for indicator in indicators:
        data = pd.concat([data, indicator.call(data)], axis=1)
    if not set(expected_columns).issubset(set(data.columns)):
        raise ValueError(
            f"Expected columns {expected_columns} not found in data columns {data.columns.tolist()}"
        )
    return data
