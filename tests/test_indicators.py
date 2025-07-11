import datetime

import numpy as np
import pandas as pd

from bullish.analysis.indicators import (
    indicators_factory,
    Indicators,
)
from bullish.analysis.functions import (
    compute_macd,
    compute_rsi,
    compute_stoch,
    compute_mfi,
    compute_roc,
    compute_patterns,
    compute_adx,
    compute_pandas_ta_adx,
    ADX,
    compute_pandas_ta_macd,
    RSI,
    compute_pandas_ta_rsi,
    compute_pandas_ta_stoch,
    STOCH,
    compute_pandas_ta_mfi,
    MFI,
    compute_sma,
    compute_pandas_ta_sma,
    SMA,
    add_indicators,
    compute_adosc,
    compute_pandas_ta_adosc,
    ADOSC,
    compute_ad,
    compute_pandas_ta_ad,
    AD,
    compute_pandas_ta_obv,
    compute_obv,
    OBV,
    compute_atr,
    compute_pandas_ta_atr,
    ATR,
    NATR,
    compute_pandas_ta_natr,
    compute_natr,
    compute_trange,
    compute_pandas_ta_trange,
    TRANGE,
    compute_price,
    PRICE,
    cross_value,
    cross_value_series,
    compute_percentile_return_after_rsi_crossover,
)


def test_indicator(data_aapl: pd.DataFrame) -> None:
    indicators = indicators_factory()
    indicator = indicators[0]
    indicator.compute(data_aapl)
    assert not indicator._data.empty
    assert all(s.date is not None for s in indicator.signals)


def test_compute(data_aapl: pd.DataFrame) -> None:
    d1 = compute_macd(data_aapl)
    d2 = compute_rsi(data_aapl)
    d3 = compute_stoch(data_aapl)
    d4 = compute_mfi(data_aapl)
    d5 = compute_patterns(data_aapl)
    d6 = compute_roc(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty
    assert not d4.empty
    assert not d5.empty
    assert not d6.empty


def test_indicators_factory(data_aapl: pd.DataFrame) -> None:
    indicators = indicators_factory()
    for indicator in indicators:
        indicator.compute(data_aapl)
        assert not indicator._data.empty
        if indicator.name not in ["CANDLESTICKS", "PRICE"]:
            try:
                assert all(
                    (s.date is not None or s.value is not None)
                    for s in indicator.signals
                )
            except:
                raise


def test_create_model() -> None:
    indicators = Indicators()
    indicators_model = indicators.create_indicator_models()
    assert indicators_model is not None


def test_indicator_function_adx(data_aapl: pd.DataFrame) -> None:
    compute_adx(data_aapl)
    compute_pandas_ta_adx(data_aapl)
    d1 = ADX.call(data_aapl)
    assert not d1.empty


def test_indicator_function_macd(data_aapl: pd.DataFrame) -> None:
    d1 = compute_macd(data_aapl)
    d2 = compute_pandas_ta_macd(data_aapl)
    d3 = ADX.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_rsi(data_aapl: pd.DataFrame) -> None:
    d1 = compute_rsi(data_aapl)
    d2 = compute_pandas_ta_rsi(data_aapl)
    d3 = RSI.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_stoch(data_aapl: pd.DataFrame) -> None:
    d1 = compute_stoch(data_aapl)
    d2 = compute_pandas_ta_stoch(data_aapl)
    d3 = STOCH.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_mfi(data_aapl: pd.DataFrame) -> None:
    d1 = compute_mfi(data_aapl)
    d2 = compute_pandas_ta_mfi(data_aapl)
    d3 = MFI.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_sma(data_aapl: pd.DataFrame) -> None:
    d1 = compute_sma(data_aapl)
    d2 = compute_pandas_ta_sma(data_aapl)
    d3 = SMA.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_add_indicators(data_aapl: pd.DataFrame) -> None:
    data = add_indicators(data_aapl)
    assert not data.empty


def test_indicator_function_adosc(data_aapl: pd.DataFrame) -> None:
    d1 = compute_adosc(data_aapl)
    d2 = compute_pandas_ta_adosc(data_aapl)
    d3 = ADOSC.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_ad(data_aapl: pd.DataFrame) -> None:
    d1 = compute_ad(data_aapl)
    d2 = compute_pandas_ta_ad(data_aapl)
    d3 = AD.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_obv(data_aapl: pd.DataFrame) -> None:
    d1 = compute_obv(data_aapl)
    d2 = compute_pandas_ta_obv(data_aapl)
    d3 = OBV.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_atr(data_aapl: pd.DataFrame) -> None:
    d1 = compute_atr(data_aapl)
    d2 = compute_pandas_ta_atr(data_aapl)
    d3 = ATR.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_natr(data_aapl: pd.DataFrame) -> None:
    d1 = compute_natr(data_aapl)
    d2 = compute_pandas_ta_natr(data_aapl)
    d3 = NATR.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_function_trange(data_aapl: pd.DataFrame) -> None:
    d1 = compute_trange(data_aapl)
    d2 = compute_pandas_ta_trange(data_aapl)
    d3 = TRANGE.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert not d3.empty


def test_indicator_price(data_aapl: pd.DataFrame) -> None:
    d1 = compute_price(data_aapl)
    d2 = PRICE.call(data_aapl)
    assert not d1.empty
    assert not d2.empty
    assert set(d2.columns) == set(PRICE.expected_columns)


def test_compute_mean_return_after_rsi_crossover(data_aapl: pd.DataFrame) -> None:

    value = compute_percentile_return_after_rsi_crossover(RSI.call(data_aapl))
    assert isinstance(value, float)
    assert value > 0
