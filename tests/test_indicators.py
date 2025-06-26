import pandas as pd

from bullish.analysis.indicators import (
    indicators_factory,
    compute_macd,
    compute_rsi,
    compute_stoch,
    Indicators,
    compute_mfi,
    compute_patterns,
    compute_roc,
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
        if indicator.name not in ["CANDLESTICKS"]:
            assert all(
                (s.date is not None or s.value is not None) for s in indicator.signals
            )


def test_create_model() -> None:
    indicators = Indicators()
    indicators_model = indicators.create_indicator_model()
    assert indicators_model is not None
