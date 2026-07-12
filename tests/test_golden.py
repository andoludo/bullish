"""Golden (snapshot) tests for the deterministic analysis layer.

These tests are intentionally self-contained: they rely only on the small price
CSV fixtures committed to the repository (``tests/data/prices*.csv``) and never
touch the large, Git-LFS-tracked ``bear.db``. They lock down the numeric output
of every technical indicator and the SQL produced by ``FilterQuery`` so that any
accidental change in behaviour is caught immediately.

To regenerate the golden file after an intentional change::

    uv run python -m tests.test_golden          # or: python tests/test_golden.py

which rewrites ``tests/data/golden/indicators.json``. Review the diff carefully
before committing it.
"""

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from bearish.models.price.prices import Prices

from bullish.analysis.functions import (
    AD,
    ADOSC,
    ADX,
    ATR,
    MACD,
    MFI,
    NATR,
    OBV,
    PRICE,
    ROC,
    RSI,
    SMA,
    STOCH,
    TRANGE,
    VOLUME,
    IndicatorFunction,
    bollinger_bands,
    support_resistance,
)
from bullish.analysis.filter import FilterQuery

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_PATH = DATA_DIR / "golden" / "indicators.json"

# Price fixtures (ticker name -> CSV file) used to drive the snapshots.
PRICE_FIXTURES = {
    "DAL": "prices.csv",
    "MSFT": "prices_msft.csv",
    "NVDA": "prices_nvda.csv",
    "PLTR": "prices_pltr.csv",
    "TSM": "prices_tsm.csv",
}

# All indicator functions whose numeric output we lock down.
INDICATORS: Dict[str, IndicatorFunction] = {
    "ADX": ADX,
    "MACD": MACD,
    "RSI": RSI,
    "STOCH": STOCH,
    "MFI": MFI,
    "ROC": ROC,
    "SMA": SMA,
    "ADOSC": ADOSC,
    "AD": AD,
    "OBV": OBV,
    "ATR": ATR,
    "NATR": NATR,
    "TRANGE": TRANGE,
    "VOLUME": VOLUME,
    "PRICE": PRICE,
}

# Deterministic filter queries (no date-dependent fields) -> golden key.
FILTER_QUERIES: Dict[str, Dict[str, Any]] = {
    "income_cashflow_price_order": {
        "last_price": (1, 1000),
        "income": ["positive_net_income"],
        "cash_flow": ["positive_free_cash_flow", "growing_operating_cash_flow"],
        "order_by_desc": "market_capitalization",
    },
    "market_cap_country": {
        "market_capitalization": [5e9, 1e12],
        "country": ["Germany", "France"],
    },
    "price_per_earning_limit": {
        "price_per_earning_ratio": [10.0, 40.0],
        "order_by_asc": "last_price",
        "limit": "50",
    },
}

_FLOAT_PRECISION = 6


def _coerce(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, (int,)):
        return int(value)
    return round(float(value), _FLOAT_PRECISION)


def _column_summary(series: pd.Series) -> Dict[str, Any]:
    valid = series.dropna()
    return {
        "count": int(valid.shape[0]),
        "last": _coerce(valid.iloc[-1]) if not valid.empty else None,
    }


def _load_prices(csv_name: str) -> pd.DataFrame:
    return Prices.from_csv(DATA_DIR / csv_name).to_dataframe()


def snapshot() -> Dict[str, Any]:
    """Compute the full golden snapshot from the committed price fixtures."""
    result: Dict[str, Any] = {"indicators": {}, "extras": {}, "queries": {}}

    for ticker, csv_name in PRICE_FIXTURES.items():
        data = _load_prices(csv_name)
        per_ticker: Dict[str, Any] = {}
        for name, indicator in INDICATORS.items():
            frame = indicator.call(data)
            per_ticker[name] = {
                col: _column_summary(frame[col]) for col in indicator.expected_columns
            }
        result["indicators"][ticker] = per_ticker

        sr = support_resistance(data, window=5)
        bb = bollinger_bands(data)
        result["extras"][ticker] = {
            "support": _coerce(sr.support.value),
            "support_previous": _coerce(sr.support.previous),
            "resistance": _coerce(sr.resistance.value),
            "resistance_previous": _coerce(sr.resistance.previous),
            "bbands_columns": sorted(str(c) for c in bb.columns),
            "bbands_last": {
                str(col): _column_summary(bb[col])["last"] for col in bb.columns
            },
        }

    for key, payload in FILTER_QUERIES.items():
        result["queries"][key] = FilterQuery.model_validate(payload).to_query()

    return result


def _write_golden() -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")


def _load_golden() -> Dict[str, Any]:
    return json.loads(GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def golden() -> Dict[str, Any]:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"Golden file {GOLDEN_PATH} is missing. "
            "Regenerate it with `python tests/test_golden.py`."
        )
    return _load_golden()


@pytest.fixture(scope="module")
def computed() -> Dict[str, Any]:
    return snapshot()


@pytest.mark.parametrize("ticker", sorted(PRICE_FIXTURES))
@pytest.mark.parametrize("indicator", sorted(INDICATORS))
def test_indicator_matches_golden(
    indicator: str, ticker: str, golden: Dict[str, Any], computed: Dict[str, Any]
) -> None:
    expected = golden["indicators"][ticker][indicator]
    actual = computed["indicators"][ticker][indicator]
    assert set(actual) == set(expected)
    for column, summary in expected.items():
        assert actual[column]["count"] == summary["count"], column
        exp_last, act_last = summary["last"], actual[column]["last"]
        if isinstance(exp_last, float):
            assert act_last == pytest.approx(exp_last, rel=1e-6, abs=1e-6), column
        else:
            assert act_last == exp_last, column


@pytest.mark.parametrize("ticker", sorted(PRICE_FIXTURES))
def test_extras_match_golden(
    ticker: str, golden: Dict[str, Any], computed: Dict[str, Any]
) -> None:
    expected = golden["extras"][ticker]
    actual = computed["extras"][ticker]
    assert actual["bbands_columns"] == expected["bbands_columns"]
    for key in ("support", "support_previous", "resistance", "resistance_previous"):
        assert actual[key] == pytest.approx(expected[key], rel=1e-6, abs=1e-6), key
    for col, exp_last in expected["bbands_last"].items():
        act_last = actual["bbands_last"][col]
        if isinstance(exp_last, float):
            assert act_last == pytest.approx(exp_last, rel=1e-6, abs=1e-6), col
        else:
            assert act_last == exp_last, col


@pytest.mark.parametrize("key", sorted(FILTER_QUERIES))
def test_filter_query_matches_golden(
    key: str, golden: Dict[str, Any], computed: Dict[str, Any]
) -> None:
    assert computed["queries"][key] == golden["queries"][key]


if __name__ == "__main__":
    _write_golden()
    print(f"Wrote golden snapshot to {GOLDEN_PATH}")
