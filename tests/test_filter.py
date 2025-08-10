import datetime
import json
from datetime import date
from pathlib import Path

import numpy as np

from bullish.analysis.filter import (
    FilterQuery,
    INCOME_GROUP,
    CASH_FLOW_GROUP,
    EPS_GROUP,
    PROPERTIES_GROUP,
)
from bullish.analysis.predefined_filters import NamedFilterQuery, read_custom_filters
from bullish.database.crud import BullishDb


def test_read_filter_query(bullish_view: BullishDb) -> None:
    today = date.today()
    start_date = today - datetime.timedelta(days=30 * 10)

    data = {
        "last_price": (1, 1000),
        "last_price_date": (start_date, today),
        "income": ["positive_net_income"],
        "cash_flow": ["positive_free_cash_flow", "growing_operating_cash_flow"],
    }
    view_query = FilterQuery.model_validate(data)
    data = bullish_view.read_filter_query(view_query)
    assert not data.empty


def test_order_by(bullish_view: BullishDb):
    today = date.today()
    start_date = today - datetime.timedelta(days=30 * 10)

    data = {
        "last_price": (1, 1000),
        "last_price_date": (start_date, today),
        "income": ["positive_net_income"],
        "cash_flow": ["positive_free_cash_flow", "growing_operating_cash_flow"],
        "order_by_desc": "market_capitalization",
    }
    view_query = FilterQuery.model_validate(data)
    assert (
        view_query.to_query()
        == "positive_net_income=1 AND positive_free_cash_flow=1 AND growing_operating_cash_flow=1 AND last_price BETWEEN 1.0 AND 1000.0 ORDER BY market_capitalization DESC LIMIT 1000"
    )
    data = bullish_view.read_filter_query(view_query)
    assert np.all(np.diff(data.market_capitalization.values) < 0)


def test_load_custom_predefined_filters(custom_filter_path: Path) -> None:
    custom_filters = read_custom_filters(custom_filter_path)
    assert custom_filters
    assert isinstance(custom_filters, list)
