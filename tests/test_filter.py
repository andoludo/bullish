import datetime
from datetime import date

import numpy as np

from bullish.analysis.filter import (
    FilterQuery,
    INCOME_GROUP,
    CASH_FLOW_GROUP,
    EPS_GROUP,
    PROPERTIES_GROUP,
)
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
        "order_by_desc": "price_per_earning_ratio",
    }
    view_query = FilterQuery.model_validate(data)
    data = bullish_view.read_filter_query(view_query)
    assert np.all(np.diff(data.price_per_earning_ratio.values) < 0)
