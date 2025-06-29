import datetime
from datetime import date

from bullish.analysis.filter import FilterQuery
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
