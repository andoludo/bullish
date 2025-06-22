from bullish.database.crud import BullishDb
from bullish.analysis.filter import FilterQuery


def test_read_filter_query(bullish_view: BullishDb) -> None:
    view_query = FilterQuery(price_per_earning_ratio=100000000)
    data = bullish_view.read_filter_query(view_query)
    assert not data.empty
