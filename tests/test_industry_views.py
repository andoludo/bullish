from bearish.models.base import Ticker  # type: ignore
from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from streamlit_file_browser import st_file_browser  # type: ignore

from bullish.analysis.industry_views import (
    IndustryView,
    compute_industry_view,
    get_industry_comparison_data,
)
from bullish.database.crud import BullishDb


def test_compute_industry_return(bullish_view: BullishDb) -> None:
    views = IndustryView.from_db(bullish_view, "Biotechnology", "Belgium")
    assert views


def test_compute_industry_view(bullish_view: BullishDb) -> None:
    compute_industry_view(bullish_view)
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol="HYL.BR")]))
    prices = bullish_view.read_series(query, months=12)
    data = Prices(prices=prices).to_dataframe()

    merged_data = get_industry_comparison_data(
        bullish_view, data, "Mean", "Biotechnology", "Belgium"
    ).dropna()
    assert not merged_data.empty
    assert "symbol" in merged_data.columns
    assert "Biotechnology" in merged_data.columns
