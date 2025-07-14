from bullish.analysis.returns import IndustryReturns, compute_industry
from bullish.database.crud import BullishDb


def test_compute_industry_return(bullish_view: BullishDb) -> None:
    returns = IndustryReturns.from_db(bullish_view, "Biotechnology", "Belgium")
    assert returns


def test_compute_industry(bullish_view: BullishDb) -> None:
    compute_industry(bullish_view)
    results = bullish_view.read_returns("Mean", "Biotechnology", "Belgium")
    assert results
