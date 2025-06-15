import tempfile
from pathlib import Path

import pytest

from bullish.database.crud import BullishDb
from bullish.view import BaseViews, TestView, View


def test_base_views(bullish_db: BullishDb) -> None:
    """Test the BaseViews class with a simple query."""
    views = BaseViews(
        view_name="test", query="SELECT symbol, name, source FROM analysis"
    )
    with tempfile.TemporaryDirectory() as d:
        views.compute(bullish_db, Path(d))
        views = bullish_db.read_query("SELECT * FROM view")
        assert not views.empty


def test_views(bullish_db: BullishDb):
    with tempfile.TemporaryDirectory() as d:
        TestView().compute(bullish_db, Path(d))
        query = """
        SELECT * FROM view;
        """
        data = bullish_db.read_query(query)
        assert not data.empty


def test_views_plot(bullish_db: BullishDb):
    view = View(symbol="NVDA", source="Yfinance", exchange="NMS")
    with tempfile.TemporaryDirectory() as d:
        view.plot(bullish_db, Path(d), show=True)
    assert True
