import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from bearish.models.base import Ticker
from bearish.models.price.prices import Prices
from bullish.analysis.analysis import run_analysis, run_signal_series_analysis
from bullish.database.crud import BullishDb
from bearish.database.crud import BearishDb  # type: ignore
from tickermood.database.scripts.upgrade import upgrade


DATABASE_PATH = Path(__file__).parent / "data" / "bear.db"


@pytest.fixture(scope="session")
def bullish_db() -> BullishDb:
    return BullishDb(database_path=DATABASE_PATH)


@pytest.fixture(scope="session")
def bullish_db_with_analysis(bullish_db: BullishDb) -> BullishDb:
    run_analysis(bullish_db)
    return bullish_db


@pytest.fixture(scope="session")
def bullish_view(bullish_db: BullishDb) -> BullishDb:
    run_analysis(bullish_db)
    return bullish_db


@pytest.fixture(scope="session")
def data_aapl(bullish_db: BullishDb) -> pd.DataFrame:
    ticker = Ticker(symbol="AAPL")
    prices = Prices.from_ticker(bullish_db, ticker)
    return prices.to_dataframe()


@pytest.fixture(scope="session")
def bullish_db_with_signal_series(bullish_view: BullishDb) -> BullishDb:
    return bullish_view


@pytest.fixture(scope="session")
def custom_filter_path() -> Path:
    return Path(__file__).parent / "data" / "custom_filter.json"
