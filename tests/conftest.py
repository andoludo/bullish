import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from bearish.models.base import Ticker
from bearish.models.price.prices import Prices
from bullish.analysis.analysis import run_analysis
from bullish.database.crud import BullishDb

DATABASE_PATH = Path(__file__).parent / "data" / "bear.db"
DATABASE_PATH_VIEW = Path(__file__).parent / "data" / "filter_bear.db"


def delete_tables(database_path: Path):
    with sqlite3.connect(database_path) as conn:
        conn.execute("DROP TABLE IF EXISTS  alembic_version;")
        conn.execute("DROP TABLE IF EXISTS jobtracker;")
        conn.execute("DROP TABLE IF EXISTS analysis;")
        conn.execute("DROP TABLE IF EXISTS view;")
        conn.execute("DROP TABLE IF EXISTS filteredresults;")
        conn.execute("DROP TABLE IF EXISTS subject;")
        conn.commit()


@pytest.fixture
def bullish_db() -> BullishDb:
    delete_tables(DATABASE_PATH)
    return BullishDb(database_path=DATABASE_PATH)


@pytest.fixture
def bullish_db_with_analysis(bullish_db: BullishDb) -> BullishDb:
    run_analysis(bullish_db)
    return bullish_db


@pytest.fixture
def bullish_view() -> BullishDb:
    return BullishDb(database_path=DATABASE_PATH_VIEW)


@pytest.fixture
def data_aapl(bullish_db: BullishDb) -> pd.DataFrame:
    ticker = Ticker(symbol="AAPL")
    prices = Prices.from_ticker(bullish_db, ticker)
    return prices.to_dataframe()
