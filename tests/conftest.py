from pathlib import Path

import pandas as pd
import pytest
from bearish.models.base import Ticker
from bearish.models.price.prices import Prices
from bullish.analysis.analysis import run_analysis
from bullish.database.crud import BullishDb


DATA_DIR = Path(__file__).parent / "data"
DATABASE_PATH = DATA_DIR / "bear.db"
# A price fixture committed to the repository (does not require the large,
# Git-LFS-tracked ``bear.db``). Used as a fallback so the price-based indicator
# tests can run even when the LFS database has not been fetched.
PRICES_CSV = DATA_DIR / "prices.csv"

_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_real_sqlite(path: Path) -> bool:
    """Return True when ``path`` is a genuine SQLite database.

    When the repository is checked out without fetching Git-LFS objects,
    ``bear.db`` is only a small text pointer file rather than the real
    database. Detect that case so DB-backed tests can be skipped cleanly
    instead of failing with an opaque ``sqlite3.DatabaseError``.
    """
    try:
        with path.open("rb") as file:
            return file.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:
        return False


DB_AVAILABLE = _is_real_sqlite(DATABASE_PATH)


@pytest.fixture(scope="session")
def bullish_db() -> BullishDb:
    if not DB_AVAILABLE:
        pytest.skip("bear.db (Git-LFS) test database is not available")
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
def data_aapl() -> pd.DataFrame:
    if DB_AVAILABLE:
        bullish_db = BullishDb(database_path=DATABASE_PATH)
        prices = Prices.from_ticker(bullish_db, Ticker(symbol="AAPL"))
        return prices.to_dataframe()
    # Fallback: deterministic price history committed to the repository.
    return Prices.from_csv(PRICES_CSV).to_dataframe()


@pytest.fixture(scope="session")
def bullish_db_with_signal_series(bullish_view: BullishDb) -> BullishDb:
    return bullish_view


@pytest.fixture(scope="session")
def custom_filter_path() -> Path:
    return DATA_DIR / "custom_filter.json"
