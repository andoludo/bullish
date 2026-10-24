from pathlib import Path

import pandas as pd
import pytest
from bearish.models.price.prices import Prices

from bullish.database.crud import BullishDb

DATA_DIR = Path(__file__).parent / "data"
DATABASE_PATH = DATA_DIR / "bear.db"
# A price fixture committed to the repository; the indicator tests never need
# the large, Git-LFS-tracked ``bear.db``.
PRICES_CSV = DATA_DIR / "prices.csv"

_SQLITE_MAGIC = b"SQLite format 3\x00"


def _is_real_sqlite(path: Path) -> bool:
    """True when ``path`` is a genuine SQLite database (not a Git-LFS pointer file)."""
    try:
        with path.open("rb") as file:
            return file.read(len(_SQLITE_MAGIC)) == _SQLITE_MAGIC
    except OSError:
        return False


DB_AVAILABLE = _is_real_sqlite(DATABASE_PATH)


@pytest.fixture(scope="session")
def legacy_db() -> BullishDb:
    """The recorded production-like database, migrated to both heads on open."""
    if not DB_AVAILABLE:
        pytest.skip("bear.db (Git-LFS) test database is not available")
    return BullishDb(database_path=DATABASE_PATH)


@pytest.fixture(scope="session")
def data_aapl() -> pd.DataFrame:
    return Prices.from_csv(PRICES_CSV).to_dataframe()
