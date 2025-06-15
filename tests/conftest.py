from pathlib import Path

import pytest

from bullish.database.crud import BullishDb

DATABASE_PATH = Path(__file__).parent / "data" / "bear.db"


@pytest.fixture
def bullish_db() -> BullishDb:
    return BullishDb(database_path=DATABASE_PATH)
