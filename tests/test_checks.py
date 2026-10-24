import pytest

from bullish.database.crud import BullishDb
from bullish.utils.checks import (
    compatible_bearish_database,
    compatible_bullish_database,
)
from tests.conftest import DB_AVAILABLE

pytestmark = pytest.mark.skipif(
    not DB_AVAILABLE, reason="bear.db (Git-LFS) test database is not available"
)


def test_compatible_bearish_database(legacy_db: BullishDb) -> None:
    assert compatible_bearish_database(legacy_db.database_path)


def test_compatible_bullish_database(legacy_db: BullishDb) -> None:
    assert compatible_bullish_database(legacy_db.database_path)
