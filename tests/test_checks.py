from bullish.utils.checks import (
    compatible_bearish_database,
    compatible_bullish_database,
)
from tests.conftest import DATABASE_PATH


def test_compatible_bearish_database() -> None:
    assert compatible_bearish_database(DATABASE_PATH)


def test_compatible_bullish_database() -> None:
    assert compatible_bullish_database(DATABASE_PATH)
