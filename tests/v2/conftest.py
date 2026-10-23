from pathlib import Path

import pytest

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.database.crud import BullishDb
from tests.v2.synthetic import AS_OF, build_synthetic_database


@pytest.fixture(scope="session")
def synthetic_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("v2") / "synthetic.db"
    build_synthetic_database(path)
    return path


@pytest.fixture(scope="session")
def bullish_db_v2(synthetic_path: Path) -> BullishDb:
    return BullishDb(database_path=synthetic_path)


@pytest.fixture(scope="session")
def settings() -> BullishSettings:
    return BullishSettings(
        base_currency="EUR", min_market_cap=0.0, min_dollar_volume=0.0
    )


@pytest.fixture(scope="session")
def ctx(bullish_db_v2: BullishDb, settings: BullishSettings) -> DataContext:
    return DataContext(db=bullish_db_v2, settings=settings, as_of=AS_OF)
