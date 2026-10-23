"""A database stamped by an older bullish (bullish revision written into alembic_version) must still upgrade."""

import os
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config
from bearish.database.crud import BearishDb

from bullish.database.crud import BullishDb
from bullish.database.scripts.upgrade import (
    BEARISH_VERSION_TABLE,
    BULLISH_VERSION_TABLE,
    _alembic_config,
    detect_bearish_revision,
)
from bullish.utils.checks import (
    compatible_bearish_database,
    compatible_bullish_database,
)

BEARISH_ALEMBIC = Path(
    BearishDb.__module__.replace(".", "/")
).parent  # bearish/database


def _bearish_config(database_url: str) -> Config:
    import bearish.database.scripts.upgrade as bearish_scripts

    root = Path(bearish_scripts.__file__).parents[1]
    os.environ.update({"DATABASE_URL": database_url})
    cfg = Config(root / "alembic" / "alembic.ini")
    cfg.set_main_option("script_location", str(root / "alembic"))
    return cfg


def _columns(path: Path, table: str) -> set[str]:
    with sqlite3.connect(path) as conn:
        return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _version(path: Path, table: str) -> str:
    with sqlite3.connect(path) as conn:
        return conn.execute(f"SELECT version_num FROM {table}").fetchone()[0]


def _legacy_database(path: Path) -> None:
    """bearish at the pre-macro revision, then the old bullish stamp (bullish revision into alembic_version)."""
    url = f"sqlite:///{path}"
    command.upgrade(_bearish_config(url), "dd9c63b1d909")
    bullish_cfg = _alembic_config(url)
    with sqlite3.connect(path) as conn:
        conn.execute("DELETE FROM alembic_version")
        conn.execute(
            "INSERT INTO alembic_version (version_num) VALUES ('037dbd721319')"
        )
        conn.commit()
    # old bullish ran its migrations on the default version table: create the
    # bullish tables for real, then emulate the old layout (bullish revision in
    # alembic_version, no bullish_alembic_version)
    os.environ["DATABASE_URL"] = url
    command.upgrade(bullish_cfg, "55553b8098f9")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE alembic_version SET version_num = '55553b8098f9'")
        conn.execute("DROP TABLE bullish_alembic_version")
        conn.commit()


def test_detect_bearish_revision_from_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    assert "filed_date" in _columns(path, "financialmetrics")
    assert "interest_expense" not in _columns(path, "financialmetrics")
    with sqlite3.connect(path) as conn:
        assert detect_bearish_revision(conn) == "dd9c63b1d909"


def test_legacy_stamped_database_upgrades_to_both_heads(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    _legacy_database(path)
    assert (
        _version(path, BEARISH_VERSION_TABLE) == "55553b8098f9"
    )  # the broken legacy layout

    db = BullishDb(database_path=path)

    assert "interest_expense" in _columns(path, "financialmetrics")
    assert (
        "coverage"
        in db.read_query("SELECT name FROM sqlite_master WHERE type='table'")[
            "name"
        ].tolist()
    )
    assert _version(path, BULLISH_VERSION_TABLE) == "a1b2c3d4e5f6"
    assert _version(path, BEARISH_VERSION_TABLE) == "a9d3e5f7c164"
    assert compatible_bearish_database(path)
    assert compatible_bullish_database(path)
    # idempotent
    BullishDb(database_path=path)
    assert _version(path, BULLISH_VERSION_TABLE) == "a1b2c3d4e5f6"


def test_fresh_bearish_database_gets_bullish_tables(tmp_path: Path) -> None:
    path = tmp_path / "fresh.db"
    BearishDb(database_path=path)
    assert _version(path, BEARISH_VERSION_TABLE) == "a9d3e5f7c164"
    BullishDb(database_path=path)
    assert _version(path, BEARISH_VERSION_TABLE) == "a9d3e5f7c164"
    assert _version(path, BULLISH_VERSION_TABLE) == "a1b2c3d4e5f6"
    assert compatible_bullish_database(path)
