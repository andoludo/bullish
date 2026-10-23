"""Bring a bearish database up to date for bullish: bearish migrations first, then bullish's own.

Both packages migrate the same SQLite file. Bearish uses alembic's default
``alembic_version`` table; bullish keeps its revision in
``bullish_alembic_version`` so that neither clobbers the other. Databases
stamped by older bullish versions wrote a bullish revision *into*
``alembic_version`` (and lost the bearish one), so the first upgrade of such
a file moves that revision to the bullish table and recovers the bearish
revision from the schema itself (:data:`BEARISH_MARKERS`).
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Literal

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from bearish.database.scripts.upgrade import upgrade as bearish_upgrade  # type: ignore

from bullish.database.scripts.stamp import drop_legacy_tables

logger = logging.getLogger(__name__)

DATABASE_PATH = Path(__file__).parents[3] / "tests" / "data" / "bear.db"
BEARISH_VERSION_TABLE = "alembic_version"
BULLISH_VERSION_TABLE = "bullish_alembic_version"

Check = Literal["table", "column", "column_absent"]
# Bearish revisions in chain order with a schema marker each: the newest marker
# present tells which bearish revision a file is at when its version row was
# overwritten by an old bullish stamp.
BEARISH_MARKERS: tuple[tuple[str, Check, str, str | None], ...] = (
    ("79d40a2eeef5", "table", "index", None),
    ("ca2d5f47456a", "table", "priceindex", None),
    ("0a4bbaa82cd1", "table", "sec", None),
    ("f7b416c59e27", "column", "sec", "value"),
    ("f94459e49cae", "table", "priceetf", None),
    ("4e44f10ee322", "table", "secshareincrease", None),
    ("ac8512e066af", "column", "secshareincrease", "name"),
    ("3977f65e4dfb", "column", "earningsdate", "is_forward"),
    ("609345acf556", "table", "pricetarget", None),
    ("d84c9fa2d2a7", "table", "filing", None),
    ("dd9c63b1d909", "column_absent", "financialmetrics", "pe_ratio"),
    ("e4fed790ac81", "table", "macroseries", None),
    ("e68e3ec73125", "table", "commodity", None),
    ("42a7c5b11570", "table", "holding13f", None),
    ("9c21f96faefe", "table", "marketevent", None),
    ("8c1f2a9d4b31", "table", "sectoralias", None),
    ("b7e4d2c90a15", "column", "filing", "items"),
    ("c3a9f1e7d250", "column", "financialmetrics", "interest_expense"),
    ("d4b8e6a1f092", "table", "equitysnapshot", None),
    ("e5c7a3b2d918", "table", "pricecurrency", None),
    ("f1a6c4d8e273", "table", "indexmember", None),
    ("a9d3e5f7c164", "table", "coverage", None),
)


def _alembic_config(database_url: str) -> Config:
    root_folder = Path(__file__).parents[1]
    os.environ.update({"DATABASE_URL": database_url})
    alembic_cfg = Config(root_folder / "alembic" / "alembic.ini")
    alembic_cfg.set_main_option("script_location", str(root_folder / "alembic"))
    return alembic_cfg


def bullish_revisions(alembic_cfg: Config) -> set[str]:
    return {
        rev.revision
        for rev in ScriptDirectory.from_config(alembic_cfg).walk_revisions()
    }


def _tables(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    ).fetchall()
    return {row[0] for row in rows}


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}


def _version(conn: sqlite3.Connection, table: str) -> str | None:
    if table not in _tables(conn):
        return None
    row = conn.execute(f"SELECT version_num FROM {table}").fetchone()  # noqa: S608
    return row[0] if row else None


def detect_bearish_revision(conn: sqlite3.Connection) -> str:
    """Newest bearish revision whose schema marker is present."""
    tables = _tables(conn)
    found = None
    for revision, check, table, column in BEARISH_MARKERS:
        if check == "table":
            present = table in tables
        elif check == "column":
            present = table in tables and column in _columns(conn, table)
        else:
            present = table in tables and column not in _columns(conn, table)
        if present:
            found = revision
    if found is None:
        raise RuntimeError(
            "Cannot tell which bearish revision this database is at; run `bearish` on it first"
        )
    return found


def reconcile_version_tables(database_path: Path, alembic_cfg: Config) -> None:
    """Move a bullish revision written into ``alembic_version`` by older bullish releases to its own table."""
    with sqlite3.connect(database_path) as conn:
        tables = _tables(conn)
        if BULLISH_VERSION_TABLE in tables:
            return
        current = _version(conn, BEARISH_VERSION_TABLE)
        if current is None:
            if {"analysis", "view"} & tables:
                drop_legacy_tables(conn)  # pre-alembic bullish tables
            return
        if current in bullish_revisions(alembic_cfg):
            bearish_revision = detect_bearish_revision(conn)
            logger.info(
                f"Moving bullish revision {current} to {BULLISH_VERSION_TABLE}; bearish is at {bearish_revision}"
            )
            conn.execute(
                f"CREATE TABLE {BULLISH_VERSION_TABLE} (version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
            )
            conn.execute(
                f"INSERT INTO {BULLISH_VERSION_TABLE} (version_num) VALUES (?)",  # noqa: S608
                (current,),
            )
            conn.execute(
                f"UPDATE {BEARISH_VERSION_TABLE} SET version_num = ?",  # noqa: S608
                (bearish_revision,),
            )
            conn.commit()


def upgrade(database_path: Path) -> None:
    database_url = f"sqlite:///{database_path}"
    alembic_cfg = _alembic_config(database_url)
    reconcile_version_tables(Path(database_path), alembic_cfg)
    bearish_upgrade(database_url)
    os.environ.update({"DATABASE_URL": database_url})
    command.upgrade(alembic_cfg, "head")


if __name__ == "__main__":
    upgrade(DATABASE_PATH)
