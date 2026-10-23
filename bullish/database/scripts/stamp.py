"""Helpers for databases that predate bullish's alembic migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from bullish.database.settings import TEST_DATABASE_PATH

LEGACY_TABLES = ("view", "analysis", "filteredresults", "jobtracker")


def drop_legacy_tables(conn: sqlite3.Connection) -> None:
    """Drop the tables an old, pre-alembic bullish created; the first bullish migration recreates them."""
    for table in LEGACY_TABLES:
        conn.execute(f"DROP TABLE IF EXISTS {table};")
    conn.commit()


def stamp(database_path: Path) -> None:
    with sqlite3.connect(database_path) as conn:
        drop_legacy_tables(conn)


if __name__ == "__main__":
    stamp(TEST_DATABASE_PATH)
