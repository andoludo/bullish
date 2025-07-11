import sqlite3
from contextlib import contextmanager
from pathlib import Path
from sqlite3 import Connection
from typing import List, Generator

from bearish.database.schemas import *  # type: ignore # noqa: F403
from bullish.database.schemas import *  # noqa: F403


@contextmanager
def get_sqlite_connection(database_path: Path) -> Generator[Connection, None, None]:
    conn = sqlite3.connect(database_path)
    try:
        yield conn
    finally:
        conn.close()


def get_table_names(module_name: str) -> List[str]:
    return [
        value.__tablename__
        for _, value in locals().items()
        if issubclass(value, SQLModel)  # type: ignore # noqa: F405
        and hasattr(value, "__tablename__")
        and hasattr(value, "__table__")
        and module_name in value.__module__
    ]


def get_table_names_from_path(database_path: Path) -> List[str]:
    with get_sqlite_connection(database_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return tables


def empty_analysis_table(database_path: Path) -> bool:
    if "analysis" not in get_table_names_from_path(database_path):
        return True
    with get_sqlite_connection(database_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM analysis")
        count = cursor.fetchone()[0]
        conn.close()
        return bool(count == 0)


def _compatible_table(database_path: Path, module_name: str) -> bool:
    if not database_path.exists() and not database_path.is_file():
        raise FileNotFoundError(f"Database file {database_path} does not exist.")
    table_names = get_table_names(module_name)
    return set(table_names).issubset(get_table_names_from_path(database_path))


def compatible_bearish_database(database_path: Path) -> bool:
    return _compatible_table(database_path, "bearish.database.schemas")


def compatible_bullish_database(database_path: Path) -> bool:
    return _compatible_table(database_path, "bullish.database.schemas")
