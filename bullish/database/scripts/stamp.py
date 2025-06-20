import os
import sqlite3
from pathlib import Path

from alembic.config import Config

from bullish.database.settings import TEST_DATABASE_PATH


def stamp(database_path: Path) -> None:
    database_url = f"sqlite:///{database_path}"
    root_folder = Path(__file__).parents[1]
    os.environ.update({"DATABASE_URL": database_url})
    with sqlite3.connect(database_path) as conn:
        conn.execute("DROP TABLE IF EXISTS  alembic_version;")
        conn.execute("DROP TABLE IF EXISTS view;")
        conn.execute("DROP TABLE IF EXISTS analysis;")
        conn.execute("DROP TABLE IF EXISTS filteredresults;")
        conn.commit()
    alembic_cfg = Config(root_folder / "alembic" / "alembic.ini")
    alembic_cfg.set_main_option("script_location", str(root_folder / "alembic"))


if __name__ == "__main__":
    stamp(TEST_DATABASE_PATH)
