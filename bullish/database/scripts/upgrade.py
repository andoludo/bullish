import os
from pathlib import Path
import sqlite3
from alembic import command
from alembic.config import Config

from bullish.database.scripts.stamp import stamp
from alembic.script import ScriptDirectory

DATABASE_PATH = Path(__file__).parents[3] / "tests" / "data" / "bear.db"


def upgrade(database_path: Path) -> None:
    root_folder = Path(__file__).parents[1]
    database_url = f"sqlite:///{database_path}"
    os.environ.update({"DATABASE_URL": database_url})
    alembic_cfg = Config(root_folder / "alembic" / "alembic.ini")
    alembic_cfg.set_main_option("script_location", str(root_folder / "alembic"))
    script = ScriptDirectory.from_config(alembic_cfg)

    versions = [rev.revision for rev in script.walk_revisions()]
    with sqlite3.connect(database_path) as conn:
        cursor = conn.execute("SELECT version_num FROM alembic_version;")
        row = cursor.fetchall()
        if row and row[0] and row[0][0] not in versions:
            stamp(database_path)
    command.upgrade(alembic_cfg, "head")


if __name__ == "__main__":
    upgrade(DATABASE_PATH)
