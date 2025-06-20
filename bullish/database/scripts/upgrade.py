import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.util import CommandError

from bullish.database.scripts.stamp import stamp

DATABASE_PATH = Path(__file__).parents[3] / "tests" / "data" / "bear.db"


def upgrade(database_path: Path) -> None:
    root_folder = Path(__file__).parents[1]
    database_url = f"sqlite:///{database_path}"
    os.environ.update({"DATABASE_URL": database_url})
    alembic_cfg = Config(root_folder / "alembic" / "alembic.ini")
    alembic_cfg.set_main_option("script_location", str(root_folder / "alembic"))
    try:
        command.upgrade(alembic_cfg, "head")
    except CommandError:
        stamp(database_path)
        command.upgrade(alembic_cfg, "head")


if __name__ == "__main__":
    upgrade(DATABASE_PATH)
