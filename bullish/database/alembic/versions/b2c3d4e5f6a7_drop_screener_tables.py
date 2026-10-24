"""drop the tables of the removed screener, jobs, news and portfolio layers

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-10-24 08:00:00.000000

"""

from typing import Sequence, Union

from alembic import op


revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

OBSOLETE_TABLES = (
    "analysis",
    "signalseries",
    "industryview",
    "industryreturns",
    "backtestresult",
    "filteredresults",
    "jobtracker",
    "openai",
    "portfolio",
    "subject",
    "view",
)


def upgrade() -> None:
    for table in OBSOLETE_TABLES:
        op.execute(f"DROP TABLE IF EXISTS {table}")


def downgrade() -> None:
    # The removed layers are gone; their tables are not recreated.
    pass
