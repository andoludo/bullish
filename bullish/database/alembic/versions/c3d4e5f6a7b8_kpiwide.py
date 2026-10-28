"""kpiwide: one compressed wide frame per (entity_type, date) for fast reads; drop the kpi index
the primary key already covers

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-10-28 09:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


revision: str = "c3d4e5f6a7b8"
down_revision: Union[str, None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kpiwide",
        sa.Column("entity_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("cols", sa.Integer(), nullable=False),
        sa.Column("index_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("columns_json", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("payload", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("entity_type", "date"),
    )
    # (entity_type, entity, name) is a prefix of kpi's primary key, so this
    # index only cost a third of every write.
    op.execute("DROP INDEX IF EXISTS ix_kpi_entity")


def downgrade() -> None:
    with op.batch_alter_table("kpi", schema=None) as batch_op:
        batch_op.create_index(
            "ix_kpi_entity", ["entity_type", "entity", "name"], unique=False
        )
    op.drop_table("kpiwide")
