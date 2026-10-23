"""kpi store: long kpi table, score table and run log

Revision ID: a1b2c3d4e5f6
Revises: 55553b8098f9
Create Date: 2026-10-23 23:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import sqlmodel


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "55553b8098f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "kpi",
        sa.Column("entity_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("entity", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.PrimaryKeyConstraint("entity_type", "entity", "name", "date"),
    )
    with op.batch_alter_table("kpi", schema=None) as batch_op:
        batch_op.create_index(
            "ix_kpi_lookup", ["entity_type", "date", "name"], unique=False
        )
        batch_op.create_index(
            "ix_kpi_entity", ["entity_type", "entity", "name"], unique=False
        )
    op.create_table(
        "score",
        sa.Column("entity_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("entity", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("score", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("value", sa.Float(), nullable=False),
        sa.Column("coverage", sa.Float(), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=True),
        sa.Column("components", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("entity_type", "entity", "score", "date"),
    )
    with op.batch_alter_table("score", schema=None) as batch_op:
        batch_op.create_index(
            "ix_score_lookup", ["entity_type", "score", "date"], unique=False
        )
    op.create_table(
        "kpirun",
        sa.Column("as_of", sa.Date(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("universe", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("entities", sa.Integer(), nullable=False),
        sa.Column("kpis", sa.Integer(), nullable=False),
        sa.Column("scores", sa.Integer(), nullable=False),
        sa.Column("base_currency", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("notes", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.PrimaryKeyConstraint("as_of"),
    )


def downgrade() -> None:
    op.drop_table("kpirun")
    with op.batch_alter_table("score", schema=None) as batch_op:
        batch_op.drop_index("ix_score_lookup")
    op.drop_table("score")
    with op.batch_alter_table("kpi", schema=None) as batch_op:
        batch_op.drop_index("ix_kpi_entity")
        batch_op.drop_index("ix_kpi_lookup")
    op.drop_table("kpi")
