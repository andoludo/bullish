"""bullish's own tables: the long KPI store, composite scores and the run log.

Everything else bullish needs lives in the bearish schema of the same file.
"""

import datetime as dt

from sqlalchemy import Index
from sqlmodel import Field, SQLModel


class KpiORM(SQLModel, table=True):
    """Long store of every KPI value: one row per (entity_type, entity, name, date)."""

    __tablename__ = "kpi"
    __table_args__ = (
        Index("ix_kpi_lookup", "entity_type", "date", "name"),
        Index("ix_kpi_entity", "entity_type", "entity", "name"),
        {"extend_existing": True},
    )
    entity_type: str = Field(primary_key=True)
    entity: str = Field(primary_key=True)
    name: str = Field(primary_key=True)
    date: dt.date = Field(primary_key=True)
    value: float


class ScoreORM(SQLModel, table=True):
    """Composite scores with their components, one row per (entity_type, entity, score, date)."""

    __tablename__ = "score"
    __table_args__ = (
        Index("ix_score_lookup", "entity_type", "score", "date"),
        {"extend_existing": True},
    )
    entity_type: str = Field(primary_key=True)
    entity: str = Field(primary_key=True)
    score: str = Field(primary_key=True)
    date: dt.date = Field(primary_key=True)
    value: float
    coverage: float
    eligible: bool = True
    rank: int | None = None
    components: str = Field(
        default="{}", description="JSON: block scores and KPI percentiles"
    )


class KpiRunORM(SQLModel, table=True):
    __tablename__ = "kpirun"
    __table_args__ = {"extend_existing": True}  # noqa: RUF012
    as_of: dt.date = Field(primary_key=True)
    created_at: dt.datetime
    universe: str = "all"
    entities: int = 0
    kpis: int = 0
    scores: int = 0
    base_currency: str = "EUR"
    notes: str = ""
