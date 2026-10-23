from datetime import date
from typing import Any

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from bullish.analysis.analysis import Analysis
from bullish.analysis.backtest import BacktestResult
from bullish.analysis.filter import FilteredResults
from bullish.analysis.indicators import SignalSeries
from bullish.analysis.industry_views import IndustryView
from bullish.analysis.openai import OpenAINews
from bullish.analysis.portfolio import Portfolio

from bullish.jobs.models import JobTracker
from sqlalchemy import Index
import datetime as dt


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


class BaseTable(SQLModel):
    symbol: str = Field(primary_key=True)
    source: str = Field(primary_key=True)


dynamic_indexes = tuple(
    Index(f"ix_analysis_{col}", col) for col in Analysis.model_fields
)


class OpenAINewsORM(SQLModel, OpenAINews, table=True):
    __tablename__ = "openai"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    symbol: str = Field(primary_key=True)
    news_date: date = Field(primary_key=True)


class PortfolioORM(SQLModel, Portfolio, table=True):
    __tablename__ = "portfolio"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    name: str = Field(primary_key=True)
    current_assets: list[dict[str, Any]] | None = Field(  # type: ignore
        default=None, sa_column=Column(JSON)
    )
    new_assets: list[dict[str, Any]] | None = Field(  # type: ignore
        default=None, sa_column=Column(JSON)
    )


class AnalysisORM(BaseTable, Analysis, table=True):
    __tablename__ = "analysis"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    news_summary: list[dict[str, Any]] | None = Field(
        default=None, sa_column=Column(JSON)
    )
    summary: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


AnalysisORM.__table_args__ = tuple(  # type: ignore # noqa: RUF005
    Index(f"ix_{AnalysisORM.__tablename__}_{col.name}", col)
    for col in AnalysisORM.__table__.columns
    if not col.primary_key and not col.index and col.name != "id"
) + (AnalysisORM.__table_args__,)


class JobTrackerORM(SQLModel, JobTracker, table=True):
    __tablename__ = "jobtracker"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    job_id: str = Field(primary_key=True)
    type: str  # type: ignore
    status: str  # type: ignore


class FilteredResultsORM(SQLModel, FilteredResults, table=True):
    __tablename__ = "filteredresults"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    name: str = Field(primary_key=True)
    symbols: list[str] = Field(sa_column=Column(JSON))
    filter_query: dict[str, Any] = Field(sa_column=Column(JSON))  # type: ignore


class SignalSeriesORM(SQLModel, SignalSeries, table=True):
    __tablename__ = "signalseries"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    date: str = Field(primary_key=True)  # type: ignore
    name: str = Field(primary_key=True)
    symbol: str = Field(primary_key=True)
    value: float | None = Field(default=None, nullable=True)  # type: ignore


class IndustryViewORM(SQLModel, IndustryView, table=True):
    __tablename__ = "industryview"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    date: str = Field(primary_key=True)  # type: ignore
    created_at: str = Field(default=None, nullable=True)  # type: ignore
    simple_return: float | None = Field(default=None, nullable=True)  # type: ignore
    log_return: float | None = Field(default=None, nullable=True)  # type: ignore
    normalized_close: float | None = Field(default=None, nullable=True)  # type: ignore
    country: str = Field(primary_key=True)  # type: ignore
    industry: str = Field(primary_key=True)  # type: ignore
    industry_group: str | None = Field(default=None, nullable=True)  # type: ignore
    sector: str | None = Field(default=None, nullable=True)  # type: ignore
    type: str = Field(primary_key=True)  # type: ignore


class BacktestResultORM(SQLModel, BacktestResult, table=True):
    __tablename__ = "backtestresult"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012
    name: str = Field(primary_key=True)
    start: str = Field(primary_key=True)  # type: ignore
    holding_period: int = Field(primary_key=True)
    extend_days: int = Field(primary_key=True)
    percentage: int = Field(primary_key=True)
    iterations: int = Field(primary_key=True)
    data: dict[str, Any] = Field(sa_column=Column(JSON))
