from typing import Dict, Any

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from bullish.analysis.analysis import Analysis
from bullish.analysis.backtest import BacktestResult
from bullish.analysis.filter import FilteredResults
from bullish.analysis.indicators import SignalSeries
from bullish.analysis.industry_views import IndustryView

from bullish.jobs.models import JobTracker
from sqlalchemy import Index


class BaseTable(SQLModel):
    symbol: str = Field(primary_key=True)
    source: str = Field(primary_key=True)


dynamic_indexes = tuple(
    Index(f"ix_analysis_{col}", col) for col in Analysis.model_fields
)


class AnalysisORM(BaseTable, Analysis, table=True):
    __tablename__ = "analysis"
    __table_args__ = {"extend_existing": True}  # noqa:RUF012


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
    filter_query: Dict[str, Any] = Field(sa_column=Column(JSON))  # type: ignore


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
    data: Dict[str, Any] = Field(sa_column=Column(JSON))
