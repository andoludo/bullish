from typing import Dict, Any

from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON
from bullish.analysis.analysis import Analysis
from bullish.analysis.filter import FilteredResults
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
