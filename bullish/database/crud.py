import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd
from bearish.database.crud import BearishDb  # type: ignore
from bearish.models.base import Ticker  # type: ignore
from pydantic import ConfigDict
from sqlalchemy import Engine, create_engine, insert, delete, update
from sqlmodel import Session, select

from bullish.analysis.analysis import Analysis
from bullish.database.schemas import AnalysisORM, JobTrackerORM, FilteredResultsORM
from bullish.database.scripts.upgrade import upgrade
from bullish.exceptions import DatabaseFileNotFoundError
from bullish.analysis.filter import FilteredResults
from bullish.interface.interface import BullishDbBase
from bullish.jobs.models import JobTracker, JobTrackerStatus

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000


class BullishDb(BearishDb, BullishDbBase):  # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    database_path: Path

    def valid(self) -> bool:
        """Check if the database is valid."""
        return self.database_path.exists() and self.database_path.is_file()

    @cached_property
    def _engine(self) -> Engine:
        if not self.valid():
            raise DatabaseFileNotFoundError("Database file not found.")
        database_url = f"sqlite:///{Path(self.database_path)}"
        upgrade(self.database_path)
        engine = create_engine(database_url)
        return engine

    def model_post_init(self, __context: Any) -> None:
        self._engine  # noqa: B018

    def _write_analysis(self, analysis: Analysis) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(AnalysisORM)
                .prefix_with("OR REPLACE")
                .values(analysis.model_dump())
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def _read_analysis(self, ticker: Ticker) -> Optional[Analysis]:
        with Session(self._engine) as session:
            query = select(AnalysisORM).where(AnalysisORM.symbol == ticker.symbol)
            analysis = session.exec(query).first()
            if not analysis:
                return None
            return Analysis.model_validate(analysis)

    def _read_analysis_data(
        self, columns: List[str], symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        columns_ = ",".join(columns)
        if symbols:
            symbols_str = ",".join([f"'{s}'" for s in symbols])
            query = f"""SELECT {columns_} FROM analysis WHERE symbol IN ({symbols_str})"""  # noqa: S608
        else:
            query = f"""SELECT {columns_} FROM analysis"""  # noqa: S608
        return pd.read_sql_query(query, self._engine)

    def _read_filter_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql(
            query,
            con=self._engine,
        )

    def _read_job_trackers(self) -> pd.DataFrame:
        query = "SELECT * FROM jobtracker ORDER BY started_at DESC"
        return pd.read_sql_query(query, self._engine)

    def write_job_tracker(self, job_tracker: JobTracker) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(JobTrackerORM)
                .prefix_with("OR REPLACE")
                .values(job_tracker.model_dump())
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def delete_job_trackers(self, job_ids: List[str]) -> None:
        with Session(self._engine) as session:
            stmt = delete(JobTrackerORM).where(JobTrackerORM.job_id.in_(job_ids))  # type: ignore
            result = session.execute(stmt)

            if result.rowcount > 0:  # type: ignore
                session.commit()
            else:
                logger.warning(f"Job tracker(s) with ID(s) {job_ids} not found.")

    def update_job_tracker_status(self, job_tracker_status: JobTrackerStatus) -> None:
        with Session(self._engine) as session:
            stmt = (
                update(JobTrackerORM)
                .where(JobTrackerORM.job_id == job_tracker_status.job_id)  # type: ignore
                .values(status=job_tracker_status.status)
            )
            result = session.execute(stmt)

            if result.rowcount > 0:  # type: ignore
                session.commit()
            else:
                logger.warning(
                    f"Job tracker with ID {job_tracker_status.job_id} not found."
                )

    def read_filtered_results(self, name: str) -> Optional[FilteredResults]:
        with Session(self._engine) as session:
            stmt = select(FilteredResultsORM).where(FilteredResultsORM.name == name)
            result = session.execute(stmt).scalar_one_or_none()

            if result:
                return FilteredResults.model_validate(
                    result.model_dump()
                )  # if you're using Pydantic or DTOs
            return None

    def read_list_filtered_results(self) -> List[str]:
        with Session(self._engine) as session:
            stmt = select(FilteredResultsORM.name)
            result = session.execute(stmt).scalars().all()
            return list(result)

    def write_filtered_results(self, filtered_results: FilteredResults) -> None:
        with Session(self._engine) as session:
            data = filtered_results.model_dump()
            stmt = insert(FilteredResultsORM).prefix_with("OR REPLACE").values(data)
            session.exec(stmt)  # type: ignore
            session.commit()
