import json
import logging
from datetime import date
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd
from bearish.database.crud import BearishDb  # type: ignore
from bearish.models.base import Ticker  # type: ignore
from bearish.database.schemas import EarningsDateORM, EquityORM, PriceORM  # type: ignore
from bearish.types import Sources  # type: ignore
from bearish.models.price.price import Price  # type: ignore
from bearish.models.price.prices import Prices  # type: ignore
from pydantic import ConfigDict
from sqlalchemy import Engine, create_engine, insert, delete, update
from sqlmodel import Session, select

from bullish.analysis.analysis import Analysis

from bullish.analysis.constants import Industry, IndustryGroup, Sector, Country
from bullish.analysis.indicators import SignalSeries
from bullish.analysis.industry_views import Type, IndustryView

from bullish.database.schemas import (
    AnalysisORM,
    JobTrackerORM,
    FilteredResultsORM,
    IndustryViewORM,
    SignalSeriesORM,
    BacktestResultORM,
)
from bullish.database.scripts.upgrade import upgrade
from bullish.exceptions import DatabaseFileNotFoundError
from bullish.analysis.filter import FilteredResults
from bullish.interface.interface import BullishDbBase
from bullish.jobs.models import JobTracker, JobTrackerStatus

if TYPE_CHECKING:
    from bullish.analysis.backtest import BacktestResult, BacktestResultQuery

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
        try:
            upgrade(self.database_path)
        except Exception as e:
            logger.warning(
                f"Failed to upgrade the database at {self.database_path}. "
                f"Reason: {e}"
                "Skipping upgrade. "
            )
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

    def _write_many_analysis(self, many_analysis: List[Analysis]) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(AnalysisORM)
                .prefix_with("OR REPLACE")
                .values([a.model_dump() for a in many_analysis])
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

    def read_symbols(self) -> List[str]:
        query = "SELECT DISTINCT symbol FROM analysis"
        data = pd.read_sql_query(query, self._engine)
        return data["symbol"].tolist()

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

    def read_job_tracker(self, task_id: str) -> Optional[JobTracker]:
        stmt = select(JobTrackerORM).where(JobTrackerORM.job_id == task_id)
        with Session(self._engine) as session:
            result = session.execute(stmt).scalar_one_or_none()
            if result:
                return JobTracker.model_validate(result.model_dump())
            return None

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
            data = filtered_results.model_dump_json(
                exclude_unset=True, exclude_defaults=True
            )
            stmt = (
                insert(FilteredResultsORM)
                .prefix_with("OR REPLACE")
                .values(json.loads(data))
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def read_dates(self, symbol: str) -> List[date]:
        with Session(self._engine) as session:
            return [
                r.date()
                for r in session.exec(
                    select(EarningsDateORM.date).where(EarningsDateORM.symbol == symbol)
                )
            ]

    def read_industry_symbols(
        self, industries: List[Industry], country: Country, source: Sources = "Yfinance"
    ) -> List[str]:
        with Session(self._engine) as session:
            stmt = select(EquityORM.symbol).where(
                EquityORM.industry.in_(industries),
                EquityORM.source == source,
                EquityORM.country == country,
            )
            result = session.exec(stmt).all()
            return list(result)

    def read_industry_group_symbols(
        self,
        industry_groups: List[IndustryGroup],
        country: Country,
        source: Sources = "Yfinance",
    ) -> List[str]:
        with Session(self._engine) as session:
            stmt = select(EquityORM.symbol).where(
                EquityORM.industry_group.in_(industry_groups),
                EquityORM.source == source,
                EquityORM.country == country,
            )
            result = session.exec(stmt).all()
            return list(result)

    def read_sector_symbols(
        self, sectors: List[Sector], country: Country, source: Sources = "Yfinance"
    ) -> List[str]:
        with Session(self._engine) as session:
            stmt = select(EquityORM.symbol).where(
                EquityORM.sector.in_(sectors),
                EquityORM.source == source,
                EquityORM.country == country,
            )
            result = session.exec(stmt).all()
            return list(result)

    def write_returns(self, industry_returns: List[IndustryView]) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(IndustryViewORM)
                .prefix_with("OR REPLACE")
                .values([a.model_dump() for a in industry_returns])
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def read_returns(
        self, type: Type, industry: Industry, country: Country
    ) -> List[IndustryView]:
        with Session(self._engine) as session:
            stmt = select(IndustryViewORM)
            if industry:
                stmt = stmt.where(IndustryViewORM.industry == industry)
            if country:
                stmt = stmt.where(IndustryViewORM.country == country)
            result = session.exec(stmt).all()
            return [IndustryView.model_validate(r) for r in result]

    def write_signal_series(self, signal_series: List[SignalSeries]) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(SignalSeriesORM)
                .prefix_with("OR REPLACE")
                .values([a.model_dump() for a in signal_series])
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def read_signal_series(
        self, name: str, start_date: date, end_date: date
    ) -> List[str]:
        with Session(self._engine) as session:
            stmt = select(SignalSeriesORM.symbol).where(
                SignalSeriesORM.name == name,
                SignalSeriesORM.date >= start_date,  # type: ignore
                SignalSeriesORM.date <= end_date,  # type: ignore
            )
            return list(set(session.exec(stmt).all()))

    def read_symbol_series(
        self, symbol: str, start_date: date, end_date: Optional[date] = None
    ) -> pd.DataFrame:

        with Session(self._engine) as session:
            query_ = select(PriceORM)
            query_ = query_.where(PriceORM.symbol == symbol)
            if end_date:
                query_ = query_.where(
                    PriceORM.date >= start_date, PriceORM.date <= end_date
                )
            else:
                query_ = query_.where(PriceORM.date >= start_date)
            series = session.exec(query_).all()
            prices = [Price.model_validate(serie) for serie in series]
            return Prices(prices=prices).to_dataframe()  # type: ignore

    def write_many_backtest_results(
        self, backtest_results: List["BacktestResult"]
    ) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(BacktestResultORM)
                .prefix_with("OR REPLACE")
                .values([a.model_dump() for a in backtest_results])
            )
            session.exec(stmt)  # type: ignore
            session.commit()

    def read_many_backtest_results(
        self, query: Optional["BacktestResultQuery"] = None
    ) -> List["BacktestResult"]:
        from bullish.analysis.backtest import BacktestResult

        with Session(self._engine) as session:
            stmt = select(BacktestResultORM)
            results = session.exec(stmt).all()
            if results:
                return [BacktestResult.model_validate(r) for r in results]
            else:
                return []
