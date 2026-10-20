import abc
import logging
from datetime import date
from typing import Optional, Any

import pandas as pd
from bearish.interface.interface import BearishDbBase  # type: ignore
from bearish.models.base import Ticker  # type: ignore
from bearish.types import Sources  # type: ignore


from bullish.analysis.analysis import Analysis, AnalysisView, SubjectAnalysis
from bullish.analysis.backtest import BacktestResult, BacktestResultQuery
from bullish.analysis.constants import Industry, Sector, IndustryGroup, Country
from bullish.analysis.filter import FilterQuery, FilteredResults
from bullish.analysis.indicators import SignalSeries
from bullish.analysis.industry_views import Type, IndustryView
from bullish.analysis.openai import OpenAINews
from bullish.jobs.models import JobTracker, JobTrackerStatus, add_icons

logger = logging.getLogger(__name__)


class BullishDbBase(BearishDbBase):  # type: ignore
    def write_analysis(self, analysis: "Analysis") -> None:
        return self._write_analysis(analysis)

    def write_many_analysis(self, many_analysis: list["Analysis"]) -> None:
        return self._write_many_analysis(many_analysis)

    def read_analysis(self, ticker: Ticker) -> Optional["Analysis"]:
        return self._read_analysis(ticker)

    def read_filter_query(self, query: FilterQuery) -> pd.DataFrame:

        query_ = query.to_query()
        columns = list(AnalysisView.model_fields)
        fields = ",".join(columns)
        query_str: str = f""" 
        SELECT {fields} FROM analysis WHERE {query_}
        """  # noqa: S608
        return self._read_filter_query(query_str)

    def read_analysis_data(
        self, columns: list[str] | None = None, symbols: list[str] | None = None
    ) -> pd.DataFrame:
        columns = columns or list(AnalysisView.model_fields)
        data = self._read_analysis_data(columns, symbols=symbols)
        if set(data.columns) != set(columns):
            raise ValueError(
                f"Expected columns {columns}, but got {data.columns.tolist()}"
            )
        return data

    def read_sec_data(self) -> pd.DataFrame:
        return self.read_query("SELECT * FROM secshareincrease")  # type: ignore

    def read_job_trackers(self) -> pd.DataFrame:
        return add_icons(self._read_job_trackers())

    @abc.abstractmethod
    def _read_job_trackers(self) -> pd.DataFrame: ...

    @abc.abstractmethod
    def write_job_tracker(self, job_tracker: JobTracker) -> None: ...

    @abc.abstractmethod
    def delete_job_trackers(self, job_ids: list[str]) -> None: ...

    @abc.abstractmethod
    def update_job_tracker_status(
        self, job_tracker_status: JobTrackerStatus
    ) -> None: ...

    @abc.abstractmethod
    def _write_analysis(self, analysis: "Analysis") -> None: ...

    @abc.abstractmethod
    def _write_many_analysis(self, many_analysis: list["Analysis"]) -> None: ...

    @abc.abstractmethod
    def _read_analysis(self, ticker: Ticker) -> Optional["Analysis"]: ...

    @abc.abstractmethod
    def _read_filter_query(self, query: str) -> pd.DataFrame: ...

    @abc.abstractmethod
    def _read_analysis_data(
        self, columns: list[str], symbols: list[str] | None = None
    ) -> pd.DataFrame: ...

    @abc.abstractmethod
    def read_filtered_results(self, name: str) -> FilteredResults | None: ...

    @abc.abstractmethod
    def read_list_filtered_results(self) -> list[str]: ...

    @abc.abstractmethod
    def write_filtered_results(self, filtered_results: FilteredResults) -> None: ...

    @abc.abstractmethod
    def read_symbols(self) -> list[str]: ...

    @abc.abstractmethod
    def read_job_tracker(self, task_id: str) -> JobTracker | None: ...

    @abc.abstractmethod
    def read_dates(self, symbol: str) -> list[date]: ...

    @abc.abstractmethod
    def read_industry_symbols(
        self, industries: list[Industry], country: Country, source: Sources = "Yfinance"
    ) -> list[str]: ...

    @abc.abstractmethod
    def read_industry_group_symbols(
        self,
        industry_groups: list[IndustryGroup],
        country: Country,
        source: Sources = "Yfinance",
    ) -> list[str]: ...

    @abc.abstractmethod
    def read_sector_symbols(
        self, sectors: list[Sector], country: Country, source: Sources = "Yfinance"
    ) -> list[str]: ...

    @abc.abstractmethod
    def write_returns(self, industry_returns: list[IndustryView]) -> None: ...

    @abc.abstractmethod
    def read_returns(
        self, type: Type, industry: Industry, country: Country
    ) -> list[IndustryView]: ...

    @abc.abstractmethod
    def write_signal_series(self, signal_series: list[SignalSeries]) -> None: ...

    @abc.abstractmethod
    def read_signal_series(
        self, name: str, start_date: date, end_date: date
    ) -> list[str]: ...

    @abc.abstractmethod
    def read_symbol_series(
        self, symbol: str, start_date: date, end_date: date | None = None
    ) -> pd.DataFrame: ...
    @abc.abstractmethod
    def write_many_backtest_results(
        self, backtest_results: list[BacktestResult]
    ) -> None: ...

    @abc.abstractmethod
    def read_many_backtest_results(
        self, query: BacktestResultQuery | None = None
    ) -> list[BacktestResult]: ...

    @abc.abstractmethod
    def read_next_earnings_date(self, symbol: str) -> date | None: ...

    @abc.abstractmethod
    def read_subject(self, symbol: str) -> SubjectAnalysis | None: ...
    @abc.abstractmethod
    def write_many_openai_news(self, openai_news: list[OpenAINews]) -> None: ...

    @abc.abstractmethod
    def update_analysis(self, symbol: str, fields: dict[str, Any]) -> None: ...
