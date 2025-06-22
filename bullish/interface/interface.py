import abc
import logging
from typing import List, Optional

import pandas as pd
from bearish.interface.interface import BearishDbBase  # type: ignore
from bearish.models.base import Ticker  # type: ignore


from bullish.analysis.analysis import Analysis, AnalysisView
from bullish.analysis.filter import FilterQuery, FilteredResults
from bullish.jobs.models import JobTracker, JobTrackerStatus, add_icons

logger = logging.getLogger(__name__)


class BullishDbBase(BearishDbBase):  # type: ignore
    def write_analysis(self, analysis: "Analysis") -> None:
        return self._write_analysis(analysis)

    def read_analysis(self, ticker: Ticker) -> Optional["Analysis"]:
        return self._read_analysis(ticker)

    def read_filter_query(self, query: FilterQuery) -> pd.DataFrame:

        if not set(query.query_parameters).issubset(set(Analysis.model_fields)):
            raise ValueError(
                f"Query parameters {query.query_parameters} are not a "
                f"subset of Analysis model fields {Analysis.model_fields}"
            )
        query_ = query.to_query()
        fields = ",".join(list(AnalysisView.model_fields))
        query_str: str = f""" 
        SELECT {fields} FROM analysis WHERE {query_} LIMIT 1000
        """  # noqa: S608
        return self._read_filter_query(query_str)

    def read_analysis_data(
        self, columns: Optional[List[str]] = None, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        columns = columns or list(AnalysisView.model_fields)
        data = self._read_analysis_data(columns, symbols=symbols)
        if set(data.columns) != set(columns):
            raise ValueError(
                f"Expected columns {columns}, but got {data.columns.tolist()}"
            )
        return data

    def read_job_trackers(self) -> pd.DataFrame:
        return add_icons(self._read_job_trackers())

    @abc.abstractmethod
    def _read_job_trackers(self) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def write_job_tracker(self, job_tracker: JobTracker) -> None:
        ...

    @abc.abstractmethod
    def delete_job_trackers(self, job_ids: List[str]) -> None:
        ...

    @abc.abstractmethod
    def update_job_tracker_status(self, job_tracker_status: JobTrackerStatus) -> None:
        ...

    @abc.abstractmethod
    def _write_analysis(self, analysis: "Analysis") -> None:
        ...

    @abc.abstractmethod
    def _read_analysis(self, ticker: Ticker) -> Optional["Analysis"]:
        ...

    @abc.abstractmethod
    def _read_filter_query(self, query: str) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def _read_analysis_data(
        self, columns: List[str], symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        ...

    @abc.abstractmethod
    def read_filtered_results(self, name: str) -> Optional[FilteredResults]:
        ...

    @abc.abstractmethod
    def read_list_filtered_results(self) -> List[str]:
        ...

    @abc.abstractmethod
    def write_filtered_results(self, filtered_results: FilteredResults) -> None:
        ...
