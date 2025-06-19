import abc
import logging
from typing import List, Optional

import pandas as pd
from bearish.interface.interface import BearishDbBase  # type: ignore
from bearish.models.base import Ticker  # type: ignore

from bullish.analysis import Analysis, AnalysisView
from bullish.filter import FilterQuery

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

    def read_analysis_data(self, columns: Optional[List[str]] = None) -> pd.DataFrame:
        columns = columns or list(AnalysisView.model_fields)
        data = self._read_analysis_data(columns)
        if set(data.columns) != set(columns):
            raise ValueError(
                f"Expected columns {columns}, but got {data.columns.tolist()}"
            )
        return data

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
    def _read_analysis_data(self, columns: List[str]) -> pd.DataFrame:
        ...
