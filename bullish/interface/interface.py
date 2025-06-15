import abc
import logging
from typing import List, Optional

from bearish.interface.interface import BearishDbBase  # type: ignore
from bearish.models.base import Ticker  # type: ignore

from bullish.analysis import Analysis
from bullish.view import View

logger = logging.getLogger(__name__)


class BullishDbBase(BearishDbBase):  # type: ignore
    def write_analysis(self, analysis: "Analysis") -> None:
        return self._write_analysis(analysis)

    def read_analysis(self, ticker: Ticker) -> Optional["Analysis"]:
        return self._read_analysis(ticker)

    def read_views(self, query: str) -> List[View]:
        return self._read_views(query)

    def write_views(self, views: List[View]) -> None:
        if not views:
            logger.warning("No views to write.")
            return
        return self._write_views(views)

    @abc.abstractmethod
    def _write_analysis(self, analysis: "Analysis") -> None:
        ...

    @abc.abstractmethod
    def _read_analysis(self, ticker: Ticker) -> Optional["Analysis"]:
        ...

    @abc.abstractmethod
    def _read_views(self, query: str) -> List[View]:
        ...

    @abc.abstractmethod
    def _write_views(self, views: List[View]) -> None:
        ...
