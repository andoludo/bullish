import logging
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import pandas as pd
from bearish.database.crud import BearishDb  # type: ignore
from bearish.models.base import Ticker  # type: ignore
from pydantic import ConfigDict
from sqlalchemy import Engine, create_engine, insert
from sqlmodel import Session, select

from bullish.analysis import Analysis
from bullish.database.schemas import AnalysisORM
from bullish.database.scripts.upgrade import upgrade
from bullish.exceptions import DatabaseFileNotFoundError
from bullish.interface.interface import BullishDbBase


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

    def _read_analysis_data(self, columns: List[str]) -> pd.DataFrame:
        columns_ = ",".join(columns)
        query = f"""SELECT {columns_} FROM analysis"""  # noqa: S608
        return pd.read_sql_query(query, self._engine)

    def _read_filter_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql(
            query,
            con=self._engine,
        )
