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
from bullish.database.schemas import AnalysisORM, ViewORM
from bullish.database.scripts.upgrade import upgrade
from bullish.interface.interface import BullishDbBase
from bullish.view import View

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000


class BullishDb(BearishDb, BullishDbBase):  # type: ignore
    model_config = ConfigDict(arbitrary_types_allowed=True)
    database_path: Path

    @cached_property
    def _engine(self) -> Engine:
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

    def _read_views(self, query: str) -> List[View]:
        views = pd.read_sql(
            query,
            con=self._engine,
        )
        return [
            View.model_validate(record) for record in views.to_dict(orient="records")
        ]

    def _write_views(self, views: List[View]) -> None:
        with Session(self._engine) as session:
            stmt = (
                insert(ViewORM)
                .prefix_with("OR REPLACE")
                .values([view.model_dump() for view in views])
            )

            session.exec(stmt)  # type: ignore
            session.commit()
