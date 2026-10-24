"""Persistence of the long ``kpi`` table, the ``score`` table and the ``kpirun`` log."""

from __future__ import annotations

import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, insert
from sqlmodel import Session, col, select

from bullish.database.schemas import KpiORM, KpiRunORM, ScoreORM
from bullish.utils.typed import as_float, row_of

if TYPE_CHECKING:
    from sqlalchemy import Engine

logger = logging.getLogger(__name__)
BATCH = 5000


class KpiRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    as_of: datetime.date
    created_at: datetime.datetime = Field(default_factory=datetime.datetime.now)
    universe: str = Field(
        default="all",
        description="Countries the run covered, comma separated, or 'all'",
    )
    entities: int = 0
    kpis: int = 0
    scores: int = 0
    base_currency: str = "EUR"
    notes: str = ""


class KpiStoreMixin:
    """Mixed into ``BullishDb``; expects ``self._engine``."""

    if TYPE_CHECKING:

        @property
        def _engine(self) -> Engine: ...

    # --- write -----------------------------------------------------------

    def write_kpi_frame(
        self, entity_type: str, frame: pd.DataFrame, as_of: datetime.date
    ) -> int:
        """Store a wide ``entity x kpi`` frame as long rows for ``as_of``; returns rows written."""
        if frame.empty:
            return 0
        index_name = str(frame.index.name or "index")
        long = frame.reset_index().melt(
            id_vars=[index_name], var_name="name", value_name="value"
        )
        long = long.rename(columns={index_name: "entity"}).dropna(subset=["value"])
        rows = [
            {
                "entity_type": entity_type,
                "entity": str(r.entity),
                "name": r.name,
                "date": as_of,
                "value": as_float(r.value),
            }
            for r in long.itertuples(index=False)
        ]
        self._replace_rows(KpiORM, rows, entity_type=entity_type, as_of=as_of)
        return len(rows)

    def write_scores(  # noqa: PLR0913
        self,
        entity_type: str,
        score_name: str,
        scores: pd.DataFrame,
        components: pd.DataFrame,
        as_of: datetime.date,
    ) -> int:
        rows = []
        for entity, row in scores.iterrows():
            if pd.isna(row["score"]):
                continue
            payload = {
                "blocks": {
                    c[len("block_") :]: _f(row[c])
                    for c in scores.columns
                    if c.startswith("block_")
                },
                "kpis": (
                    {k: _f(v) for k, v in row_of(components, str(entity)).items()}
                    if entity in components.index
                    else {}
                ),
            }
            rows.append(
                {
                    "entity_type": entity_type,
                    "entity": str(entity),
                    "score": score_name,
                    "date": as_of,
                    "value": float(row["score"]),
                    "coverage": float(row["coverage"]),
                    "eligible": bool(row["eligible"]),
                    "rank": None if pd.isna(row["rank"]) else int(row["rank"]),
                    "components": json.dumps(payload),
                }
            )
        with Session(self._engine) as session:
            session.exec(  # type: ignore[call-overload]
                delete(ScoreORM).where(
                    col(ScoreORM.entity_type) == entity_type,
                    col(ScoreORM.score) == score_name,
                    col(ScoreORM.date) == as_of,
                )
            )
            for i in range(0, len(rows), BATCH):
                session.exec(insert(ScoreORM), params=rows[i : i + BATCH])  # type: ignore[call-overload]
            session.commit()
        return len(rows)

    def write_kpi_run(self, run: KpiRun) -> None:
        with Session(self._engine) as session:
            session.exec(insert(KpiRunORM).prefix_with("OR REPLACE").values(run.model_dump()))  # type: ignore[call-overload]
            session.commit()

    def _replace_rows(
        self,
        orm: type[KpiORM],
        rows: list[dict[str, Any]],
        entity_type: str,
        as_of: datetime.date,
    ) -> None:
        with Session(self._engine) as session:
            session.exec(  # type: ignore[call-overload]
                delete(orm).where(
                    col(orm.entity_type) == entity_type, col(orm.date) == as_of
                )
            )
            for i in range(0, len(rows), BATCH):
                session.exec(insert(orm), params=rows[i : i + BATCH])  # type: ignore[call-overload]
            session.commit()

    # --- read ------------------------------------------------------------

    def latest_kpi_date(
        self, entity_type: str, as_of: datetime.date | None = None
    ) -> datetime.date | None:
        with Session(self._engine) as session:
            stmt = select(col(KpiORM.date)).where(
                col(KpiORM.entity_type) == entity_type
            )
            if as_of is not None:
                stmt = stmt.where(col(KpiORM.date) <= as_of)
            dates = session.exec(stmt.distinct()).all()
        return max(dates) if dates else None

    def kpi_dates(self, entity_type: str = "equity") -> list[datetime.date]:
        with Session(self._engine) as session:
            return sorted(
                session.exec(
                    select(col(KpiORM.date))
                    .where(col(KpiORM.entity_type) == entity_type)
                    .distinct()
                ).all()
            )

    def read_kpi_frame(
        self,
        entity_type: str,
        as_of: datetime.date | None = None,
        entities: list[str] | None = None,
    ) -> pd.DataFrame:
        """Wide ``entity x kpi`` frame of the latest run at or before ``as_of``."""
        date = self.latest_kpi_date(entity_type, as_of)
        if date is None:
            return pd.DataFrame()
        with Session(self._engine) as session:
            stmt = select(
                col(KpiORM.entity), col(KpiORM.name), col(KpiORM.value)
            ).where(col(KpiORM.entity_type) == entity_type, col(KpiORM.date) == date)
            if entities:
                stmt = stmt.where(col(KpiORM.entity).in_(entities))
            rows = session.exec(stmt).all()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(  # noqa: PD010
            rows, columns=["entity", "name", "value"]
        ).pivot(index="entity", columns="name", values="value")
        frame.index.name = entity_type
        frame.attrs["as_of"] = date
        return frame

    def read_kpi_series(  # noqa: PLR0913
        self,
        entity_type: str,
        entity: str,
        name: str,
        start: datetime.date | None = None,
        end: datetime.date | None = None,
    ) -> pd.Series:
        with Session(self._engine) as session:
            stmt = select(col(KpiORM.date), col(KpiORM.value)).where(
                col(KpiORM.entity_type) == entity_type,
                col(KpiORM.entity) == entity,
                col(KpiORM.name) == name,
            )
            if start is not None:
                stmt = stmt.where(col(KpiORM.date) >= start)
            if end is not None:
                stmt = stmt.where(col(KpiORM.date) <= end)
            rows = session.exec(stmt.order_by(col(KpiORM.date))).all()
        return pd.Series(dict(rows), dtype=float, name=name)

    def read_scores(
        self, entity_type: str, score_name: str, as_of: datetime.date | None = None
    ) -> pd.DataFrame:
        with Session(self._engine) as session:
            dates = session.exec(
                select(col(ScoreORM.date))
                .where(
                    col(ScoreORM.entity_type) == entity_type,
                    col(ScoreORM.score) == score_name,
                )
                .distinct()
            ).all()
            valid = [d for d in dates if as_of is None or d <= as_of]
            if not valid:
                return pd.DataFrame()
            date = max(valid)
            rows = session.exec(
                select(ScoreORM).where(
                    col(ScoreORM.entity_type) == entity_type,
                    col(ScoreORM.score) == score_name,
                    col(ScoreORM.date) == date,
                )
            ).all()
        frame = pd.DataFrame([r.model_dump() for r in rows])
        frame["components"] = frame["components"].map(json.loads)
        frame.attrs["as_of"] = date
        return frame.set_index("entity").sort_values("rank")

    def read_kpi_runs(self) -> list[KpiRun]:
        with Session(self._engine) as session:
            rows = session.exec(select(KpiRunORM).order_by(col(KpiRunORM.as_of))).all()
        return [KpiRun.model_validate(r.model_dump()) for r in rows]


def _f(value: object) -> float | None:
    number = as_float(value)
    return None if pd.isna(number) else round(number, 4)
