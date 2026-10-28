"""Persistence of the long ``kpi`` table, the ``score`` table and the ``kpirun`` log."""

from __future__ import annotations

import datetime
import json
import logging
import sqlite3
import zlib
from contextlib import contextmanager
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import delete, insert
from sqlmodel import Session, col, select

from bullish.database.schemas import KpiORM, KpiRunORM, KpiWideORM, ScoreORM
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
        """Store a wide ``entity x kpi`` frame for ``as_of``: long rows plus the wide read cache.

        Returns the number of long rows written (one per non-null cell).
        """
        if frame.empty:
            return 0
        values = frame.to_numpy(dtype=float)
        mask = ~np.isnan(values)
        row_pos, col_pos = np.nonzero(mask)
        entities = frame.index.astype(str).to_numpy()[row_pos].tolist()
        names = frame.columns.astype(str).to_numpy()[col_pos].tolist()
        date = as_of.isoformat()
        rows = [
            (entity_type, entity, name, date, value)
            for entity, name, value in zip(
                entities, names, values[mask].tolist(), strict=True
            )
        ]
        with self._bulk_connection() as conn:
            conn.execute(
                "DELETE FROM kpi WHERE entity_type = ? AND date = ?",
                (entity_type, date),
            )
            conn.executemany(
                "INSERT INTO kpi (entity_type, entity, name, date, value) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._write_wide(conn, entity_type, frame, as_of)
        return len(rows)

    @contextmanager
    def _bulk_connection(self) -> Iterator[sqlite3.Connection]:
        """A raw sqlite connection in one transaction with fsync off for the duration of a bulk load."""
        raw = self._engine.raw_connection()
        try:
            conn: sqlite3.Connection = raw.driver_connection  # type: ignore[assignment]
            previous = conn.execute("PRAGMA synchronous").fetchone()[0]
            conn.execute("PRAGMA synchronous = OFF")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                conn.execute(f"PRAGMA synchronous = {int(previous)}")
        finally:
            raw.close()

    @staticmethod
    def _write_wide(
        conn: sqlite3.Connection,
        entity_type: str,
        frame: pd.DataFrame,
        as_of: datetime.date,
    ) -> None:
        values = np.ascontiguousarray(frame.to_numpy(dtype=np.float64))
        conn.execute(
            "INSERT OR REPLACE INTO kpiwide "
            "(entity_type, date, created_at, rows, cols, index_json, columns_json, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entity_type,
                as_of.isoformat(),
                datetime.datetime.now().isoformat(sep=" "),
                int(values.shape[0]),
                int(values.shape[1]),
                json.dumps(frame.index.astype(str).tolist()),
                json.dumps(frame.columns.astype(str).tolist()),
                zlib.compress(values.tobytes(), level=1),
            ),
        )

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

    # --- read ------------------------------------------------------------

    def latest_run(
        self, entity_type: str, as_of: datetime.date | None = None
    ) -> tuple[datetime.date, datetime.datetime | None] | None:
        """``(date, created_at)`` of the latest stored run at or before ``as_of``.

        Answered from the wide cache when it has any run at or before ``as_of``,
        else from the long table (runs written before the cache existed).
        """
        with Session(self._engine) as session:
            stmt = select(col(KpiWideORM.date), col(KpiWideORM.created_at)).where(
                col(KpiWideORM.entity_type) == entity_type
            )
            if as_of is not None:
                stmt = stmt.where(col(KpiWideORM.date) <= as_of)
            wide = session.exec(
                stmt.order_by(col(KpiWideORM.date).desc()).limit(1)
            ).first()
            if wide is not None:
                return wide[0], wide[1]
            # Runs written before the wide cache existed: scan the long table.
            legacy = select(col(KpiORM.date)).where(
                col(KpiORM.entity_type) == entity_type
            )
            if as_of is not None:
                legacy = legacy.where(col(KpiORM.date) <= as_of)
            dates = session.exec(legacy.distinct()).all()
        return (max(dates), None) if dates else None

    def latest_kpi_date(
        self, entity_type: str, as_of: datetime.date | None = None
    ) -> datetime.date | None:
        run = self.latest_run(entity_type, as_of)
        return run[0] if run else None

    def kpi_dates(self, entity_type: str = "equity") -> list[datetime.date]:
        with Session(self._engine) as session:
            dates = set(
                session.exec(
                    select(col(KpiORM.date))
                    .where(col(KpiORM.entity_type) == entity_type)
                    .distinct()
                ).all()
            )
            dates |= set(
                session.exec(
                    select(col(KpiWideORM.date)).where(
                        col(KpiWideORM.entity_type) == entity_type
                    )
                ).all()
            )
        return sorted(dates)

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
        frame = self._read_wide(entity_type, date)
        if frame is None:
            frame = self._read_long(entity_type, date)
        if frame.empty:
            return frame
        if entities:
            frame = frame[frame.index.isin(entities)]
        frame.index.name = entity_type
        frame.attrs["as_of"] = date
        return frame

    def _read_wide(self, entity_type: str, date: datetime.date) -> pd.DataFrame | None:
        with Session(self._engine) as session:
            row = session.exec(
                select(KpiWideORM).where(
                    col(KpiWideORM.entity_type) == entity_type,
                    col(KpiWideORM.date) == date,
                )
            ).first()
        if row is None:
            return None
        values = np.frombuffer(zlib.decompress(row.payload), dtype=np.float64)
        return pd.DataFrame(
            values.reshape(row.rows, row.cols).copy(),
            index=pd.Index(json.loads(row.index_json)),
            columns=pd.Index(json.loads(row.columns_json), name="name"),
        )

    def _read_long(self, entity_type: str, date: datetime.date) -> pd.DataFrame:
        with Session(self._engine) as session:
            stmt = select(
                col(KpiORM.entity), col(KpiORM.name), col(KpiORM.value)
            ).where(col(KpiORM.entity_type) == entity_type, col(KpiORM.date) == date)
            rows = session.exec(stmt).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(  # noqa: PD010
            rows, columns=["entity", "name", "value"]
        ).pivot(index="entity", columns="name", values="value")

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
