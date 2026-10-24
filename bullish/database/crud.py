"""``BullishDb``: the bearish database plus bullish's KPI store, migrated on open."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path
from typing import Any

from bearish.database.crud import BearishDb
from pydantic import ConfigDict
from sqlalchemy import Engine, create_engine

from bullish.database.kpi_store import KpiStoreMixin
from bullish.database.scripts.upgrade import upgrade
from bullish.exceptions import DatabaseFileNotFoundError

logger = logging.getLogger(__name__)


class BullishDb(KpiStoreMixin, BearishDb):
    """Opens an existing bearish file, brings both schemas to head and exposes the KPI store."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    database_path: Path

    def valid(self) -> bool:
        return self.database_path.exists() and self.database_path.is_file()

    @cached_property
    def _engine(self) -> Engine:
        if not self.valid():
            raise DatabaseFileNotFoundError("Database file not found.")
        upgrade(self.database_path)
        return create_engine(f"sqlite:///{Path(self.database_path)}")

    def model_post_init(self, __context: Any) -> None:
        self._engine  # noqa: B018
