"""Runtime settings of the KPI engine and the MCP server.

Every value can be set through the environment (``BULLISH_DATABASE_PATH``,
``BULLISH_BASE_CURRENCY``, ...) or a ``.env`` file in the working directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PeerGroup = Literal["sector", "industry", "country"]


class BullishSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BULLISH_", env_file=".env", extra="ignore"
    )

    database_path: Path | None = Field(
        default=None, description="SQLite file produced by bearish"
    )
    base_currency: str = Field(
        default="EUR",
        description="Currency every cross-country comparison is expressed in",
    )
    peer_group: PeerGroup = Field(
        default="sector", description="Group inside which equity KPIs are ranked"
    )
    lookback_years: int = Field(
        default=5, ge=1, description="Price history loaded for the KPIs"
    )
    statement_lag_days: int = Field(
        default=90,
        ge=0,
        description="Publication lag assumed for statements without a filed_date",
    )
    min_dollar_volume: float = Field(
        default=1_000_000.0, ge=0.0, description="Liquidity floor for the presets"
    )
    min_market_cap: float = Field(
        default=300_000_000.0, ge=0.0, description="Size floor for the presets"
    )
    winsor_limit: float = Field(
        default=0.01, ge=0.0, le=0.25, description="Tail share clipped before ranking"
    )
    min_peers: int = Field(
        default=5,
        ge=2,
        description="Peers needed to rank inside a group; else the universe",
    )
    price_table: str = Field(default="price")
    min_price_rows: int = Field(
        default=60,
        ge=0,
        description=(
            "Price rows inside the lookback a symbol needs to enter the KPI run; "
            "0 keeps every equity row"
        ),
    )
    max_rows: int = Field(
        default=200, ge=1, description="Hard cap on rows an MCP tool returns"
    )
    allow_sql: bool = Field(
        default=False, description="Expose the read-only SQL tool on the MCP server"
    )

    def require_database(self) -> Path:
        if self.database_path is None:
            raise ValueError(
                "No database configured: set BULLISH_DATABASE_PATH or pass --db"
            )
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found: {self.database_path}")
        return self.database_path
