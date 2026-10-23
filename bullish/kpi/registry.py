"""Declarative catalogue of every KPI the engine produces.

A block (``bullish.kpi.equity.momentum``, ...) declares its KPIs with
:func:`declare_many` and returns a frame whose columns are exactly those
names. The registry is what the MCP server exposes as ``list_kpis`` /
``describe_kpi`` and what the scoring layer reads to know the direction
(``higher_is_better``) and the evidence tier of each input.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

EntityType = Literal[
    "equity", "sector", "industry", "country", "etf", "market", "commodity"
]
Tier = Literal["A", "B", "C"]
TIER_MEANING: dict[str, str] = {
    "A": "strong academic evidence",
    "B": "practitioner evidence / widely used",
    "C": "heuristic or context only",
}


class KpiDefinition(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Snake-case identifier, unique per entity type")
    entity_type: EntityType
    block: str = Field(
        description="Family the KPI belongs to: momentum, growth, quality, ..."
    )
    tier: Tier
    unit: str = Field(description="ratio, pct, days, count, score, flag, currency, ...")
    description: str
    higher_is_better: bool | None = Field(
        default=True,
        description="None when the KPI is context (no direction), e.g. a quadrant code",
    )

    @property
    def key(self) -> tuple[str, str]:
        return self.entity_type, self.name


class Registry:
    def __init__(self) -> None:
        self._definitions: dict[tuple[str, str], KpiDefinition] = {}

    def add(self, definition: KpiDefinition) -> KpiDefinition:
        if definition.key in self._definitions:
            raise ValueError(
                f"KPI {definition.entity_type}.{definition.name} declared twice"
            )
        self._definitions[definition.key] = definition
        return definition

    def get(self, entity_type: str, name: str) -> KpiDefinition:
        try:
            return self._definitions[(entity_type, name)]
        except KeyError as e:
            raise KeyError(f"Unknown KPI {entity_type}.{name}") from e

    def find(self, name: str) -> list[KpiDefinition]:
        return [d for d in self._definitions.values() if d.name == name]

    def all(
        self, entity_type: EntityType | None = None, block: str | None = None
    ) -> list[KpiDefinition]:
        return [
            d
            for d in self._definitions.values()
            if (entity_type is None or d.entity_type == entity_type)
            and (block is None or d.block == block)
        ]

    def names(self, entity_type: EntityType, block: str | None = None) -> list[str]:
        return [d.name for d in self.all(entity_type, block)]

    def direction(self, entity_type: str, name: str) -> bool | None:
        return self.get(entity_type, name).higher_is_better

    def validate_frame(
        self, entity_type: EntityType, block: str, frame: pd.DataFrame
    ) -> pd.DataFrame:
        """Keep the declared columns of ``block`` in declaration order; fail on undeclared ones."""
        declared = self.names(entity_type, block)
        undeclared = [column for column in frame.columns if column not in declared]
        if undeclared:
            raise ValueError(
                f"Block {entity_type}.{block} produced undeclared KPIs: {undeclared}"
            )
        return frame.reindex(columns=declared)


REGISTRY = Registry()

DeclarationRow = tuple[str, Tier, str, str, bool | None]


def declare_many(
    entity_type: EntityType, block: str, rows: list[DeclarationRow]
) -> list[str]:
    """Register ``(name, tier, unit, description, higher_is_better)`` rows; returns the names."""
    names = []
    for name, tier, unit, description, higher_is_better in rows:
        REGISTRY.add(
            KpiDefinition(
                name=name,
                entity_type=entity_type,
                block=block,
                tier=tier,
                unit=unit,
                description=description,
                higher_is_better=higher_is_better,
            )
        )
        names.append(name)
    return names
