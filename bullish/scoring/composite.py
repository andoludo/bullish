"""Weighted composites of KPI percentiles, with every component kept for explanation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from bullish.config import BullishSettings
from bullish.kpi.registry import REGISTRY, EntityType
from bullish.scoring.normalize import directed_percentile
from bullish.utils.typed import as_float, row_of

Operator = Literal[">=", "<=", ">", "<"]


class KpiWeight(BaseModel):
    model_config = ConfigDict(frozen=True)

    kpi: str
    weight: float = Field(default=1.0, gt=0)
    invert: bool = Field(
        default=False,
        description="Rank against the registry direction (e.g. low ret_1m is good)",
    )


class BlockSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    weight: float = Field(gt=0)
    kpis: tuple[KpiWeight, ...]
    rationale: str = ""


class Filter(BaseModel):
    """Hard eligibility rule; a missing KPI value fails the rule unless ``allow_missing``."""

    model_config = ConfigDict(frozen=True)

    kpi: str
    op: Operator
    value: float | str = Field(
        description="Number, or the name of a settings field (e.g. min_dollar_volume)"
    )
    allow_missing: bool = False

    def threshold(self, settings: BullishSettings) -> float:
        if isinstance(self.value, str):
            return float(getattr(settings, self.value))
        return float(self.value)

    def apply(self, values: pd.Series, settings: BullishSettings) -> pd.Series:
        threshold = self.threshold(settings)
        ops = {
            ">=": values >= threshold,
            "<=": values <= threshold,
            ">": values > threshold,
            "<": values < threshold,
        }
        passed = ops[self.op]
        return passed | values.isna() if self.allow_missing else passed & values.notna()


class CompositeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    entity_type: EntityType
    description: str
    blocks: tuple[BlockSpec, ...]
    filters: tuple[Filter, ...] = ()
    evidence: str = Field(
        default="", description="One line on the literature behind the composite"
    )

    def kpis(self) -> list[str]:
        return [k.kpi for block in self.blocks for k in block.kpis]

    def validate_against_registry(self) -> None:
        for name in {*self.kpis(), *(f.kpi for f in self.filters)}:
            REGISTRY.get(self.entity_type, name)


class ScoreResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: CompositeSpec
    scores: pd.DataFrame = Field(
        description="score, coverage, eligible and one block_<name> column per block"
    )
    components: pd.DataFrame = Field(
        description="Directed percentile of every KPI used by the composite"
    )


def score_composite(
    spec: CompositeSpec,
    kpis: pd.DataFrame,
    settings: BullishSettings,
    groups: pd.Series | None = None,
) -> ScoreResult:
    """Rank every KPI, average inside blocks, then across blocks; weights renormalise over what is available."""
    spec.validate_against_registry()
    components = pd.DataFrame(index=kpis.index)
    scores = pd.DataFrame(index=kpis.index)
    total_weight = 0.0
    weighted_sum = pd.Series(0.0, index=kpis.index)
    weight_available = pd.Series(0.0, index=kpis.index)
    for block in spec.blocks:
        block_sum = pd.Series(0.0, index=kpis.index)
        block_available = pd.Series(0.0, index=kpis.index)
        for item in block.kpis:
            values = (
                kpis[item.kpi]
                if item.kpi in kpis
                else pd.Series(np.nan, index=kpis.index)
            )
            pct = directed_percentile(
                values,
                REGISTRY.direction(spec.entity_type, item.kpi),
                groups,
                settings.winsor_limit,
                settings.min_peers,
                invert=item.invert,
            )
            components[item.kpi] = pct
            block_sum += pct.fillna(0.0) * item.weight
            block_available += pct.notna().astype(float) * item.weight
        block_score = (block_sum / block_available.replace(0.0, np.nan)).rename(
            f"block_{block.name}"
        )
        scores[f"block_{block.name}"] = block_score
        weighted_sum += block_score.fillna(0.0) * block.weight
        weight_available += block_score.notna().astype(float) * block.weight
        total_weight += block.weight
    scores["score"] = weighted_sum / weight_available.replace(0.0, np.nan)
    scores["coverage"] = weight_available / total_weight
    eligible = pd.Series(True, index=kpis.index)
    for rule in spec.filters:
        values = (
            kpis[rule.kpi] if rule.kpi in kpis else pd.Series(np.nan, index=kpis.index)
        )
        eligible &= rule.apply(values.astype(float), settings)
    scores["eligible"] = eligible
    scores["rank"] = scores["score"].where(eligible).rank(ascending=False, method="min")
    return ScoreResult(spec=spec, scores=scores, components=components)


def explain(result: ScoreResult, entity: str) -> dict[str, object]:
    """Blocks, their weights and the percentile of every KPI behind ``entity``'s score."""
    if entity not in result.scores.index:
        raise KeyError(f"{entity} is not scored by {result.spec.name}")
    row = row_of(result.scores, entity)
    blocks = []
    for block in result.spec.blocks:
        kpis = []
        for item in block.kpis:
            definition = REGISTRY.get(result.spec.entity_type, item.kpi)
            pct = as_float(row_of(result.components, entity)[item.kpi])
            kpis.append(
                {
                    "kpi": item.kpi,
                    "weight": item.weight,
                    "percentile": None if pd.isna(pct) else round(pct, 1),
                    "tier": definition.tier,
                    "inverted": item.invert,
                    "description": definition.description,
                }
            )
        value = row[f"block_{block.name}"]
        blocks.append(
            {
                "block": block.name,
                "weight": block.weight,
                "score": None if pd.isna(value) else round(float(value), 1),
                "rationale": block.rationale,
                "kpis": kpis,
            }
        )
    return {
        "entity": entity,
        "composite": result.spec.name,
        "score": None if pd.isna(row["score"]) else round(float(row["score"]), 1),
        "coverage": round(as_float(row["coverage"]), 2),
        "eligible": bool(row["eligible"]),
        "rank": None if pd.isna(row["rank"]) else int(row["rank"]),
        "evidence": result.spec.evidence,
        "blocks": blocks,
    }
