"""Equity KPI blocks; :func:`compute_equity_kpis` runs them all on a :class:`DataContext`."""

from __future__ import annotations

import logging
from types import ModuleType

import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.equity import (
    growth,
    macro_exposure,
    momentum,
    ownership,
    quality,
    revisions,
    valuation,
)
from bullish.kpi.registry import REGISTRY

logger = logging.getLogger(__name__)

BLOCKS: tuple[ModuleType, ...] = (
    momentum,
    growth,
    quality,
    valuation,
    revisions,
    ownership,
    macro_exposure,
)


def compute_equity_kpis(ctx: DataContext) -> pd.DataFrame:
    """One row per symbol, one column per declared equity KPI."""
    frames = []
    for block in BLOCKS:
        frame = block.compute(ctx)
        frames.append(REGISTRY.validate_frame("equity", block.BLOCK, frame))
        logger.info(
            f"equity.{block.BLOCK}: {frame.shape[1]} KPIs for {len(frame)} symbols"
        )
    result = pd.concat(frames, axis=1)
    result.index.name = "symbol"
    return result.astype(float)
