"""Group-level KPI blocks and the orchestrator that runs them after the equity KPIs."""

from __future__ import annotations

import logging

import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.group import commodity, country_macro, etf, market
from bullish.kpi.group.aggregate import GROUP_TYPES, compute_group, market_index
from bullish.kpi.profile import PhaseTimer
from bullish.kpi.registry import REGISTRY

logger = logging.getLogger(__name__)


def compute_group_kpis(
    ctx: DataContext, equity: pd.DataFrame, timer: PhaseTimer | None = None
) -> dict[str, pd.DataFrame]:
    """``{entity_type: frame}`` for sector, industry, country, market, etf and commodity."""
    timer = timer or PhaseTimer()
    frames: dict[str, pd.DataFrame] = {}
    benchmark = (
        market_index(ctx, equity) if not ctx.close.empty else pd.Series(dtype=float)
    )
    for group_type in GROUP_TYPES:
        with timer.phase(f"group.{group_type}"):
            frame = (
                compute_group(ctx, equity, group_type, benchmark)
                if not ctx.close.empty
                else pd.DataFrame()
            )
            if group_type == "country":
                countries = sorted(
                    set(frame.index) | set(ctx.universe["country"].dropna())
                )
                macro = country_macro.compute(ctx, countries)
                frame = frame.reindex(index=macro.index).join(macro)
                frame = frame.reindex(columns=REGISTRY.names("country"))
        frames[group_type] = frame
        logger.info(f"{group_type}: {len(frame)} groups")
    with timer.phase("group.market"):
        frames["market"] = market.compute(ctx, equity)
    with timer.phase("group.etf"):
        frames["etf"] = etf.compute(ctx)
    with timer.phase("group.commodity"):
        frames["commodity"] = commodity.compute(ctx)
    return frames
