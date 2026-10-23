"""Group-level KPI blocks and the orchestrator that runs them after the equity KPIs."""

from __future__ import annotations

import logging

import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.group import commodity, country_macro, etf, market
from bullish.kpi.group.aggregate import GROUP_TYPES, compute_group, market_index
from bullish.kpi.registry import REGISTRY

logger = logging.getLogger(__name__)


def compute_group_kpis(
    ctx: DataContext, equity: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """``{entity_type: frame}`` for sector, industry, country, market, etf and commodity."""
    frames: dict[str, pd.DataFrame] = {}
    benchmark = (
        market_index(ctx, equity) if not ctx.close.empty else pd.Series(dtype=float)
    )
    for group_type in GROUP_TYPES:
        frame = (
            compute_group(ctx, equity, group_type, benchmark)
            if not ctx.close.empty
            else pd.DataFrame()
        )
        if group_type == "country":
            countries = sorted(set(frame.index) | set(ctx.universe["country"].dropna()))
            macro = country_macro.compute(ctx, countries)
            frame = frame.reindex(index=macro.index).join(macro)
            frame = frame.reindex(columns=REGISTRY.names("country"))
        frames[group_type] = frame
        logger.info(f"{group_type}: {len(frame)} groups")
    frames["market"] = market.compute(ctx, equity)
    frames["etf"] = etf.compute(ctx)
    frames["commodity"] = commodity.compute(ctx)
    return frames
