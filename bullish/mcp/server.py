"""The bullish MCP server: read-only tools, resources and prompts over the stored KPI runs.

Run it with ``bullish mcp --db /path/to/bear.db`` (stdio, for Claude Desktop
and Claude Code) or ``bullish mcp --transport streamable-http --port 8765``.
Every tool takes an optional ``as_of`` (ISO date) and answers from the
latest KPI run at or before that date, so questions about the past are
answered with what was known then.
"""

from __future__ import annotations

import datetime
import json
import logging
from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from bullish.config import BullishSettings
from bullish.mcp.service import (
    BullishService,
    EtfComparison,
    PortfolioAnalysis,
    Ranking,
    Scorecard,
)
from bullish.scoring.coverage import CoverageReport
from bullish.scoring.presets import EQUITY_PRESETS
from bullish.scoring.verdict import StockVerdict

logger = logging.getLogger(__name__)
P = ParamSpec("P")
R = TypeVar("R")

READ_ONLY = ToolAnnotations(
    read_only_hint=True, open_world_hint=False, idempotent_hint=True
)
INSTRUCTIONS = """bullish exposes decision KPIs computed from a local bearish database: market regime,
sector / industry / country rankings, ETF rankings and comparisons, equity screens (presets), per-stock
scorecards and a verdict ('fundamentally sound but temporarily oversold?'), smart-money and filing detail,
macro and commodity snapshots, data-coverage grades and portfolio look-through. Every answer carries the
date of the KPI run it is based on and the evidence tier (A academic, B practitioner, C heuristic) of each
input. Start with get_market_regime, then rank_sectors / rank_etfs / screen_equities, and finish single
names with get_equity_scorecard and assess_stock. Check get_data_coverage before trusting a thin name."""


def _as_of(value: str | None) -> datetime.date | None:
    if not value:
        return None
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as e:
        raise ToolError(f"as_of must be an ISO date (YYYY-MM-DD), got {value!r}") from e


def _guard(fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Call a service method and turn its domain errors into ``ToolError`` for the client."""
    try:
        return fn(*args, **kwargs)
    except (KeyError, PermissionError, ValueError, FileNotFoundError) as e:
        raise ToolError(str(e)) from e


def build_server(  # noqa: C901, PLR0915
    settings: BullishSettings | None = None, service: BullishService | None = None
) -> MCPServer:
    settings = settings or BullishSettings()
    svc = service or BullishService(settings)
    mcp = MCPServer("bullish", instructions=INSTRUCTIONS, version="2.0")

    # --- catalogue -------------------------------------------------------

    @mcp.tool(title="List KPIs", annotations=READ_ONLY)
    def list_kpis(
        entity_type: str | None = None, block: str | None = None
    ) -> list[dict[str, Any]]:
        """Catalogue of every KPI: name, entity type (equity, sector, industry, country, etf, market,
        commodity), block, evidence tier, unit, direction.
        """
        return svc.list_kpis(entity_type, block)

    @mcp.tool(title="Describe a KPI", annotations=READ_ONLY)
    def describe_kpi(name: str, entity_type: str | None = None) -> list[dict[str, Any]]:
        """Definition, unit, evidence tier and direction of one KPI (all entity types that define it
        unless one is given).
        """
        return _guard(svc.describe_kpi, name, entity_type)

    @mcp.tool(title="List composites and presets", annotations=READ_ONLY)
    def list_composites() -> list[dict[str, Any]]:
        """Every composite score and equity preset with its blocks, weights, filters and evidence line."""
        return svc.list_composites()

    @mcp.tool(title="List KPI runs", annotations=READ_ONLY)
    def list_runs() -> list[dict[str, Any]]:
        """Dates of the stored KPI runs (what as_of values can be answered), with row counts and base currency."""
        return [r.model_dump(mode="json") for r in svc.runs()]

    # --- regime and rankings ---------------------------------------------

    @mcp.tool(title="Market regime", annotations=READ_ONLY)
    def get_market_regime(
        region: str = "ALL", as_of: str | None = None
    ) -> dict[str, Any]:
        """Risk-on / neutral / risk-off label with the trend, breadth, credit, rates and macro signals
        behind it, the growth-inflation quadrant and the factor leading over three months. region:
        ALL or an ISO-3 country with enough stocks.
        """
        return _guard(svc.market_regime, region, _as_of(as_of))

    @mcp.tool(title="Rank sectors", annotations=READ_ONLY)
    def rank_sectors(as_of: str | None = None, limit: int = 20) -> Ranking:
        """Sectors ranked by the sector score (relative strength, breadth, revisions and growth,
        valuation, flows, macro tailwind) with block scores and key KPIs.
        """
        return _guard(
            svc.ranking,
            "sector",
            "sector_score",
            _as_of(as_of),
            limit,
            (
                "mom_12_1",
                "rs_vs_market_3m",
                "rrg_quadrant",
                "breadth_above_200",
                "revision_ratio",
                "median_pe_vs_5y_z",
                "members",
            ),
        )

    @mcp.tool(title="Rank industries", annotations=READ_ONLY)
    def rank_industries(as_of: str | None = None, limit: int = 30) -> Ranking:
        """Industries ranked by the industry score (same blocks as the sector score)."""
        return _guard(
            svc.ranking,
            "industry",
            "industry_score",
            _as_of(as_of),
            limit,
            (
                "mom_12_1",
                "rs_vs_market_3m",
                "rrg_quadrant",
                "breadth_above_200",
                "revision_ratio",
                "members",
            ),
        )

    @mcp.tool(title="Rank countries", annotations=READ_ONLY)
    def rank_countries(as_of: str | None = None, limit: int = 30) -> Ranking:
        """Countries ranked by the country score (market strength, growth momentum, policy, stability,
        sentiment, external) with the macro KPIs behind it.
        """
        return _guard(
            svc.ranking,
            "country",
            "country_score",
            _as_of(as_of),
            limit,
            (
                "mom_12_1",
                "cli_change_3m",
                "pmi",
                "cpi_yoy",
                "policy_rate_change_6m",
                "fx_vs_base_3m",
                "members",
            ),
        )

    @mcp.tool(title="Rank ETFs", annotations=READ_ONLY)
    def rank_etfs(
        exposure: str | None = None, as_of: str | None = None, limit: int = 20
    ) -> Ranking:
        """ETFs ranked by the ETF score = 0.5 x attractiveness of their exposures (sector / country /
        commodity scores) + 0.3 x vehicle quality (cost, tracking, liquidity, concentration,
        risk-adjusted momentum) + 0.2 x momentum. exposure filters to funds exposed to a sector,
        country or commodity name.
        """
        return _guard(svc.rank_etfs, exposure, _as_of(as_of), limit)

    @mcp.tool(title="Compare ETFs", annotations=READ_ONLY)
    def compare_etfs(symbols: list[str], as_of: str | None = None) -> EtfComparison:
        """Side by side vehicle KPIs (expense ratio, AUM and flows, liquidity, tracking, concentration,
        premium to NAV) plus the pairwise overlap of the reported top holdings and each fund's
        derived exposures.
        """
        return _guard(svc.compare_etfs, symbols, _as_of(as_of))

    @mcp.tool(title="Screen equities", annotations=READ_ONLY)
    def screen_equities(  # noqa: PLR0913
        preset: str = "quality_compounder",
        country: str | None = None,
        sector: str | None = None,
        as_of: str | None = None,
        limit: int = 25,
        only_eligible: bool = True,
    ) -> Ranking:
        """Ranked equities for a preset: emerging_leader, quality_compounder, value_with_catalyst,
        turnaround, dividend_quality or quality_on_sale. country is ISO-3, sector a GICS sector
        name. Rows carry block scores and key KPIs; use explain_score for the full breakdown.
        """
        if preset not in EQUITY_PRESETS:
            raise ToolError(
                f"Unknown preset {preset!r}; choose from {sorted(EQUITY_PRESETS)}"
            )
        where = {k: v for k, v in (("country", country), ("sector", sector)) if v}
        return _guard(
            svc.ranking,
            "equity",
            preset,
            _as_of(as_of),
            limit,
            (
                "market_cap_base",
                "mom_12_1",
                "revenue_ttm_growth",
                "roic",
                "fcf_yield",
                "drawdown_52w",
                "coverage_grade",
            ),
            only_eligible,
            where or None,
        )

    @mcp.tool(title="Explain a score", annotations=READ_ONLY)
    def explain_score(
        entity: str,
        composite: str,
        entity_type: str = "equity",
        as_of: str | None = None,
    ) -> dict[str, Any]:
        """Every block and KPI behind an entity's composite score: weight, percentile inside the peer
        group, raw value, unit, evidence tier and rationale.
        """
        return _guard(svc.explain, entity_type, entity, composite, _as_of(as_of))

    # --- single names ----------------------------------------------------

    @mcp.tool(title="Equity scorecard", annotations=READ_ONLY)
    def get_equity_scorecard(symbol: str, as_of: str | None = None) -> Scorecard:
        """Every KPI of one stock grouped by block (momentum, growth, quality, valuation, revisions,
        ownership, macro exposure), its preset scores and ranks, the sector context, the
        data-coverage grade and data freshness.
        """
        return _guard(svc.scorecard, symbol, _as_of(as_of))

    @mcp.tool(title="Assess a stock", annotations=READ_ONLY)
    def assess_stock(symbol: str, as_of: str | None = None) -> StockVerdict:
        """Verdict on one stock: is the business fundamentally sound (checklist), is it oversold
        (signals), is the drop company-specific or sector / market wide, does the evidence read
        temporary (insiders buying, estimates holding) or structural (8-K restatement, offering,
        departures), recovery evidence, and a label: quality_on_sale, falling_knife,
        sound_not_cheap, oversold_unclear, avoid.
        """
        return _guard(svc.assess, symbol, _as_of(as_of))

    @mcp.tool(title="Technical signals", annotations=READ_ONLY)
    def get_technical_signals(symbol: str, as_of: str | None = None) -> dict[str, Any]:
        """TA-Lib indicators of one stock (RSI, MACD, ADX, stochastic, MFI, moving averages, candlestick
        patterns): current values and the trading-day age of the last crossover or pattern, most recent first.
        """
        return _guard(svc.technical_signals, symbol, _as_of(as_of))

    @mcp.tool(title="KPI time series", annotations=READ_ONLY)
    def get_kpi_series(
        entity: str,
        kpi: str,
        entity_type: str = "equity",
        start: str | None = None,
        end: str | None = None,
    ) -> dict[str, Any]:
        """Stored history of one KPI for one entity across KPI runs (one point per run date)."""
        return _guard(
            svc.kpi_series, entity_type, entity, kpi, _as_of(start), _as_of(end)
        )

    @mcp.tool(title="Smart money detail", annotations=READ_ONLY)
    def get_smart_money(symbol: str, as_of: str | None = None) -> dict[str, Any]:
        """Raw rows behind the ownership KPIs: insider transactions of the last year, 13F holder
        summary per quarter, Schedule 13D/13G beneficial owners.
        """
        return _guard(svc.smart_money, symbol, _as_of(as_of))

    @mcp.tool(title="SEC filings", annotations=READ_ONLY)
    def get_filings(
        symbol: str, forms: list[str] | None = None, since: str | None = None
    ) -> list[dict[str, Any]]:
        """10-K, 10-Q and 8-K filings of a US issuer with the 8-K item codes (4.02 restatement, 4.01
        auditor change, 5.02 officer departure, 1.03 bankruptcy, 2.02 results ...).
        """
        return _guard(svc.filings, symbol, forms, _as_of(since))

    @mcp.tool(title="Earnings calendar", annotations=READ_ONLY)
    def get_earnings_calendar(
        days: int = 30, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        """Scheduled earnings reports in the next ``days`` days across the universe (event risk)."""
        return _guard(svc.earnings_calendar, days, _as_of(as_of))

    # --- context data ----------------------------------------------------

    @mcp.tool(title="Macro snapshot", annotations=READ_ONLY)
    def get_macro_snapshot(
        entity: str = "USA", concepts: list[str] | None = None, as_of: str | None = None
    ) -> dict[str, Any]:
        """Latest value, date and 3 / 12-month change of the macro series of a country (ISO-3), region
        or commodity, point-in-time as of the date given.
        """
        return _guard(svc.macro_snapshot, entity, concepts, _as_of(as_of))

    @mcp.tool(title="Commodity snapshot", annotations=READ_ONLY)
    def get_commodity_snapshot(
        code: str | None = None, as_of: str | None = None
    ) -> list[dict[str, Any]]:
        """Momentum, volatility, drawdown and speculative positioning of each commodity, with the
        sectors each one helps or hurts.
        """
        return _guard(svc.commodity_snapshot, code, _as_of(as_of))

    @mcp.tool(title="Data coverage", annotations=READ_ONLY)
    def get_data_coverage(
        symbol: str | None = None, as_of: str | None = None
    ) -> dict[str, Any]:
        """Can the database support a judgement? For a symbol: grade A / B / C / insufficient per
        dimension (statements, prices, expectations, ownership, reference), what is missing and the
        bearish command that fills it. Without a symbol: the grade distribution of the universe by
        country.
        """
        return _guard(svc.data_coverage, symbol, _as_of(as_of))

    @mcp.tool(title="Analyze a portfolio", annotations=READ_ONLY)
    def analyze_portfolio(
        holdings: dict[str, float], as_of: str | None = None
    ) -> PortfolioAnalysis:
        """Look-through sector, country and commodity exposure of a holdings map (symbol -> amount,
        equities and ETFs), concentration and one-year pairwise correlations.
        """
        return _guard(svc.analyze_portfolio, holdings, _as_of(as_of))

    if settings.allow_sql:

        @mcp.tool(title="Read-only SQL", annotations=READ_ONLY)
        def query(sql: str) -> list[dict[str, Any]]:
            """Run a read-only SELECT against the database (row-capped). Enabled because BULLISH_ALLOW_SQL is set."""
            return _guard(svc.query, sql)

    # --- resources -------------------------------------------------------

    @mcp.resource(
        "bullish://catalog/kpis", title="KPI catalogue", mime_type="application/json"
    )
    def kpi_catalogue() -> str:
        """Every KPI definition as JSON."""
        return json.dumps(svc.list_kpis(), indent=2)

    @mcp.resource(
        "bullish://catalog/composites",
        title="Composites and presets",
        mime_type="application/json",
    )
    def composite_catalogue() -> str:
        """Every composite and preset with blocks, weights and filters."""
        return json.dumps(svc.list_composites(), indent=2)

    @mcp.resource(
        "bullish://methodology/{composite}",
        title="Methodology of a composite",
        mime_type="application/json",
    )
    def methodology(composite: str) -> str:
        """Blocks, rationale, KPIs with tiers and the ranking method of one composite."""
        try:
            return json.dumps(svc.methodology(composite), indent=2)
        except KeyError as e:
            raise ToolError(f"Unknown composite {composite!r}") from e

    # --- prompts ---------------------------------------------------------

    @mcp.prompt(title="Weekly market review")
    def weekly_market_review(region: str = "ALL") -> str:
        """Regime, sector rotation, best ETFs and notable screens in one briefing."""
        return (
            f"Write a weekly market review for {region}. 1) Call get_market_regime for the regime, "
            f"quadrant and factor leadership. "
            "2) Call rank_sectors and describe the leaders, laggards and any RRG quadrant changes. 3) "
            "Call rank_countries. "
            "4) Call rank_etfs for the top vehicles on the strongest exposures. 5) Call "
            "screen_equities with emerging_leader and "
            "quality_on_sale, top 10 each. Quote the KPI run date, flag any coverage below B, and "
            "state the evidence tier behind each claim."
        )

    @mcp.prompt(title="Sector briefing")
    def sector_briefing(sector: str) -> str:
        """Why a sector ranks where it does, and the names inside it."""
        return (
            f"Brief me on the {sector} sector. Call explain_score with entity_type sector and "
            f"composite sector_score, then "
            f"screen_equities with sector='{sector}' for quality_compounder and emerging_leader, and "
            f"rank_etfs with exposure='{sector}'. "
            "Separate what has academic evidence (tier A) from context (tier C) and end with the "
            "risks the KPIs cannot see."
        )

    @mcp.prompt(title="ETF selection")
    def etf_selection(exposure: str) -> str:
        """Pick the vehicle for an exposure."""
        return (
            f"I want exposure to {exposure}. Call rank_etfs with exposure='{exposure}', then "
            f"compare_etfs on the top three. Recommend one, "
            "explaining cost, tracking, liquidity, concentration and overlap, and say what the "
            "exposure score is built from."
        )

    @mcp.prompt(title="Stock due diligence")
    def stock_due_diligence(symbol: str) -> str:
        """Fundamentally good? Temporarily oversold? What would change the view?"""
        return (
            f"Do a due diligence on {symbol}. Call get_data_coverage first and stop if the grade is "
            f"insufficient. Then get_equity_scorecard, "
            "assess_stock, get_smart_money and get_filings with forms ['8-K']. Answer: is the "
            "business sound (which checks fail), is the "
            "price depressed, is the drop company-specific or sector-wide, does the evidence read "
            "temporary or structural, and what would "
            "change the verdict. Keep the evidence tiers explicit and never present a KPI as a forecast."
        )

    return mcp


def serve(
    settings: BullishSettings | None = None,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    server = build_server(settings)
    if transport == "stdio":
        server.run()
    else:
        server.run(transport="streamable-http", host=host, port=port)


_ = CoverageReport  # re-exported for type checkers reading the tool signatures
