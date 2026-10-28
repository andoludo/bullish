"""One KPI run: equity KPIs, group KPIs, composites, look-through ETF score, coverage grades, persisted."""

from __future__ import annotations

import datetime
import logging

import pandas as pd

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.database.crud import BullishDb
from bullish.database.kpi_store import KpiRun
from bullish.kpi.equity import compute_equity_kpis
from bullish.kpi.group import compute_group_kpis
from bullish.kpi.registry import declare_many
from bullish.scoring.composite import ScoreResult, score_composite
from bullish.scoring.coverage import grade_universe
from bullish.utils.typed import as_float
from bullish.scoring.presets import (
    COUNTRY_SCORE,
    EQUITY_PRESETS,
    ETF_VEHICLE_SCORE,
    INDUSTRY_SCORE,
    SECTOR_SCORE,
)

logger = logging.getLogger(__name__)

ETF_SCORE_NAME = "etf_score"
COVERAGE_GRADE = {"A": 3.0, "B": 2.0, "C": 1.0, "insufficient": 0.0}
declare_many(
    "etf",
    "lookthrough",
    [
        (
            "exposure_score",
            "B",
            "score",
            "Weighted sector / country / commodity attractiveness of the fund's exposures",
            True,
        ),
        (
            "etf_score",
            "B",
            "score",
            "0.5 x exposure score + 0.3 x vehicle score + 0.2 x momentum percentile",
            True,
        ),
    ],
)
declare_many(
    "equity",
    "coverage",
    [
        (
            "coverage_grade",
            "C",
            "grade",
            "Data coverage grade: 3 = A, 2 = B, 1 = C, 0 = insufficient",
            True,
        ),
    ],
)


class RunResult:
    def __init__(self, as_of: datetime.date) -> None:
        self.as_of = as_of
        self.equity = pd.DataFrame()
        self.groups: dict[str, pd.DataFrame] = {}
        self.scores: dict[tuple[str, str], ScoreResult] = {}
        self.coverage = pd.DataFrame()
        self.run: KpiRun | None = None


def lookthrough_scores(
    ctx: DataContext,
    scores: dict[tuple[str, str], ScoreResult],
    commodity: pd.DataFrame,
) -> pd.Series:
    """Weighted attractiveness of each ETF's exposures: sector / industry scores, country scores, commodity momentum."""
    exposures = ctx.etf_exposures
    if exposures.empty:
        return pd.Series(dtype=float)
    lookups: dict[str, pd.Series] = {}
    if ("sector", SECTOR_SCORE.name) in scores:
        lookups["sector"] = scores[("sector", SECTOR_SCORE.name)].scores["score"]
    if ("country", COUNTRY_SCORE.name) in scores:
        lookups["country"] = scores[("country", COUNTRY_SCORE.name)].scores["score"]
    if not commodity.empty and "ret_3m" in commodity:
        lookups["commodity"] = commodity["ret_3m"].rank(pct=True) * 100.0

    def _lookup(entity_type: str, entity: str) -> float:
        table = lookups.get(entity_type)
        return as_float(table.get(entity)) if table is not None else float("nan")

    values = [
        _lookup(entity_type, entity)
        for entity_type, entity in zip(
            exposures["entity_type"], exposures["entity"], strict=True
        )
    ]
    frame = exposures.assign(_score=values).dropna(subset=["_score"])
    if frame.empty:
        return pd.Series(dtype=float)
    weighted = (frame["_score"] * frame["weight"]).groupby(frame["symbol"]).sum()
    weights = frame["weight"].groupby(frame["symbol"]).sum()
    return weighted / weights.replace(0.0, pd.NA)


def run(  # noqa: PLR0913
    db: BullishDb,
    settings: BullishSettings,
    as_of: datetime.date | None = None,
    countries: list[str] | None = None,
    persist: bool = True,
    symbols: list[str] | None = None,
) -> RunResult:
    """Build the point-in-time context for ``as_of`` and run every KPI, score and grade on it."""
    as_of = as_of or datetime.date.today()
    ctx = DataContext(
        db=db, settings=settings, as_of=as_of, countries=countries, symbols=symbols
    )
    return run_context(ctx, persist=persist)


def run_context(ctx: DataContext, persist: bool = True) -> RunResult:
    """One KPI run on an existing context (``bullish history`` derives one per date from a span)."""
    db, settings, as_of, countries = ctx.db, ctx.settings, ctx.as_of, ctx.countries
    result = RunResult(as_of)
    if ctx.universe.empty:
        logger.warning("Empty universe: nothing to compute")
        return result
    logger.info(f"KPI run as of {as_of}: {len(ctx.universe)} equities")
    equity = compute_equity_kpis(ctx)
    coverage = (
        grade_universe(ctx.coverage_index, ctx.universe, ctx.as_of_ts)
        if not ctx.coverage.empty
        else pd.DataFrame()
    )
    if not coverage.empty:
        equity["coverage_grade"] = (
            coverage["overall"].map(COVERAGE_GRADE).reindex(equity.index)
        )
    groups = compute_group_kpis(ctx, equity)
    peers = ctx.peer_group()
    scores: dict[tuple[str, str], ScoreResult] = {}
    for spec in EQUITY_PRESETS.values():
        scores[("equity", spec.name)] = score_composite(spec, equity, settings, peers)
    for spec, frame in (
        (SECTOR_SCORE, groups["sector"]),
        (INDUSTRY_SCORE, groups["industry"]),
        (COUNTRY_SCORE, groups["country"]),
    ):
        if not frame.empty:
            scores[(spec.entity_type, spec.name)] = score_composite(
                spec, frame, settings
            )
    etf = groups["etf"]
    if not etf.empty:
        vehicle = score_composite(ETF_VEHICLE_SCORE, etf, settings)
        scores[("etf", ETF_VEHICLE_SCORE.name)] = vehicle
        exposure = (
            lookthrough_scores(ctx, scores, groups["commodity"])
            .reindex(etf.index)
            .astype(float)
        )
        momentum = (
            vehicle.components["mom_12_1"]
            if "mom_12_1" in vehicle.components
            else pd.Series(dtype=float)
        )
        etf["exposure_score"] = exposure
        etf["etf_score"] = (
            0.5 * exposure.fillna(vehicle.scores["score"])
            + 0.3 * vehicle.scores["score"]
            + 0.2 * momentum.reindex(etf.index).fillna(50.0)
        )
    result.equity, result.groups, result.scores, result.coverage = (
        equity,
        groups,
        scores,
        coverage,
    )
    if persist:
        _persist(db, settings, result, countries)
    return result


def _persist(
    db: BullishDb,
    settings: BullishSettings,
    result: RunResult,
    countries: list[str] | None,
) -> None:
    written = db.write_kpi_frame("equity", result.equity, result.as_of)
    for entity_type, frame in result.groups.items():
        written += db.write_kpi_frame(entity_type, frame, result.as_of)
    scores = 0
    for (entity_type, name), score in result.scores.items():
        scores += db.write_scores(
            entity_type, name, score.scores, score.components, result.as_of
        )
    if "etf_score" in result.groups.get("etf", pd.DataFrame()):
        etf = result.groups["etf"]
        frame = pd.DataFrame(
            {"score": etf["etf_score"], "coverage": 1.0, "eligible": True}
        )
        frame["rank"] = frame["score"].rank(ascending=False, method="min")
        components = pd.DataFrame({"exposure_score": etf["exposure_score"]})
        scores += db.write_scores(
            "etf", ETF_SCORE_NAME, frame, components, result.as_of
        )
    run = KpiRun(
        as_of=result.as_of,
        universe=",".join(countries) if countries else "all",
        entities=int(len(result.equity)),
        kpis=written,
        scores=scores,
        base_currency=settings.base_currency,
    )
    db.write_kpi_run(run)
    result.run = run
    logger.info(
        f"Persisted {written} KPI values and {scores} scores for {result.as_of}"
    )
