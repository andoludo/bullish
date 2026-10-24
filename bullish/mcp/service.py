"""Plain-Python service behind the MCP tools: reads the stored KPI runs and the bearish tables.

Every method returns Pydantic models or JSON-ready dicts so the server
layer stays a thin registration of tools, resources and prompts.
"""

from __future__ import annotations

import datetime
import json
import logging
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.database.crud import BullishDb
from bullish.database.kpi_store import KpiRun
from bullish.kpi.registry import REGISTRY, TIER_MEANING, KpiDefinition
from bullish.scoring.coverage import CoverageReport, grade_symbol
from bullish.scoring.presets import ALL_COMPOSITES, EQUITY_PRESETS
from bullish.scoring.regime import MarketRegime, classify
from bullish.scoring.verdict import StockVerdict, assess
from bullish.utils.typed import as_float, iso_date, row_of

logger = logging.getLogger(__name__)

GRADE_BY_CODE = {3.0: "A", 2.0: "B", 1.0: "C", 0.0: "insufficient"}
STRUCTURAL_RISK_KPIS = (
    "red_flag_8k_12m",
    "officer_departure_12m",
    "equity_offering_12m",
    "short_interest_pct_float",
)


class Freshness(BaseModel):
    kpi_run: datetime.date | None = Field(
        description="Date of the KPI run the answer is based on"
    )
    last_price: datetime.date | None = None
    last_statement_filed: datetime.date | None = None
    last_macro_vintage: datetime.date | None = None
    base_currency: str


class RankedRow(BaseModel):
    rank: int | None
    entity: str
    score: float
    coverage: float
    eligible: bool
    blocks: dict[str, float | None]
    kpis: dict[str, float | None] = Field(
        default_factory=dict, description="Selected raw KPIs for context"
    )


class Ranking(BaseModel):
    composite: str
    entity_type: str
    as_of: datetime.date | None
    description: str
    evidence: str
    freshness: Freshness
    rows: list[RankedRow]
    note: str = ""


class Scorecard(BaseModel):
    symbol: str
    as_of: datetime.date | None
    reference: dict[str, Any]
    kpis: dict[str, dict[str, float | None]] = Field(
        description="Raw KPI values grouped by block"
    )
    tiers: dict[str, str] = Field(description="Evidence tier per KPI")
    preset_scores: dict[str, dict[str, float | int | bool | None]]
    sector_context: dict[str, float | None]
    coverage: CoverageReport | None
    freshness: Freshness


class EtfComparison(BaseModel):
    as_of: datetime.date | None
    vehicles: list[dict[str, Any]]
    overlap: dict[str, dict[str, float]] = Field(
        description="Pairwise weighted overlap of the reported top holdings"
    )
    freshness: Freshness


class PortfolioAnalysis(BaseModel):
    as_of: datetime.date | None
    weights: dict[str, float]
    sector_exposure: dict[str, float]
    country_exposure: dict[str, float]
    commodity_exposure: dict[str, float]
    concentration_hhi: float
    average_pairwise_correlation_1y: float | None
    correlations: dict[str, dict[str, float]]
    unknown_symbols: list[str]
    freshness: Freshness


def _clean(value: Any) -> Any:
    if isinstance(value, np.floating | float):
        return None if np.isnan(value) else round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, pd.Timestamp | datetime.datetime):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    return value


def records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    rows = frame.head(limit) if limit else frame
    return [
        {str(k): _clean(v) for k, v in row.items()}
        for row in rows.reset_index().to_dict(orient="records")
    ]


class BullishService:
    def __init__(self, settings: BullishSettings, db: BullishDb | None = None) -> None:
        self.settings = settings
        self._db = db

    @property
    def db(self) -> BullishDb:
        if self._db is None:
            self._db = BullishDb(database_path=self.settings.require_database())
        return self._db

    # --- context ---------------------------------------------------------

    @lru_cache(maxsize=4)  # noqa: B019
    def context(self, as_of: datetime.date | None) -> DataContext:
        return DataContext(
            db=self.db, settings=self.settings, as_of=as_of or datetime.date.today()
        )

    def frame(
        self, entity_type: str, as_of: datetime.date | None = None
    ) -> pd.DataFrame:
        return self.db.read_kpi_frame(entity_type, as_of)

    def freshness(self, as_of: datetime.date | None = None) -> Freshness:
        run_date = self.db.latest_kpi_date("equity", as_of)
        last_price = self.db.read_query("SELECT MAX(date) AS d FROM price")["d"].iloc[0]
        last_filed = self.db.read_query(
            "SELECT MAX(filed_date) AS d FROM financialmetrics"
        )["d"].iloc[0]
        last_macro = self.db.read_query("SELECT MAX(as_of) AS d FROM macroobservation")[
            "d"
        ].iloc[0]
        return Freshness(
            kpi_run=run_date,
            last_price=_date(last_price),
            last_statement_filed=_date(last_filed),
            last_macro_vintage=_date(last_macro),
            base_currency=self.settings.base_currency,
        )

    # --- catalogue -------------------------------------------------------

    def list_kpis(
        self, entity_type: str | None = None, block: str | None = None
    ) -> list[dict[str, Any]]:
        return [d.model_dump() | {"tier_meaning": TIER_MEANING[d.tier]} for d in REGISTRY.all(entity_type, block)]  # type: ignore[arg-type]

    def describe_kpi(
        self, name: str, entity_type: str | None = None
    ) -> list[dict[str, Any]]:
        found = (
            [REGISTRY.get(entity_type, name)] if entity_type else REGISTRY.find(name)
        )
        if not found:
            raise KeyError(f"No KPI named {name}")
        return [d.model_dump() | {"tier_meaning": TIER_MEANING[d.tier]} for d in found]

    def list_composites(self) -> list[dict[str, Any]]:
        return [
            {
                "name": spec.name,
                "entity_type": spec.entity_type,
                "description": spec.description,
                "evidence": spec.evidence,
                "blocks": [
                    {
                        "name": b.name,
                        "weight": b.weight,
                        "kpis": [k.kpi for k in b.kpis],
                    }
                    for b in spec.blocks
                ],
                "filters": [f.model_dump() for f in spec.filters],
            }
            for spec in ALL_COMPOSITES.values()
        ]

    def methodology(self, composite: str) -> dict[str, Any]:
        spec = ALL_COMPOSITES[composite]
        return {
            "name": spec.name,
            "entity_type": spec.entity_type,
            "description": spec.description,
            "evidence": spec.evidence,
            "blocks": [
                {
                    "name": b.name,
                    "weight": b.weight,
                    "rationale": b.rationale,
                    "kpis": [
                        {
                            "kpi": k.kpi,
                            "weight": k.weight,
                            "inverted": k.invert,
                            **_definition(spec.entity_type, k.kpi),
                        }
                        for k in b.kpis
                    ],
                }
                for b in spec.blocks
            ],
            "filters": [f.model_dump() for f in spec.filters],
            "method": "Each KPI is winsorised and ranked 0-100 inside its peer group "
            "(direction-aware); blocks average their "
            "KPIs, the composite averages its blocks; weights renormalise over the KPIs available.",
        }

    # --- rankings --------------------------------------------------------

    def ranking(  # noqa: PLR0913
        self,
        entity_type: str,
        composite: str,
        as_of: datetime.date | None = None,
        limit: int | None = None,
        context_kpis: tuple[str, ...] = (),
        only_eligible: bool = True,
        where: dict[str, str] | None = None,
    ) -> Ranking:
        spec = ALL_COMPOSITES.get(composite)
        scores = self.db.read_scores(entity_type, composite, as_of)
        if scores.empty:
            return Ranking(
                composite=composite,
                entity_type=entity_type,
                as_of=None,
                description=spec.description if spec else "",
                evidence=spec.evidence if spec else "",
                freshness=self.freshness(as_of),
                rows=[],
                note="No stored run: execute `bullish compute` first.",
            )
        run_date = scores.attrs["as_of"]
        kpis = self.frame(entity_type, run_date)
        if where and entity_type == "equity":
            universe = self.context(run_date).universe
            keep = pd.Series(True, index=scores.index)
            for column, value in where.items():
                if column in universe:
                    keep &= universe[column].reindex(scores.index).eq(value)
            scores = scores[keep]
        if only_eligible:
            scores = scores[scores["eligible"]]
        rows = []
        for entity, row in scores.head(limit or self.settings.max_rows).iterrows():
            context = {
                k: (
                    _clean(row_of(kpis, str(entity))[k])
                    if entity in kpis.index and k in kpis
                    else None
                )
                for k in context_kpis
            }
            rows.append(
                RankedRow(
                    rank=None if pd.isna(row["rank"]) else int(row["rank"]),
                    entity=str(entity),
                    score=round(float(row["value"]), 1),
                    coverage=round(float(row["coverage"]), 2),
                    eligible=bool(row["eligible"]),
                    blocks=row["components"].get("blocks", {}),
                    kpis=context,
                )
            )
        return Ranking(
            composite=composite,
            entity_type=entity_type,
            as_of=run_date,
            description=spec.description if spec else "",
            evidence=spec.evidence if spec else "",
            freshness=self.freshness(as_of),
            rows=rows,
        )

    def explain(
        self,
        entity_type: str,
        entity: str,
        composite: str,
        as_of: datetime.date | None = None,
    ) -> dict[str, Any]:
        spec = ALL_COMPOSITES[composite]
        scores = self.db.read_scores(entity_type, composite, as_of)
        if scores.empty or entity not in scores.index:
            raise KeyError(
                f"{entity} has no stored {composite} score; run `bullish compute`"
            )
        row = row_of(scores, entity)
        kpis = self.frame(entity_type, scores.attrs["as_of"])
        blocks = []
        for block in spec.blocks:
            items = []
            for item in block.kpis:
                definition = REGISTRY.get(entity_type, item.kpi)
                items.append(
                    {
                        "kpi": item.kpi,
                        "weight": item.weight,
                        "inverted": item.invert,
                        "tier": definition.tier,
                        "percentile": row["components"].get("kpis", {}).get(item.kpi),
                        "value": (
                            _clean(kpis.loc[entity, item.kpi])
                            if entity in kpis.index and item.kpi in kpis
                            else None
                        ),
                        "unit": definition.unit,
                        "description": definition.description,
                    }
                )
            blocks.append(
                {
                    "block": block.name,
                    "weight": block.weight,
                    "score": row["components"].get("blocks", {}).get(block.name),
                    "rationale": block.rationale,
                    "kpis": items,
                }
            )
        return {
            "entity": entity,
            "entity_type": entity_type,
            "composite": composite,
            "as_of": iso_date(scores.attrs["as_of"]),
            "score": round(as_float(row["value"]), 1),
            "rank": None if pd.isna(row["rank"]) else int(row["rank"]),
            "coverage": round(as_float(row["coverage"]), 2),
            "eligible": bool(row["eligible"]),
            "evidence": spec.evidence,
            "blocks": blocks,
        }

    # --- single entities -------------------------------------------------

    def scorecard(self, symbol: str, as_of: datetime.date | None = None) -> Scorecard:
        kpis = self.frame("equity", as_of)
        if kpis.empty or symbol not in kpis.index:
            raise KeyError(
                f"{symbol} is not in the stored equity KPIs (unknown symbol, or run `bullish compute`)"
            )
        run_date = kpis.attrs["as_of"]
        row = kpis.loc[symbol]
        grouped: dict[str, dict[str, float | None]] = {}
        tiers: dict[str, str] = {}
        for definition in REGISTRY.all("equity"):
            if definition.name in row.index:
                grouped.setdefault(definition.block, {})[definition.name] = _clean(
                    row[definition.name]
                )
                tiers[definition.name] = definition.tier
        presets = {}
        for name in EQUITY_PRESETS:
            scores = self.db.read_scores("equity", name, run_date)
            if not scores.empty and symbol in scores.index:
                s = row_of(scores, symbol)
                presets[name] = {
                    "score": round(as_float(s["value"]), 1),
                    "rank": None if pd.isna(s["rank"]) else int(s["rank"]),
                    "eligible": bool(s["eligible"]),
                    "coverage": round(as_float(s["coverage"]), 2),
                }
        ctx = self.context(run_date)
        reference = (
            {k: _clean(v) for k, v in ctx.universe.loc[symbol].items()}
            if symbol in ctx.universe.index
            else {}
        )
        sector_name = reference.get("sector")
        sectors = self.frame("sector", run_date)
        sector_context = (
            {k: _clean(v) for k, v in sectors.loc[sector_name].items()}
            if sector_name in sectors.index
            else {}
        )
        coverage = (
            grade_symbol(
                symbol,
                ctx.coverage,
                row_of(ctx.universe, symbol) if symbol in ctx.universe.index else None,
                ctx.as_of_ts,
            )
            if not ctx.coverage.empty
            else None
        )
        return Scorecard(
            symbol=symbol,
            as_of=run_date,
            reference=reference,
            kpis=grouped,
            tiers=tiers,
            preset_scores=presets,
            sector_context=sector_context,
            coverage=coverage,
            freshness=self.freshness(as_of),
        )

    def assess(self, symbol: str, as_of: datetime.date | None = None) -> StockVerdict:
        kpis = self.frame("equity", as_of)
        if kpis.empty or symbol not in kpis.index:
            raise KeyError(
                f"{symbol} is not in the stored equity KPIs (unknown symbol, or run `bullish compute`)"
            )
        run_date = kpis.attrs["as_of"]
        row = row_of(kpis, symbol).copy()
        for name in ("roic", "fcf_conversion", "piotroski_f", "mom_12_1"):
            scores = self.db.read_scores("equity", "quality_on_sale", run_date)
            pct = (
                scores.loc[symbol]["components"].get("kpis", {}).get(name)
                if not scores.empty and symbol in scores.index
                else None
            )
            row[f"pct_{name}"] = np.nan if pct is None else pct
        ctx = self.context(run_date)
        sector_name = (
            str(ctx.universe.loc[symbol, "sector"])
            if symbol in ctx.universe.index
            else None
        )
        sectors = self.frame("sector", run_date)
        sector_row = (
            row_of(sectors, sector_name)
            if sector_name is not None and sector_name in sectors.index
            else None
        )
        grade = (
            GRADE_BY_CODE.get(as_float(row.get("coverage_grade", np.nan)))
            if not pd.isna(row.get("coverage_grade", np.nan))
            else None
        )
        return assess(
            symbol, row, run_date.isoformat(), sector_row, coverage_grade=grade
        )

    def technical_signals(
        self, symbol: str, as_of: datetime.date | None = None
    ) -> dict[str, Any]:
        """TA-Lib signals of one stock: current indicator values and the age of the last dated signals."""
        kpis = self.frame("equity", as_of)
        if kpis.empty or symbol not in kpis.index:
            raise KeyError(
                f"{symbol} is not in the stored equity KPIs (unknown symbol, or run `bullish compute`)"
            )
        row = kpis.loc[symbol]
        values: dict[str, float | None] = {}
        days_since: dict[str, float | None] = {}
        for definition in REGISTRY.all("equity", "technical"):
            if definition.name not in row.index:
                continue
            target = days_since if definition.name.endswith("_days_since") else values
            target[definition.name] = _clean(row[definition.name])
        recent = sorted(
            (
                (name.removesuffix("_days_since"), age)
                for name, age in days_since.items()
                if age is not None
            ),
            key=lambda item: item[1],
        )
        return {
            "symbol": symbol,
            "as_of": kpis.attrs["as_of"].isoformat(),
            "values": values,
            "days_since": days_since,
            "most_recent_signals": [
                {"signal": name, "trading_days_ago": age} for name, age in recent[:10]
            ],
        }

    def market_regime(
        self, region: str = "ALL", as_of: datetime.date | None = None
    ) -> dict[str, Any]:
        market = self.frame("market", as_of)
        if market.empty:
            raise KeyError("No stored market KPIs; run `bullish compute`")
        if region not in market.index:
            raise KeyError(
                f"No market row for {region}; available: {list(market.index)}"
            )
        regime: MarketRegime = classify(region, row_of(market, region))
        return regime.model_dump() | {
            "as_of": market.attrs["as_of"].isoformat(),
            "kpis": {k: _clean(v) for k, v in market.loc[region].items()},
            "available_regions": list(market.index),
        }

    def kpi_series(  # noqa: PLR0913
        self,
        entity_type: str,
        entity: str,
        name: str,
        start: datetime.date | None,
        end: datetime.date | None,
    ) -> dict[str, Any]:
        REGISTRY.get(entity_type, name)
        series = self.db.read_kpi_series(entity_type, entity, name, start, end)
        return {
            "entity_type": entity_type,
            "entity": entity,
            "kpi": name,
            "points": [
                {"date": iso_date(d), "value": _clean(v)} for d, v in series.items()
            ],
        }

    # --- etfs ------------------------------------------------------------

    def compare_etfs(
        self, symbols: list[str], as_of: datetime.date | None = None
    ) -> EtfComparison:
        etf = self.frame("etf", as_of)
        run_date = etf.attrs.get("as_of") if not etf.empty else None
        ctx = self.context(run_date)
        rows = []
        holdings: dict[str, pd.Series] = {}
        for symbol in symbols:
            row: dict[str, Any] = {"symbol": symbol}
            if not etf.empty and symbol in etf.index:
                row |= {k: _clean(v) for k, v in etf.loc[symbol].items()}
            if symbol in ctx.etfs.index:
                meta = ctx.etfs.loc[symbol]
                row |= {
                    "name": meta.get("long_name") or meta.get("name"),
                    "category": meta.get("category"),
                    "currency": meta.get("currency"),
                }
                raw = meta.get("holding_percent")
                parsed = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(parsed, dict) and parsed:
                    weights = pd.Series(
                        {k: float(v) for k, v in parsed.items() if v is not None}
                    )
                    holdings[symbol] = weights / 100 if weights.max() > 1.5 else weights
                exposures = ctx.etf_exposures
                row["exposures"] = records(
                    exposures[exposures["symbol"] == symbol].drop(columns=["symbol"])
                )
            rows.append(row)
        overlap: dict[str, dict[str, float]] = {}
        for a, wa in holdings.items():
            overlap[a] = {}
            for b, wb in holdings.items():
                common = wa.index.intersection(wb.index)
                overlap[a][b] = (
                    round(float(np.minimum(wa[common], wb[common]).sum()), 4)
                    if len(common)
                    else 0.0
                )
        return EtfComparison(
            as_of=run_date,
            vehicles=rows,
            overlap=overlap,
            freshness=self.freshness(as_of),
        )

    def rank_etfs(
        self, exposure: str | None, as_of: datetime.date | None, limit: int | None
    ) -> Ranking:
        ranking = self.ranking(
            "etf",
            "etf_score",
            as_of,
            limit=None,
            context_kpis=(
                "exposure_score",
                "expense_ratio",
                "aum",
                "tracking_error_1y",
                "holdings_hhi",
                "mom_12_1",
            ),
        )
        if exposure:
            exposures = self.context(ranking.as_of).etf_exposures
            wanted = set(
                exposures[exposures["entity"].str.casefold() == exposure.casefold()][
                    "symbol"
                ]
            )
            ranking.rows = [r for r in ranking.rows if r.entity in wanted]
            ranking.note = f"Filtered to funds exposed to {exposure}"
        ranking.rows = ranking.rows[: limit or self.settings.max_rows]
        return ranking

    # --- raw data --------------------------------------------------------

    def smart_money(
        self, symbol: str, as_of: datetime.date | None = None
    ) -> dict[str, Any]:
        ctx = self.context(as_of)
        year_ago = ctx.as_of_ts - pd.Timedelta(days=365)
        insiders = ctx.insiders
        insiders = (
            insiders[
                (insiders["symbol"] == symbol)
                & (pd.to_datetime(insiders["transaction_date"]) >= year_ago)
            ]
            if not insiders.empty
            else insiders
        )
        summary = ctx.holdings_summary
        summary = summary[summary["symbol"] == symbol] if not summary.empty else summary
        owners = ctx.ownership_13dg
        owners = owners[owners["symbol"] == symbol] if not owners.empty else owners
        return {
            "symbol": symbol,
            "as_of": ctx.as_of.isoformat(),
            "insider_transactions_12m": records(
                insiders.sort_values("transaction_date", ascending=False),
                self.settings.max_rows,
            ),
            "institutional_summary": records(
                summary.sort_values("period", ascending=False)
            ),
            "beneficial_owners_13dg": records(
                owners.sort_values("filed_date", ascending=False)
            ),
        }

    def filings(
        self, symbol: str, forms: list[str] | None, since: datetime.date | None
    ) -> list[dict[str, Any]]:
        rows = [f.model_dump() for f in self.db.read_filings(symbol, forms=forms)]
        if since:
            rows = [r for r in rows if r["filed_date"] >= since]
        return [
            {k: _clean(v) for k, v in r.items()}
            | {
                "item_codes": [
                    c.strip() for c in (r.get("items") or "").split(",") if c.strip()
                ]
            }
            for r in rows[: self.settings.max_rows]
        ]

    def earnings_calendar(
        self, days: int, as_of: datetime.date | None = None
    ) -> list[dict[str, Any]]:
        ctx = self.context(as_of)
        earnings = ctx.earnings
        if earnings.empty:
            return []
        window = earnings[
            (earnings["date"] > ctx.as_of_ts)
            & (earnings["date"] <= ctx.as_of_ts + pd.Timedelta(days=days))
        ]
        return records(
            window.sort_values("date")[
                ["symbol", "date", "eps_estimate", "revenue_estimate"]
            ],
            self.settings.max_rows,
        )

    def macro_snapshot(
        self,
        entity: str,
        concepts: list[str] | None,
        as_of: datetime.date | None = None,
    ) -> dict[str, Any]:
        ctx = self.context(as_of)
        frame = ctx.macro[ctx.macro["series_id"].str.endswith(f".{entity}")]
        if concepts:
            frame = frame[frame["series_id"].str.split(".").str[0].isin(concepts)]
        latest = frame.sort_values("date").groupby("series_id").tail(1)
        points = {}
        for series_id, row in latest.set_index("series_id").iterrows():
            history = frame[frame["series_id"] == series_id].set_index("date")["value"]
            points[str(series_id).split(".")[0]] = {
                "value": _clean(row["value"]),
                "date": _clean(row["date"]),
                "change_3m": (
                    _clean(history.iloc[-1] - history.iloc[-4])
                    if len(history) > 3
                    else None
                ),
                "change_12m": (
                    _clean(history.iloc[-1] - history.iloc[-13])
                    if len(history) > 12
                    else None
                ),
            }
        return {"entity": entity, "as_of": ctx.as_of.isoformat(), "series": points}

    def commodity_snapshot(
        self, code: str | None, as_of: datetime.date | None = None
    ) -> list[dict[str, Any]]:
        frame = self.frame("commodity", as_of)
        if frame.empty:
            return []
        if code:
            frame = frame.loc[[code]] if code in frame.index else frame.iloc[0:0]
        ctx = self.context(frame.attrs.get("as_of"))
        sensitivities = ctx.sector_sensitivities
        rows = []
        for commodity, row in frame.dropna(how="all").iterrows():
            affected = sensitivities[sensitivities["commodity"] == commodity]
            rows.append(
                {"commodity": commodity}
                | {k: _clean(v) for k, v in row.items()}
                | {
                    "benefits": sorted(
                        affected[affected["direction"] > 0]["sector"].tolist()
                    ),
                    "hurts": sorted(
                        affected[affected["direction"] < 0]["sector"].tolist()
                    ),
                }
            )
        return rows

    def data_coverage(
        self, symbol: str | None, as_of: datetime.date | None = None
    ) -> dict[str, Any]:
        ctx = self.context(as_of)
        if symbol:
            row = row_of(ctx.universe, symbol) if symbol in ctx.universe.index else None
            return grade_symbol(symbol, ctx.coverage, row, ctx.as_of_ts).model_dump()
        kpis = self.frame("equity", as_of)
        if kpis.empty or "coverage_grade" not in kpis:
            return {"note": "No stored run with coverage grades; run `bullish compute`"}
        grades = kpis["coverage_grade"].map(GRADE_BY_CODE)
        by_country = ctx.universe["country"].reindex(kpis.index)
        table = pd.crosstab(by_country.fillna("unknown"), grades.fillna("insufficient"))
        return {
            "as_of": kpis.attrs["as_of"].isoformat(),
            "symbols": int(len(grades)),
            "grades": grades.value_counts().to_dict(),
            "by_country": {
                str(c): {str(g): int(n) for g, n in row.items()}
                for c, row in table.iterrows()
            },
        }

    def runs(self) -> list[KpiRun]:
        return self.db.read_kpi_runs()

    # --- portfolio -------------------------------------------------------

    def analyze_portfolio(
        self, holdings: dict[str, float], as_of: datetime.date | None = None
    ) -> PortfolioAnalysis:
        kpis = self.frame("equity", as_of)
        run_date = kpis.attrs.get("as_of") if not kpis.empty else None
        ctx = self.context(run_date)
        total = float(sum(abs(v) for v in holdings.values())) or 1.0
        weights = {s: abs(v) / total for s, v in holdings.items()}
        sector: dict[str, float] = {}
        country: dict[str, float] = {}
        commodity: dict[str, float] = {}
        unknown = []
        exposures = ctx.etf_exposures
        for symbol, weight in weights.items():
            if symbol in ctx.universe.index:
                row = ctx.universe.loc[symbol]
                sector[row["sector"] or "unknown"] = (
                    sector.get(row["sector"] or "unknown", 0.0) + weight
                )
                country[row["country"] or "unknown"] = (
                    country.get(row["country"] or "unknown", 0.0) + weight
                )
            elif symbol in ctx.etfs.index and not exposures.empty:
                for _, e in exposures[exposures["symbol"] == symbol].iterrows():
                    target = {
                        "sector": sector,
                        "country": country,
                        "region": country,
                        "commodity": commodity,
                    }.get(e["entity_type"])
                    if target is not None:
                        target[e["entity"]] = target.get(
                            e["entity"], 0.0
                        ) + weight * float(e["weight"])
            else:
                unknown.append(symbol)
        prices = pd.concat(
            [
                ctx.close.reindex(
                    columns=[s for s in weights if s in ctx.close.columns]
                ),
                ctx.etf_close.reindex(
                    columns=[s for s in weights if s in ctx.etf_close.columns]
                ),
            ],
            axis=1,
        )
        returns = prices.ffill().pct_change().iloc[-252:].dropna(how="all")
        corr = returns.corr() if returns.shape[1] > 1 else pd.DataFrame()
        avg = None
        if not corr.empty:
            upper = corr.where(  # noqa: PD013
                np.triu(np.ones(corr.shape), k=1).astype(bool)
            ).stack()
            avg = round(as_float(upper.mean()), 3) if len(upper) else None
        return PortfolioAnalysis(
            as_of=run_date,
            weights={k: round(v, 4) for k, v in weights.items()},
            sector_exposure={k: round(v, 4) for k, v in sector.items()},
            country_exposure={k: round(v, 4) for k, v in country.items()},
            commodity_exposure={k: round(v, 4) for k, v in commodity.items()},
            concentration_hhi=round(float(sum(w * w for w in weights.values())), 4),
            average_pairwise_correlation_1y=avg,
            correlations={
                a: {b: round(float(v), 3) for b, v in row.items() if not np.isnan(v)}
                for a, row in corr.iterrows()
            },
            unknown_symbols=unknown,
            freshness=self.freshness(as_of),
        )

    def query(self, sql: str) -> list[dict[str, Any]]:
        if not self.settings.allow_sql:
            raise PermissionError(
                "The SQL tool is disabled; set BULLISH_ALLOW_SQL=1 to enable read-only queries"
            )
        statement = sql.strip().rstrip(";")
        lowered = statement.casefold()
        if not lowered.startswith(("select", "with")) or any(
            word in lowered
            for word in (
                "insert",
                "update",
                "delete",
                "drop",
                "alter",
                "create",
                "pragma",
                "attach",
            )
        ):
            raise PermissionError("Only read-only SELECT statements are allowed")
        return records(
            self.db.read_query(
                f"SELECT * FROM ({statement}) LIMIT {self.settings.max_rows}"  # noqa: S608
            )
        )


def _definition(entity_type: str, name: str) -> dict[str, Any]:
    definition: KpiDefinition = REGISTRY.get(entity_type, name)
    return {
        "tier": definition.tier,
        "unit": definition.unit,
        "description": definition.description,
        "higher_is_better": definition.higher_is_better,
    }


def _date(value: Any) -> datetime.date | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return pd.Timestamp(value).date()
