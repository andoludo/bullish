import datetime
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, get_args, Tuple

from bullish.analysis.analysis import AnalysisView
from bullish.analysis.backtest import (
    BacktestQueryDate,
    BacktestQueries,
    BacktestQueryRange,
    BacktestQuerySelection,
)
from bullish.analysis.constants import (
    Europe,
    Us,
    HighGrowthIndustry,
    DefensiveIndustries,
)
from bullish.analysis.filter import FilterQuery, BOOLEAN_GROUP_MAPPING
from pydantic import BaseModel, Field

from bullish.analysis.indicators import Indicators
from bullish.database.crud import BullishDb

DATE_THRESHOLD = [
    datetime.date.today() - datetime.timedelta(days=2),
    datetime.date.today(),
]


def _get_variants(variants: List[str]) -> List[Tuple[str, ...]]:
    return [tuple(variants[:i]) for i in range(1, len(variants) + 1)]


class NamedFilterQuery(FilterQuery):
    name: str
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(
            exclude_unset=True,
            exclude_none=True,
            exclude_defaults=True,
            exclude={"name"},
        )

    def to_backtesting_query(
        self, backtest_start_date: datetime.date
    ) -> BacktestQueries:
        queries: List[
            Union[BacktestQueryRange, BacktestQueryDate, BacktestQuerySelection]
        ] = []
        in_use_backtests = Indicators().in_use_backtest()
        for in_use in in_use_backtests:
            value = self.to_dict().get(in_use)
            if value and self.model_fields[in_use].annotation == List[datetime.date]:
                delta = value[1] - value[0]
                queries.append(
                    BacktestQueryDate(
                        name=in_use.upper(),
                        start=backtest_start_date - delta,
                        end=backtest_start_date,
                        table="signalseries",
                    )
                )
        for field in self.to_dict():
            if field in BOOLEAN_GROUP_MAPPING:
                value = self.to_dict().get(field)
                if value and self.model_fields[field].annotation == Optional[List[str]]:  # type: ignore
                    queries.extend(
                        [
                            BacktestQueryDate(
                                name=v.upper(),
                                start=backtest_start_date - timedelta(days=252),
                                end=backtest_start_date,
                                table="signalseries",
                            )
                            for v in value
                        ]
                    )

            if field in AnalysisView.model_fields:
                value = self.to_dict().get(field)
                if (
                    value
                    and self.model_fields[field].annotation == Optional[List[float]]  # type: ignore
                    and len(value) == 2
                ):
                    queries.append(
                        BacktestQueryRange(
                            name=field.lower(),
                            min=value[0],
                            max=value[1],
                            table="analysis",
                        )
                    )
                if value and self.model_fields[field].annotation == Optional[List[str]]:  # type: ignore
                    queries.append(
                        BacktestQuerySelection(
                            name=field.lower(),
                            selections=value,
                            table="analysis",
                        )
                    )

        return BacktestQueries(queries=queries)

    def get_backtesting_symbols(
        self, bullish_db: BullishDb, backtest_start_date: datetime.date
    ) -> List[str]:
        queries = self.to_backtesting_query(backtest_start_date)

        return bullish_db.read_query(queries.to_query())["symbol"].tolist()  # type: ignore

    def country_variant(self, suffix: str, countries: List[str]) -> "NamedFilterQuery":
        return NamedFilterQuery.model_validate(
            self.model_dump()
            | {"name": f"{self.name} ({suffix})", "country": countries}
        )

    def update_indicator_filter(
        self, suffix: str, rsi_parameter_name: str
    ) -> "NamedFilterQuery":
        return NamedFilterQuery.model_validate(
            self.model_dump()
            | {"name": f"{self.name} ({suffix})", rsi_parameter_name: DATE_THRESHOLD}
        )

    def _custom_variant(
        self, suffix: str, properties: Dict[str, Any]
    ) -> "NamedFilterQuery":
        return NamedFilterQuery.model_validate(
            self.model_dump() | {"name": f"{self.name} ({suffix})", **properties}
        )

    def week_top_performers(self) -> "NamedFilterQuery":
        properties = {
            "volume_above_average": DATE_THRESHOLD,
            "weekly_growth": [1, 100],
        }
        return self._custom_variant("Week Top Performers", properties)

    def month_top_performers(self) -> "NamedFilterQuery":
        properties = {
            "monthly_growth": [8, 100],
        }
        return self._custom_variant("Month Top Performers", properties)

    def year_top_performers(self) -> "NamedFilterQuery":
        properties = {
            "volume_above_average": DATE_THRESHOLD,
            "sma_50_above_sma_200": [
                datetime.date.today() - datetime.timedelta(days=5000),
                datetime.date.today(),
            ],
            "yearly_growth": [30, 100],
        }
        return self._custom_variant("Yearly Top Performers", properties)

    def poor_performers(self) -> "NamedFilterQuery":
        properties = {
            "sma_50_below_sma_200": [
                datetime.date.today() - datetime.timedelta(days=5000),
                datetime.date.today(),
            ],
            "price_below_sma_50": [
                datetime.date.today() - datetime.timedelta(days=5000),
                datetime.date.today(),
            ],
            "monthly_growth": [-100, 0],
        }
        return self._custom_variant("Poor Performers", properties)

    def yearly_fundamentals(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "positive_operating_income",
                "positive_net_income",
                "growing_net_income",
                "growing_operating_income",
            ],
            "cash_flow": ["positive_free_cash_flow", "growing_operating_cash_flow"],
            "properties": [
                "positive_return_on_equity",
                "operating_cash_flow_is_higher_than_net_income",
            ],
        }
        return self._custom_variant("Yearly Fundamentals", properties)

    def quarterly_fundamentals(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "quarterly_positive_operating_income",
                "quarterly_positive_net_income",
            ],
            "cash_flow": [
                "quarterly_positive_free_cash_flow",
            ],
            "properties": [
                "quarterly_operating_cash_flow_is_higher_than_net_income",
            ],
        }
        return self._custom_variant("Quarterly Fundamentals", properties)

    def growing_quarterly_fundamentals(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "quarterly_positive_operating_income",
                "quarterly_positive_net_income",
                "quarterly_growing_net_income",
            ],
            "cash_flow": [
                "quarterly_positive_free_cash_flow",
                "quarterly_growing_operating_cash_flow",
            ],
            "properties": [
                "quarterly_operating_cash_flow_is_higher_than_net_income",
            ],
        }
        return self._custom_variant("Growing Quarterly Fundamentals", properties)

    def min_fundamentals(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "positive_operating_income",
                "positive_net_income",
            ],
            "cash_flow": [
                "positive_free_cash_flow",
            ],
            "eps": [
                "positive_diluted_eps",  # or positive_basic_eps if diluted not available
            ],
            "properties": [
                "positive_return_on_equity",
                "operating_cash_flow_is_higher_than_net_income",
            ],
        }
        return self._custom_variant("Min Fundamentals", properties)

    def high_growth(self) -> "NamedFilterQuery":
        properties = {"industry": list(get_args(HighGrowthIndustry))}
        return self._custom_variant("Growth", properties)

    def defensive(self) -> "NamedFilterQuery":
        properties = {"industry": list(get_args(DefensiveIndustries))}
        return self._custom_variant("Defensive", properties)

    def cheap(self) -> "NamedFilterQuery":
        properties = {"last_price": [1, 30]}
        return self._custom_variant("Cheap", properties)

    def europe(self) -> "NamedFilterQuery":
        return self.country_variant("Europe", list(get_args(Europe)))

    def us(self) -> "NamedFilterQuery":
        return self.country_variant("Us", list(get_args(Us)))

    def rsi_30(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("RSI 30", "rsi_bullish_crossover_30")

    def rsi_40(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("RSI 40", "rsi_bullish_crossover_40")

    def macd(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover")

    def rsi_neutral_(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("RSI Neutral", "rsi_neutral")

    def rsi_oversold_(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("RSI Oversold", "rsi_oversold")

    def rsi_overbought_(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("RSI Overbought", "rsi_overbought")

    def adx(self) -> "NamedFilterQuery":
        return self.update_indicator_filter("ADX 14", "adx_14")

    def earnings_date(self) -> "NamedFilterQuery":
        return NamedFilterQuery.model_validate(
            self.model_dump()
            | {
                "name": f"{self.name} (Earnings Date)",
                "next_earnings_date": [
                    datetime.date.today(),
                    datetime.date.today() + timedelta(days=20),
                ],
            }
        )

    def variants(
        self,
        variants: Optional[List[List[str]]] = None,
        filters: Optional[List[str]] = None,
    ) -> List["NamedFilterQuery"]:
        if filters and self.name not in filters:
            return [self]
        variants = variants or [["europe"], ["us"]]

        _variants = {v for variant in variants for v in _get_variants(variant)}
        filters_ = []
        for attributes in _variants:
            filter__ = self
            for attr in attributes:
                filter__ = getattr(filter__, attr)()
            filters_.append(filter__)

        return [self, *filters_]


def load_custom_filters() -> List[NamedFilterQuery]:
    if "CUSTOM_FILTERS_PATH" in os.environ:
        custom_filters_path = os.environ["CUSTOM_FILTERS_PATH"]
        return [
            variant
            for f in read_custom_filters(Path(custom_filters_path))
            for variant in f.variants(
                variants=[["rsi_overbought_"]], filters=["portfolio", "Portfolio"]
            )
        ]
    return []


def read_custom_filters(custom_filters_path: Path) -> List[NamedFilterQuery]:
    if custom_filters_path.exists():
        filters = json.loads(custom_filters_path.read_text())
        return [NamedFilterQuery.model_validate(filter) for filter in filters]
    return []


SMALL_CAP = NamedFilterQuery(
    name="Small Cap",
    last_price=[1, 20],
    market_capitalization=[5e7, 5e8],
    properties=["positive_debt_to_equity"],
    average_volume_30=[50000, 5e9],
    order_by_desc="market_capitalization",
).variants(
    variants=[
        ["week_top_performers", "min_fundamentals"],
        ["month_top_performers", "min_fundamentals"],
        ["earnings_date", "min_fundamentals"],
        ["rsi_oversold_", "min_fundamentals"],
    ]
)

LARGE_CAPS = NamedFilterQuery(
    name="Large Cap",
    order_by_desc="market_capitalization",
    market_capitalization=[1e10, 1e14],
).variants(
    variants=[
        ["rsi_oversold_", "macd", "yearly_fundamentals"],
        ["rsi_neutral_", "macd", "adx", "yearly_fundamentals"],
        ["rsi_30", "macd", "adx", "yearly_fundamentals"],
        ["rsi_oversold_", "macd", "quarterly_fundamentals"],
        ["rsi_neutral_", "macd", "adx", "quarterly_fundamentals"],
        ["rsi_30", "macd", "adx", "quarterly_fundamentals"],
        ["earnings_date", "quarterly_fundamentals", "yearly_fundamentals"],
    ]
)

MID_CAPS = NamedFilterQuery(
    name="Mid Cap",
    order_by_desc="market_capitalization",
    market_capitalization=[5e8, 1e10],
).variants(
    variants=[
        ["week_top_performers"],
        ["month_top_performers"],
        ["earnings_date", "quarterly_fundamentals", "yearly_fundamentals"],
        ["rsi_oversold_", "macd", "adx"],
    ]
)


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        *load_custom_filters(),
        *SMALL_CAP,
        *MID_CAPS,
        *LARGE_CAPS,
    ]


class PredefinedFilters(BaseModel):
    filters: list[NamedFilterQuery] = Field(default_factory=predefined_filters)

    def get_predefined_filter_names(self) -> list[str]:
        return [filter.name for filter in self.filters]

    def get_predefined_filter(self, name: str) -> Dict[str, Any]:
        for filter in self.filters:
            if filter.name == name:
                return filter.to_dict()
        raise ValueError(f"Filter with name '{name}' not found.")
