import datetime
import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, get_args

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

    def top_performers(self) -> "NamedFilterQuery":
        properties = {
            "volume_above_average": DATE_THRESHOLD,
            "sma_50_above_sma_200": [
                datetime.date.today() - datetime.timedelta(days=5000),
                datetime.date.today(),
            ],
            "weekly_growth": [1, 100],
            "monthly_growth": [8, 100],
        }
        return self._custom_variant("Top Performers", properties)

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

    def short_term_profitability(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "positive_operating_income",
                "positive_net_income",
                "quarterly_positive_operating_income",
                "quarterly_positive_net_income",
            ],
            "cash_flow": [
                "positive_free_cash_flow",
                "quarterly_positive_free_cash_flow",
            ],
            "eps": [
                "positive_basic_eps",
                "positive_diluted_eps",
                "quarterly_positive_basic_eps",
                "quarterly_positive_diluted_eps",
            ],
            "properties": [
                "positive_return_on_assets",
                "positive_return_on_equity",
                "positive_debt_to_equity",
                "operating_cash_flow_is_higher_than_net_income",
                "quarterly_positive_return_on_assets",
                "quarterly_positive_return_on_equity",
                "quarterly_positive_debt_to_equity",
                "quarterly_operating_cash_flow_is_higher_than_net_income",
            ],
        }
        return self._custom_variant("Short-term profitability", properties)

    def long_term_profitability(self) -> "NamedFilterQuery":
        properties = {
            "income": [
                "growing_net_income",
                "growing_operating_income",
                "quarterly_growing_net_income",
                "quarterly_growing_operating_income",
            ],
            "cash_flow": [
                "growing_operating_cash_flow",
                "quarterly_growing_operating_cash_flow",
            ],
            "eps": [
                "growing_basic_eps",
                "growing_diluted_eps",
                "quarterly_growing_basic_eps",
                "quarterly_growing_diluted_eps",
            ],
        }
        return self._custom_variant("Long-term profitability", properties)

    def high_growth(self) -> "NamedFilterQuery":
        properties = {"industry": list(get_args(HighGrowthIndustry))}
        return self._custom_variant("Growth", properties)

    def defensive(self) -> "NamedFilterQuery":
        properties = {"industry": list(get_args(DefensiveIndustries))}
        return self._custom_variant("Defensive", properties)

    def variants(self) -> List["NamedFilterQuery"]:
        variants_ = [
            self.country_variant("Europe", list(get_args(Europe))),
            self.country_variant("Us", list(get_args(Us))),
            self.country_variant("Europe", list(get_args(Europe))).top_performers(),
            self.country_variant("Us", list(get_args(Us))).top_performers(),
            self.country_variant("Europe", list(get_args(Europe))).poor_performers(),
            self.country_variant("Us", list(get_args(Us))).poor_performers(),
            self.country_variant("Europe", list(get_args(Europe)))
            .update_indicator_filter("RSI 30", "rsi_bullish_crossover_30")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
            self.country_variant("Europe", list(get_args(Europe)))
            .update_indicator_filter("RSI 40", "rsi_bullish_crossover_40")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
            self.country_variant("Europe", list(get_args(Europe)))
            .update_indicator_filter("RSI Neutral", "rsi_neutral")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
            self.country_variant("Us", list(get_args(Us)))
            .update_indicator_filter("RSI 30", "rsi_bullish_crossover_30")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
            self.country_variant("Us", list(get_args(Us)))
            .update_indicator_filter("RSI 40", "rsi_bullish_crossover_40")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
            self.country_variant("Us", list(get_args(Us)))
            .update_indicator_filter("RSI Neutral", "rsi_neutral")
            .update_indicator_filter("MACD", "macd_12_26_9_bullish_crossover"),
        ]
        variants_short_term_profitability = [
            v.short_term_profitability() for v in variants_
        ]
        variants_long_term_profitability = [
            v.long_term_profitability() for v in variants_
        ]
        variants_growth = [v.high_growth() for v in variants_]
        variants_defensive = [v.defensive() for v in variants_]
        return [
            *variants_,
            *variants_short_term_profitability,
            *variants_long_term_profitability,
            *variants_growth,
            *variants_defensive,
        ]


def load_custom_filters() -> List[NamedFilterQuery]:
    if "CUSTOM_FILTERS_PATH" in os.environ:
        custom_filters_path = os.environ["CUSTOM_FILTERS_PATH"]
        return read_custom_filters(Path(custom_filters_path))
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
).variants()

LARGE_CAPS = NamedFilterQuery(
    name="Large Cap",
    order_by_desc="market_capitalization",
    market_capitalization=[1e10, 1e14],
).variants()

MID_CAPS = NamedFilterQuery(
    name="Mid Cap",
    order_by_desc="market_capitalization",
    market_capitalization=[5e8, 1e10],
).variants()

NEXT_EARNINGS_DATE = NamedFilterQuery(
    name="Next Earnings date",
    order_by_desc="market_capitalization",
    next_earnings_date=[
        datetime.date.today(),
        datetime.date.today() + timedelta(days=20),
    ],
).variants()


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        *load_custom_filters(),
        *SMALL_CAP,
        *MID_CAPS,
        *LARGE_CAPS,
        *NEXT_EARNINGS_DATE,
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
