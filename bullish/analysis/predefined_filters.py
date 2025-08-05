import datetime
from datetime import timedelta
from typing import Dict, Any, Optional, List, Union, get_args

from bullish.analysis.analysis import AnalysisView
from bullish.analysis.backtest import (
    BacktestQueryDate,
    BacktestQueries,
    BacktestQueryRange,
    BacktestQuerySelection,
)
from bullish.analysis.constants import Europe, Us
from bullish.analysis.filter import FilterQuery, BOOLEAN_GROUP_MAPPING
from pydantic import BaseModel, Field

from bullish.analysis.indicators import Indicators
from bullish.database.crud import BullishDb

DATE_THRESHOLD = [
    datetime.date.today() - datetime.timedelta(days=7),
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

    def variants(self) -> List["NamedFilterQuery"]:
        return [
            self.country_variant("Europe", list(get_args(Europe))),
            self.country_variant("Us", list(get_args(Us))),
        ]


SMALL_CAP = NamedFilterQuery(
    name="Small Cap",
    last_price=[1, 20],
    market_capitalization=[5e7, 5e8],
    properties=["positive_debt_to_equity"],
    average_volume_30=[50000, 5e9],
    order_by_desc="market_capitalization",
).variants()

TOP_PERFORMERS = NamedFilterQuery(
    name="Top Performers",
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    price_above_sma_50=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    volume_above_average=DATE_THRESHOLD,
    weekly_growth=[1, 100],
    monthly_growth=[8, 100],
    order_by_desc="market_capitalization",
).variants()

LARGE_CAPS = NamedFilterQuery(
    name="Large caps",
    order_by_desc="market_capitalization",
    limit="50",
).variants()

NEXT_EARNINGS_DATE = NamedFilterQuery(
    name="Next Earnings date",
    order_by_desc="market_capitalization",
    next_earnings_date=[
        datetime.date.today(),
        datetime.date.today() + timedelta(days=10),
    ],
).variants()

RSI_CROSSOVER_40 = NamedFilterQuery(
    name="RSI cross-over 40",
    rsi_bullish_crossover_40=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e13],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
).variants()

RSI_CROSSOVER_30 = NamedFilterQuery(
    name="RSI cross-over 30",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e13],
    order_by_desc="market_capitalization",
).variants()


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        *SMALL_CAP,
        *TOP_PERFORMERS,
        *LARGE_CAPS,
        *NEXT_EARNINGS_DATE,
        *RSI_CROSSOVER_40,
        *RSI_CROSSOVER_30,
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
