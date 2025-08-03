import datetime
from datetime import timedelta
from typing import Dict, Any, Optional, List, Union

from bullish.analysis.analysis import AnalysisView
from bullish.analysis.backtest import (
    BacktestQueryDate,
    BacktestQueries,
    BacktestQueryRange,
    BacktestQuerySelection,
)
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


STRONG_FUNDAMENTALS = NamedFilterQuery(
    name="Strong Fundamentals",
    income=[
        "positive_operating_income",
        "growing_operating_income",
        "positive_net_income",
        "growing_net_income",
    ],
    cash_flow=["positive_free_cash_flow", "growing_operating_cash_flow"],
    eps=["positive_diluted_eps", "growing_diluted_eps"],
    properties=[
        "operating_cash_flow_is_higher_than_net_income",
        "positive_return_on_equity",
        "positive_return_on_assets",
        "positive_debt_to_equity",
    ],
    market_capitalization=[1e10, 1e12],  # 1 billion to 1 trillion
    rsi_bullish_crossover_30=DATE_THRESHOLD,
)

GOOD_FUNDAMENTALS = NamedFilterQuery(
    name="Good Fundamentals",
    income=[
        "positive_operating_income",
        "positive_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    eps=["positive_diluted_eps"],
    properties=[
        "positive_return_on_equity",
        "positive_return_on_assets",
        "positive_debt_to_equity",
    ],
    market_capitalization=[1e10, 1e12],  # 1 billion to 1 trillion
    rsi_bullish_crossover_30=DATE_THRESHOLD,
)

RSI_CROSSOVER_30_GROWTH_STOCK_STRONG_FUNDAMENTAL = NamedFilterQuery(
    name="RSI cross-over 30 growth stock strong fundamental",
    income=[
        "positive_operating_income",
        "growing_operating_income",
        "positive_net_income",
        "growing_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[10, 100],
    rsi_bullish_crossover_30=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e12],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)
RSI_CROSSOVER_40_GROWTH_STOCK_STRONG_FUNDAMENTAL = NamedFilterQuery(
    name="RSI cross-over 40 growth stock strong fundamental",
    income=[
        "positive_operating_income",
        "growing_operating_income",
        "positive_net_income",
        "growing_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_40=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e12],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)

RSI_CROSSOVER_30_GROWTH_STOCK = NamedFilterQuery(
    name="RSI cross-over 30 growth stock",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=DATE_THRESHOLD,
    market_capitalization=[1e10, 1e13],
    order_by_desc="market_capitalization",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)

MEDIAN_YEARLY_GROWTH = NamedFilterQuery(
    name="Median yearly growth",
    market_capitalization=[1e6, 1e13],
    median_yearly_growth=[40, 1000],
    last_price=[1, 100],
    order_by_asc="last_price",
    country=[
        "Germany",
        "United states",
        "France",
        "Belgium",
    ],
)
RSI_CROSSOVER_40_GROWTH_STOCK = NamedFilterQuery(
    name="RSI cross-over 40 growth stock",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_40=DATE_THRESHOLD,
    market_capitalization=[1e10, 1e13],
    order_by_desc="market_capitalization",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)


MOMENTUM_GROWTH_GOOD_FUNDAMENTALS = NamedFilterQuery(
    name="Momentum Growth Good Fundamentals (RSI 30)",
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)

MOMENTUM_GROWTH_STRONG_FUNDAMENTALS = NamedFilterQuery(
    name="Momentum Growth Strong Fundamentals (RSI 30)",
    income=[
        "positive_operating_income",
        "growing_operating_income",
        "positive_net_income",
        "growing_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)
MOMENTUM_GROWTH_RSI_30 = NamedFilterQuery(
    name="Momentum Growth Screener (RSI 30)",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)
MOMENTUM_GROWTH_RSI_40 = NamedFilterQuery(
    name="Momentum Growth Screener (RSI 40)",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_40=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)

GOLDEN_CROSS_LAST_SEVEN_DAYS = NamedFilterQuery(
    name="Golden cross in the last five days",
    price_per_earning_ratio=[10, 500],
    last_price=[1, 10000],
    golden_cross=[
        datetime.date.today() - datetime.timedelta(days=7),
        datetime.date.today(),
    ],
    order_by_desc="market_capitalization",
    country=[
        "Germany",
        "United states",
        "France",
        "United kingdom",
        "Canada",
        "Japan",
        "Belgium",
    ],
)


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        STRONG_FUNDAMENTALS,
        GOOD_FUNDAMENTALS,
        RSI_CROSSOVER_30_GROWTH_STOCK_STRONG_FUNDAMENTAL,
        RSI_CROSSOVER_40_GROWTH_STOCK_STRONG_FUNDAMENTAL,
        RSI_CROSSOVER_30_GROWTH_STOCK,
        RSI_CROSSOVER_40_GROWTH_STOCK,
        MOMENTUM_GROWTH_GOOD_FUNDAMENTALS,
        MOMENTUM_GROWTH_STRONG_FUNDAMENTALS,
        MOMENTUM_GROWTH_RSI_30,
        MOMENTUM_GROWTH_RSI_40,
        GOLDEN_CROSS_LAST_SEVEN_DAYS,
        MEDIAN_YEARLY_GROWTH,
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
