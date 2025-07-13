import datetime
from typing import Dict, Any, Optional

from bullish.analysis.filter import FilterQuery
from pydantic import BaseModel, Field


DATE_THRESHOLD = [
    datetime.date.today() - datetime.timedelta(days=5),
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
    price_per_earning_ratio=[20, 40],
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
    price_per_earning_ratio=[20, 40],
    rsi_bullish_crossover_40=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e12],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)

RSI_CROSSOVER_30_GROWTH_STOCK = NamedFilterQuery(
    name="RSI cross-over 30 growth stock",
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[20, 40],
    rsi_bullish_crossover_30=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e12],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)
RSI_CROSSOVER_40_GROWTH_STOCK = NamedFilterQuery(
    name="RSI cross-over 40 growth stock",
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[20, 40],
    rsi_bullish_crossover_40=DATE_THRESHOLD,
    market_capitalization=[5e8, 1e12],
    order_by_desc="market_capitalization",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)


MOMENTUM_GROWTH = NamedFilterQuery(
    name="Momentum Growth",
    price_per_earning_ratio=[10, 500],
    last_price=[1, 10000],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=30),
    ],
    price_above_sma_50=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=30),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    order_by_desc="momentum",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)
MOMENTUM_GROWTH_STRONG_FUNDAMENTALS = NamedFilterQuery(
    name="Momentum Growth Strong Fundamentals",
    income=[
        "positive_operating_income",
        "growing_operating_income",
        "positive_net_income",
        "growing_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    price_per_earning_ratio=[10, 500],
    last_price=[1, 10000],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=30),
    ],
    price_above_sma_50=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=30),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    order_by_desc="momentum",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)
MOMENTUM_GROWTH_RSI_30 = NamedFilterQuery(
    name="Momentum Growth Screener (RSI 30)",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_30=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)
MOMENTUM_GROWTH_RSI_40 = NamedFilterQuery(
    name="Momentum Growth Screener (RSI 40)",
    price_per_earning_ratio=[10, 500],
    rsi_bullish_crossover_40=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    macd_12_26_9_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    sma_50_above_sma_200=[
        datetime.date.today() - datetime.timedelta(days=5000),
        datetime.date.today() - datetime.timedelta(days=10),
    ],
    market_capitalization=[5e8, 1e12],
    order_by_desc="momentum",
    country=["Germany", "United states", "France", "United kingdom", "Canada", "Japan"],
)


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        STRONG_FUNDAMENTALS,
        GOOD_FUNDAMENTALS,
        RSI_CROSSOVER_30_GROWTH_STOCK_STRONG_FUNDAMENTAL,
        RSI_CROSSOVER_40_GROWTH_STOCK_STRONG_FUNDAMENTAL,
        RSI_CROSSOVER_30_GROWTH_STOCK,
        RSI_CROSSOVER_40_GROWTH_STOCK,
        MOMENTUM_GROWTH,
        MOMENTUM_GROWTH_STRONG_FUNDAMENTALS,
        MOMENTUM_GROWTH_RSI_30,
        MOMENTUM_GROWTH_RSI_40,
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
