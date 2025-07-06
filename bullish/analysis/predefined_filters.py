import datetime
from typing import Dict, Any, Optional

from bullish.analysis.filter import FilterQuery
from pydantic import BaseModel, Field


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
)

MICRO_CAP_EVENT_SPECULATION = NamedFilterQuery(
    name="Micro-Cap Event Speculation",
    description="seeks tiny names where unusual volume and price gaps hint at "
    "pending corporate events (patent win, FDA news, buy-out rumors).",
    positive_adosc_20_day_breakout=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    cdltasukigap=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    rate_of_change_30=[20, 100],  # 10% to 50% in the last 30 days
    market_capitalization=[0, 5e8],
)

MOMENTUM_BREAKOUT_HUNTER = NamedFilterQuery(
    name="Momentum Breakout Hunter",
    description="A confluence of medium-term (50/200 MA) and "
    "shorter oscillators suggests fresh upside momentum with fuel left.",
    income=[
        "positive_operating_income",
        "positive_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    golden_cross=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    adx_14_long=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    rate_of_change_30=[0, 100],
    rsi_neutral=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
)

DEEP_VALUE_PLUS_CATALYST = NamedFilterQuery(
    name="Deep-Value Plus Catalyst",
    description="Seeks beaten-down names that just printed a bullish "
    "candle and early accumulation signals—often the first leg of a bottom.",
    income=[
        "positive_operating_income",
        "positive_net_income",
    ],
    lower_than_200_day_high=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    rate_of_change_30=[3, 100],
    rsi_bullish_crossover=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
)
END_OF_TREND_REVERSAL = NamedFilterQuery(
    name="End of trend reversal",
    description="Layers long-term MA breach with momentum exhaustion and a "
    "bullish candle—classic setup for mean-reversion traders.",
    death_cross=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    rsi_oversold=[
        datetime.date.today() - datetime.timedelta(days=5),
        datetime.date.today(),
    ],
    candlesticks=["cdlmorningstart", "cdlabandonedbaby", "cdl3whitesoldiers"],
)


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        STRONG_FUNDAMENTALS,
        GOOD_FUNDAMENTALS,
        MICRO_CAP_EVENT_SPECULATION,
        MOMENTUM_BREAKOUT_HUNTER,
        DEEP_VALUE_PLUS_CATALYST,
        END_OF_TREND_REVERSAL,
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
