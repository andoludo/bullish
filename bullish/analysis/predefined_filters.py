import datetime
from typing import Dict, Any, Optional

from bullish.analysis.filter import FilterQuery
from pydantic import BaseModel, Field

DATE_THRESHOLD = [
    datetime.date.today() - datetime.timedelta(days=10),
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


SHOOTING_STARS = NamedFilterQuery(
    name="Shooting stars",
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    market_capitalization=[1e9, 1e12],  # 1 billion to 1 trillion
    order_by_desc="median_yearly_growth",
    order_by_asc="last_price",
)

RSI_CROSSOVER = NamedFilterQuery(
    name="RSI cross-over",
    cash_flow=["positive_free_cash_flow"],
    properties=["operating_cash_flow_is_higher_than_net_income"],
    return_after_rsi_crossover_45_period_90=[0.0, 100],
    rsi_bullish_crossover_45=DATE_THRESHOLD,
    market_capitalization=[1e9, 1e12],  # 1 billion to 1 trillion
    order_by_desc="market_capitalization",
)
MICRO_CAP_EVENT_SPECULATION = NamedFilterQuery(
    name="Micro-Cap Event Speculation",
    description="seeks tiny names where unusual volume and price gaps hint at "
    "pending corporate events (patent win, FDA news, buy-out rumors).",
    positive_adosc_20_day_breakout=DATE_THRESHOLD,
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
    golden_cross=DATE_THRESHOLD,
    adx_14_long=DATE_THRESHOLD,
    rate_of_change_30=[0, 100],
    rsi_neutral=DATE_THRESHOLD,
)

DEEP_VALUE_PLUS_CATALYST = NamedFilterQuery(
    name="Deep-Value Plus Catalyst",
    description="Seeks beaten-down names that just printed a bullish "
    "candle and early accumulation signals—often the first leg of a bottom.",
    income=[
        "positive_operating_income",
        "positive_net_income",
    ],
    lower_than_200_day_high=DATE_THRESHOLD,
    rate_of_change_30=[3, 100],
    rsi_bullish_crossover_30=DATE_THRESHOLD,
)
END_OF_TREND_REVERSAL = NamedFilterQuery(
    name="End of trend reversal",
    description="Layers long-term MA breach with momentum exhaustion and a "
    "bullish candle—classic setup for mean-reversion traders.",
    death_cross=DATE_THRESHOLD,
    rsi_oversold=DATE_THRESHOLD,
    candlesticks=["cdlmorningstart", "cdlabandonedbaby", "cdl3whitesoldiers"],
)

HIGH_QUALITY_CASH_GENERATOR = NamedFilterQuery(
    name="High Quality Cash Generator",
    description="This quartet isolates companies that are profitable, cash-rich, and disciplined with leverage. "
    "Ideal first pass for “quality” or “compounder” "
    "portfolios where downside protection matters as much as upside.",
    income=[
        "positive_net_income",
    ],
    cash_flow=["positive_free_cash_flow"],
    properties=[
        "operating_cash_flow_is_higher_than_net_income",
        "positive_return_on_equity",
        "positive_return_on_assets",
        "positive_debt_to_equity",
    ],
)

EARNINGS_ACCELERATION_TREND_CONFIRMATION = NamedFilterQuery(
    name="Earnings Acceleration Trend Confirmation",
    description="Pairs fundamental acceleration with momentum confirmation. Research shows this “double positive” "
    "outperforms simple momentum because it filters out purely sentiment-driven rallies.",
    income=[
        "growing_operating_income",
        "positive_net_income",
    ],
    eps=["growing_basic_eps"],
    golden_cross=DATE_THRESHOLD,
    macd_12_26_9_bullish_crossover=DATE_THRESHOLD,
    adx_14_long=DATE_THRESHOLD,
)
DIVIDEND_GROWTH_COMPOUNDER = NamedFilterQuery(
    name="Dividend-Growth Compounders",
    description="Separates true dividend growers from high-yield traps. "
    "Critical for income portfolios that need both yield and growth to beat inflation.",
    mean_dividend_payout_ratio=[0, 0.6],  # 0% to 60% payout ratio
    cash_flow=[
        "positive_free_cash_flow",
        "quarterly_positive_free_cash_flow",
        "growing_operating_cash_flow",
    ],
    properties=["quarterly_positive_return_on_equity"],
)

BREAK_OUT_MOMENTUM = NamedFilterQuery(
    name="Break-out Momentum",
    description="Combines price, volume, and pattern confirmation. Great for tactical traders seeking "
    "quick continuation moves with statistically higher follow-through.",
    adosc_crosses_above_0=DATE_THRESHOLD,
    positive_adosc_20_day_breakout=DATE_THRESHOLD,
    rsi_bullish_crossover_30=DATE_THRESHOLD,
)

OVERSOLD_MEAN_REVERSION = NamedFilterQuery(
    name="Oversold Mean Reversion",
    description="Gives contrarian traders a high-probability bounce setup by "
    "stacking three different oversold measures plus a reversal pattern.",
    rsi_oversold=DATE_THRESHOLD,
    stoch_oversold=DATE_THRESHOLD,
    mfi_oversold=DATE_THRESHOLD,
    lower_than_200_day_high=DATE_THRESHOLD,
)


def predefined_filters() -> list[NamedFilterQuery]:
    return [
        STRONG_FUNDAMENTALS,
        GOOD_FUNDAMENTALS,
        MICRO_CAP_EVENT_SPECULATION,
        MOMENTUM_BREAKOUT_HUNTER,
        DEEP_VALUE_PLUS_CATALYST,
        END_OF_TREND_REVERSAL,
        HIGH_QUALITY_CASH_GENERATOR,
        EARNINGS_ACCELERATION_TREND_CONFIRMATION,
        DIVIDEND_GROWTH_COMPOUNDER,
        BREAK_OUT_MOMENTUM,
        OVERSOLD_MEAN_REVERSION,
        SHOOTING_STARS,
        RSI_CROSSOVER,
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
