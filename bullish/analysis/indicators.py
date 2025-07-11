import logging
from datetime import date
from typing import Optional, List, Callable, Any, Literal, Dict, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, create_model

from bullish.analysis.functions import (
    cross,
    cross_value,
    ADX,
    MACD,
    RSI,
    STOCH,
    MFI,
    ROC,
    CANDLESTOCK_PATTERNS,
    SMA,
    ADOSC,
    PRICE,
    compute_percentile_return_after_rsi_crossover,
)

logger = logging.getLogger(__name__)
SignalType = Literal["Short", "Long", "Oversold", "Overbought", "Value"]


class Signal(BaseModel):
    name: str
    type_info: SignalType
    type: Any
    range: Optional[List[float]] = None
    function: Callable[[pd.DataFrame], Optional[Union[date, float]]]
    description: str
    date: Optional[date] = None
    value: Optional[float] = None

    def is_date(self) -> bool:
        if self.type == Optional[date]:
            return True
        elif self.type == Optional[float]:
            return False
        else:
            raise NotImplementedError

    def compute(self, data: pd.DataFrame) -> None:
        if self.is_date():
            self.date = self.function(data)  # type: ignore
        else:
            self.value = self.function(data)  # type: ignore


class Indicator(BaseModel):
    name: str
    description: str
    expected_columns: List[str]
    function: Callable[[pd.DataFrame], pd.DataFrame]
    _data: pd.DataFrame = PrivateAttr(default=pd.DataFrame())
    signals: List[Signal] = Field(default_factory=list)

    def compute(self, data: pd.DataFrame) -> None:
        results = self.function(data)
        if not set(self.expected_columns).issubset(results.columns):
            raise ValueError(
                f"Expected columns {self.expected_columns}, but got {results.columns.tolist()}"
            )
        self._data = results
        self._signals()

    def _signals(self) -> None:
        for signal in self.signals:
            try:
                signal.compute(self._data)
            except Exception as e:  # noqa: PERF203
                logger.error(
                    f"Fail to compute signal {signal.name} for indicator {self.name}: {e}"
                )


def indicators_factory() -> List[Indicator]:
    return [
        Indicator(
            name="ADX_14",
            description="Average Directional Movement Index",
            expected_columns=["ADX_14", "MINUS_DI", "PLUS_DI"],
            function=ADX.call,
            signals=[
                Signal(
                    name="ADX_14_LONG",
                    description="ADX 14 Long Signal",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.ADX_14 > 20) & (d.PLUS_DI > d.MINUS_DI)
                    ].last_valid_index(),
                ),
                Signal(
                    name="ADX_14_SHORT",
                    description="ADX 14 Short Signal",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.ADX_14 > 20) & (d.MINUS_DI > d.PLUS_DI)
                    ].last_valid_index(),
                ),
            ],
        ),
        Indicator(
            name="MACD_12_26_9",
            description="Moving Average Convergence/Divergence",
            expected_columns=[
                "MACD_12_26_9",
                "MACD_12_26_9_SIGNAL",
                "MACD_12_26_9_HIST",
            ],
            function=MACD.call,
            signals=[
                Signal(
                    name="MACD_12_26_9_BULLISH_CROSSOVER",
                    description="MACD 12-26-9 Bullish Crossover",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross(d.MACD_12_26_9, d.MACD_12_26_9_SIGNAL),
                ),
                Signal(
                    name="MACD_12_26_9_BEARISH_CROSSOVER",
                    description="MACD 12-26-9 Bearish Crossover",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross(d.MACD_12_26_9_SIGNAL, d.MACD_12_26_9),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_UP",
                    description="MACD 12-26-9 Zero Line Cross Up",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.MACD_12_26_9, 0),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_DOWN",
                    description="MACD 12-26-9 Zero Line Cross Down",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.MACD_12_26_9, 0, above=False),
                ),
            ],
        ),
        Indicator(
            name="RSI",
            description="Relative Strength Index",
            expected_columns=RSI.expected_columns,
            function=RSI.call,
            signals=[
                Signal(
                    name="RSI_BULLISH_CROSSOVER_30",
                    description="RSI Bullish Crossover",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 30),
                ),
                Signal(
                    name="RSI_BULLISH_CROSSOVER_40",
                    description="RSI Bullish Crossover 40",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 40),
                ),
                Signal(
                    name="RSI_BULLISH_CROSSOVER_45",
                    description="RSI Bullish Crossover 45",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 45),
                ),
                Signal(
                    name="RSI_BEARISH_CROSSOVER",
                    description="RSI Bearish Crossover",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 70, above=False),
                ),
                Signal(
                    name="RSI_OVERSOLD",
                    description="RSI Oversold Signal",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[(d.RSI < 30) & (d.RSI > 0)].last_valid_index(),
                ),
                Signal(
                    name="RSI_OVERBOUGHT",
                    description="RSI Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.RSI < 100) & (d.RSI > 70)
                    ].last_valid_index(),
                ),
                Signal(
                    name="RSI_NEUTRAL",
                    description="RSI Neutral Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.RSI < 60) & (d.RSI > 40)
                    ].last_valid_index(),
                ),
                Signal(
                    name="RETURN_AFTER_RSI_CROSSOVER_45_PERIOD_90",
                    description="Percentile 30 return after RSI crossover 45 in the next 90 days",
                    type_info="Long",
                    type=Optional[float],
                    function=lambda d: compute_percentile_return_after_rsi_crossover(d),
                ),
            ],
        ),
        Indicator(
            name="STOCH",
            description="Stochastic",
            expected_columns=["SLOW_K", "SLOW_D"],
            function=STOCH.call,
            signals=[
                Signal(
                    name="STOCH_OVERSOLD",
                    description="Stoch Oversold Signal",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.SLOW_K < 20) & (d.SLOW_K > 0)
                    ].last_valid_index(),
                ),
                Signal(
                    name="STOCH_OVERBOUGHT",
                    description="Stoch Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.SLOW_K < 100) & (d.SLOW_K > 80)
                    ].last_valid_index(),
                ),
            ],
        ),
        Indicator(
            name="MFI",
            description="Money Flow Index",
            expected_columns=["MFI"],
            function=MFI.call,
            signals=[
                Signal(
                    name="MFI_OVERSOLD",
                    description="MFI Oversold Signal",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[(d.MFI < 20)].last_valid_index(),
                ),
                Signal(
                    name="MFI_OVERBOUGHT",
                    description="MFI Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[(d.MFI > 80)].last_valid_index(),
                ),
            ],
        ),
        Indicator(
            name="SMA",
            description="Money Flow Index",
            expected_columns=["SMA_50", "SMA_200"],
            function=SMA.call,
            signals=[
                Signal(
                    name="GOLDEN_CROSS",
                    description="Golden cross: SMA 50 crosses above SMA 200",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: cross(d.SMA_50, d.SMA_200),
                ),
                Signal(
                    name="DEATH_CROSS",
                    description="Death cross: SMA 50 crosses below SMA 200",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: cross(d.SMA_50, d.SMA_200, above=False),
                ),
            ],
        ),
        Indicator(
            name="PRICE",
            description="Price based indicators",
            expected_columns=PRICE.expected_columns,
            function=PRICE.call,
            signals=[
                Signal(
                    name="LOWER_THAN_200_DAY_HIGH",
                    description="Current price is lower than the 200-day high",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[
                        0.6 * d["200_DAY_HIGH"] > d.LAST_PRICE
                    ].last_valid_index(),
                ),
                Signal(
                    name="LOWER_THAN_20_DAY_HIGH",
                    description="Current price is lower than the 20-day high",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[
                        0.6 * d["20_DAY_HIGH"] > d.LAST_PRICE
                    ].last_valid_index(),
                ),
                Signal(
                    name="MEDIAN_WEEKLY_GROWTH",
                    description="Median weekly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: np.median(d.WEEKLY_GROWTH.unique()),
                ),
                Signal(
                    name="MEDIAN_MONTHLY_GROWTH",
                    description="Median monthly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: np.median(d.MONTHLY_GROWTH.unique()),
                ),
                Signal(
                    name="MEDIAN_YEARLY_GROWTH",
                    description="Median yearly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: np.median(d.YEARLY_GROWTH.unique()),
                ),
            ],
        ),
        Indicator(
            name="ROC",
            description="Rate Of Change",
            expected_columns=["ROC_7", "ROC_30", "ROC_1"],
            function=ROC.call,
            signals=[
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_1",
                    type_info="Value",
                    description="Median daily Rate of Change of the last 30 days",
                    type=Optional[float],
                    function=lambda d: np.median(d.ROC_1.tolist()[-30:]),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_7_4",
                    type_info="Value",
                    description="Median weekly Rate of Change of the last 4 weeks",
                    type=Optional[float],
                    function=lambda d: np.median(d.ROC_7.tolist()[-4:]),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_7_12",
                    type_info="Value",
                    description="Median weekly Rate of Change of the last 12 weeks",
                    type=Optional[float],
                    function=lambda d: np.median(d.ROC_7.tolist()[-12:]),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_30",
                    type_info="Value",
                    description="Median monthly Rate of Change of the last 12 Months",
                    type=Optional[float],
                    function=lambda d: np.median(d.ROC_30.tolist()[-12:]),
                ),
                Signal(
                    name="RATE_OF_CHANGE_30",
                    type_info="Value",
                    description="30-day Rate of Change",
                    type=Optional[float],
                    function=lambda d: d.ROC_30.tolist()[-1],
                ),
                Signal(
                    name="RATE_OF_CHANGE_7",
                    type_info="Value",
                    description="7-day Rate of Change",
                    type=Optional[float],
                    function=lambda d: d.ROC_7.tolist()[-1],
                ),
            ],
        ),
        Indicator(
            name="ADOSC",
            description="Chaikin A/D Oscillator",
            expected_columns=["ADOSC", "ADOSC_SIGNAL"],
            function=ADOSC.call,
            signals=[
                Signal(
                    name="ADOSC_CROSSES_ABOVE_0",
                    type_info="Oversold",
                    description="Bullish momentum in money flow",
                    type=Optional[date],
                    function=lambda d: cross_value(d.ADOSC, 0, above=True),
                ),
                Signal(
                    name="POSITIVE_ADOSC_20_DAY_BREAKOUT",
                    type_info="Oversold",
                    description="20-day breakout confirmed by positive ADOSC",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.ADOSC_SIGNAL == True)  # noqa: E712
                    ].last_valid_index(),
                ),
            ],
        ),
        Indicator(
            name="CANDLESTICKS",
            description="Candlestick Patterns",
            expected_columns=[
                "CDLMORNINGSTAR",
                "CDL3LINESTRIKE",
                "CDL3WHITESOLDIERS",
                "CDLABANDONEDBABY",
                "CDLTASUKIGAP",
                "CDLPIERCING",
                "CDLENGULFING",
            ],
            function=CANDLESTOCK_PATTERNS.call,
            signals=[
                Signal(
                    name="CDLMORNINGSTAR",
                    type_info="Long",
                    description="Morning Star Candlestick Pattern",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLMORNINGSTAR == 100)].last_valid_index(),
                ),
                Signal(
                    name="CDL3LINESTRIKE",
                    description="3 Line Strike Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDL3LINESTRIKE == 100)].last_valid_index(),
                ),
                Signal(
                    name="CDL3WHITESOLDIERS",
                    description="3 White Soldiers Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.CDL3WHITESOLDIERS == 100)
                    ].last_valid_index(),
                ),
                Signal(
                    name="CDLABANDONEDBABY",
                    description="Abandoned Baby Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.CDLABANDONEDBABY == 100)
                    ].last_valid_index(),
                ),
                Signal(
                    name="CDLTASUKIGAP",
                    description="Tasukigap Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLTASUKIGAP == 100)].last_valid_index(),
                ),
                Signal(
                    name="CDLPIERCING",
                    description="Piercing Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLPIERCING == 100)].last_valid_index(),
                ),
                Signal(
                    name="CDLENGULFING",
                    description="Engulfing Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLENGULFING == 100)].last_valid_index(),
                ),
            ],
        ),
    ]


class Indicators(BaseModel):
    indicators: List[Indicator] = Field(default_factory=indicators_factory)

    def compute(self, data: pd.DataFrame) -> None:
        for indicator in self.indicators:
            try:
                indicator.compute(data)
            except Exception as e:
                logger.error(f"Failed to compute indicator {indicator.name}: {e}")
                continue
            logger.info(
                f"Computed {indicator.name} with {len(indicator.signals)} signals"
            )

    def to_dict(self, data: pd.DataFrame) -> Dict[str, Any]:
        self.compute(data)
        res = {}
        for indicator in self.indicators:
            for signal in indicator.signals:
                res[signal.name.lower()] = (
                    signal.date if signal.is_date() else signal.value
                )
        return res

    def create_indicator_models(self) -> List[type[BaseModel]]:
        models = []
        for indicator in self.indicators:
            model_parameters = {}
            for signal in indicator.signals:
                range_ = {}
                if signal.range:
                    range_ = {"ge": signal.range[0], "le": signal.range[1]}
                model_parameters[signal.name.lower()] = (
                    signal.type,
                    Field(  # type: ignore
                        None,
                        **range_,
                        description=(
                            signal.description
                            or " ".join(signal.name.lower().capitalize().split("_"))
                        ),
                    ),
                )
            model = create_model(indicator.name, **model_parameters)  # type: ignore
            model._description = indicator.description
            models.append(model)
        return models


IndicatorModels = Indicators().create_indicator_models()
