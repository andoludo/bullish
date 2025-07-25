import logging
from datetime import date
from typing import Optional, List, Callable, Any, Literal, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, create_model

from bullish.analysis.functions import (
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
    cross_simple,
    cross_value_series,
    find_last_true_run_start,
)

logger = logging.getLogger(__name__)
SignalType = Literal["Short", "Long", "Oversold", "Overbought", "Value"]


def _last_date(d: pd.Series) -> Optional[date]:
    d_valid = d[d == 1]
    if d_valid.empty:
        return None
    last_index = d_valid.last_valid_index()
    return last_index.date() if last_index is not None else None  # type: ignore


class ProcessingFunction(BaseModel):
    date: Callable[[pd.Series], Optional[date]] = Field(default=_last_date)
    number: Callable[[pd.Series], Optional[float]] = Field(
        default=lambda d: d.iloc[-1] if not d.dropna().empty else None
    )


class SignalSeries(BaseModel):
    name: str
    date: date
    value: float
    symbol: str


class Signal(BaseModel):
    name: str
    type_info: SignalType
    type: Any
    range: Optional[List[float]] = None
    function: Callable[[pd.DataFrame], pd.Series]
    processing: ProcessingFunction = Field(default_factory=ProcessingFunction)
    description: str
    date: Optional[date] = None
    value: Optional[float] = None
    in_use_backtest: bool = False

    def is_date(self) -> bool:
        if self.type == Optional[date]:
            return True
        elif self.type == Optional[float]:
            return False
        else:
            raise NotImplementedError

    def apply_function(self, data: pd.DataFrame) -> pd.Series:
        result = self.function(data)
        if not isinstance(result, pd.Series):
            raise ValueError(
                f"Function for signal {self.name} must return a pandas Series"
            )
        return result

    def compute(self, data: pd.DataFrame) -> None:
        if self.is_date():
            self.date = self.processing.date(self.apply_function(data))
        else:
            self.value = self.processing.number(self.apply_function(data))

    def compute_series(self, data: pd.DataFrame) -> pd.Series:
        return self.apply_function(data)


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
        self.compute_signals()

    def compute_signals(self) -> None:
        for signal in self.signals:
            try:
                signal.compute(self._data)
            except Exception as e:  # noqa: PERF203
                logger.error(
                    f"Fail to compute signal {signal.name} for indicator {self.name}: {e}"
                )

    def compute_series(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        series = []
        try:
            results = self.function(data)
        except Exception as e:
            logger.error(
                f"Failed to compute indicator {self.name} for symbol {symbol}: {e}"
            )
            return pd.DataFrame()
        if not set(self.expected_columns).issubset(results.columns):
            raise ValueError(
                f"Expected columns {self.expected_columns}, but got {results.columns.tolist()}"
            )
        for signal in self.signals:
            if not signal.in_use_backtest:
                continue
            try:
                series_ = signal.compute_series(results)
                if signal.type == Optional[date]:
                    series__ = pd.DataFrame(series_[series_ == 1].rename("value"))
                else:
                    series__ = pd.DataFrame(
                        series_[series_ != None].rename("value")  # noqa: E711
                    )

                series__["name"] = signal.name
                series__["date"] = series__.index.date  # type: ignore
                series__["symbol"] = symbol
                series__ = series__.reset_index(drop=True)
                series.append(series__)
            except Exception as e:
                logger.error(
                    f"Fail to compute signal {signal.name} for indicator {self.name}: {e}"
                )
        if not series:
            return pd.DataFrame()
        data = pd.concat(series).reset_index(drop=True)
        return data


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
                    function=lambda d: (d.ADX_14 > 20) & (d.PLUS_DI > d.MINUS_DI),
                ),
                Signal(
                    name="ADX_14_SHORT",
                    description="ADX 14 Short Signal",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: (d.ADX_14 > 20) & (d.MINUS_DI > d.PLUS_DI),
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
                    function=lambda d: cross_simple(
                        d.MACD_12_26_9, d.MACD_12_26_9_SIGNAL
                    ),
                    in_use_backtest=True,
                ),
                Signal(
                    name="MACD_12_26_9_BEARISH_CROSSOVER",
                    description="MACD 12-26-9 Bearish Crossover",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross_simple(
                        d.MACD_12_26_9_SIGNAL, d.MACD_12_26_9
                    ),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_UP",
                    description="MACD 12-26-9 Zero Line Cross Up",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value_series(d.MACD_12_26_9, 0),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_DOWN",
                    description="MACD 12-26-9 Zero Line Cross Down",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value_series(
                        d.MACD_12_26_9, 0, above=False
                    ),
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
                    function=lambda d: cross_value_series(d.RSI, 30),
                    in_use_backtest=True,
                ),
                Signal(
                    name="RSI_BULLISH_CROSSOVER_40",
                    description="RSI Bullish Crossover 40",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value_series(d.RSI, 40),
                    in_use_backtest=True,
                ),
                Signal(
                    name="RSI_BULLISH_CROSSOVER_45",
                    description="RSI Bullish Crossover 45",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value_series(d.RSI, 45),
                    in_use_backtest=True,
                ),
                Signal(
                    name="RSI_BEARISH_CROSSOVER",
                    description="RSI Bearish Crossover",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross_value_series(d.RSI, 70, above=False),
                ),
                Signal(
                    name="RSI_OVERSOLD",
                    description="RSI Oversold Signal",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: (d.RSI < 30) & (d.RSI > 0),
                    in_use_backtest=True,
                ),
                Signal(
                    name="RSI_OVERBOUGHT",
                    description="RSI Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: (d.RSI < 100) & (d.RSI > 70),
                ),
                Signal(
                    name="RSI_NEUTRAL",
                    description="RSI Neutral Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: (d.RSI < 60) & (d.RSI > 40),
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
                    function=lambda d: (d.SLOW_K < 20) & (d.SLOW_K > 0),
                ),
                Signal(
                    name="STOCH_OVERBOUGHT",
                    description="Stoch Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: (d.SLOW_K < 100) & (d.SLOW_K > 80),
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
                    function=lambda d: (d.MFI < 20),
                ),
                Signal(
                    name="MFI_OVERBOUGHT",
                    description="MFI Overbought Signal",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: (d.MFI > 80),
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
                    function=lambda d: cross_simple(d.SMA_50, d.SMA_200),
                    in_use_backtest=True,
                ),
                Signal(
                    name="DEATH_CROSS",
                    description="Death cross: SMA 50 crosses below SMA 200",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: cross_simple(d.SMA_50, d.SMA_200, above=False),
                ),
                Signal(
                    name="SMA_50_ABOVE_SMA_200",
                    description="SMA 50 is above SMA 200",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d.SMA_50 > d.SMA_200,
                    in_use_backtest=True,
                    processing=ProcessingFunction(date=find_last_true_run_start),
                ),
                Signal(
                    name="PRICE_ABOVE_SMA_50",
                    description="Price is above SMA 50",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d.SMA_50 < d.CLOSE,
                    in_use_backtest=True,
                    processing=ProcessingFunction(date=find_last_true_run_start),
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
                    function=lambda d: 0.6 * d["200_DAY_HIGH"] > d.LAST_PRICE,
                ),
                Signal(
                    name="LOWER_THAN_20_DAY_HIGH",
                    description="Current price is lower than the 20-day high",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: 0.6 * d["20_DAY_HIGH"] > d.LAST_PRICE,
                ),
                Signal(
                    name="MEDIAN_WEEKLY_GROWTH",
                    description="Median weekly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: d.WEEKLY_GROWTH,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.unique())
                    ),
                ),
                Signal(
                    name="MEDIAN_MONTHLY_GROWTH",
                    description="Median monthly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: d.MONTHLY_GROWTH,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.unique())
                    ),
                ),
                Signal(
                    name="MEDIAN_YEARLY_GROWTH",
                    description="Median yearly growth",
                    type_info="Oversold",
                    type=Optional[float],
                    function=lambda d: d.YEARLY_GROWTH,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.unique())
                    ),
                ),
            ],
        ),
        Indicator(
            name="ROC",
            description="Rate Of Change",
            expected_columns=ROC.expected_columns,
            function=ROC.call,
            signals=[
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_1",
                    type_info="Value",
                    description="Median daily Rate of Change of the last 30 days",
                    type=Optional[float],
                    function=lambda d: d.ROC_1,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.tolist()[-30:])
                    ),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_7_4",
                    type_info="Value",
                    description="Median weekly Rate of Change of the last 4 weeks",
                    type=Optional[float],
                    function=lambda d: d.ROC_7,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.tolist()[-4:])
                    ),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_7_12",
                    type_info="Value",
                    description="Median weekly Rate of Change of the last 12 weeks",
                    type=Optional[float],
                    function=lambda d: d.ROC_7,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.tolist()[-12:])
                    ),
                ),
                Signal(
                    name="MEDIAN_RATE_OF_CHANGE_30",
                    type_info="Value",
                    description="Median monthly Rate of Change of the last 12 Months",
                    type=Optional[float],
                    function=lambda d: d.ROC_30,
                    processing=ProcessingFunction(
                        number=lambda v: np.median(v.tolist()[-12:])
                    ),
                ),
                Signal(
                    name="RATE_OF_CHANGE_30",
                    type_info="Value",
                    description="30-day Rate of Change",
                    type=Optional[float],
                    function=lambda d: d.ROC_30,
                ),
                Signal(
                    name="RATE_OF_CHANGE_7",
                    type_info="Value",
                    description="7-day Rate of Change",
                    type=Optional[float],
                    function=lambda d: d.ROC_7,
                ),
                Signal(
                    name="MOMENTUM",
                    type_info="Value",
                    description="7-day Rate of Change",
                    type=Optional[float],
                    function=lambda d: d.MOM,
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
                    function=lambda d: cross_value_series(d.ADOSC, 0, above=True),
                ),
                Signal(
                    name="POSITIVE_ADOSC_20_DAY_BREAKOUT",
                    type_info="Oversold",
                    description="20-day breakout confirmed by positive ADOSC",
                    type=Optional[date],
                    function=lambda d: (d.ADOSC_SIGNAL == True),  # noqa: E712
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
                    function=lambda d: d.CDLMORNINGSTAR == 100,
                ),
                Signal(
                    name="CDL3LINESTRIKE",
                    description="3 Line Strike Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDL3LINESTRIKE == 100,
                ),
                Signal(
                    name="CDL3WHITESOLDIERS",
                    description="3 White Soldiers Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDL3WHITESOLDIERS == 100,
                ),
                Signal(
                    name="CDLABANDONEDBABY",
                    description="Abandoned Baby Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDLABANDONEDBABY == 100,
                ),
                Signal(
                    name="CDLTASUKIGAP",
                    description="Tasukigap Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDLTASUKIGAP == 100,
                ),
                Signal(
                    name="CDLPIERCING",
                    description="Piercing Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDLPIERCING == 100,
                ),
                Signal(
                    name="CDLENGULFING",
                    description="Engulfing Candlestick Pattern",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d.CDLENGULFING == 100,
                ),
            ],
        ),
    ]


class Indicators(BaseModel):
    indicators: List[Indicator] = Field(default_factory=indicators_factory)

    def in_use_backtest(self) -> List[str]:
        return [
            signal.name.lower()
            for indicator in self.indicators
            for signal in indicator.signals
            if signal.in_use_backtest
        ]

    def _compute(self, data: pd.DataFrame) -> None:
        for indicator in self.indicators:
            try:
                indicator.compute(data)
            except Exception as e:
                logger.error(f"Failed to compute indicator {indicator.name}: {e}")
                continue
            logger.info(
                f"Computed {indicator.name} with {len(indicator.signals)} signals"
            )

    def compute_series(self, data: pd.DataFrame, symbol: str) -> List[SignalSeries]:
        data__ = pd.concat(
            [indicator.compute_series(data, symbol) for indicator in self.indicators]
        )
        return [
            SignalSeries.model_validate(s) for s in data__.to_dict(orient="records")
        ]

    def compute(self, data: pd.DataFrame) -> Dict[str, Any]:
        self._compute(data)
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
