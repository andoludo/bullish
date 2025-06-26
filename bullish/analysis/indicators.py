import logging
from datetime import date
from typing import Optional, List, Callable, Any, Literal, Dict, Union

import pandas as pd
import talib
from pydantic import BaseModel, Field, PrivateAttr, create_model
import pandas_ta as ta  # type: ignore

logger = logging.getLogger(__name__)
SignalType = Literal["Short", "Long", "Oversold", "Overbought", "Value"]


def cross(
    series_a: pd.Series, series_b: pd.Series, above: bool = True  # type: ignore
) -> Optional[date]:
    crossing = ta.cross(series_a=series_a, series_b=series_b, above=above)
    if not crossing[crossing == 1].index.empty:
        return pd.Timestamp(crossing[crossing == 1].index[-1]).date()
    return None


def cross_value(series: pd.Series, number: int, above: bool = True) -> Optional[date]:  # type: ignore
    return cross(series, pd.Series(number, index=series.index), above=above)


class Signal(BaseModel):
    name: str
    type_info: SignalType
    type: Any
    function: Callable[[pd.DataFrame], Optional[Union[date, float]]]
    description: Optional[str] = None
    date: Optional[date] = None
    value: Optional[float] = None

    def compute(self, data: pd.DataFrame) -> None:
        if self.type == Optional[date]:
            self.date = self.function(data)  # type: ignore
        elif self.type == Optional[float]:
            self.value = self.function(data)  # type: ignore
        else:
            raise NotImplementedError


class Indicator(BaseModel):
    name: str
    description: Optional[str] = None
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


def compute_adx(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ADX_14"] = talib.ADX(data.high, data.low, close=data.close)  # type: ignore
    results["MINUS_DM"] = talib.MINUS_DI(data.high, data.low, data.close)  # type: ignore
    results["PLUS_DM"] = talib.PLUS_DI(data.high, data.low, data.close)  # type: ignore
    return results


def compute_macd(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    (
        results["MACD_12_26_9"],
        results["MACD_12_26_9_SIGNAL"],
        results["MACD_12_26_9_HIST"],
    ) = talib.MACD(
        data.close  # type: ignore
    )
    return results


def compute_rsi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["RSI"] = talib.RSI(data.close)  # type: ignore
    return results


def compute_stoch(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["SLOW_K"], results["SLOW_D"] = talib.STOCH(data.high, data.low, data.close)  # type: ignore
    return results


def compute_mfi(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["MFI"] = talib.MFI(data.high, data.low, data.close, data.volume)  # type: ignore
    return results


def compute_roc(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["ROC_7"] = talib.ROC(data.close, timeperiod=7)  # type: ignore
    results["ROC_30"] = talib.ROC(data.close, timeperiod=30)  # type: ignore
    results["ROC_90"] = talib.ROC(data.close, timeperiod=90)  # type: ignore
    results["ROC_180"] = talib.ROC(data.close, timeperiod=180)  # type: ignore
    return results


def compute_patterns(data: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame(index=data.index)
    results["CDLMORNINGSTAR"] = talib.CDLMORNINGSTAR(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDL3LINESTRIKE"] = talib.CDL3LINESTRIKE(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDL3WHITESOLDIERS"] = talib.CDL3WHITESOLDIERS(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLABANDONEDBABY"] = talib.CDLABANDONEDBABY(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLTASUKIGAP"] = talib.CDLTASUKIGAP(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLPIERCING"] = talib.CDLPIERCING(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    results["CDLENGULFING"] = talib.CDLENGULFING(
        data.open, data.high, data.low, data.close  # type: ignore
    )
    return results


def indicators_factory() -> List[Indicator]:
    return [
        Indicator(
            name="ADX_14",
            description="Average Directional Movement Index",
            expected_columns=["ADX_14", "MINUS_DM", "PLUS_DM"],
            function=compute_adx,
            signals=[
                Signal(
                    name="ADX_14_LONG",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.ADX_14 > 20) & (d.PLUS_DM > d.MINUS_DM)
                    ].index[-1],
                ),
                Signal(
                    name="ADX_14_SHORT",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: d[
                        (d.ADX_14 > 20) & (d.MINUS_DM > d.PLUS_DM)
                    ].index[-1],
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
            function=compute_macd,
            signals=[
                Signal(
                    name="MACD_12_26_9_BULLISH_CROSSOVER",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross(d.MACD_12_26_9, d.MACD_12_26_9_SIGNAL),
                ),
                Signal(
                    name="MACD_12_26_9_BEARISH_CROSSOVER",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross(d.MACD_12_26_9_SIGNAL, d.MACD_12_26_9),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_UP",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.MACD_12_26_9, 0),
                ),
                Signal(
                    name="MACD_12_26_9_ZERO_LINE_CROSS_DOWN",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.MACD_12_26_9, 0, above=False),
                ),
            ],
        ),
        Indicator(
            name="RSI",
            description="Relative Strength Index",
            expected_columns=["RSI"],
            function=compute_rsi,
            signals=[
                Signal(
                    name="RSI_BULLISH_CROSSOVER",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 30),
                ),
                Signal(
                    name="RSI_BEARISH_CROSSOVER",
                    type_info="Short",
                    type=Optional[date],
                    function=lambda d: cross_value(d.RSI, 70, above=False),
                ),
                Signal(
                    name="RSI_OVERSOLD",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[(d.RSI < 30) & (d.RSI > 0)].index[-1],
                ),
                Signal(
                    name="RSI_OVERBOUGHT",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[(d.RSI < 100) & (d.RSI > 70)].index[-1],
                ),
            ],
        ),
        Indicator(
            name="STOCH",
            description="Stochastic",
            expected_columns=["SLOW_K", "SLOW_D"],
            function=compute_stoch,
            signals=[
                Signal(
                    name="STOCH_OVERSOLD",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[(d.SLOW_K < 20) & (d.SLOW_K > 0)].index[-1],
                ),
                Signal(
                    name="STOCH_OVERBOUGHT",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[(d.SLOW_K < 100) & (d.SLOW_K > 80)].index[-1],
                ),
            ],
        ),
        Indicator(
            name="MFI",
            description="Money Flow Index",
            expected_columns=["MFI"],
            function=compute_mfi,
            signals=[
                Signal(
                    name="MFI_OVERSOLD",
                    type_info="Oversold",
                    type=Optional[date],
                    function=lambda d: d[(d.MFI < 20)].index[-1],
                ),
                Signal(
                    name="MFI_OVERBOUGHT",
                    type_info="Overbought",
                    type=Optional[date],
                    function=lambda d: d[(d.MFI > 80)].index[-1],
                ),
            ],
        ),
        Indicator(
            name="ROC",
            description="Rate Of Change",
            expected_columns=["ROC_7", "ROC_30", "ROC_90", "ROC_180"],
            function=compute_roc,
            signals=[
                Signal(
                    name="RATE_OF_CHANGE_7",
                    type_info="Value",
                    type=Optional[float],
                    function=lambda d: d.ROC_7.tolist()[-1],
                ),
                Signal(
                    name="RATE_OF_CHANGE_30",
                    type_info="Value",
                    type=Optional[float],
                    function=lambda d: d.ROC_30.tolist()[-1],
                ),
                Signal(
                    name="RATE_OF_CHANGE_90",
                    type_info="Value",
                    type=Optional[float],
                    function=lambda d: d.ROC_90.tolist()[-1],
                ),
                Signal(
                    name="RATE_OF_CHANGE_180",
                    type_info="Value",
                    type=Optional[float],
                    function=lambda d: d.ROC_180.tolist()[-1],
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
            function=compute_patterns,
            signals=[
                Signal(
                    name="CDLMORNINGSTAR",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLMORNINGSTAR == 100)].index[-1],
                ),
                Signal(
                    name="CDL3LINESTRIKE",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDL3LINESTRIKE == 100)].index[-1],
                ),
                Signal(
                    name="CDL3WHITESOLDIERS",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDL3WHITESOLDIERS == 100)].index[-1],
                ),
                Signal(
                    name="CDLABANDONEDBABY",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLABANDONEDBABY == 100)].index[-1],
                ),
                Signal(
                    name="CDLTASUKIGAP",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLTASUKIGAP == 100)].index[-1],
                ),
                Signal(
                    name="CDLPIERCING",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLPIERCING == 100)].index[-1],
                ),
                Signal(
                    name="CDLENGULFING",
                    type_info="Long",
                    type=Optional[date],
                    function=lambda d: d[(d.CDLENGULFING == 100)].index[-1],
                ),
            ],
        ),
    ]


class Indicators(BaseModel):
    indicators: List[Indicator] = Field(default_factory=indicators_factory)

    def compute(self, data: pd.DataFrame) -> None:
        for indicator in self.indicators:
            indicator.compute(data)
            logger.info(
                f"Computed {indicator.name} with {len(indicator.signals)} signals"
            )

    def to_dict(self, data: pd.DataFrame) -> Dict[str, Any]:
        self.compute(data)
        res = {}
        for indicator in self.indicators:
            for signal in indicator.signals:
                res[signal.name.lower()] = signal.date
        return res

    def create_indicator_model(self) -> type[BaseModel]:
        model_parameters = {}
        for indicator in self.indicators:
            for signal in indicator.signals:
                model_parameters[signal.name.lower()] = (
                    signal.type,
                    Field(
                        None,
                        description=(
                            signal.description
                            or " ".join(signal.name.lower().capitalize().split("_"))
                        ),
                    ),
                )
        return create_model("IndicatorModel", **model_parameters)  # type: ignore
