import json
import logging
import random
from datetime import date, timedelta
from io import StringIO
from typing import TYPE_CHECKING, Optional, Union, List, Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


import plotly.graph_objects as go

if TYPE_CHECKING:
    from bullish.analysis.predefined_filters import NamedFilterQuery
    from bullish.database.crud import BullishDb

logger = logging.getLogger(__name__)
COLOR = {
    "mean": "#1f77b4",  # A refined blue (Plotly default)
    "upper": "#d62728",  # Strong red
    "lower": "#2ca02c",  # Rich green
    "median": "#ff7f0e",  # Bright orange
}


class BacktestQueryBase(BaseModel):
    name: str
    table: str


class BacktestQueryDate(BacktestQueryBase):

    start: date
    end: date


class BacktestQueryRange(BacktestQueryBase):

    min: float
    max: float


class BacktestQuerySelection(BacktestQueryBase):

    selections: List[str]

    def to_selections(self) -> str:
        return ", ".join([f"'{s}'" for s in self.selections])


class BacktestQueries(BaseModel):
    queries: list[Union[BacktestQueryDate, BacktestQueryRange, BacktestQuerySelection]]

    def to_query(self) -> str:
        query_parts = []
        for query in self.queries:
            if isinstance(query, (BacktestQueryDate)):
                query_parts.append(
                    f"SELECT symbol FROM {query.table} WHERE name='{query.name}' "  # noqa: S608
                    f"AND date >='{query.start}' AND date <='{query.end}'"
                )
            if isinstance(query, (BacktestQueryRange)):
                query_parts.append(
                    f"SELECT symbol FROM {query.table} WHERE "  # noqa: S608
                    f"{query.name} >= {query.min} AND {query.name} <= {query.max}"
                )
            if isinstance(query, (BacktestQuerySelection)):
                query_parts.append(
                    f"SELECT symbol FROM {query.table} WHERE "  # noqa: S608
                    f"{query.name} IN ({query.to_selections()})"
                )

        if len(query_parts) == 1:
            return query_parts[0]
        else:
            return " INTERSECT ".join(query_parts)


class ReturnPercentage(BaseModel):
    return_percentage: float = Field(
        default=12, description="Return percentage of the backtest"
    )


class BaseBacktestResult(BaseModel):
    start: date = Field(default=date.today() - timedelta(days=252))
    end: date = Field(default=date.today())
    investment: float = Field(default=1000)
    holding_period: int = Field(default=30 * 3)
    extend_days: int = Field(
        default=5,
        description="Extend the backtest by this many days if no symbols are found",
    )
    percentage: int = Field(default=12, description="Return percentage of the backtest")
    iterations: int = Field(default=200, description="Number of iterations to run")


class BacktestResultQuery(BaseBacktestResult):
    name: str


class BacktestResult(BacktestResultQuery):
    data: Dict[str, Any]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_json(StringIO(json.dumps(self.data))).sort_index()


class BacktestResults(BaseModel):
    results: List[BacktestResult]

    def figure(self, type: str = "mean") -> go.Figure:
        fig = go.Figure()
        for result in self.results:
            data = result.to_dataframe()[type]
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data,
                    mode="lines",
                    name=f"{result.name} ({type})",
                    line={"width": 1},
                    hovertemplate=(
                        "Date: %{x}<br>"
                        + "Price: %{y:.2f}<br>"
                        + f"Percentage: {result.percentage}<br>"
                        + f"Iterations: {result.iterations}<br>"
                        + f"Investment: {result.investment}<extra></extra>"
                    ),
                )
            )
        fig.update_layout(
            height=800,
            showlegend=True,
            margin={"t": 60, "b": 40},
        )

        return fig


class BackTestConfig(BaseBacktestResult):
    exit_strategy: ReturnPercentage = Field(default=ReturnPercentage)

    def to_base_backtest_result(self) -> BaseBacktestResult:
        return BaseBacktestResult(
            start=self.start,
            end=self.end,
            investment=self.investment,
            holding_period=self.holding_period,
            extend_days=self.extend_days,
            percentage=self.percentage,
            iterations=self.iterations,
        )


class Equity(BaseModel):
    symbol: str
    start: date
    end: date
    buy: float
    sell: float
    investment_in: float
    investment_out: Optional[float] = None

    def profit(self) -> float:
        return (self.sell - self.buy) * (self.investment_in / self.buy)

    def current_value(self) -> float:
        return self.investment_in + self.profit()

    def set_investment_out(self) -> None:
        self.investment_out = self.current_value()


class BackTest(BaseModel):
    equities: list[Equity] = Field(
        default_factory=list, description="List of equities bought during the backtest"
    )
    end: date = Field(default=date.today(), description="End date of the backtest")

    def valid(self) -> bool:
        return bool(self.equities)

    def total_profit(self) -> float:
        return sum(equity.profit() for equity in self.equities)

    def symbols(self) -> list[str]:
        return [equity.symbol for equity in self.equities]

    def show(self) -> None:
        for eq in self.equities:
            print(
                f"\n{eq.symbol} ({eq.type}): {eq.start}:{eq.investment_in} ({eq.buy}) - "
                f"{eq.end}:{eq.investment_out} ({eq.sell})"
            )

    def to_dataframe(self) -> pd.DataFrame:
        prices = [
            self.equities[0].investment_in,
            *[e.investment_out for e in self.equities],
        ]
        symbols = [self.equities[0].symbol, *[e.symbol for e in self.equities]]
        index = [self.equities[0].start, *[e.end for e in self.equities]]
        buy = [self.equities[0].buy, *[e.buy for e in self.equities]]
        sell = [self.equities[0].sell, *[e.sell for e in self.equities]]
        data = pd.DataFrame(
            np.array([prices, symbols, buy, sell]).T,
            index=index,
            columns=["prices", "symbols", "buy", "sell"],
        )
        data = data[~data.index.duplicated(keep="first")]
        return data

    def __hash__(self) -> int:
        return hash(tuple(sorted(equity.symbol for equity in self.equities)))


class BackTests(BaseModel):
    tests: list[BackTest] = Field(default_factory=list, description="List of backtests")
    config: BackTestConfig
    name: str

    @model_validator(mode="after")
    def _validate(self) -> "BackTests":
        self.tests = list(set(self.tests))  # Remove duplicates
        return self

    def to_dataframe(self) -> pd.DataFrame:

        data = (
            pd.concat([t.to_dataframe() for t in self.tests if t.valid()], axis=1)
            .sort_index()
            .fillna(method="ffill")
        )
        data = data[~data.index.duplicated(keep="first")]
        return data

    def to_error(self) -> pd.DataFrame:
        data_ = self.to_dataframe()
        mean = data_.prices.astype(float).mean(axis=1).rename("mean")
        std = data_.prices.astype(float).std(axis=1)
        median = data_.prices.astype(float).median(axis=1).rename("median")
        upper = (mean + std).rename("upper")
        lower = (mean - std).rename("lower")
        return pd.concat([mean, upper, lower, median], axis=1).sort_index()

    def to_backtest_result(self) -> BacktestResult:

        return BacktestResult.model_validate(
            self.config.to_base_backtest_result().model_dump()
            | {"data": json.loads(self.to_error().to_json()), "name": self.name}
        )

    def to_figure(self) -> go.Figure:

        data_ = self.to_dataframe()
        self.to_error()
        column_chunks = [data_.iloc[:, i : i + 4] for i in range(0, data_.shape[1], 4)]
        fig = go.Figure()
        for data in column_chunks:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.prices.astype(float),
                    mode="lines",
                    showlegend=False,
                    customdata=data[
                        ["symbols", "sell", "buy"]
                    ],  # Include multiple overlay columns
                    line={"color": "grey", "width": 0.5},  # normal grey
                    opacity=0.5,
                    hovertemplate=(
                        "Date: %{x}<br>"
                        + "Price: %{y:.2f}<br>"
                        + "Symbols: %{customdata[0]}<br>"
                        + "Sell: %{customdata[1]}<br>"
                        + "Buy: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
        for name, column in self.to_error().items():
            fig.add_trace(
                go.Scatter(
                    x=column.index,
                    y=column,
                    mode="lines",
                    line={"color": COLOR[name], "width": 1},
                    showlegend=True,
                    name=name,
                )
            )
        fig.update_layout(
            title="Predefined filter performance",
            xaxis_title="Date",
            yaxis_title="Prices [Currency]",
        )
        fig.show()
        return fig


def run_backtest(  # noqa: C901, PLR0915
    bullish_db: "BullishDb", named_filter: "NamedFilterQuery", config: BackTestConfig
) -> BackTest:
    equities = []
    start_date = config.start
    presence_delta = timedelta(days=config.holding_period)
    investment = config.investment
    exclude_symbols = []
    while True:
        symbols = []
        while not symbols:
            symbols = named_filter.get_backtesting_symbols(bullish_db, start_date)
            symbols = [b for b in symbols if b not in exclude_symbols]
            if symbols:
                break
            start_date = start_date + timedelta(days=config.extend_days)
            if start_date > config.end:
                logger.debug("No symbols found for the given date range.")
                break
        if symbols:
            symbol = random.choice(symbols)  # noqa: S311
            logger.debug(f"Found symbol: {symbol}, for date: {start_date}")
            enter_position = start_date
            end_position = None
            counter = 0
            buy_price = None
            while True:

                data = bullish_db.read_symbol_series(
                    symbol,
                    start_date=enter_position + counter * presence_delta,
                    end_date=enter_position + (counter + 1) * presence_delta,
                )
                if data.empty:
                    logger.debug(f"No data found for symbol: {symbol}")
                    exclude_symbols.append(symbol)
                    end_position = start_date
                    break
                data.index = data.index.tz_localize(None)
                if counter == 0:
                    enter_position_timestamp = data.close.first_valid_index()
                    enter_position = enter_position_timestamp.date()
                    buy_price = data.close.loc[enter_position_timestamp]

                mask = data.close >= buy_price * (
                    1 + config.percentage / (100 * (counter + 1))
                )
                mask_ = mask[mask == True]  # noqa: E712

                if mask_.empty:
                    if enter_position + (counter + 1) * presence_delta > config.end:
                        end_position = data.close.index[-1].date()
                        sell_price = data.close.iloc[-1]
                        equity = Equity(
                            symbol=symbol,
                            start=enter_position,
                            end=end_position,
                            buy=buy_price,
                            sell=sell_price,
                            investment_in=investment,
                        )
                        equity.set_investment_out()
                        equities.append(equity)
                        investment = equity.current_value()
                        end_position = config.end
                        break
                    counter += 1
                    continue
                else:
                    end_position_timestamp = data[mask].first_valid_index()
                    end_position = end_position_timestamp.date()
                    equity = Equity(
                        symbol=symbol,
                        start=enter_position,
                        end=end_position,
                        buy=buy_price,
                        sell=data[mask].close.loc[end_position_timestamp],
                        investment_in=investment,
                    )
                    equity.set_investment_out()
                    equities.append(equity)
                    investment = equity.current_value()
                    break

            start_date = end_position
        if start_date >= config.end:
            break
    back_test = BackTest(equities=equities)
    return back_test


def run_tests(
    bullish_db: "BullishDb", named_filter: "NamedFilterQuery", config: BackTestConfig
) -> BackTests:
    return BackTests(
        config=config,
        name=named_filter.name,
        tests=[
            run_backtest(bullish_db, named_filter, config)
            for _ in range(config.iterations)
        ],
    )


def run_many_tests(
    bullish_db: "BullishDb",
    named_filters: List["NamedFilterQuery"],
    config: BackTestConfig,
) -> None:
    back_tests = []
    for named_filter in named_filters:
        try:
            back_tests.append(
                run_tests(bullish_db, named_filter, config).to_backtest_result()
            )
        except Exception as e:  # noqa: PERF203
            logger.error(e)
            continue

    if back_tests:
        bullish_db.write_many_backtest_results(back_tests)
