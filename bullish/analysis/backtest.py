import random
from datetime import date, timedelta, datetime
from typing import TYPE_CHECKING, Optional

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from bullish.database.crud import BullishDb

if TYPE_CHECKING:
    from bullish.analysis.predefined_filters import NamedFilterQuery


class BacktestQuery(BaseModel):
    name: str
    start: date
    end: date

class BacktestQueries(BaseModel):
    queries: list[BacktestQuery]

    def to_query(self) -> str:
        query_parts = []
        for query in self.queries:
            query_parts.append(f"SELECT symbol FROM signalseries WHERE name='{query.name}' AND date >='{query.start}' AND date <='{query.end}'")
        if len(query_parts) ==1:
            return query_parts[0]
        else:
            return " INTERSECT ".join(query_parts)


class ReturnPercentage(BaseModel):
    return_percentage: float = Field(default=12, description="Return percentage of the backtest")

class BackTestConfig(BaseModel):
    start: date
    end: date = Field(default=date.today())
    investment: float = Field(default=1000)
    exit_strategy:ReturnPercentage = Field(default=ReturnPercentage)


class Equity(BaseModel):
    symbol: str
    start: date
    end: date
    buy: float
    sell: float
    investment_in: float
    investment_out: Optional[float] = None
    type: str

    def profit(self) -> float:
        return (self.sell - self.buy) * (self.investment_in / self.buy)

    def current_value(self) -> float:
        return self.investment_in + self.profit()

    def set_investment_out(self):
        self.investment_out = self.current_value()

class BackTest(BaseModel):
    equities: list[Equity] = Field(default_factory=list, description="List of equities bought during the backtest")
    end: date = Field(default=date.today(), description="End date of the backtest")

    def valid(self) -> bool:
        return bool(self.equities)

    def total_profit(self)-> float:
        return sum(equity.profit() for equity in self.equities)

    def symbols(self) -> list[str]:
        return [equity.symbol for equity in self.equities]

    def show(self):
        for eq in self.equities:
            print(f"\n{eq.symbol} ({eq.type}): {eq.start}:{eq.investment_in} ({eq.buy}) - {eq.end}:{eq.investment_out} ({eq.sell})")

    def to_dataframe(self) ->pd.DataFrame:
        return pd.DataFrame([self.equities[0].investment_in]+[e.investment_out for e in self.equities]+[self.equities[-1].investment_out], index = [self.equities[0].start]+[e.end for e in self.equities]+[self.end], columns=["test"])



    def __hash__(self) -> int:
        return hash(tuple(sorted(equity.symbol for equity in self.equities)))

class BackTests(BaseModel):
    tests: list[BackTest] = Field(default_factory=list, description="List of backtests")

    @model_validator(mode="after")
    def _validate(self):
        self.tests = list(set(self.tests))  # Remove duplicates
        return self

    def show(self):
        for test in self.tests:
            test.show()
    def to_dataframe(self) -> pd.DataFrame:

        return pd.concat([t.to_dataframe() for t in self.tests if t.valid()], axis=1).sort_index().fillna(method="ffill")





def run_backtest(bullish_db: BullishDb, named_filter: "NamedFilterQuery", config: BackTestConfig) -> BackTest:
    equities = []
    start_date= config.start
    presence_delta = timedelta(days=30*3)
    investment = config.investment
    while True:
        symbols = []
        while not symbols:
            symbols = named_filter.get_backtesting_symbols(bullish_db, start_date)
            if symbols:
                break
            start_date = start_date+ timedelta(days=5)
            if start_date > config.end:
                break
        if symbols:
            symbol = random.choice(symbols)
            enter_position = start_date
            end_position = None
            counter = 0
            buy_price = None
            while True:

                data = bullish_db.read_symbol_series(symbol, start_date=enter_position + counter*presence_delta, end_date=enter_position+ (counter+1)*presence_delta)

                if data.empty:
                    data__ = bullish_db.read_symbol_series(symbol, start_date=config.start, end_date=enter_position+ (counter+1)*presence_delta)
                    data__.index = data__.index.tz_localize(None)
                    end_position = data__.close.index[-1].date()
                    sell_price = data__.close.iloc[-1]
                    equity = Equity(symbol=symbol, start=enter_position, end=end_position, buy=buy_price, sell=sell_price, investment_in=investment, type= "nodata")
                    equity.set_investment_out()
                    equities.append(equity)
                    investment = equity.current_value()
                    break
                data.index = data.index.tz_localize(None)
                if counter == 0:
                    enter_position_timestamp = data.close.first_valid_index()
                    enter_position = enter_position_timestamp.date()
                    buy_price = data.close.loc[enter_position_timestamp]

                mask = data.close >=buy_price*(1 + 12/(100*(counter+1)))
                mask_ = mask[mask==True]

                if mask_.empty:
                    if enter_position + (counter + 1) * presence_delta > config.end:
                        end_position = enter_position + (counter + 1) * presence_delta
                        break
                    counter+=1
                    continue
                else:
                    end_position_timestamp = data[mask].first_valid_index()
                    end_position= end_position_timestamp.date()
                    equity = Equity(symbol=symbol, start=enter_position, end=end_position, buy=buy_price, sell=data[mask].close.loc[end_position_timestamp], investment_in=investment, type= "noral")
                    equity.set_investment_out()
                    equities.append(equity)
                    investment = equity.current_value()
                    break

            start_date = end_position
        if start_date > config.end:
            break
    back_test = BackTest(equities=equities)
    return back_test


def run_tests(bullish_db: BullishDb, named_filter: "NamedFilterQuery", config: BackTestConfig) :
    test = []
    for _ in range(100):
        try:
            test.append(run_backtest(bullish_db, named_filter, config) )
        except Exception as e:
            continue
    back_tests = BackTests(tests=test)

    back_tests.show()
    return back_tests.to_dataframe()
