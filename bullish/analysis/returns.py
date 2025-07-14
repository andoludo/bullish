import datetime
from typing import (
    Optional,
    Any,
    Annotated,
    Literal,
    Dict,
    List,
    TYPE_CHECKING,
    get_args,
)

import numpy as np
import pandas as pd
from bearish.models.base import Ticker  # type: ignore
from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from pydantic import BaseModel, BeforeValidator, Field, model_validator

from bullish.analysis.constants import Industry, IndustryGroup, Sector, Country

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb

Type = Literal["Mean", "Median", "Max", "Min", "StdDev", "Variance"]

FUNCTIONS = {
    "Mean": np.mean,
    "Median": np.median,
    "Max": np.max,
    "Min": np.min,
    "StdDev": np.std,
    "Variance": np.var,
}


class PricesReturns(Prices):  # type: ignore

    def returns(self) -> pd.DataFrame:
        data = self.to_dataframe()
        data["simple_return"] = data.close.pct_change() * 100
        data["log_return"] = (data.close / data.close.shift(1)).apply(np.log) * 100
        return data[["simple_return", "log_return"]].dropna()  # type: ignore


def to_float(value: Any) -> Optional[float]:
    if value == "None":
        return None
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return float(value)


class IndustryReturns(BaseModel):
    date: datetime.date
    created_at: datetime.date
    simple_return: Annotated[float, BeforeValidator(to_float), Field(None)]
    log_return: Annotated[float, BeforeValidator(to_float), Field(None)]
    country: Country
    industry: Industry
    industry_group: Optional[IndustryGroup] = None
    sector: Optional[Sector] = None
    type: Type

    @model_validator(mode="before")
    def _validate(cls, values: Dict[str, Any]) -> Dict[str, Any]:  # noqa: N805
        created_at = datetime.date.today()
        current_date = values.get("date", created_at)
        return (
            {"date": current_date}
            | values
            | {
                "created_at": created_at,
            }
        )

    @classmethod
    def from_data(
        cls,
        data: pd.DataFrame,
        function_name: Type,
        industry: Industry,
        country: Country,
    ) -> List["IndustryReturns"]:
        function = FUNCTIONS[function_name]
        if data.shape[1] > 2:
            simple_return = (
                data["simple_return"].apply(function, axis=1).rename("simple_return")  # type: ignore
            )
            log_return = data["log_return"].apply(function, axis=1).rename("log_return")  # type: ignore
        else:
            simple_return = data["simple_return"]
            log_return = data["log_return"]

        data = pd.concat([simple_return, log_return], axis=1)
        data["date"] = data.index
        return [
            cls.model_validate(
                r | {"industry": industry, "type": function_name, "country": country}
            )
            for r in data.to_dict(orient="records")
        ]

    @classmethod
    def from_db(
        cls, bullish: "BullishDb", industry: Industry, country: Country
    ) -> List["IndustryReturns"]:
        returns = []
        symbols = bullish.read_industry_symbols(industries=[industry], country=country)
        query = AssetQuery(
            symbols=Symbols(equities=[Ticker(symbol=s) for s in symbols])
        )
        data = bullish.read_series(query, months=6)
        raw_data = [
            PricesReturns(prices=[d for d in data if d.symbol == s]).returns()
            for s in symbols
        ]
        if raw_data:
            data_ = pd.concat(raw_data, axis=1)
            for function_name in FUNCTIONS:
                returns.extend(cls.from_data(data_, function_name, industry, country))  # type: ignore
        return returns


def compute_industry(bullish: "BullishDb") -> None:
    for country in get_args(Country):
        for industry in get_args(Industry):
            returns = IndustryReturns.from_db(bullish, industry, country)
            if returns:
                bullish.write_returns(returns)
