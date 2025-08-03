import datetime
import logging
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

from bullish.analysis.constants import (
    Industry,
    IndustryGroup,
    Sector,
    Country,
    SubCountry,
)

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb

logger = logging.getLogger(__name__)
Type = Literal["Mean"]

FUNCTIONS = {"Mean": np.mean}
BASELINE_DATE = datetime.date.today() - datetime.timedelta(days=60)


def compute_normalized_close(close_: pd.Series) -> pd.Series:
    close = close_.copy()
    close.index = close.index.tz_localize(None)  # type: ignore
    closest_ts = close.index[
        close.index.get_indexer([BASELINE_DATE], method="nearest")[0]
    ]
    normalized_close = (close / close.loc[closest_ts]).rename("normalized_close")
    normalized_close.index = close_.index
    return normalized_close  # type: ignore


def get_industry_comparison_data(
    bullish_db: "BullishDb",
    symbol_data: pd.DataFrame,
    type: Type,
    industry: Industry,
    country: Country,
) -> pd.DataFrame:
    try:

        views = bullish_db.read_returns(type, industry, country)
        industry_data = IndustryViews.from_views(views).to_dataframe()
        normalized_symbol = compute_normalized_close(symbol_data.close).rename("symbol")
        normalized_industry = industry_data.normalized_close.rename(industry)
        data = [normalized_symbol, normalized_industry]
        for country in get_args(SubCountry):
            views = bullish_db.read_returns(type, industry, country)
            if views:
                industry_data = IndustryViews.from_views(views).to_dataframe()
                normalized_industry = industry_data.normalized_close.rename(
                    f"{industry}-{country}"
                )
                data.append(normalized_industry)
        return pd.concat(data, axis=1)
    except Exception as e:
        logger.error(e)
        return pd.DataFrame()


class PricesReturns(Prices):  # type: ignore

    def returns(self) -> pd.DataFrame:
        try:
            data = self.to_dataframe()
            data["simple_return"] = data.close.pct_change() * 100
            data["log_return"] = (data.close / data.close.shift(1)).apply(np.log) * 100
            data["normalized_close"] = compute_normalized_close(data.close)
            return data[["simple_return", "log_return", "normalized_close"]]  # type: ignore
        except Exception:
            return pd.DataFrame(
                columns=["simple_return", "log_return", "normalized_close"]
            )


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


class Basedate(BaseModel):
    date: datetime.date


class KPI(BaseModel):
    simple_return: Annotated[float, BeforeValidator(to_float), Field(None)]
    log_return: Annotated[float, BeforeValidator(to_float), Field(None)]
    normalized_close: Annotated[float, BeforeValidator(to_float), Field(None)]


class BaseIndustryView(Basedate, KPI): ...


class IndustryView(BaseIndustryView):
    created_at: datetime.date
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
    ) -> List["IndustryView"]:
        function = FUNCTIONS[function_name]
        data_ = []
        for field in KPI.model_fields:

            data__ = (
                data[field].apply(function, axis=1).rename(field)
                if data[[field]].shape[1] > 1
                else data[field]
            )

            data_.append(data__)

        data_final = pd.concat(data_, axis=1)
        data_final["date"] = data_final.index
        return [
            cls.model_validate(
                r | {"industry": industry, "type": function_name, "country": country}
            )
            for r in data_final.to_dict(orient="records")
        ]

    @classmethod
    def from_db(
        cls, bullish: "BullishDb", industry: Industry, country: Country
    ) -> List["IndustryView"]:
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
        raw_data = [r for r in raw_data if not r.empty]

        if raw_data:
            data_ = pd.concat(raw_data, axis=1)
            for function_name in FUNCTIONS:
                returns.extend(cls.from_data(data_, function_name, industry, country))  # type: ignore
        return returns


class IndustryViews(BaseModel):
    views: List[IndustryView]

    def to_dataframe(self) -> pd.DataFrame:
        data = pd.DataFrame.from_records(
            [
                p.model_dump(include=set(BaseIndustryView.model_fields))
                for p in self.views
            ]
        )
        if data.empty:
            return data
        data = data.set_index("date", inplace=False)
        data = data.sort_index(inplace=False)

        data.index = pd.to_datetime(data.index, utc=True)
        data = data[~data.index.duplicated(keep="first")]
        return data

    @classmethod
    def from_views(cls, views: List[IndustryView]) -> "IndustryViews":
        return cls(views=views)


def compute_industry_view(bullish: "BullishDb") -> None:
    for country in get_args(Country):
        for industry in get_args(Industry):
            returns = IndustryView.from_db(bullish, industry, country)
            if returns:
                bullish.write_returns(returns)
