import datetime
from datetime import date
from typing import get_args, Any, Optional, List, Tuple, Type, Dict

from bearish.types import SeriesLength  # type: ignore
from pydantic import BaseModel, Field, ConfigDict
from pydantic import create_model
from pydantic.fields import FieldInfo

from bullish.analysis.analysis import (
    YearlyFundamentalAnalysis,
    QuarterlyFundamentalAnalysis,
    TechnicalAnalysisModels,
    TechnicalAnalysis,
    AnalysisView,
)
from bullish.analysis.constants import Industry, IndustryGroup, Sector, Country

SIZE_RANGE = 2


def _get_type(name: str, info: FieldInfo) -> Tuple[Any, Any]:
    alias = info.alias or " ".join(name.capitalize().split("_")).strip()
    if info.annotation == Optional[float]:  # type: ignore
        ge = next((item.ge for item in info.metadata if hasattr(item, "ge")), 0)
        le = next((item.le for item in info.metadata if hasattr(item, "le")), 100)
        default = [ge, le]
        return (
            Optional[List[float]],
            Field(default=default, alias=alias, description=info.description),
        )
    elif info.annotation == Optional[date]:  # type: ignore
        le = date.today()
        ge = le - datetime.timedelta(days=30 * 2)  # 30 days * 12 months
        return (
            List[date],
            Field(default=[ge, le], alias=alias, description=info.description),
        )
    else:
        raise NotImplementedError


FUNDAMENTAL_ANALYSIS_GROUP = ["income", "cash_flow", "eps"]


def _get_fundamental_analysis_boolean_fields() -> List[str]:
    return [
        name
        for name, info in {
            **YearlyFundamentalAnalysis.model_fields,
            **QuarterlyFundamentalAnalysis.model_fields,
        }.items()
        if info.annotation == Optional[bool]
    ]


def _get_technical_analysis_float_fields() -> List[str]:
    return [
        name
        for name, info in (
            TechnicalAnalysis.model_fields | AnalysisView.model_fields
        ).items()
        if info.annotation == Optional[float]
    ]


def get_boolean_field_group(group: str) -> List[str]:
    groups = FUNDAMENTAL_ANALYSIS_GROUP.copy()
    groups.remove(group)
    return [
        name
        for name in _get_fundamental_analysis_boolean_fields()
        if group in name and not any(g in name for g in groups)
    ]


INCOME_GROUP = get_boolean_field_group("income")
CASH_FLOW_GROUP = get_boolean_field_group("cash_flow")
EPS_GROUP = get_boolean_field_group("eps")
PROPERTIES_GROUP = list(
    set(_get_fundamental_analysis_boolean_fields()).difference(
        {*INCOME_GROUP, *CASH_FLOW_GROUP, *EPS_GROUP}
    )
)
BOOLEAN_GROUP_MAPPING: Dict[str, List[str]] = {
    "income": INCOME_GROUP,
    "cash_flow": CASH_FLOW_GROUP,
    "eps": EPS_GROUP,
    "properties": PROPERTIES_GROUP,
}
GROUP_MAPPING: Dict[str, List[str]] = {
    **BOOLEAN_GROUP_MAPPING,
    "properties": PROPERTIES_GROUP,
    "country": list(get_args(Country)),
    "industry": list(get_args(Industry)),
    "industry_group": list(get_args(IndustryGroup)),
    "sector": list(get_args(Sector)),
    "symbol": [],
    "order_by_asc": _get_technical_analysis_float_fields(),
    "order_by_desc": _get_technical_analysis_float_fields(),
}


def _create_fundamental_analysis_models() -> List[Type[BaseModel]]:
    models = []
    boolean_fields = {
        "income": (Optional[List[str]], Field(default=None, description="Income")),
        "cash_flow": (
            Optional[List[str]],
            Field(default=None, description="Cash flow"),
        ),
        "eps": (
            Optional[List[str]],
            Field(default=None, description="Earnings per share"),
        ),
        "properties": (
            Optional[List[str]],
            Field(default=None, description="General properties"),
        ),
    }
    yearly_fields = {
        name: _get_type(name, info)
        for name, info in YearlyFundamentalAnalysis.model_fields.items()
        if info.annotation != Optional[bool]  # type: ignore
    }
    quarterly_fields = {
        name: _get_type(name, info)
        for name, info in QuarterlyFundamentalAnalysis.model_fields.items()
        if info.annotation != Optional[bool]
    }
    for property in [
        (boolean_fields, "Selection filter", "SelectionFilter"),
        (yearly_fields, "Yearly properties", "YearlyFilter"),
        (quarterly_fields, "Quarterly properties", "QuarterlyFilter"),
    ]:
        model_ = create_model(  # type: ignore
            property[-1],
            __config__=ConfigDict(populate_by_name=True),
            **property[0],
        )
        model_._description = property[1]
        models.append(model_)

    return models


def create_technical_analysis_models() -> List[Type[BaseModel]]:
    models = []
    for model in TechnicalAnalysisModels:
        model_ = create_model(  # type: ignore
            f"{model.__name__}Filter",  # type: ignore
            __config__=ConfigDict(populate_by_name=True),
            **{
                name: _get_type(name, info) for name, info in model.model_fields.items()  # type: ignore
            },
        )

        model_._description = model._description  # type: ignore
        models.append(model_)
    return models


TechnicalAnalysisFilters = create_technical_analysis_models()
FundamentalAnalysisFilters = _create_fundamental_analysis_models()


class GeneralFilter(BaseModel):
    country: Optional[List[str]] = None
    order_by_asc: Optional[str] = None
    order_by_desc: Optional[str] = None
    industry: Optional[List[str]] = None
    industry_group: Optional[List[str]] = None
    sector: Optional[List[str]] = None
    symbol: Optional[List[str]] = None
    market_capitalization: Optional[List[float]] = Field(default=[5e8, 1e12])
    price_per_earning_ratio: Optional[List[float]] = Field(default=[0.0, 1000.0])


class FilterQuery(GeneralFilter, *TechnicalAnalysisFilters, *FundamentalAnalysisFilters):  # type: ignore

    def valid(self) -> bool:
        return any(
            bool(v)
            for _, v in self.model_dump(
                exclude_defaults=True, exclude_unset=True
            ).items()
        )

    def to_query(self) -> str:
        parameters = self.model_dump(exclude_defaults=True, exclude_unset=True)
        query = []
        order_by_desc = ""
        order_by_asc = ""
        for parameter, value in parameters.items():
            if not value:
                continue

            if (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and parameter not in GeneralFilter.model_fields
            ):
                query.append(" AND ".join([f"{v}=1" for v in value]))
            elif (
                isinstance(value, str) and bool(value) and parameter == "order_by_desc"
            ):
                order_by_desc = f"ORDER BY {value} DESC"
            elif isinstance(value, str) and bool(value) and parameter == "order_by_asc":
                order_by_asc = f"ORDER BY {value} ASC"
            elif (
                isinstance(value, list)
                and len(value) == SIZE_RANGE
                and all(isinstance(item, (int, float)) for item in value)
            ):
                query.append(f"{parameter} BETWEEN {value[0]} AND {value[1]}")
            elif (
                isinstance(value, list)
                and len(value) == SIZE_RANGE
                and all(isinstance(item, date) for item in value)
            ):
                query.append(f"{parameter} BETWEEN '{value[0]}' AND '{value[1]}'")
            elif (
                isinstance(value, list)
                and all(isinstance(item, str) for item in value)
                and parameter in GeneralFilter.model_fields
            ):
                general_filters = [f"'{v}'" for v in value]
                query.append(f"{parameter} IN ({', '.join(general_filters)})")
            else:
                raise NotImplementedError
        query_ = " AND ".join(query)
        return f"{query_} {order_by_desc.strip()} {order_by_asc.strip()}".strip()


class FilterQueryStored(FilterQuery): ...


class FilterUpdate(BaseModel):
    window_size: SeriesLength = Field("5d")
    data_age_in_days: int = 1
    update_financials: bool = False
    update_analysis_only: bool = False


class FilteredResults(BaseModel):
    name: str
    filter_query: FilterQueryStored
    symbols: list[str] = Field(
        default_factory=list, description="List of filtered tickers."
    )
