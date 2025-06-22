from functools import cached_property
from typing import Literal, Set, get_args, Dict, Any, Optional, List

from bearish.types import SeriesLength  # type: ignore
from pydantic import BaseModel, Field

Industry = Literal[
    "Food & Staples Retailing",
    "Packaged Foods",
    "Grocery Stores",
    "Household Products",
    "Household & Personal Products",
    "Confectioners",
    "Beverages",
    "Beverages - Non - Alcoholic",
    "Beverages - Wineries & Distilleries",
    "Pharmaceuticals",
    "Health Care Providers & Services",
    "Health Care Equipment & Supplies",
    "Healthcare Plans",
    "Medical Devices",
    "Medical Instruments & Supplies",
    "Medical Care Facilities",
    "Diagnostics & Research",
    "Drug Manufacturers - General",
    "Drug Manufacturers - Specialty & Generic",
    "Pharmaceutical Retailers",
    "Health Information Services",
    "Medical Distribution",
    "Electric Utilities",
    "Gas Utilities",
    "Water Utilities",
    "Utilities - Diversified",
    "Utilities - Regulated Electric",
    "Utilities - Regulated Gas",
    "Utilities - Renewable",
    "Utilities - Independent Power Producers",
    "Waste Management",
    "Pollution & Treatment Controls",
    "Security & Protection Services",
    "Insurance",
    "Insurance - Property & Casual",
]

Country = Literal[
    "Germany",
    "France",
    "Netherlands",
    "Belgium",
    "Italy",
    "Spain",
    "Switzerland",
    "Sweden",
    "Denmark",
    "Norway",
    "Finland",
    "Portugal",
    "Austria",
    "United states",
]
SIGNS = {
    "price_per_earning_ratio": "<",
    "market_capitalization": ">",
    "industry": " IN ",
    "country": " IN ",
}


class FilterQuery(BaseModel):
    positive_free_cash_flow: bool = Field(
        False, description="The username for the database."
    )
    positive_net_income: bool = False
    positive_operating_income: bool = False
    quarterly_positive_free_cash_flow: bool = False
    quarterly_positive_net_income: bool = False
    quarterly_positive_operating_income: bool = False
    growing_net_income: bool = False
    quarterly_operating_cash_flow_is_higher_than_net_income: bool = False
    operating_cash_flow_is_higher_than_net_income: bool = False
    rsi_last_value_exists: bool = False
    market_capitalization: int = Field(
        0, ge=0, multiple_of=1000, description="Positive integer with step count of 10."
    )
    price_per_earning_ratio: int = Field(
        0, ge=0, multiple_of=10, description="Positive integer with step count of 10."
    )
    industry: Set[Industry] = Field(None, description="Industry name.")  # type: ignore
    country: Set[Country] = Field(None, description="Country name.")  # type: ignore

    @cached_property
    def query_parameters(self) -> Dict[str, Any]:
        if not bool(self.industry):
            self.industry = tuple(get_args(Industry))  # type: ignore
        if not bool(self.country):
            self.country = tuple(get_args(Country))  # type: ignore
        return self.model_dump(exclude_defaults=True, exclude_unset=True)

    def to_query(self) -> str:
        query = " AND ".join(
            [f"{k}{SIGNS.get(k,'=')}{v}" for k, v in self.query_parameters.items()]
        )
        return query


class FilterQueryStored(FilterQuery):
    industry: Optional[List[Industry]] = None  # type: ignore
    country: Optional[List[Country]] = None  # type: ignore


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
