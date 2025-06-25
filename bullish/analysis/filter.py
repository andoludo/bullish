import datetime
from datetime import date
from typing import Literal, get_args, Any, Optional, List, Tuple, Type

from bearish.types import SeriesLength  # type: ignore
from pydantic import BaseModel, Field, ConfigDict
from pydantic import create_model
from pydantic.fields import FieldInfo

from bullish.analysis.analysis import (
    TechnicalAnalysis,
    YearlyFundamentalAnalysis,
    QuarterlyFundamentalAnalysis,
)

Industry = Literal[
    "Publishing",
    "Internet Retail",
    "Scientific & Technical Instruments",
    "Engineering & Construction",
    "Diagnostics & Research",
    "Software - Infrastructure",
    "Thermal Coal",
    "Software - Application",
    "Auto Manufacturers",
    "Farm Products",
    "Medical Devices",
    "Education & Training Services",
    "Auto Parts",
    "Specialty Chemicals",
    "Marine Shipping",
    "Biotechnology",
    "Real Estate Services",
    "Gold",
    "Entertainment",
    "Specialty Retail",
    "Utilities - Independent Power Producers",
    "Steel",
    "Mortgage Finance",
    "Communication Equipment",
    "Drug Manufacturers - Specialty & Generic",
    "Electronic Gaming & Multimedia",
    "Banks - Regional",
    "Oil & Gas E&P",
    "Travel Services",
    "Real Estate - Diversified",
    "Telecom Services",
    "Uranium",
    "Consulting Services",
    "Waste Management",
    "Agricultural Inputs",
    "Utilities - Diversified",
    "Auto & Truck Dealerships",
    "Confectioners",
    "Other Industrial Metals & Mining",
    "Beverages - Wineries & Distilleries",
    "Oil & Gas Midstream",
    "Recreational Vehicles",
    "Electrical Equipment & Parts",
    "Household & Personal Products",
    "Packaging & Containers",
    "REIT - Specialty",
    "Home Improvement Retail",
    "Electronic Components",
    "Asset Management",
    "Consumer Electronics",
    "Conglomerates",
    "Health Information Services",
    "Medical Instruments & Supplies",
    "Building Products & Equipment",
    "Information Technology Services",
    "Specialty Industrial Machinery",
    "Food Distribution",
    "Packaged Foods",
    "Rental & Leasing Services",
    "Medical Distribution",
    "Grocery Stores",
    "Advertising Agencies",
    "Beverages - Non - Alcoholic",
    "Apparel Manufacturing",
    "Oil & Gas Equipment & Services",
    "Coking Coal",
    "Industrial Distribution",
    "Restaurants",
    "Beverages - Brewers",
    "Chemicals",
    "Real Estate - Development",
    "Credit Services",
    "Tobacco",
    "Metal Fabrication",
    "Building Materials",
    "Residential Construction",
    "Specialty Business Services",
    "REIT - Hotel & Motel",
    "Internet Content & Information",
    "Lodging",
    "Furnishings, Fixtures & Appliances",
    "Airlines",
    "Computer Hardware",
    "Integrated Freight & Logistics",
    "Solar",
    "Capital Markets",
    "Leisure",
    "Airports & Air Services",
    "Aluminum",
    "Insurance Brokers",
    "Semiconductors",
    "REIT - Retail",
    "Luxury Goods",
    "Lumber & Wood Production",
    "REIT - Mortgage",
    "Semiconductor Equipment & Materials",
    "Aerospace & Defense",
    "Security & Protection Services",
    "Utilities - Renewable",
    "Utilities - Regulated Gas",
    "Apparel Retail",
    "Pollution & Treatment Controls",
    "Broadcasting",
    "Resorts & Casinos",
    "Other Precious Metals & Mining",
    "Financial Data & Stock Exchanges",
    "Footwear & Accessories",
    "Medical Care Facilities",
    "Electronics & Computer Distribution",
    "Gambling",
    "Tools & Accessories",
    "Insurance - Property & Casualty",
    "Utilities - Regulated Water",
    "Insurance - Specialty",
    "Personal Services",
    "Pharmaceutical Retailers",
    "Farm & Heavy Construction Machinery",
    "Utilities - Regulated Electric",
    "Department Stores",
    "Staffing & Employment Services",
    "Textile Manufacturing",
    "Silver",
    "REIT - Industrial",
    "REIT - Diversified",
    "Copper",
    "Business Equipment & Supplies",
    "Infrastructure Operations",
    "Trucking",
    "Insurance - Reinsurance",
    "Insurance - Diversified",
    "Drug Manufacturers - General",
    "Oil & Gas Drilling",
    "Banks - Diversified",
    "REIT - Residential",
    "Oil & Gas Refining & Marketing",
    "Shell Companies",
    "Financial Conglomerates",
    "Paper & Paper Products",
    "Insurance - Life",
    "REIT - Office",
    "Railroads",
    "Oil & Gas Integrated",
    "Healthcare Plans",
    "REIT - Healthcare Facilities",
    "Discount Stores",
]

IndustryGroup = Literal[
    "publishing",
    "internet-retail",
    "scientific-technical-instruments",
    "engineering-construction",
    "diagnostics-research",
    "software-infrastructure",
    "thermal-coal",
    "software-application",
    "auto-manufacturers",
    "farm-products",
    "medical-devices",
    "education-training-services",
    "auto-parts",
    "specialty-chemicals",
    "marine-shipping",
    "biotechnology",
    "real-estate-services",
    "gold",
    "entertainment",
    "specialty-retail",
    "utilities-independent-power-producers",
    "steel",
    "mortgage-finance",
    "communication-equipment",
    "drug-manufacturers-specialty-generic",
    "electronic-gaming-multimedia",
    "banks-regional",
    "oil-gas-e-p",
    "travel-services",
    "real-estate-diversified",
    "telecom-services",
    "uranium",
    "consulting-services",
    "waste-management",
    "agricultural-inputs",
    "utilities-diversified",
    "auto-truck-dealerships",
    "confectioners",
    "other-industrial-metals-mining",
    "beverages-wineries-distilleries",
    "oil-gas-midstream",
    "recreational-vehicles",
    "electrical-equipment-parts",
    "household-personal-products",
    "packaging-containers",
    "reit-specialty",
    "home-improvement-retail",
    "electronic-components",
    "asset-management",
    "consumer-electronics",
    "conglomerates",
    "health-information-services",
    "medical-instruments-supplies",
    "building-products-equipment",
    "information-technology-services",
    "specialty-industrial-machinery",
    "food-distribution",
    "packaged-foods",
    "rental-leasing-services",
    "medical-distribution",
    "grocery-stores",
    "advertising-agencies",
    "beverages-non-alcoholic",
    "apparel-manufacturing",
    "oil-gas-equipment-services",
    "coking-coal",
    "industrial-distribution",
    "restaurants",
    "beverages-brewers",
    "chemicals",
    "real-estate-development",
    "credit-services",
    "tobacco",
    "metal-fabrication",
    "building-materials",
    "residential-construction",
    "specialty-business-services",
    "reit-hotel-motel",
    "internet-content-information",
    "lodging",
    "furnishings-fixtures-appliances",
    "airlines",
    "computer-hardware",
    "integrated-freight-logistics",
    "solar",
    "capital-markets",
    "leisure",
    "airports-air-services",
    "aluminum",
    "insurance-brokers",
    "semiconductors",
    "reit-retail",
    "luxury-goods",
    "lumber-wood-production",
    "reit-mortgage",
    "semiconductor-equipment-materials",
    "aerospace-defense",
    "security-protection-services",
    "utilities-renewable",
    "utilities-regulated-gas",
    "apparel-retail",
    "pollution-treatment-controls",
    "broadcasting",
    "resorts-casinos",
    "other-precious-metals-mining",
    "financial-data-stock-exchanges",
    "footwear-accessories",
    "medical-care-facilities",
    "electronics-computer-distribution",
    "gambling",
    "tools-accessories",
    "insurance-property-casualty",
    "utilities-regulated-water",
    "insurance-specialty",
    "personal-services",
    "pharmaceutical-retailers",
    "farm-heavy-construction-machinery",
    "utilities-regulated-electric",
    "department-stores",
    "staffing-employment-services",
    "textile-manufacturing",
    "silver",
    "reit-industrial",
    "reit-diversified",
    "copper",
    "business-equipment-supplies",
    "infrastructure-operations",
    "trucking",
    "insurance-reinsurance",
    "insurance-diversified",
    "drug-manufacturers-general",
    "oil-gas-drilling",
    "banks-diversified",
    "reit-residential",
    "oil-gas-refining-marketing",
    "shell-companies",
    "financial-conglomerates",
    "paper-paper-products",
    "insurance-life",
    "reit-office",
    "railroads",
    "oil-gas-integrated",
    "healthcare-plans",
    "reit-healthcare-facilities",
    "discount-stores",
]

Sector = Literal[
    "Communication Services",
    "Consumer Cyclical",
    "Technology",
    "Industrials",
    "Healthcare",
    "Energy",
    "Consumer Defensive",
    "Basic Materials",
    "Real Estate",
    "Utilities",
    "Financial Services",
    "Conglomerates",
]

Country = Literal[
    "Australia",
    "China",
    "Japan",
    "United kingdom",
    "United states",
    "Poland",
    "Switzerland",
    "Canada",
    "Greece",
    "Spain",
    "Germany",
    "Indonesia",
    "Belgium",
    "France",
    "Netherlands",
    "British virgin islands",
    "Italy",
    "Hungary",
    "Austria",
    "Finland",
    "Sweden",
    "Bermuda",
    "Taiwan",
    "Israel",
    "Ukraine",
    "Singapore",
    "Jersey",
    "Ireland",
    "Luxembourg",
    "Cyprus",
    "Cayman islands",
    "Norway",
    "Denmark",
    "Hong kong",
    "New zealand",
    "Kazakhstan",
    "Nigeria",
    "Argentina",
    "Brazil",
    "Czech republic",
    "Mauritius",
    "South africa",
    "India",
    "Mexico",
    "Mongolia",
    "Slovenia",
    "Thailand",
    "Malaysia",
    "Costa rica",
    "Isle of man",
    "Egypt",
    "Turkey",
    "United arab emirates",
    "Colombia",
    "Gibraltar",
    "Malta",
    "Liechtenstein",
    "Guernsey",
    "Peru",
    "Estonia",
    "French guiana",
    "Portugal",
    "Uruguay",
    "Chile",
    "Martinique",
    "Monaco",
    "Panama",
    "Papua new guinea",
    "South korea",
    "Macau",
    "Gabon",
    "Romania",
    "Senegal",
    "Morocco",
    "Jordan",
    "Lithuania",
    "Dominican republic",
    "Reunion",
    "Zambia",
    "Cambodia",
    "Myanmar",
    "Bahamas",
    "Philippines",
    "Bangladesh",
    "Latvia",
    "Vietnam",
    "Iceland",
    "Azerbaijan",
    "Georgia",
    "Liberia",
    "Kenya",
]
SIZE_RANGE = 2


def _get_type(name: str, info: FieldInfo) -> Tuple[Any, Any]:
    alias = info.alias or " ".join(name.capitalize().split("_")).strip()
    if info.annotation == Optional[float]:  # type: ignore
        ge = next((item.ge for item in info.metadata if hasattr(item, "ge")), 0)
        le = next((item.le for item in info.metadata if hasattr(item, "le")), 100)
        return (Optional[List[float]], Field(default=[ge, le], alias=alias))
    elif info.annotation == Optional[date]:  # type: ignore
        le = date.today()
        ge = le - datetime.timedelta(days=30 * 12)  # 30 days * 12 months
        return (List[date], Field(default=[ge, le], alias=alias))
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

GROUP_MAPPING = {
    "income": INCOME_GROUP,
    "cash_flow": CASH_FLOW_GROUP,
    "eps": EPS_GROUP,
    "properties": PROPERTIES_GROUP,
    "country": get_args(Country),
    "industry": get_args(Industry),
    "industry_group": get_args(IndustryGroup),
    "sector": get_args(Sector),
}


def _create_fundamental_analysis_model() -> Type[BaseModel]:
    boolean_fields = {
        "income": (Optional[List[str]], Field(default=None)),
        "cash_flow": (Optional[List[str]], Field(default=None)),
        "eps": (Optional[List[str]], Field(default=None)),
        "properties": (Optional[List[str]], Field(default=None)),
    }
    remaining_fields = {
        name: _get_type(name, info)
        for name, info in {
            **YearlyFundamentalAnalysis.model_fields,
            **QuarterlyFundamentalAnalysis.model_fields,
        }.items()
        if info.annotation != Optional[bool]
    }
    return create_model(
        "FundamentalAnalysisFilter",
        __config__=ConfigDict(populate_by_name=True),
        **(boolean_fields | remaining_fields),
    )


TechnicalAnalysisFilter = create_model(  # type: ignore
    "TechnicalAnalysisFilter",
    __config__=ConfigDict(populate_by_name=True),
    **{
        name: _get_type(name, info)
        for name, info in TechnicalAnalysis.model_fields.items()
    },
)
FundamentalAnalysisFilter = _create_fundamental_analysis_model()


class GeneralFilter(BaseModel):
    country: Optional[List[str]] = None
    industry: Optional[List[str]] = None
    industry_group: Optional[List[str]] = None
    sector: Optional[List[str]] = None
    market_capitalization: Optional[List[float]] = Field(
        default_factory=lambda: [5e8, 1e12]
    )


class FilterQuery(GeneralFilter, TechnicalAnalysisFilter, FundamentalAnalysisFilter):  # type: ignore

    def valid(self) -> bool:
        return bool(self.model_dump(exclude_defaults=True, exclude_unset=True))

    def to_query(self) -> str:
        parameters = self.model_dump(exclude_defaults=True, exclude_unset=True)
        query = []
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
        return query_


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
