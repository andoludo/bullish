import logging
import time
from itertools import batched, chain
from pathlib import Path
from typing import (
    Annotated,
    Any,
    List,
    Optional,
    Sequence,
    Type,
    get_args,
    TYPE_CHECKING,
    ClassVar,
)

import pandas as pd
from bearish.interface.interface import BearishDbBase  # type: ignore
from bearish.models.assets.equity import BaseEquity  # type: ignore
from bearish.models.base import (  # type: ignore
    DataSourceBase,
    Ticker,
    PriceTracker,
    TrackerQuery,
    FinancialsTracker,
)
from bearish.models.financials.balance_sheet import (  # type: ignore
    BalanceSheet,
    QuarterlyBalanceSheet,
)
from bearish.models.financials.base import Financials, FinancialsWithDate  # type: ignore
from bearish.models.financials.cash_flow import (  # type: ignore
    CashFlow,
    QuarterlyCashFlow,
)
from bearish.models.financials.metrics import (  # type: ignore
    FinancialMetrics,
    QuarterlyFinancialMetrics,
)
from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from bearish.types import TickerOnlySources  # type: ignore
from pydantic import BaseModel, BeforeValidator, Field, create_model

from bullish.analysis.indicators import Indicators, IndicatorModels, SignalSeries
from joblib import Parallel, delayed  # type: ignore

from bullish.analysis.industry_views import compute_industry_view

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb

QUARTERLY = "quarterly"
logger = logging.getLogger(__name__)


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


def _load_data(
    data: Sequence[DataSourceBase], symbol: str, class_: Type[DataSourceBase]
) -> pd.DataFrame:
    try:
        records = pd.DataFrame.from_records(
            [f.model_dump() for f in data if f.symbol == symbol]
        )
        return records.set_index("date").sort_index()
    except Exception as e:
        logger.warning(f"Failed to load data from {symbol}: {e}")
        columns = list(class_.model_fields)
        return pd.DataFrame(columns=columns).sort_index()


def _compute_growth(series: pd.Series) -> bool:
    if series.empty:
        return False
    return all(series.pct_change(fill_method=None).dropna() > 0)


def _all_positive(series: pd.Series) -> bool:
    if series.empty:
        return False
    return all(series.dropna() > 0)


def _get_last(data: pd.Series) -> Optional[float]:
    return data.iloc[-1] if not data.empty else None


def _abs(data: pd.Series) -> pd.Series:
    try:
        return abs(data)
    except Exception as e:
        logger.warning(f"Failed to compute absolute value: {e}")
        return data


class TechnicalAnalysisBase(BaseModel):
    _description: ClassVar[str] = "General technical indicators"
    last_price: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
        ),
    ]
    max_year_loss: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
        ),
    ]


TechnicalAnalysisModels = [*IndicatorModels, TechnicalAnalysisBase]


class TechnicalAnalysis(*TechnicalAnalysisModels):  # type: ignore

    @classmethod
    def from_data(cls, prices: pd.DataFrame, ticker: Ticker) -> "TechnicalAnalysis":
        if "close" not in prices.columns:
            logger.warning(
                f"Ticker {ticker.symbol} does not have valid 'close' values.",
                exc_info=True,
            )
            return cls()
        try:
            res = Indicators().compute(prices)
            last_price = prices.close.iloc[-1]
            max_price = prices.close.iloc[-253 * 2 :].max()
            max_year_loss = (max_price - last_price) / max_price
            return cls(last_price=last_price, max_year_loss=max_year_loss, **res)
        except Exception as e:
            logger.error(
                f"Failing to calculate technical analysis for {ticker.symbol}: {e}",
                exc_info=True,
            )
            return cls()


class BaseFundamentalAnalysis(BaseModel):
    positive_debt_to_equity: Optional[bool] = Field(
        None,
        description="True if the company's debt-to-equity ratio is favorable (typically low or improving).",
    )
    positive_return_on_assets: Optional[bool] = Field(
        None,
        description="True if the company reports a positive return on assets (ROA), "
        "indicating efficient use of its assets.",
    )
    positive_return_on_equity: Optional[bool] = Field(
        None,
        description="True if the return on equity (ROE) is positive, "
        "showing profitability relative to shareholder equity.",
    )
    positive_diluted_eps: Optional[bool] = Field(
        None,
        description="True if the diluted earnings per share (EPS), "
        "which includes the effect of convertible securities, is positive.",
    )
    positive_basic_eps: Optional[bool] = Field(
        None,
        description="True if the basic earnings per share (EPS) is positive, reflecting profitable operations.",
    )
    growing_basic_eps: Optional[bool] = Field(
        None,
        description="True if the basic EPS has shown consistent growth over a defined time period.",
    )
    growing_diluted_eps: Optional[bool] = Field(
        None,
        description="True if the diluted EPS has consistently increased over time.",
    )
    positive_net_income: Optional[bool] = Field(
        None,
        description="True if the net income is positive, indicating overall profitability.",
    )
    positive_operating_income: Optional[bool] = Field(
        None,
        description="True if the company has positive operating income from its core business operations.",
    )
    growing_net_income: Optional[bool] = Field(
        None, description="True if net income has shown consistent growth over time."
    )
    growing_operating_income: Optional[bool] = Field(
        None,
        description="True if the operating income has consistently increased over a period.",
    )
    positive_free_cash_flow: Optional[bool] = Field(
        None,
        description="True if the company has positive free cash flow, indicating financial flexibility and health.",
    )
    growing_operating_cash_flow: Optional[bool] = Field(
        None,
        description="True if the company's operating cash flow is growing steadily.",
    )
    operating_cash_flow_is_higher_than_net_income: Optional[bool] = Field(
        None,
        description="True if the operating cash flow exceeds net income, often a sign of high-quality earnings.",
    )

    # Capital Expenditure Ratios
    mean_capex_ratio: Optional[float] = Field(
        None,
        description="Average capital expenditure (CapEx) ratio, usually "
        "calculated as CapEx divided by revenue or operating cash flow.",
    )
    max_capex_ratio: Optional[float] = Field(
        None, description="Maximum observed CapEx ratio over the evaluation period."
    )
    min_capex_ratio: Optional[float] = Field(
        None, description="Minimum observed CapEx ratio over the evaluation period."
    )

    # Dividend Payout Ratios
    mean_dividend_payout_ratio: Optional[float] = Field(
        None,
        description="Average dividend payout ratio, representing the proportion of earnings paid out as dividends.",
    )
    max_dividend_payout_ratio: Optional[float] = Field(
        None, description="Maximum dividend payout ratio observed over the period."
    )
    min_dividend_payout_ratio: Optional[float] = Field(
        None, description="Minimum dividend payout ratio observed over the period."
    )

    # EPS Value
    earning_per_share: Optional[float] = Field(
        None,
        description="The latest or most relevant value of earnings per share (EPS), indicating net income per share.",
    )

    def is_empty(self) -> bool:
        return all(getattr(self, field) is None for field in self.model_fields)

    @classmethod
    def from_financials(
        cls, financials: "Financials", ticker: Ticker
    ) -> "BaseFundamentalAnalysis":
        return cls._from_financials(
            balance_sheets=financials.balance_sheets,
            financial_metrics=financials.financial_metrics,
            cash_flows=financials.cash_flows,
            ticker=ticker,
        )

    @classmethod
    def _from_financials(
        cls,
        balance_sheets: List[BalanceSheet] | List[QuarterlyBalanceSheet],
        financial_metrics: List[FinancialMetrics] | List[QuarterlyFinancialMetrics],
        cash_flows: List[CashFlow] | List[QuarterlyCashFlow],
        ticker: Ticker,
    ) -> "BaseFundamentalAnalysis":
        try:
            symbol = ticker.symbol

            balance_sheet = _load_data(balance_sheets, symbol, BalanceSheet)
            financial = _load_data(financial_metrics, symbol, FinancialMetrics)
            cash_flow = _load_data(cash_flows, symbol, CashFlow)

            # Debt-to-equity
            debt_to_equity = (
                balance_sheet.total_liabilities / balance_sheet.total_shareholder_equity
            ).dropna()
            positive_debt_to_equity = _all_positive(debt_to_equity)

            # Add relevant balance sheet data to financials
            financial["total_shareholder_equity"] = balance_sheet[
                "total_shareholder_equity"
            ]
            financial["common_stock_shares_outstanding"] = balance_sheet[
                "common_stock_shares_outstanding"
            ]

            # EPS and income checks
            earning_per_share = _get_last(
                (
                    financial.net_income / financial.common_stock_shares_outstanding
                ).dropna()
            )
            positive_net_income = _all_positive(financial.net_income)
            positive_operating_income = _all_positive(financial.operating_income)
            growing_net_income = _compute_growth(financial.net_income)
            growing_operating_income = _compute_growth(financial.operating_income)
            positive_diluted_eps = _all_positive(financial.diluted_eps)
            positive_basic_eps = _all_positive(financial.basic_eps)
            growing_basic_eps = _compute_growth(financial.basic_eps)
            growing_diluted_eps = _compute_growth(financial.diluted_eps)

            # Profitability ratios
            return_on_equity = (
                financial.net_income * 100 / financial.total_shareholder_equity
            ).dropna()
            return_on_assets = (
                financial.net_income * 100 / balance_sheet.total_assets
            ).dropna()
            positive_return_on_assets = _all_positive(return_on_assets)
            positive_return_on_equity = _all_positive(return_on_equity)
            # Cash flow analysis
            cash_flow["net_income"] = financial["net_income"]
            free_cash_flow = (
                cash_flow["operating_cash_flow"] - cash_flow["capital_expenditure"]
            )
            positive_free_cash_flow = _all_positive(free_cash_flow)
            growing_operating_cash_flow = _compute_growth(
                cash_flow["operating_cash_flow"]
            )
            operating_income_net_income = cash_flow[
                ["operating_cash_flow", "net_income"]
            ].dropna()
            operating_cash_flow_is_higher_than_net_income = all(
                operating_income_net_income["operating_cash_flow"]
                >= operating_income_net_income["net_income"]
            )
            cash_flow["capex_ratio"] = (
                cash_flow["capital_expenditure"] / cash_flow["operating_cash_flow"]
            ).dropna()
            mean_capex_ratio = cash_flow["capex_ratio"].mean()
            max_capex_ratio = cash_flow["capex_ratio"].max()
            min_capex_ratio = cash_flow["capex_ratio"].min()
            dividend_payout_ratio = (
                _abs(cash_flow["cash_dividends_paid"]) / free_cash_flow
            ).dropna()
            mean_dividend_payout_ratio = dividend_payout_ratio.mean()
            max_dividend_payout_ratio = dividend_payout_ratio.max()
            min_dividend_payout_ratio = dividend_payout_ratio.min()

            return cls(
                earning_per_share=earning_per_share,
                positive_debt_to_equity=positive_debt_to_equity,
                positive_return_on_assets=positive_return_on_assets,
                positive_return_on_equity=positive_return_on_equity,
                growing_net_income=growing_net_income,
                growing_operating_income=growing_operating_income,
                positive_diluted_eps=positive_diluted_eps,
                positive_basic_eps=positive_basic_eps,
                growing_basic_eps=growing_basic_eps,
                growing_diluted_eps=growing_diluted_eps,
                positive_net_income=positive_net_income,
                positive_operating_income=positive_operating_income,
                positive_free_cash_flow=positive_free_cash_flow,
                growing_operating_cash_flow=growing_operating_cash_flow,
                operating_cash_flow_is_higher_than_net_income=operating_cash_flow_is_higher_than_net_income,
                mean_capex_ratio=mean_capex_ratio,
                max_capex_ratio=max_capex_ratio,
                min_capex_ratio=min_capex_ratio,
                mean_dividend_payout_ratio=mean_dividend_payout_ratio,
                max_dividend_payout_ratio=max_dividend_payout_ratio,
                min_dividend_payout_ratio=min_dividend_payout_ratio,
            )
        except Exception as e:
            logger.error(
                f"Failed to compute fundamental analysis for {ticker}: {e}",
                exc_info=True,
            )
            return cls()


class YearlyFundamentalAnalysis(BaseFundamentalAnalysis): ...


fields_with_prefix = {
    f"{QUARTERLY}_{name}": (
        field_info.annotation,
        Field(default=None, description=field_info.description),
    )
    for name, field_info in BaseFundamentalAnalysis.model_fields.items()
}

# Create the new model
BaseQuarterlyFundamentalAnalysis = create_model(  # type: ignore
    "BaseQuarterlyFundamentalAnalysis", **fields_with_prefix
)


class QuarterlyFundamentalAnalysis(BaseQuarterlyFundamentalAnalysis):  # type: ignore
    @classmethod
    def from_quarterly_financials(
        cls, financials: "Financials", ticker: Ticker
    ) -> "QuarterlyFundamentalAnalysis":
        base_financial_analisys = BaseFundamentalAnalysis._from_financials(
            balance_sheets=financials.quarterly_balance_sheets,
            financial_metrics=financials.quarterly_financial_metrics,
            cash_flows=financials.quarterly_cash_flows,
            ticker=ticker,
        )
        return cls.model_validate({f"{QUARTERLY}_{k}": v for k, v in base_financial_analisys.model_dump().items()})  # type: ignore


class FundamentalAnalysis(YearlyFundamentalAnalysis, QuarterlyFundamentalAnalysis):
    @classmethod
    def from_financials(
        cls, financials: Financials, ticker: Ticker
    ) -> "FundamentalAnalysis":
        yearly_analysis = YearlyFundamentalAnalysis.from_financials(
            financials=financials, ticker=ticker
        )
        quarterly_analysis = QuarterlyFundamentalAnalysis.from_quarterly_financials(
            financials=financials, ticker=ticker
        )
        return FundamentalAnalysis.model_validate(
            yearly_analysis.model_dump() | quarterly_analysis.model_dump()
        )

    @classmethod
    def compute_series(
        cls, financials: FinancialsWithDate, ticker: Ticker
    ) -> List[SignalSeries]:
        fundamendal_analysis = FundamentalAnalysis.from_financials(financials, ticker)
        fundamental_analysis_ = fundamendal_analysis.model_dump(
            exclude_none=True, exclude_unset=True, exclude_defaults=True
        )
        fundamental_analysis_ = {
            k: v for k, v in fundamental_analysis_.items() if v is True
        }
        return [
            SignalSeries(
                name=k.upper(), symbol=ticker.symbol, value=v, date=financials.date
            )
            for k, v in fundamental_analysis_.items()
        ]


class AnalysisView(BaseModel):
    sector: Annotated[
        Optional[str],
        Field(
            None,
            description="Broad sector to which the company belongs, "
            "such as 'Real Estate' or 'Technology'",
        ),
    ]
    industry: Annotated[
        Optional[str],
        Field(
            None,
            description="Detailed industry categorization for the company, "
            "like 'Real Estate Management & Development'",
        ),
    ]
    market_capitalization: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
            description="Market capitalization value",
        ),
    ]
    country: Annotated[
        Optional[str],
        Field(None, description="Country where the company's headquarters is located"),
    ]
    symbol: str = Field(
        description="Unique ticker symbol identifying the company on the stock exchange"
    )
    name: Annotated[
        Optional[str],
        Field(None, description="Full name of the company"),
    ]
    price_per_earning_ratio: Optional[float] = None
    last_price: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
        ),
    ]
    max_year_loss: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
        ),
    ]
    median_yearly_growth: Optional[float] = None
    median_weekly_growth: Optional[float] = None
    median_monthly_growth: Optional[float] = None


class Analysis(AnalysisView, BaseEquity, TechnicalAnalysis, FundamentalAnalysis):  # type: ignore

    @classmethod
    def from_ticker(cls, bearish_db: BearishDbBase, ticker: Ticker) -> "Analysis":
        asset = bearish_db.read_assets(
            AssetQuery(
                symbols=Symbols(equities=[ticker]),
                excluded_sources=get_args(TickerOnlySources),
            )
        )
        equity = asset.get_one_equity()
        financials = Financials.from_ticker(bearish_db, ticker)
        fundamental_analysis = FundamentalAnalysis.from_financials(financials, ticker)
        prices = Prices.from_ticker(bearish_db, ticker)
        technical_analysis = TechnicalAnalysis.from_data(prices.to_dataframe(), ticker)
        return cls.model_validate(
            equity.model_dump()
            | fundamental_analysis.model_dump()
            | technical_analysis.model_dump()
            | {
                "price_per_earning_ratio": (
                    (
                        technical_analysis.last_price
                        / fundamental_analysis.earning_per_share
                    )
                    if technical_analysis.last_price is not None
                    and fundamental_analysis.earning_per_share != 0
                    and fundamental_analysis.earning_per_share is not None
                    else None
                )
            }
        )


def compute_financials_series(
    financials_: Financials, ticker: Ticker
) -> List[SignalSeries]:
    financials_with_dates = FinancialsWithDate.from_financials(financials_)
    series = []
    for f in financials_with_dates:
        series.extend(FundamentalAnalysis.compute_series(f, ticker))
    return series


def compute_analysis(database_path: Path, ticker: Ticker) -> Analysis:
    from bullish.database.crud import BullishDb

    bullish_db = BullishDb(database_path=database_path)
    return Analysis.from_ticker(bullish_db, ticker)


def compute_signal_series(database_path: Path, ticker: Ticker) -> List[SignalSeries]:
    from bullish.database.crud import BullishDb

    bullish_db = BullishDb(database_path=database_path)
    indicators = Indicators()
    prices = Prices.from_ticker(bullish_db, ticker)
    signal_series = indicators.compute_series(prices.to_dataframe(), ticker.symbol)
    financials = Financials.from_ticker(bullish_db, ticker)
    financial_series = compute_financials_series(financials, ticker)
    return signal_series + financial_series


def run_signal_series_analysis(bullish_db: "BullishDb") -> None:
    price_trackers = set(bullish_db._read_tracker(TrackerQuery(), PriceTracker))
    finance_trackers = set(bullish_db._read_tracker(TrackerQuery(), FinancialsTracker))
    tickers = list(price_trackers.intersection(finance_trackers))
    parallel = Parallel(n_jobs=-1)

    for batch_ticker in batched(tickers, 1):
        many_signal_series = parallel(
            delayed(compute_signal_series)(bullish_db.database_path, ticker)
            for ticker in batch_ticker
        )
        series = list(chain.from_iterable(many_signal_series))
        try:
            bullish_db.write_signal_series(series)
        except Exception as e:
            logger.error(f"Failed to compute signal series for {batch_ticker}: {e}")


def run_analysis(bullish_db: "BullishDb") -> None:
    compute_industry_view(bullish_db)
    price_trackers = set(bullish_db._read_tracker(TrackerQuery(), PriceTracker))
    finance_trackers = set(bullish_db._read_tracker(TrackerQuery(), FinancialsTracker))
    tickers = list(price_trackers.intersection(finance_trackers))
    parallel = Parallel(n_jobs=-1)

    for batch_ticker in batched(tickers, 1000):
        start = time.perf_counter()
        many_analysis = parallel(
            delayed(compute_analysis)(bullish_db.database_path, ticker)
            for ticker in batch_ticker
        )
        bullish_db.write_many_analysis(many_analysis)
        elapsed_time = time.perf_counter() - start
        print(
            f"Computed analysis for {len(batch_ticker)} tickers in {elapsed_time:.2f} seconds."
        )
