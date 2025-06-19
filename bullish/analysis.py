import logging
from datetime import date
from typing import (
    Annotated,
    Any,
    List,
    Optional,
    Sequence,
    Type,
    cast,
    get_args,
    TYPE_CHECKING,
)

import pandas as pd
import pandas_ta as ta  # type: ignore
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
from bearish.models.financials.base import Financials  # type: ignore
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


def price_growth(prices: pd.DataFrame, days: int, max: bool = False) -> Optional[float]:
    prices_ = prices.copy()
    last_index = prices_.last_valid_index()
    delta = pd.Timedelta(days=days)
    start_index = last_index - delta  # type: ignore

    try:
        closest_index = prices_.index.unique().asof(start_index)  # type: ignore
        price = (
            prices_.loc[closest_index].close
            if not max
            else prices_[closest_index:].close.max()
        )
    except Exception as e:
        logger.warning(
            f"""Failing to calculate price growth: {e}.""",
            exc_info=True,
        )
        return None
    return (  # type: ignore
        (prices_.loc[last_index].close - price) * 100 / prices_.loc[last_index].close
    )


def buy_opportunity(
    series_a: pd.Series, series_b: pd.Series  # type: ignore
) -> Optional[date]:
    sell = ta.cross(series_a=series_a, series_b=series_b)
    buy = ta.cross(series_a=series_b, series_b=series_a)
    if not buy[buy == 1].index.empty and not sell[sell == 1].index.empty:
        last_buy_signal = pd.Timestamp(buy[buy == 1].index[-1])
        last_sell_signal = pd.Timestamp(sell[sell == 1].index[-1])
        if last_buy_signal > last_sell_signal:
            return last_buy_signal
    return None


def perc(data: pd.Series) -> float:  # type: ignore
    return cast(float, ((data.iloc[-1] - data.iloc[0]) / data.iloc[0]) * 100)


def yoy(prices: pd.DataFrame) -> pd.Series:  # type: ignore
    return prices.close.resample("YE").apply(perc)  # type: ignore


def mom(prices: pd.DataFrame) -> pd.Series:  # type: ignore
    return prices.close.resample("ME").apply(perc)  # type: ignore


def wow(prices: pd.DataFrame) -> pd.Series:  # type: ignore
    return prices.close.resample("W").apply(perc)  # type: ignore


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


def _compute_growth(series: pd.Series) -> bool:  # type: ignore
    if series.empty:
        return False
    return all(series.pct_change(fill_method=None).dropna() > 0)


def _all_positive(series: pd.Series) -> bool:  # type: ignore
    if series.empty:
        return False
    return all(series.dropna() > 0)


def _get_last(data: pd.Series) -> Optional[float]:  # type: ignore
    return data.iloc[-1] if not data.empty else None


def _abs(data: pd.Series) -> pd.Series:  # type: ignore
    try:
        return abs(data)
    except Exception as e:
        logger.warning(f"Failed to compute absolute value: {e}")
        return data


class TechnicalAnalysis(BaseModel):
    rsi_last_value: Optional[float] = None
    macd_12_26_9_buy_date: Optional[date] = None
    ma_50_200_buy_date: Optional[date] = None
    slope_7: Optional[float] = None
    slope_14: Optional[float] = None
    slope_30: Optional[float] = None
    slope_60: Optional[float] = None
    last_adx: Optional[float] = None
    last_dmp: Optional[float] = None
    last_dmn: Optional[float] = None
    last_price: Annotated[
        Optional[float],
        BeforeValidator(to_float),
        Field(
            default=None,
        ),
    ]
    last_price_date: Annotated[
        Optional[date],
        Field(
            default=None,
        ),
    ]
    year_to_date_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_52_weeks_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_week_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_month_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_year_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    year_to_date_max_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_week_max_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_month_max_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    last_year_max_growth: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    macd_12_26_9_buy: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    star_yoy: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    star_wow: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]
    star_mom: Annotated[
        Optional[float],
        Field(
            default=None,
        ),
    ]

    @classmethod
    def from_data(cls, prices: pd.DataFrame) -> "TechnicalAnalysis":
        try:
            last_index = prices.last_valid_index()
            year_to_date_days = (
                last_index
                - pd.Timestamp(year=last_index.year, month=1, day=1, tz="UTC")  # type: ignore
            ).days
            year_to_date_growth = price_growth(prices, year_to_date_days)
            last_52_weeks_growth = price_growth(prices=prices, days=399)
            last_week_growth = price_growth(prices=prices, days=7)
            last_month_growth = price_growth(prices=prices, days=31)
            last_year_growth = price_growth(prices=prices, days=365)
            year_to_date_max_growth = price_growth(prices, year_to_date_days, max=True)
            last_week_max_growth = price_growth(prices=prices, days=7, max=True)
            last_month_max_growth = price_growth(prices=prices, days=31, max=True)
            last_year_max_growth = price_growth(prices=prices, days=365, max=True)
            prices.ta.sma(50, append=True)
            prices.ta.sma(200, append=True)
            prices.ta.adx(append=True)
            prices["SLOPE_14"] = ta.linreg(prices.close, slope=True, length=14)
            prices["SLOPE_7"] = ta.linreg(prices.close, slope=True, length=7)
            prices["SLOPE_30"] = ta.linreg(prices.close, slope=True, length=30)
            prices["SLOPE_60"] = ta.linreg(prices.close, slope=True, length=60)
            prices.ta.macd(append=True)
            prices.ta.rsi(append=True)

            rsi_last_value = prices.RSI_14.iloc[-1]
            macd_12_26_9_buy_date = buy_opportunity(
                prices.MACDs_12_26_9, prices.MACD_12_26_9
            )
            star_yoy = yoy(prices).median()
            star_mom = mom(prices).median()
            star_wow = wow(prices).median()
            try:
                macd_12_26_9_buy = (
                    prices.MACD_12_26_9.iloc[-1] > prices.MACDs_12_26_9.iloc[-1]
                )
            except Exception as e:
                logger.warning(
                    f"Failing to calculate MACD buy date: {e}", exc_info=True
                )
                macd_12_26_9_buy = None
            ma_50_200_buy_date = buy_opportunity(prices.SMA_200, prices.SMA_50)
            return cls(
                rsi_last_value=rsi_last_value,
                macd_12_26_9_buy_date=macd_12_26_9_buy_date,
                macd_12_26_9_buy=macd_12_26_9_buy,
                ma_50_200_buy_date=ma_50_200_buy_date,
                last_price=prices.close.iloc[-1],
                last_price_date=prices.index[-1],
                last_adx=prices.ADX_14.iloc[-1],
                last_dmp=prices.DMP_14.iloc[-1],
                last_dmn=prices.DMN_14.iloc[-1],
                slope_7=prices.SLOPE_7.iloc[-1],
                slope_14=prices.SLOPE_14.iloc[-1],
                slope_30=prices.SLOPE_30.iloc[-1],
                slope_60=prices.SLOPE_60.iloc[-1],
                year_to_date_growth=year_to_date_growth,
                last_52_weeks_growth=last_52_weeks_growth,
                last_week_growth=last_week_growth,
                last_month_growth=last_month_growth,
                last_year_growth=last_year_growth,
                year_to_date_max_growth=year_to_date_max_growth,
                last_week_max_growth=last_week_max_growth,
                last_month_max_growth=last_month_max_growth,
                last_year_max_growth=last_year_max_growth,
                star_yoy=star_yoy,
                star_mom=star_mom,
                star_wow=star_wow,
            )
        except Exception as e:
            logger.error(f"Failing to calculate technical analysis: {e}", exc_info=True)
            return cls()  # type: ignore


class BaseFundamentalAnalysis(BaseModel):
    positive_free_cash_flow: Optional[float] = None
    growing_operating_cash_flow: Optional[float] = None
    operating_cash_flow_is_higher_than_net_income: Optional[float] = None
    mean_capex_ratio: Optional[float] = None
    max_capex_ratio: Optional[float] = None
    min_capex_ratio: Optional[float] = None
    mean_dividend_payout_ratio: Optional[float] = None
    max_dividend_payout_ratio: Optional[float] = None
    min_dividend_payout_ratio: Optional[float] = None
    positive_net_income: Optional[float] = None
    positive_operating_income: Optional[float] = None
    growing_net_income: Optional[float] = None
    growing_operating_income: Optional[float] = None
    positive_diluted_eps: Optional[float] = None
    positive_basic_eps: Optional[float] = None
    growing_basic_eps: Optional[float] = None
    growing_diluted_eps: Optional[float] = None
    positive_debt_to_equity: Optional[float] = None
    positive_return_on_assets: Optional[float] = None
    positive_return_on_equity: Optional[float] = None
    earning_per_share: Optional[float] = None

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


class YearlyFundamentalAnalysis(BaseFundamentalAnalysis):
    ...


fields_with_prefix = {
    f"{QUARTERLY}_{name}": (Optional[float], Field(default=None))
    for name in BaseFundamentalAnalysis.model_fields
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
        return cls.model_validate({f"{QUARTERLY}_{k}": v for k, v in base_financial_analisys.model_dump().items()})  # type: ignore # noqa: E501


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


class Analysis(AnalysisView, BaseEquity, TechnicalAnalysis, FundamentalAnalysis):  # type: ignore
    price_per_earning_ratio: Optional[float] = None

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
        technical_analysis = TechnicalAnalysis.from_data(prices.to_dataframe())
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


def run_analysis(bullish_db: "BullishDb") -> None:
    price_trackers = set(bullish_db._read_tracker(TrackerQuery(), PriceTracker))
    finance_trackers = set(bullish_db._read_tracker(TrackerQuery(), FinancialsTracker))
    tickers = list(price_trackers.intersection(finance_trackers))
    for ticker in tickers:
        analysis = Analysis.from_ticker(bullish_db, ticker)
        bullish_db.write_analysis(analysis)
