"""A small, deterministic bearish database for the KPI, scoring and MCP tests.

Twelve equities over three sectors and three countries, four years of daily
prices, five fiscal years and twelve quarters of statements, analyst and
ownership rows, filings, snapshots, three ETFs, one index, commodity prices
and a handful of macro series. Two symbols are shaped on purpose:

* ``QSALE`` is a sound business whose price fell 30 % in the last two months
  on no fundamental change, with insiders buying afterwards (quality on sale);
* ``KNIFE`` fell as much but restated its accounts, lost its CFO, raised
  equity and saw insiders sell (falling knife).
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bearish.database.crud import BearishDb
from bearish.macro.commodities import COMMODITIES
from bearish.macro.etf_exposure import write_etf_exposures
from bearish.macro.models import Observation
from bearish.macro.pipelines.linking import (
    write_country_references,
    write_sector_references,
)
from bearish.macro.sensitivities import write_sector_sensitivities
from bearish.coverage import write_coverage
from bearish.database.schemas import PriceCommodityORM, PriceEtfORM, PriceIndexORM
from bearish.models.assets.assets import Assets
from bearish.models.assets.equity import Equity
from bearish.models.assets.etfs import Etf
from bearish.models.assets.index import Index
from bearish.models.assets.index_members import IndexMember
from bearish.models.assets.snapshots import EquitySnapshot, EtfSnapshot, Snapshots
from bearish.models.financials.analyst import AnalystRecommendation, PriceTarget
from bearish.models.financials.balance_sheet import BalanceSheet, QuarterlyBalanceSheet
from bearish.models.financials.base import Financials
from bearish.models.financials.cash_flow import CashFlow, QuarterlyCashFlow
from bearish.models.financials.earnings_date import EarningsDate
from bearish.models.financials.metrics import (
    FinancialMetrics,
    QuarterlyFinancialMetrics,
)
from bearish.models.price.price import Price
from bearish.models.sec.filing import Filing
from bearish.models.sec.holdings import Holding13FSummary, Ownership13DG
from bearish.models.sec.insider import InsiderTransaction

AS_OF = datetime.date(2026, 10, 23)
YEARS = 4
SEED = 42


class YfPrice(Price):
    __source__ = "Yfinance"


# symbol: (name, yahoo sector, yahoo industry, country, currency, drift per year, market cap bn, shape)
UNIVERSE: dict[str, tuple[str, str, str, str, str, float, float, str]] = {
    "ACCEL": (
        "Accelerate Semis",
        "Technology",
        "Semiconductors",
        "United states",
        "USD",
        0.45,
        40.0,
        "leader",
    ),
    "STEADY": (
        "Steady Software",
        "Technology",
        "Software - Application",
        "United states",
        "USD",
        0.18,
        60.0,
        "compounder",
    ),
    "QSALE": (
        "Quality On Sale",
        "Technology",
        "Software - Infrastructure",
        "United states",
        "USD",
        0.15,
        25.0,
        "quality_on_sale",
    ),
    "KNIFE": (
        "Falling Knife",
        "Technology",
        "Electronic Components",
        "United states",
        "USD",
        0.05,
        8.0,
        "falling_knife",
    ),
    "MEDCO": (
        "Medco Devices",
        "Healthcare",
        "Medical Devices",
        "United states",
        "USD",
        0.10,
        30.0,
        "plain",
    ),
    "PHARM": (
        "Pharm Generic",
        "Healthcare",
        "Drug Manufacturers - Specialty & Generic",
        "United states",
        "USD",
        0.02,
        12.0,
        "plain",
    ),
    "OILCO": (
        "Oil Producer",
        "Energy",
        "Oil & Gas E&P",
        "United states",
        "USD",
        0.06,
        50.0,
        "plain",
    ),
    "DRILL": (
        "Drilling Services",
        "Energy",
        "Oil & Gas Equipment & Services",
        "United states",
        "USD",
        -0.05,
        5.0,
        "weak",
    ),
    "SAPDE": (
        "German Software",
        "Technology",
        "Software - Application",
        "Germany",
        "EUR",
        0.16,
        150.0,
        "compounder",
    ),
    "BAYDE": (
        "German Pharma",
        "Healthcare",
        "Drug Manufacturers - General",
        "Germany",
        "EUR",
        -0.02,
        30.0,
        "weak",
    ),
    "SOLBE": (
        "Belgian Chemicals",
        "Basic Materials",
        "Specialty Chemicals",
        "Belgium",
        "EUR",
        0.04,
        12.0,
        "plain",
    ),
    "UCBBE": (
        "Belgian Biotech",
        "Healthcare",
        "Biotechnology",
        "Belgium",
        "EUR",
        0.22,
        20.0,
        "leader",
    ),
}
ETFS: dict[str, dict[str, object]] = {
    "TECHX": {
        "name": "Tech Sector ETF",
        "weights": {"technology": 100.0},
        "expense": 0.10,
        "aum": 5e10,
        "category": "Technology",
    },
    "HLTHX": {
        "name": "Health Care ETF",
        "weights": {"healthcare": 100.0},
        "expense": 0.35,
        "aum": 2e9,
        "category": "Health",
    },
    "WORLDX": {
        "name": "World Equity ETF",
        "weights": {
            "technology": 40.0,
            "healthcare": 30.0,
            "energy": 15.0,
            "basic_materials": 15.0,
        },
        "expense": 0.20,
        "aum": 1e11,
        "category": "Global Large-Cap Blend Equity",
    },
}
MACRO: dict[str, tuple[float, float]] = {  # series_id: (start level, monthly drift)
    "VIX.USA": (18.0, 0.0),
    "HY_SPREAD.USA": (3.5, 0.0),
    "IG_SPREAD.USA": (1.0, 0.0),
    "Y10.USA": (4.2, 0.0),
    "Y2.USA": (3.9, 0.0),
    "CURVE_10_2.USA": (0.3, 0.0),
    "REAL_Y10.USA": (1.8, 0.0),
    "CLI.USA": (99.5, 0.05),
    "PMI_MFG.USA": (49.0, 0.2),
    "CPI_YOY.USA": (3.2, -0.05),
    "POLICY_RATE.USA": (4.5, -0.05),
    "RECESSION_PROB.USA": (5.0, 0.0),
    "DXY.USA": (104.0, -0.2),
    "FF_MOM.USA": (0.5, 0.0),
    "FF_HML.USA": (-0.2, 0.0),
    "FF_SMB.USA": (0.1, 0.0),
    "FF_RMW.USA": (0.3, 0.0),
    "CLI.DEU": (99.0, 0.03),
    "CPI_YOY.DEU": (2.5, -0.03),
    "Y10.DEU": (2.4, 0.0),
    "POLICY_RATE.DEU": (3.0, -0.05),
    "GDP_FCST.USA": (2.1, 0.0),
    "GDP_FCST.DEU": (0.8, 0.0),
}


def business_days(years: int = YEARS, end: datetime.date = AS_OF) -> pd.DatetimeIndex:
    return pd.bdate_range(end=end, periods=years * 252)


def _path(close: np.ndarray, shape: str) -> np.ndarray:
    n = len(close)
    if shape in {"quality_on_sale", "falling_knife"}:
        drop = np.linspace(1.0, 0.70, 42)  # 30 % slide over the last two months
        close[n - 42 :] = close[n - 43] * drop
    if shape == "weak":
        close[n - 252 :] *= np.linspace(1.0, 0.85, 252)
    if shape == "leader":
        close[n - 252 :] *= np.linspace(1.0, 1.35, 252)
    return close


def price_frames(rng: np.random.Generator) -> dict[str, pd.DataFrame]:
    dates = business_days()
    frames = {}
    for i, (symbol, spec) in enumerate(UNIVERSE.items()):
        drift, shape = spec[5], spec[7]
        daily = rng.normal(drift / 252, 0.018, len(dates))
        close = 50.0 * (1 + i * 0.1) * np.exp(np.cumsum(daily))
        close = _path(close, shape)
        volume = rng.integers(500_000, 5_000_000, len(dates)).astype(float)
        if shape in {"quality_on_sale", "falling_knife"}:
            volume[-42:] *= 2.5
        frames[symbol] = pd.DataFrame({"date": dates, "close": close, "volume": volume})
    return frames


def _prices(
    symbol: str, frame: pd.DataFrame, dividend_per_quarter: float = 0.0
) -> list[Price]:
    rows = []
    for k, (day, close, volume) in enumerate(
        zip(frame["date"], frame["close"], frame["volume"], strict=True)
    ):
        dividend = dividend_per_quarter if dividend_per_quarter and k % 63 == 0 else 0.0
        rows.append(
            YfPrice(
                symbol=symbol,
                date=day.date(),
                open=close,
                high=close * 1.01,
                low=close * 0.99,
                close=close,
                volume=volume,
                dividends=dividend,
                stock_splits=0.0,
            )
        )
    return rows


def fiscal_years(n: int = 5) -> list[datetime.date]:
    return [datetime.date(AS_OF.year - n + i, 12, 31) for i in range(n)]


def quarters(n: int = 12) -> list[datetime.date]:
    last = pd.Timestamp(AS_OF) - pd.offsets.QuarterEnd(1)
    return [(last - pd.offsets.QuarterEnd(k)).date() for k in range(n)][::-1]


def _statements(symbol: str, spec: tuple[str, ...], mcap: float) -> Financials:
    shape = spec[7]
    source = "SEC" if spec[3] == "United states" else "Yfinance"
    growth = {
        "leader": 0.35,
        "compounder": 0.14,
        "quality_on_sale": 0.16,
        "falling_knife": -0.10,
        "plain": 0.05,
        "weak": -0.04,
    }[shape]
    margin = {
        "leader": 0.22,
        "compounder": 0.28,
        "quality_on_sale": 0.26,
        "falling_knife": 0.02,
        "plain": 0.12,
        "weak": 0.06,
    }[shape]
    revenue0 = mcap * 1e9 / 6
    shares = 1e8
    years = fiscal_years()
    metrics, balance, cash = [], [], []
    for i, year in enumerate(years):
        revenue = revenue0 * (1 + growth) ** i
        m = margin + 0.01 * i if shape != "falling_knife" else margin - 0.02 * i
        operating = revenue * m
        net = operating * 0.8
        assets = revenue * 1.5
        equity = assets * 0.55
        debt = assets * (0.4 if shape == "falling_knife" else 0.15)
        cfo = net * (0.5 if shape == "falling_knife" else 1.15)
        capex = -revenue * 0.05
        filed = (pd.Timestamp(year) + pd.Timedelta(days=50)).date()
        share_count = shares * (1.12**i if shape == "falling_knife" else 0.99**i)
        metrics.append(
            FinancialMetrics(
                symbol=symbol,
                source=source,
                date=year,
                filed_date=filed,
                total_revenue=revenue,
                gross_profit=revenue
                * (0.6 if shape != "falling_knife" else 0.3 - 0.02 * i),
                operating_income=operating,
                net_income=net,
                ebitda=operating + revenue * 0.04,
                diluted_eps=net / share_count,
                basic_eps=net / share_count,
                tax_rate=0.21,
                interest_expense=debt * 0.05,
                research_and_development=revenue * 0.12,
                selling_general_and_administrative=revenue * 0.2,
                diluted_average_shares=share_count,
            )
        )
        balance.append(
            BalanceSheet(
                symbol=symbol,
                source=source,
                date=year,
                filed_date=filed,
                total_assets=assets,
                total_current_assets=assets * 0.4,
                total_current_liabilities=assets * 0.2,
                total_liabilities=assets - equity,
                total_shareholder_equity=equity,
                long_term_debt=debt,
                short_long_term_debt_total=debt,
                cash_and_short_term_investments=assets * 0.15,
                retained_earnings=equity * 0.6,
                common_stock_shares_outstanding=share_count,
            )
        )
        cash.append(
            CashFlow(
                symbol=symbol,
                source=source,
                date=year,
                filed_date=filed,
                operating_cash_flow=cfo,
                capital_expenditure=capex,
                depreciation_amortization_depletion=revenue * 0.04,
                cash_dividends_paid=(
                    -net * 0.3 * (1.05**i) if shape == "compounder" else 0.0
                ),
                proceeds_from_issuance_of_common_stock=(
                    mcap * 1e9 * 0.05
                    if shape == "falling_knife" and i == len(years) - 1
                    else 0.0
                ),
                stock_based_compensation=revenue * 0.03,
            )
        )
    q_metrics, q_balance, q_cash = [], [], []
    for k, quarter in enumerate(quarters()):
        q_growth = growth + (0.03 * (k - 8) if shape == "leader" else 0.0)
        revenue = revenue0 * (1 + q_growth) ** (k / 4) / 4
        m = margin + 0.003 * k if shape != "falling_knife" else margin - 0.01 * k
        operating = revenue * m
        net = operating * 0.8
        filed = (pd.Timestamp(quarter) + pd.Timedelta(days=40)).date()
        share_count = shares * (1.03**k if shape == "falling_knife" else 0.998**k)
        q_metrics.append(
            QuarterlyFinancialMetrics(
                symbol=symbol,
                source=source,
                date=quarter,
                filed_date=filed,
                total_revenue=revenue,
                gross_profit=revenue * 0.6,
                operating_income=operating,
                net_income=net,
                ebitda=operating + revenue * 0.04,
                diluted_eps=net / share_count,
                interest_expense=revenue * 0.01,
                research_and_development=revenue * 0.12,
                diluted_average_shares=share_count,
            )
        )
        assets = revenue * 6
        q_balance.append(
            QuarterlyBalanceSheet(
                symbol=symbol,
                source=source,
                date=quarter,
                filed_date=filed,
                total_assets=assets,
                total_current_assets=assets * 0.4,
                total_current_liabilities=assets * 0.2,
                total_liabilities=assets * 0.45,
                total_shareholder_equity=assets * 0.55,
                short_long_term_debt_total=assets * 0.15,
                cash_and_short_term_investments=assets * 0.15,
                retained_earnings=assets * 0.3,
                common_stock_shares_outstanding=share_count,
            )
        )
        q_cash.append(
            QuarterlyCashFlow(
                symbol=symbol,
                source=source,
                date=quarter,
                filed_date=filed,
                operating_cash_flow=net * (0.5 if shape == "falling_knife" else 1.15),
                capital_expenditure=-revenue * 0.05,
                depreciation_amortization_depletion=revenue * 0.04,
                stock_based_compensation=revenue * 0.03,
                cash_dividends_paid=-net * 0.3 if shape == "compounder" else 0.0,
                proceeds_from_issuance_of_common_stock=(
                    mcap * 1e9 * 0.05 if shape == "falling_knife" and k == 10 else 0.0
                ),
            )
        )
    return Financials(
        financial_metrics=metrics,
        balance_sheets=balance,
        cash_flows=cash,
        quarterly_financial_metrics=q_metrics,
        quarterly_balance_sheets=q_balance,
        quarterly_cash_flows=q_cash,
    )


def _expectations(
    symbol: str, shape: str, last_close: float
) -> tuple[list[EarningsDate], list[PriceTarget], list[AnalystRecommendation]]:
    earnings = []
    beats = {
        "leader": 0.08,
        "compounder": 0.03,
        "quality_on_sale": 0.04,
        "falling_knife": -0.12,
        "plain": 0.0,
        "weak": -0.04,
    }[shape]
    for k, quarter in enumerate(quarters(8)):
        report = (pd.Timestamp(quarter) + pd.Timedelta(days=30)).date()
        estimate = 1.0 + 0.05 * k
        earnings.append(
            EarningsDate(
                symbol=symbol,
                source="Yfinance",
                date=report,
                eps_estimate=estimate,
                eps_reported=estimate * (1 + beats),
                revenue_estimate=1e9,
                revenue_reported=1e9 * (1 + beats / 2),
                is_forward=False,
            )
        )
    earnings.append(
        EarningsDate(
            symbol=symbol,
            source="Yfinance",
            date=AS_OF + datetime.timedelta(days=20),
            eps_estimate=1.5,
            is_forward=True,
        )
    )
    targets = []
    trend = {
        "leader": 0.04,
        "compounder": 0.01,
        "quality_on_sale": 0.0,
        "falling_knife": -0.08,
        "plain": 0.0,
        "weak": -0.02,
    }[shape]
    for k in range(6):
        day = AS_OF - datetime.timedelta(days=30 * (5 - k))
        mean = (
            last_close * 1.2 * (1 + trend) ** k
            if shape != "quality_on_sale"
            else last_close * 1.45
        )
        targets.append(
            PriceTarget(
                symbol=symbol,
                source="Yfinance",
                date=day,
                target_mean=mean,
                target_high=mean * 1.2,
                target_low=mean * 0.8,
                target_median=mean,
                number_of_analysts=12,
                recommendation_mean=2.0 - trend * 5 * k,
                recommendation_key="buy",
            )
        )
    recs = []
    for period, buy in (
        ("0m", 8 + int(trend * 50)),
        ("-1m", 8),
        ("-2m", 8),
        ("-3m", 8 - int(trend * 50)),
    ):
        recs.append(
            AnalystRecommendation(
                symbol=symbol,
                source="Yfinance",
                date=AS_OF,
                period=period,
                strong_buy=2,
                buy=max(buy, 0),
                hold=4,
                sell=1,
                strong_sell=0,
            )
        )
    return earnings, targets, recs


def _ownership(
    symbol: str, shape: str, last_close: float
) -> tuple[
    list[InsiderTransaction], list[Holding13FSummary], list[Ownership13DG], list[Filing]
]:
    insiders: list[InsiderTransaction] = []
    if shape == "quality_on_sale":
        for n, owner in enumerate(("CEO Alice", "CFO Bob", "Director Carol")):
            day = AS_OF - datetime.timedelta(days=5 + n)
            insiders.append(
                InsiderTransaction(
                    symbol=symbol,
                    cik="1",
                    accession_number=f"a-{symbol}-{n}",
                    line_no=1,
                    filed_date=day,
                    transaction_date=day,
                    reporting_owner=owner,
                    owner_relationship="officer",
                    security_title="Common",
                    transaction_code="P",
                    acquired_disposed="A",
                    shares=10_000,
                    price=last_close,
                    value=10_000 * last_close,
                )
            )
    if shape == "falling_knife":
        for n in range(2):
            day = AS_OF - datetime.timedelta(days=10 + n)
            insiders.append(
                InsiderTransaction(
                    symbol=symbol,
                    cik="2",
                    accession_number=f"s-{symbol}-{n}",
                    line_no=1,
                    filed_date=day,
                    transaction_date=day,
                    reporting_owner=f"Officer {n}",
                    owner_relationship="officer",
                    security_title="Common",
                    transaction_code="S",
                    acquired_disposed="D",
                    shares=50_000,
                    price=last_close,
                    value=50_000 * last_close,
                )
            )
    holders = {
        "leader": (120, 25, 5),
        "compounder": (200, 10, 8),
        "quality_on_sale": (90, 12, 4),
        "falling_knife": (40, 2, 15),
    }.get(shape, (60, 5, 5))
    period = quarters(1)[0]
    summary = [
        Holding13FSummary(
            cusip=f"{symbol:<9}"[:9],
            period=period,
            symbol=symbol,
            issuer_name=symbol,
            holders=holders[0],
            total_shares=3e7,
            total_value=3e7 * last_close,
            prev_holders=holders[0] - holders[1] + holders[2],
            new_holders=holders[1],
            closed_holders=holders[2],
            shares_change=1e6,
            value_change=1e6 * last_close,
            latest_filed_date=AS_OF - datetime.timedelta(days=20),
        )
    ]
    activists = []
    if shape == "quality_on_sale":
        activists.append(
            Ownership13DG(
                accession_number=f"d-{symbol}",
                symbol=symbol,
                issuer_cik="1",
                form_type="SCHEDULE 13D",
                filed_date=AS_OF - datetime.timedelta(days=3),
                reporting_person="Activist Fund",
                percent_of_class=6.1,
            )
        )
    filings = [
        Filing(
            symbol=symbol,
            cik="1",
            form_type="10-K",
            period_end=fiscal_years()[-1],
            filed_date=AS_OF - datetime.timedelta(days=250),
            accession_number=f"k-{symbol}",
        ),
        Filing(
            symbol=symbol,
            cik="1",
            form_type="8-K",
            period_end=AS_OF - datetime.timedelta(days=40),
            filed_date=AS_OF - datetime.timedelta(days=40),
            accession_number=f"e-{symbol}-1",
            items="2.02,9.01",
        ),
    ]
    if shape == "falling_knife":
        filings.append(
            Filing(
                symbol=symbol,
                cik="1",
                form_type="8-K",
                period_end=AS_OF - datetime.timedelta(days=30),
                filed_date=AS_OF - datetime.timedelta(days=30),
                accession_number=f"e-{symbol}-2",
                items="4.02,5.02,9.01",
            )
        )
    return insiders, summary, activists, filings


def _macro(db: BearishDb) -> None:
    observations = []
    months = pd.date_range(end=AS_OF, periods=48, freq="MS")
    for series_id, (level, drift) in MACRO.items():
        for k, month in enumerate(months):
            value = (
                level + drift * k + (0.3 * np.sin(k / 3) if "FF_" in series_id else 0.0)
            )
            observations.append(
                Observation(
                    series_id=series_id,
                    date=month.date(),
                    value=value,
                    as_of=(month + pd.Timedelta(days=15)).date(),
                )
            )
    db.write_observations(observations)


def _etfs(db: BearishDb, rng: np.random.Generator, dates: pd.DatetimeIndex) -> None:
    etfs, snapshots, prices = [], [], []
    for n, (symbol, spec) in enumerate(ETFS.items()):
        close = 100.0 * np.exp(np.cumsum(rng.normal(0.08 / 252, 0.012, len(dates))))
        volume = rng.integers(1_000_000, 10_000_000, len(dates)).astype(float)
        nav = close[-1] * (1.002 if n == 1 else 1.0)
        etfs.append(
            Etf.model_validate(
                {
                    "symbol": symbol,
                    "source": "Yfinance",
                    "name": spec["name"],
                    "long_name": spec["name"],
                    "currency": "USD",
                    "exchange": "PCX",
                    "category": spec["category"],
                    "sector_weightings": spec["weights"],
                    "annual_report_expense_ratio": spec["expense"],
                    "total_net_assets": spec["aum"],
                    "nav_price": nav,
                    "holding_percent": (
                        {"ACCEL": 0.2, "STEADY": 0.15, "QSALE": 0.1}
                        if symbol == "TECHX"
                        else {"MEDCO": 0.3, "PHARM": 0.2}
                    ),
                    "fund_inception_date": 1420070400.0,
                    "trailing_annual_dividend_yield": 0.012,
                }
            )
        )
        prices.extend(
            _prices(
                symbol, pd.DataFrame({"date": dates, "close": close, "volume": volume})
            )
        )
        for k in range(4):
            day = AS_OF - datetime.timedelta(days=30 * k)
            snapshots.append(
                EtfSnapshot(
                    symbol=symbol,
                    source="Yfinance",
                    as_of=day,
                    total_net_assets=float(spec["aum"])
                    * (1 - 0.02 * k if n == 1 else 1 + 0.02 * k),
                    nav_price=nav,
                )
            )
    db.write_assets(Assets(etfs=etfs))
    db.write_series(prices, table=PriceEtfORM)
    db.write_snapshots(Snapshots(etfs=snapshots))
    write_etf_exposures(db)


def _index_and_commodities(
    db: BearishDb,
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    frames: dict[str, pd.DataFrame],
) -> None:
    db.write_assets(
        Assets(
            index=[
                Index.model_validate(
                    {
                        "symbol": "^GSPC",
                        "source": "Yfinance",
                        "name": "S&P 500",
                        "currency": "USD",
                    }
                )
            ]
        )
    )
    index_close = (
        np.mean(
            [f["close"].to_numpy() / f["close"].iloc[0] for f in frames.values()],
            axis=0,
        )
        * 5000
    )
    db.write_series(
        _prices(
            "^GSPC", pd.DataFrame({"date": dates, "close": index_close, "volume": 1e9})
        ),
        table=PriceIndexORM,
    )
    us = [symbol for symbol, spec in UNIVERSE.items() if spec[3] == "United states"]
    db.write_index_members(
        [
            IndexMember(
                index_symbol="^GSPC",
                symbol=symbol,
                as_of=AS_OF - datetime.timedelta(days=10),
            )
            for symbol in us
        ]
    )
    db.write_commodities(list(COMMODITIES))
    for commodity, drift in (
        ("BRENT", 0.30),
        ("COPPER", 0.10),
        ("GOLD", 0.05),
        ("NATGAS_HH", -0.20),
    ):
        symbol = next(c.yfinance_symbol for c in COMMODITIES if c.code == commodity)
        if not symbol:
            continue
        close = 80.0 * np.exp(np.cumsum(rng.normal(drift / 252, 0.015, len(dates))))
        db.write_series(
            _prices(
                symbol, pd.DataFrame({"date": dates, "close": close, "volume": 1e5})
            ),
            table=PriceCommodityORM,
        )


def _fx(db: BearishDb, dates: pd.DatetimeIndex, rng: np.random.Generator) -> None:
    from bearish.database.schemas import PriceCurrencyORM

    rows = []
    for pair, level, drift in (
        ("USDEUR=X", 0.92, -0.03),
        ("USDGBP=X", 0.78, 0.0),
        ("USDJPY=X", 150.0, 0.05),
    ):
        close = level * np.exp(np.cumsum(rng.normal(drift / 252, 0.004, len(dates))))
        rows.extend(
            _prices(pair, pd.DataFrame({"date": dates, "close": close, "volume": 0.0}))
        )
    db.write_series(rows, table=PriceCurrencyORM)


def build_synthetic_database(path: Path) -> Path:
    """Create and fill the database; returns ``path``."""
    rng = np.random.default_rng(SEED)
    db = BearishDb(database_path=path)
    frames = price_frames(rng)
    dates = business_days()
    equities, prices, financials, snapshots = [], [], [], []
    earnings_rows, target_rows, rec_rows = [], [], []
    insider_rows, summary_rows, activist_rows, filing_rows = [], [], [], []
    for symbol, spec in UNIVERSE.items():
        name, sector, industry, country, currency, _, mcap, shape = spec
        last_close = float(frames[symbol]["close"].iloc[-1])
        shares = 1e8
        equities.append(
            Equity.model_validate(
                {
                    "symbol": symbol,
                    "source": "Yfinance",
                    "name": name,
                    "sector": sector,
                    "industry": industry,
                    "country": country,
                    "currency": currency,
                    "exchange": "NMS" if currency == "USD" else "GER",
                    "market_capitalization": last_close * shares,
                    "shares_outstanding": shares,
                    "float_shares": shares * 0.9,
                    "short_shares": shares
                    * (0.15 if shape == "falling_knife" else 0.02),
                    "short_ratio": 6.0 if shape == "falling_knife" else 1.5,
                    "trailing_price_to_earnings": 25.0,
                    "forward_price_to_earnings": 20.0,
                    "trailing_peg_ratio": 1.5,
                    "price_to_book": 4.0,
                    "dividend_yield": 0.01,
                }
            )
        )
        prices.extend(
            _prices(
                symbol,
                frames[symbol],
                dividend_per_quarter=0.3 if shape == "compounder" else 0.0,
            )
        )
        financials.append(_statements(symbol, spec, mcap))
        e, t, r = _expectations(symbol, shape, last_close)
        earnings_rows += e
        target_rows += t
        rec_rows += r
        i, s, a, f = _ownership(symbol, shape, last_close)
        insider_rows += i
        summary_rows += s
        activist_rows += a
        filing_rows += f
        for k in range(3):
            day = AS_OF - datetime.timedelta(days=30 * k)
            short = shares * (0.15 - 0.03 * k if shape == "falling_knife" else 0.02)
            snapshots.append(
                EquitySnapshot(
                    symbol=symbol,
                    source="Yfinance",
                    as_of=day,
                    short_shares=short,
                    market_capitalization=last_close * shares,
                )
            )
    db.write_assets(Assets(equities=equities))
    db.write_series(prices)
    db.write_financials(financials)
    db.write_financials(
        [
            Financials(
                earnings_date=earnings_rows,
                price_targets=target_rows,
                analyst_recommendations=rec_rows,
            )
        ]
    )
    db.write_insider_transactions(insider_rows)
    db.write_holdings13f_summary(summary_rows)
    db.write_ownership13dg(activist_rows)
    db.write_filings(filing_rows)
    db.write_snapshots(Snapshots(equities=snapshots))
    write_country_references(db)
    write_sector_references(db)
    write_sector_sensitivities(db)
    _etfs(db, rng, dates)
    _index_and_commodities(db, rng, dates, frames)
    _fx(db, dates, rng)
    _macro(db)
    write_coverage(db)
    return path
