from pathlib import Path

import pandas as pd
from bearish.models.base import Ticker
from bearish.models.price.prices import Prices
from bearish.models.query.query import AssetQuery, Symbols

from bullish.analysis import Analysis, TechnicalAnalysis, mom, wow, yoy
from bullish.database.crud import BullishDb


def test_read_financials(bullish_db: BullishDb) -> None:
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol="AAPL")]))
    financials = bullish_db._read_financials(query)
    assert financials.balance_sheets
    assert financials.cash_flows
    assert financials.earnings_date


def test_analysis(bullish_db: BullishDb) -> None:

    analysis = Analysis.from_ticker(bullish_db, Ticker(symbol="AAPL"))
    assert analysis.last_adx is not None
    assert analysis.positive_net_income is not None
    bullish_db.write_analysis(analysis)
    analysis_db = bullish_db.read_analysis(Ticker(symbol="AAPL"))
    assert analysis_db.last_adx is not None
    assert analysis_db.positive_net_income is not None


def test_star_prices() -> None:
    yoy_ = []
    mom_ = []
    wow_ = []
    for ticker in ["NVDA", "RHM.DE", "TSM", "PLTR", "SMCI", "MSFT"]:
        prices = Prices.from_csv(
            Path(__file__).parent / "data" / f"prices_{ticker.lower()}.csv"
        ).to_dataframe()
        yoy_.append(yoy(prices))
        mom_.append(mom(prices))
        wow_.append(wow(prices))
    median_yoy = pd.concat(yoy_).median()
    median_mom = pd.concat(mom_).median()
    median_wow = pd.concat(wow_).median()
    assert median_yoy > 30
    assert median_mom > 1
    assert median_wow > 0


def test_technical_analysis():

    prices = Prices.from_csv(Path(__file__).parent / "data" / "prices.csv")
    ta = TechnicalAnalysis.from_data(prices.to_dataframe())
    assert ta
