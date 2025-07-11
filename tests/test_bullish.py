import datetime
from pathlib import Path

import pytest
from bearish.main import Bearish
from bearish.models.base import Ticker
from bearish.models.price.prices import Prices
from bearish.models.query.query import AssetQuery, Symbols
from sqlalchemy import inspect

from bullish.analysis.analysis import (
    Analysis,
    TechnicalAnalysis,
    run_analysis,
)
from bullish.analysis.filter import FilteredResults, FilterQueryStored
from bullish.database.crud import BullishDb
from bullish.jobs.models import JobTracker


def test_read_financials(bullish_db: BullishDb) -> None:
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol="AAPL")]))
    financials = bullish_db._read_financials(query)
    assert financials.balance_sheets
    assert financials.cash_flows
    assert financials.earnings_date


def test_analysis(bullish_db: BullishDb) -> None:

    analysis = Analysis.from_ticker(bullish_db, Ticker(symbol="AAPL"))
    assert analysis.rsi_bullish_crossover_30 is not None
    assert analysis.positive_net_income is not None
    bullish_db.write_analysis(analysis)
    analysis_db = bullish_db.read_analysis(Ticker(symbol="AAPL"))
    assert analysis_db.rsi_bullish_crossover_30 is not None
    assert analysis_db.positive_net_income is not None


def test_technical_analysis() -> None:

    prices = Prices.from_csv(Path(__file__).parent / "data" / "prices.csv")
    ta = TechnicalAnalysis.from_data(prices.to_dataframe(), Ticker(symbol="AAPL"))
    assert ta


def test_run_analysis(bullish_db: BullishDb) -> None:
    run_analysis(bullish_db)
    analysis_db = bullish_db.read_analysis(Ticker(symbol="AAPL"))
    assert analysis_db.rsi_bullish_crossover_30 is not None
    assert analysis_db.positive_net_income is not None


def test_read_analysis_view(bullish_db_with_analysis: BullishDb) -> None:
    data = bullish_db_with_analysis.read_analysis_data()
    assert not data.empty


def test_read_write_job_tracker(bullish_db: BullishDb) -> None:
    job_tracker_1 = JobTracker(
        job_id="test_job_1", status="Running", type="Update data"
    )
    job_tracker_2 = JobTracker(
        job_id="test_job_2", status="Running", type="Update data"
    )
    bullish_db.write_job_tracker(job_tracker_1)
    bullish_db.write_job_tracker(job_tracker_2)
    job_trackers = bullish_db.read_job_trackers()
    assert not job_trackers.empty
    bullish_db.delete_job_trackers(job_trackers["job_id"].tolist())
    job_trackers = bullish_db.read_job_trackers()
    assert job_trackers.empty


def test_bearish_update_price(bullish_db: BullishDb) -> None:
    bearish = Bearish(path=bullish_db.database_path, auto_migration=False)
    reference_date = datetime.date.today() + datetime.timedelta(days=5)
    bearish.update_prices(["AAPL"], reference_date=reference_date, delay=1)
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol="AAPL")]))
    res = bearish.read_series(query)
    assert res


def test_filtered_results(bullish_db: BullishDb) -> None:
    filtered_results = FilteredResults(
        name="test_results_3",
        filter_query=FilterQueryStored(market_capitalization=[0, 100]),
        symbols=["AAPL", "GOOGL"],
    )
    bullish_db.write_filtered_results(filtered_results)
    filtered_results = bullish_db.read_filtered_results(filtered_results.name)
    assert filtered_results is not None
    results = bullish_db.read_list_filtered_results()
    assert results


@pytest.mark.skip
def test_ticker_mood_instantiation(bullish_db: BullishDb) -> None:
    from tickermood.database.crud import TickerMoodDb

    TickerMoodDb(database_path=bullish_db.database_path, no_migration=True)
    insp = inspect(bullish_db._engine)
    assert insp.has_table("subject")


def test_read_dates(bullish_db: BullishDb) -> None:
    dates = bullish_db.read_dates(symbol="AAPL")
    assert len(dates) > 0
