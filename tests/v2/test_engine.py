import datetime

import pandas as pd
import pytest

from bullish.config import BullishSettings
from bullish.database.crud import BullishDb
from bullish.kpi import engine
from bullish.kpi.registry import REGISTRY
from tests.v2.synthetic import AS_OF


@pytest.fixture(scope="module")
def run_result(bullish_db_v2: BullishDb, settings: BullishSettings) -> engine.RunResult:
    return engine.run(bullish_db_v2, settings, as_of=AS_OF)


def test_run_persists_kpis_scores_and_log(
    bullish_db_v2: BullishDb, run_result: engine.RunResult
) -> None:
    assert run_result.run is not None
    assert run_result.run.entities == 12
    assert run_result.run.kpis > 1000
    assert run_result.run.scores > 20
    runs = bullish_db_v2.read_kpi_runs()
    assert AS_OF in [r.as_of for r in runs]
    frame = bullish_db_v2.read_kpi_frame("equity")
    assert frame.attrs["as_of"] == AS_OF
    assert set(frame.index) == set(run_result.equity.index)
    assert frame.loc["ACCEL", "ret_3m"] == pytest.approx(
        run_result.equity.loc["ACCEL", "ret_3m"]
    )
    assert "coverage_grade" in frame and frame["coverage_grade"].eq(3.0).all()
    sector = bullish_db_v2.read_kpi_frame("sector")
    assert "Information Technology" in sector.index
    assert bullish_db_v2.read_kpi_frame("equity", as_of=datetime.date(2000, 1, 1)).empty


def test_scores_round_trip_with_components(
    bullish_db_v2: BullishDb, run_result: engine.RunResult
) -> None:
    scores = bullish_db_v2.read_scores("equity", "quality_on_sale")
    assert scores.attrs["as_of"] == AS_OF
    assert scores.index[0] == "QSALE"  # best rank first
    row = scores.loc["QSALE"]
    assert row["eligible"]
    assert set(row["components"]["blocks"]) == {
        "fundamentals",
        "oversold",
        "temporary_evidence",
    }
    assert row["components"]["kpis"]["insider_cluster_buy"] == 100.0
    sectors = bullish_db_v2.read_scores("sector", "sector_score")
    assert len(sectors) == 4
    etfs = bullish_db_v2.read_scores("etf", "etf_score")
    assert set(etfs.index) == {"TECHX", "HLTHX", "WORLDX"}
    assert etfs["value"].between(0, 100).all()


def test_etf_lookthrough_uses_sector_scores(run_result: engine.RunResult) -> None:
    etf = run_result.groups["etf"]
    sector = run_result.scores[("sector", "sector_score")].scores["score"]
    assert etf.loc["TECHX", "exposure_score"] == pytest.approx(
        sector["Information Technology"]
    )
    assert etf.loc["HLTHX", "exposure_score"] == pytest.approx(sector["Health Care"])
    assert etf["etf_score"].notna().all()


def test_kpi_series_and_second_run(
    bullish_db_v2: BullishDb, settings: BullishSettings, run_result: engine.RunResult
) -> None:
    earlier = AS_OF - datetime.timedelta(days=90)
    engine.run(bullish_db_v2, settings, as_of=earlier)
    series = bullish_db_v2.read_kpi_series("equity", "ACCEL", "ret_3m")
    assert {earlier, AS_OF} <= set(series.index)
    assert series.index[-1] == AS_OF
    assert {earlier, AS_OF} <= set(bullish_db_v2.kpi_dates())
    # the latest frame is still today's, and asking for the past gives the past run
    assert bullish_db_v2.read_kpi_frame("equity").attrs["as_of"] == AS_OF
    assert (
        bullish_db_v2.read_kpi_frame("equity", as_of=earlier).attrs["as_of"] == earlier
    )
    # every stored KPI name is declared
    stored = set(
        bullish_db_v2.read_query(
            "SELECT DISTINCT entity_type, name FROM kpi"
        ).itertuples(index=False, name=None)
    )
    declared = {(d.entity_type, d.name) for d in REGISTRY.all()}
    assert stored <= declared
    assert isinstance(pd.Timestamp(AS_OF), pd.Timestamp)
