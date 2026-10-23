import numpy as np
import pandas as pd
import pytest

from bullish.data.panels import DataContext
from bullish.kpi.equity import compute_equity_kpis
from bullish.kpi.group import compute_group_kpis
from bullish.kpi.registry import REGISTRY


@pytest.fixture(scope="module")
def frames(ctx: DataContext) -> dict[str, pd.DataFrame]:
    return compute_group_kpis(ctx, compute_equity_kpis(ctx))


def test_sector_frame_has_every_sector_and_declared_column(
    frames: dict[str, pd.DataFrame],
) -> None:
    sector = frames["sector"]
    assert set(sector.index) == {
        "Information Technology",
        "Health Care",
        "Energy",
        "Materials",
    }
    assert list(sector.columns) == REGISTRY.names("sector")
    assert sector.loc["Information Technology", "members"] == 5
    assert sector["rrg_quadrant"].dropna().isin([1, 2, 3, 4]).all()
    assert 0 <= sector.loc["Information Technology", "breadth_above_200"] <= 1


def test_group_index_relative_strength_is_consistent(
    frames: dict[str, pd.DataFrame],
) -> None:
    sector = frames["sector"]
    # relative strengths vs the universe net out roughly to zero when cap-weighted; leaders beat laggards
    assert (
        sector.loc["Information Technology", "ret_12m"]
        > sector.loc["Energy", "ret_12m"] - 1.0
    )
    assert np.isfinite(sector["rs_ratio"]).all()


def test_country_frame_combines_equity_and_macro(
    frames: dict[str, pd.DataFrame],
) -> None:
    country = frames["country"]
    assert {"USA", "DEU", "BEL"} <= set(country.index)
    assert country.loc["USA", "members"] == 8
    assert country.loc["USA", "cli_change_3m"] > 0  # the fixture drifts the CLI upward
    assert (
        country.loc["USA", "policy_rate_change_6m"] < 0
    )  # cutting cycle in the fixture
    assert country.loc["DEU", "fx_vs_base_3m"] == 0.0  # EUR is the base currency
    assert np.isfinite(country.loc["USA", "fx_vs_base_3m"])
    assert pd.isna(country.loc["BEL", "pmi"])  # no Belgian macro rows in the fixture


def test_market_regime_rows(frames: dict[str, pd.DataFrame]) -> None:
    market = frames["market"]
    assert {"ALL", "USA"} <= set(market.index)
    assert market.loc["ALL", "members"] == 12
    assert (
        market.loc["ALL", "growth_inflation_quadrant"] == 1
    )  # growth up, inflation down in the fixture
    assert market.loc["ALL", "vix"] == pytest.approx(18.0)
    assert market.loc["ALL", "ff_mom_3m"] == pytest.approx(
        market.loc["USA", "ff_mom_3m"]
    )
    assert np.isfinite(market.loc["ALL", "copper_vs_gold_3m"])
    assert np.isfinite(market.loc["ALL", "cyclicals_vs_defensives_3m"])


def test_etf_vehicle_kpis(frames: dict[str, pd.DataFrame]) -> None:
    etf = frames["etf"]
    assert set(etf.index) == {"TECHX", "HLTHX", "WORLDX"}
    assert etf.loc["TECHX", "expense_ratio"] == pytest.approx(0.10)
    assert etf.loc["HLTHX", "aum_change_3m"] > 0 > etf.loc["TECHX", "aum_change_3m"]
    assert etf.loc["HLTHX", "premium_to_nav"] < 0  # NAV above price in the fixture
    assert etf.loc["TECHX", "top10_weight"] == pytest.approx(0.45)
    assert etf.loc["TECHX", "n_exposures"] >= 1
    assert np.isfinite(etf.loc["WORLDX", "tracking_error_1y"])
    assert etf.loc["WORLDX", "age_years"] > 10


def test_commodity_kpis(frames: dict[str, pd.DataFrame]) -> None:
    commodity = frames["commodity"]
    assert "BRENT" in commodity.index
    assert np.isfinite(commodity.loc["BRENT", "ret_3m"])
    assert pd.isna(commodity.loc["WHEAT", "ret_3m"])  # no price in the fixture
