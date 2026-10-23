import numpy as np
import pandas as pd
import pytest

from bullish.data.panels import DataContext
from bullish.kpi.equity import BLOCKS, compute_equity_kpis
from bullish.kpi.registry import REGISTRY
from tests.v2.synthetic import UNIVERSE


@pytest.fixture(scope="module")
def kpis(ctx: DataContext) -> pd.DataFrame:
    return compute_equity_kpis(ctx)


def test_universe_resolves_sector_and_country(ctx: DataContext) -> None:
    universe = ctx.universe
    assert set(universe.index) == set(UNIVERSE)
    assert universe.loc["ACCEL", "sector"] == "Information Technology"
    assert universe.loc["SOLBE", "sector"] == "Materials"
    assert (
        universe.loc["ACCEL", "industry"] == "Semiconductors & Semiconductor Equipment"
    )
    assert universe.loc["SAPDE", "country"] == "DEU"
    assert universe.loc["ACCEL", "country"] == "USA"


def test_every_declared_kpi_is_a_column(kpis: pd.DataFrame) -> None:
    declared_by_blocks = {
        name for block in BLOCKS for name in REGISTRY.names("equity", block.BLOCK)
    }
    assert set(kpis.columns) == declared_by_blocks
    assert set(kpis.columns) <= set(REGISTRY.names("equity"))
    assert kpis.index.name == "symbol"
    assert len(kpis) == len(UNIVERSE)


def test_momentum_values_match_a_manual_computation(
    ctx: DataContext, kpis: pd.DataFrame
) -> None:
    close = ctx.close["ACCEL"].ffill()
    assert kpis.loc["ACCEL", "ret_3m"] == pytest.approx(
        close.iloc[-1] / close.iloc[-64] - 1
    )
    assert kpis.loc["ACCEL", "mom_12_1"] == pytest.approx(
        close.iloc[-22] / close.iloc[-253] - 1
    )
    assert kpis.loc["ACCEL", "drawdown_52w"] == pytest.approx(
        close.iloc[-1] / close.iloc[-252:].max() - 1
    )
    assert kpis.loc["ACCEL", "price_history_days"] == len(close.dropna())


def test_shaped_symbols_show_their_shape(kpis: pd.DataFrame) -> None:
    # the two sliding stocks are oversold, the leader is not
    assert kpis.loc["QSALE", "drawdown_52w"] < -0.25
    assert kpis.loc["KNIFE", "drawdown_52w"] < -0.25
    assert kpis.loc["QSALE", "rsi_14"] < 35
    assert kpis.loc["ACCEL", "drawdown_52w"] > -0.15
    # idiosyncratic: the drop is far below the sector median drawdown
    assert kpis.loc["QSALE", "excess_drawdown_vs_sector"] < -0.15
    # fundamentals: the sound business is sound, the knife is not
    assert kpis.loc["QSALE", "fcf_conversion"] > 0.8
    knife_conversion = kpis.loc["KNIFE", "fcf_conversion"]
    assert (
        pd.isna(knife_conversion) or knife_conversion < 0.5
    )  # loss-making: conversion undefined
    assert kpis.loc["KNIFE", "share_count_change_1y"] > 0.05
    assert kpis.loc["QSALE", "share_count_change_1y"] < 0
    assert kpis.loc["ACCEL", "revenue_growth_yoy"] > 0.3
    assert kpis.loc["ACCEL", "revenue_acceleration"] > 0
    assert kpis.loc["ACCEL", "piotroski_f"] >= 5
    assert kpis.loc["KNIFE", "piotroski_f"] <= 4
    assert kpis.loc["KNIFE", "red_flag_8k_12m"] == 1
    assert kpis.loc["KNIFE", "officer_departure_12m"] == 1
    assert kpis.loc["QSALE", "red_flag_8k_12m"] == 0
    assert kpis.loc["QSALE", "insider_cluster_buy"] == 1
    assert kpis.loc["QSALE", "insider_buyers_3m"] == 3
    assert kpis.loc["KNIFE", "insider_net_buy_3m"] < 0
    assert kpis.loc["QSALE", "activist_13d"] == 1
    assert kpis.loc["KNIFE", "equity_offering_12m"] == 1
    assert kpis.loc["KNIFE", "short_interest_pct_float"] > 0.1
    assert kpis.loc["KNIFE", "short_interest_change_1m"] > 0
    assert kpis.loc["ACCEL", "eps_beat_rate_4q"] == 1.0
    assert kpis.loc["KNIFE", "eps_surprise_last"] < 0
    assert kpis.loc["QSALE", "target_upside"] > 0.3
    assert kpis.loc["KNIFE", "target_mean_change_3m"] < 0
    assert kpis.loc["ACCEL", "buy_ratio_change_3m"] > 0


def test_valuation_and_fx_conversion(ctx: DataContext, kpis: pd.DataFrame) -> None:
    eur_per_usd = ctx.usd_rate("EUR")
    assert 0.7 < eur_per_usd < 1.1
    usd_cap = ctx.universe.loc["ACCEL", "market_capitalization"]
    assert kpis.loc["ACCEL", "market_cap_base"] == pytest.approx(usd_cap * eur_per_usd)
    eur_cap = ctx.universe.loc["SAPDE", "market_capitalization"]
    assert kpis.loc["SAPDE", "market_cap_base"] == pytest.approx(eur_cap)
    assert kpis.loc["ACCEL", "pe_trailing"] > 0
    assert kpis.loc["STEADY", "fcf_yield"] > 0
    assert np.isfinite(kpis.loc["ACCEL", "earnings_yield_minus_10y"])
    assert kpis.loc["STEADY", "dividend_yield"] > 0


def test_macro_exposure(kpis: pd.DataFrame) -> None:
    # oil rose 30 % a year in the fixture: producers get a tailwind, airlines would not
    assert kpis.loc["OILCO", "commodity_tailwind_3m"] > 0
    assert kpis.loc["SAPDE", "fx_momentum_3m"] == 0.0  # listed in the base currency
    assert np.isfinite(kpis.loc["ACCEL", "fx_momentum_3m"])


def test_point_in_time_hides_unfiled_statements(bullish_db_v2, settings) -> None:  # type: ignore[no-untyped-def]
    import datetime

    early = DataContext(
        db=bullish_db_v2, settings=settings, as_of=datetime.date(2024, 1, 15)
    )
    annual = early.metrics_annual
    # FY2023 (filed ~Feb 2024) must not be visible in January 2024
    assert annual["date"].max() < pd.Timestamp("2023-12-31")
    assert early.close.index.max() <= pd.Timestamp("2024-01-15")
