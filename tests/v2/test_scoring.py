import numpy as np
import pandas as pd
import pytest

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.kpi.equity import compute_equity_kpis
from bullish.kpi.group import compute_group_kpis
from bullish.scoring.composite import explain, score_composite
from bullish.scoring.coverage import grade_symbol, grade_universe
from bullish.scoring.normalize import directed_percentile, percentile_rank, winsorize
from bullish.scoring.presets import EMERGING_LEADER, QUALITY_ON_SALE, SECTOR_SCORE
from bullish.scoring.regime import classify
from bullish.scoring.verdict import assess
from tests.v2.synthetic import AS_OF


@pytest.fixture(scope="module")
def equity(ctx: DataContext) -> pd.DataFrame:
    return compute_equity_kpis(ctx)


@pytest.fixture(scope="module")
def groups(ctx: DataContext, equity: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return compute_group_kpis(ctx, equity)


def test_percentile_rank_inside_groups_and_fallback() -> None:
    values = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, np.nan, 10.0], index=list("abcdefgh")
    )
    groups = pd.Series(["x"] * 6 + ["y", "y"], index=values.index)
    pct = percentile_rank(values, groups, min_peers=5)
    assert pct["a"] == 0.0 and pct["f"] == 100.0
    assert np.isnan(pct["g"])
    # group y has one value: too small, so it ranks against everything and lands at the top
    assert pct["h"] == 100.0
    assert percentile_rank(pd.Series([5.0], index=["only"])).iloc[0] == 50.0


def test_winsorize_and_direction() -> None:
    values = pd.Series(np.r_[np.arange(20.0), 1000.0])
    clipped = winsorize(values, 0.05)
    assert clipped.max() < 1000.0
    lower_better = directed_percentile(values, False, None, 0.0, 5)
    assert lower_better.iloc[0] == 100.0
    inverted = directed_percentile(values, True, None, 0.0, 5, invert=True)
    assert inverted.iloc[0] == 100.0


def test_equity_presets_rank_the_shaped_symbols(
    equity: pd.DataFrame, ctx: DataContext, settings: BullishSettings
) -> None:
    sector = ctx.universe["sector"]
    leader = score_composite(EMERGING_LEADER, equity, settings, sector)
    assert leader.scores.loc["ACCEL", "score"] > leader.scores.loc["DRILL", "score"]
    assert leader.scores["coverage"].between(0, 1).all()
    # size and liquidity floors are zero in the test settings; only the G-score floor (missing allowed) bites
    assert ((~leader.scores["eligible"]) == (equity["mohanram_g"] < 3)).all()
    on_sale = score_composite(QUALITY_ON_SALE, equity, settings, sector)
    assert on_sale.scores.loc["QSALE", "eligible"]
    assert on_sale.scores.loc["KNIFE", "eligible"]
    assert not on_sale.scores.loc["ACCEL", "eligible"]  # not drawn down
    assert on_sale.scores.loc["QSALE", "score"] > on_sale.scores.loc["KNIFE", "score"]
    explanation = explain(on_sale, "QSALE")
    assert explanation["composite"] == "quality_on_sale"
    assert {b["block"] for b in explanation["blocks"]} == {
        "fundamentals",
        "oversold",
        "temporary_evidence",
    }
    assert any(
        k["kpi"] == "insider_cluster_buy" and k["percentile"] == 100.0
        for b in explanation["blocks"]
        for k in b["kpis"]
    )


def test_sector_score_has_blocks(
    groups: dict[str, pd.DataFrame], settings: BullishSettings
) -> None:
    result = score_composite(SECTOR_SCORE, groups["sector"], settings)
    assert set(result.scores.index) == set(groups["sector"].index)
    assert result.scores["score"].notna().all()
    assert result.scores["rank"].min() == 1


def test_verdict_separates_quality_on_sale_from_falling_knife(
    equity: pd.DataFrame, groups: dict[str, pd.DataFrame]
) -> None:
    sector = groups["sector"].loc["Information Technology"]
    qsale = assess("QSALE", equity.loc["QSALE"], str(AS_OF), sector)
    knife = assess("KNIFE", equity.loc["KNIFE"], str(AS_OF), sector)
    accel = assess("ACCEL", equity.loc["ACCEL"], str(AS_OF), sector)
    assert qsale.label == "quality_on_sale"
    assert qsale.fundamentally_sound and qsale.oversold
    assert qsale.drawdown_nature == "idiosyncratic"
    assert (
        qsale.temporary_vs_structural is not None and qsale.temporary_vs_structural > 0
    )
    assert "Insider cluster buy" in qsale.recovery_evidence
    assert knife.label == "falling_knife"
    assert not knife.fundamentally_sound
    assert knife.structural_red_flags
    assert knife.likely_cause == "structural event filed on 8-K"
    assert accel.label == "sound_not_cheap"
    assert accel.drawdown_nature == "none"
    assert all(c.name for c in qsale.fundamental_checks)


def test_coverage_grades(ctx: DataContext) -> None:
    report = grade_symbol(
        "ACCEL", ctx.coverage, ctx.universe.loc["ACCEL"], ctx.as_of_ts
    )
    assert report.grade_of("statements") == "A"
    assert report.grade_of("prices") == "A"
    assert report.grade_of("reference") == "A"
    assert report.overall == "A"
    eu = grade_symbol("SAPDE", ctx.coverage, ctx.universe.loc["SAPDE"], ctx.as_of_ts)
    assert eu.grade_of("ownership") == "B"  # SEC data does not cover German filers
    unknown = grade_symbol("NOPE", ctx.coverage, None, ctx.as_of_ts)
    assert unknown.overall == "insufficient"
    assert unknown.dimensions[0].fill_with is not None
    frame = grade_universe(ctx.coverage, ctx.universe, ctx.as_of_ts)
    assert (frame["overall"] == "A").all()


def test_market_regime_classification(groups: dict[str, pd.DataFrame]) -> None:
    regime = classify("ALL", groups["market"].loc["ALL"])
    assert regime.label in {"risk_on", "neutral", "risk_off"}
    assert regime.quadrant is not None and regime.quadrant.startswith(
        "growth up, inflation down"
    )
    assert regime.factor_leadership in {
        "momentum",
        "value",
        "small caps",
        "profitability",
    }
    assert len(regime.signals) == 12
    assert regime.score is not None
