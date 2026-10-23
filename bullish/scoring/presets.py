"""The composites: sector / industry / country / ETF vehicle scores and the equity presets.

Weights are defaults; the rank backtest (``bullish.backtest.rank``) is the
place to revisit them. Each block carries its rationale so a client can
always answer "why".
"""

from __future__ import annotations

import bullish.kpi.equity  # (registers the equity KPIs)
import bullish.kpi.group  # noqa: F401  (registers the group KPIs)
from bullish.scoring.composite import BlockSpec, CompositeSpec, Filter, KpiWeight

K = KpiWeight

_GROUP_BLOCKS = (
    BlockSpec(
        name="relative_strength",
        weight=30,
        kpis=(
            K(kpi="mom_12_1", weight=2),
            K(kpi="rs_vs_market_3m"),
            K(kpi="rs_vs_market_12m"),
            K(kpi="rs_momentum"),
            K(kpi="rs_ratio"),
        ),
        rationale=(
            "Industry momentum and 12-1 relative strength carry the strongest evidence "
            "(Moskowitz-Grinblatt, Jegadeesh-Titman)"
        ),
    ),
    BlockSpec(
        name="breadth",
        weight=15,
        kpis=(
            K(kpi="breadth_above_200"),
            K(kpi="pct_golden_cross"),
            K(kpi="pct_positive_3m"),
            K(kpi="new_highs_minus_lows"),
        ),
        rationale="A move carried by most members is more durable than one carried by a few large caps",
    ),
    BlockSpec(
        name="fundamentals",
        weight=25,
        kpis=(
            K(kpi="median_revenue_acceleration"),
            K(kpi="median_revenue_growth"),
            K(kpi="beat_rate"),
            K(kpi="revision_ratio", weight=2),
            K(kpi="median_margin_trend"),
        ),
        rationale="Revisions and surprise drift predict returns; acceleration leads the estimate cycle",
    ),
    BlockSpec(
        name="valuation",
        weight=10,
        kpis=(K(kpi="median_pe_vs_5y_z"), K(kpi="median_fcf_yield")),
        rationale="Cheap against own history limits the downside of a momentum tilt",
    ),
    BlockSpec(
        name="flows",
        weight=10,
        kpis=(
            K(kpi="inst_holders_change_sum"),
            K(kpi="insider_buyers_sum"),
            K(kpi="median_short_interest"),
        ),
        rationale="Insider buying has strong evidence; 13F holder flows are a lagging confirmation",
    ),
    BlockSpec(
        name="macro",
        weight=10,
        kpis=(K(kpi="commodity_tailwind_3m"),),
        rationale="Commodity moves reach sector earnings through the curated sensitivities (context, tier C)",
    ),
)

SECTOR_SCORE = CompositeSpec(
    name="sector_score",
    entity_type="sector",
    description=(
        "Sector attractiveness: relative strength, breadth, revisions and growth, valuation, flows, "
        "macro tailwind"
    ),
    blocks=_GROUP_BLOCKS,
    evidence=(
        "Momentum / relative strength (A), revisions and surprises (A), insider buying (A); cycle "
        "mapping deliberately absent"
    ),
)
INDUSTRY_SCORE = CompositeSpec(
    name="industry_score",
    entity_type="industry",
    description="Industry attractiveness with the same blocks as the sector score",
    blocks=_GROUP_BLOCKS,
    evidence=SECTOR_SCORE.evidence,
)

COUNTRY_SCORE = CompositeSpec(
    name="country_score",
    entity_type="country",
    description=(
        "Country attractiveness: equity market strength, growth momentum, policy, stability, "
        "sentiment, external"
    ),
    blocks=(
        BlockSpec(
            name="market",
            weight=35,
            kpis=(
                K(kpi="mom_12_1", weight=2),
                K(kpi="rs_vs_market_12m"),
                K(kpi="breadth_above_200"),
                K(kpi="median_pe_vs_5y_z"),
            ),
            rationale="The market's own momentum and breadth, valuation against history",
        ),
        BlockSpec(
            name="growth",
            weight=25,
            kpis=(
                K(kpi="cli_change_3m", weight=2),
                K(kpi="pmi"),
                K(kpi="gdp_fcst"),
                K(kpi="gdp_nowcast_minus_fcst"),
            ),
            rationale="Leading indicators turning up precede earnings upgrades",
        ),
        BlockSpec(
            name="policy",
            weight=15,
            kpis=(
                K(kpi="cpi_change_6m"),
                K(kpi="policy_rate_change_6m"),
                K(kpi="real_y10"),
                K(kpi="curve_10_2"),
            ),
            rationale="Falling inflation and easing policy support multiples",
        ),
        BlockSpec(
            name="stability",
            weight=10,
            kpis=(K(kpi="credit_gdp_gap"), K(kpi="debt_gdp"), K(kpi="ca_gdp")),
            rationale="Credit gaps and twin deficits precede financial stress",
        ),
        BlockSpec(
            name="sentiment",
            weight=10,
            kpis=(K(kpi="cons_conf"), K(kpi="biz_conf"), K(kpi="epu"), K(kpi="gpr")),
            rationale="Confidence and uncertainty, context only",
        ),
        BlockSpec(
            name="external",
            weight=5,
            kpis=(K(kpi="fx_vs_base_3m"), K(kpi="terms_of_trade_tailwind_3m")),
            rationale="Currency momentum against the base currency and commodity terms of trade",
        ),
    ),
    evidence="Macro momentum (B), valuation vs history (B); most macro blocks are context",
)

ETF_VEHICLE_SCORE = CompositeSpec(
    name="etf_vehicle_score",
    entity_type="etf",
    description=(
        "Quality of the fund as a vehicle: cost and tracking, liquidity and flows, concentration, "
        "risk-adjusted momentum"
    ),
    blocks=(
        BlockSpec(
            name="cost",
            weight=30,
            kpis=(
                K(kpi="expense_ratio", weight=2),
                K(kpi="tracking_difference_1y"),
                K(kpi="tracking_error_1y"),
            ),
            rationale="Total cost of ownership is fee plus tracking difference",
        ),
        BlockSpec(
            name="liquidity",
            weight=25,
            kpis=(K(kpi="dollar_volume_60d"), K(kpi="aum"), K(kpi="aum_change_3m")),
            rationale="Traded value and assets drive spreads and closure risk; flows confirm demand",
        ),
        BlockSpec(
            name="concentration",
            weight=15,
            kpis=(K(kpi="holdings_hhi"), K(kpi="top10_weight")),
            rationale="A fund that is three stocks is not diversification",
        ),
        BlockSpec(
            name="risk_adjusted",
            weight=30,
            kpis=(
                K(kpi="sharpe_1y"),
                K(kpi="mom_12_1"),
                K(kpi="max_drawdown_1y"),
                K(kpi="dist_52w_high"),
            ),
            rationale="Momentum with a drawdown penalty",
        ),
    ),
    evidence="Costs and tracking (A), momentum (A), liquidity (B)",
)

_LIQUIDITY = (
    Filter(kpi="avg_dollar_volume_60d", op=">=", value="min_dollar_volume"),
    Filter(kpi="market_cap_base", op=">=", value="min_market_cap"),
)

EMERGING_LEADER = CompositeSpec(
    name="emerging_leader",
    entity_type="equity",
    description=(
        "Candidates for the next leader: accelerating growth, expanding margins, rising estimates, "
        "institutional accumulation, momentum"
    ),
    blocks=(
        BlockSpec(
            name="growth_acceleration",
            weight=30,
            kpis=(
                K(kpi="revenue_acceleration", weight=2),
                K(kpi="revenue_yoy_last_q"),
                K(kpi="revenue_ttm_growth"),
                K(kpi="eps_yoy_last_q"),
            ),
            rationale="Acceleration, not level, is what estimates miss",
        ),
        BlockSpec(
            name="margin_expansion",
            weight=20,
            kpis=(
                K(kpi="gross_margin_trend"),
                K(kpi="operating_margin_trend"),
                K(kpi="operating_leverage"),
                K(kpi="rule_of_40"),
            ),
            rationale="Margin expansion separated multi-baggers from the rest in outperformer studies",
        ),
        BlockSpec(
            name="revisions",
            weight=20,
            kpis=(
                K(kpi="target_mean_change_3m"),
                K(kpi="buy_ratio_change_3m"),
                K(kpi="eps_surprise_mean_4q"),
                K(kpi="eps_beat_rate_4q"),
            ),
            rationale="Post-earnings-announcement drift and revision momentum",
        ),
        BlockSpec(
            name="accumulation",
            weight=10,
            kpis=(
                K(kpi="inst_holders_change"),
                K(kpi="inst_value_change"),
                K(kpi="insider_net_buy_3m"),
            ),
            rationale="New holders and insider buying confirm the story",
        ),
        BlockSpec(
            name="momentum",
            weight=20,
            kpis=(
                K(kpi="mom_12_1"),
                K(kpi="rs_vs_sector_3m"),
                K(kpi="price_vs_sma200"),
            ),
            rationale="Price confirms; a liquidity floor keeps the list investable",
        ),
    ),
    filters=(
        *_LIQUIDITY,
        Filter(kpi="mohanram_g", op=">=", value=3, allow_missing=True),
    ),
    evidence=(
        "Revisions and PEAD (A), momentum (A), acceleration and margin expansion (B); "
        "survivorship-free backtest required"
    ),
)

QUALITY_COMPOUNDER = CompositeSpec(
    name="quality_compounder",
    entity_type="equity",
    description="High returns on capital, clean earnings, steady growth, strong balance sheet, reasonable price",
    blocks=(
        BlockSpec(
            name="returns_on_capital",
            weight=40,
            kpis=(
                K(kpi="roic", weight=2),
                K(kpi="fcf_conversion"),
                K(kpi="accruals_ratio"),
                K(kpi="gross_margin"),
                K(kpi="operating_margin"),
                K(kpi="piotroski_f"),
            ),
            rationale="Profitability and accruals are the quality factors with academic support (Novy-Marx, Sloan)",
        ),
        BlockSpec(
            name="stability",
            weight=20,
            kpis=(
                K(kpi="years_of_revenue_growth"),
                K(kpi="profitable_years_4"),
                K(kpi="fcf_positive_years_4"),
                K(kpi="revenue_cagr_3y"),
            ),
            rationale="Consistency over the cycle",
        ),
        BlockSpec(
            name="balance_sheet",
            weight=20,
            kpis=(
                K(kpi="net_debt_to_ebitda"),
                K(kpi="interest_coverage"),
                K(kpi="share_count_change_1y"),
            ),
            rationale="Low leverage and no dilution",
        ),
        BlockSpec(
            name="valuation",
            weight=10,
            kpis=(K(kpi="fcf_yield"), K(kpi="pe_vs_5y_z")),
            rationale="Quality at a reasonable price",
        ),
        BlockSpec(
            name="momentum",
            weight=10,
            kpis=(K(kpi="mom_12_1"),),
            rationale="Avoid catching a falling quality name",
        ),
    ),
    filters=_LIQUIDITY,
    evidence="Quality / profitability (A), accruals (A), momentum (A)",
)

VALUE_WITH_CATALYST = CompositeSpec(
    name="value_with_catalyst",
    entity_type="equity",
    description="Cheap against history and peers, with estimates, insiders or price turning up",
    blocks=(
        BlockSpec(
            name="valuation",
            weight=40,
            kpis=(
                K(kpi="fcf_yield", weight=2),
                K(kpi="earnings_yield"),
                K(kpi="ev_to_ebitda"),
                K(kpi="pe_vs_5y_z"),
                K(kpi="price_to_book"),
            ),
            rationale="Value factor (Fama-French HML); own-history discount",
        ),
        BlockSpec(
            name="catalyst",
            weight=40,
            kpis=(
                K(kpi="target_mean_change_3m"),
                K(kpi="buy_ratio_change_3m"),
                K(kpi="eps_surprise_last"),
                K(kpi="insider_net_buy_3m"),
                K(kpi="insider_cluster_buy"),
                K(kpi="rsi_turning_up"),
            ),
            rationale="Value works when something changes; revisions and insiders are the change",
        ),
        BlockSpec(
            name="safety",
            weight=20,
            kpis=(K(kpi="piotroski_f", weight=2), K(kpi="altman_z")),
            rationale="Piotroski: high F-score among cheap stocks avoids the value traps",
        ),
    ),
    filters=_LIQUIDITY,
    evidence="Value (A), F-score within value (A), revisions (A), insiders (A)",
)

TURNAROUND = CompositeSpec(
    name="turnaround",
    entity_type="equity",
    description="Fundamentals improving from a low base with the price starting to confirm",
    blocks=(
        BlockSpec(
            name="improvement",
            weight=40,
            kpis=(
                K(kpi="piotroski_f", weight=2),
                K(kpi="operating_margin_trend"),
                K(kpi="revenue_acceleration"),
                K(kpi="eps_surprise_mean_4q"),
            ),
            rationale="Direction of change matters more than level",
        ),
        BlockSpec(
            name="price_confirmation",
            weight=30,
            kpis=(
                K(kpi="rs_momentum"),
                K(kpi="rsi_turning_up"),
                K(kpi="above_20d_high"),
                K(kpi="up_down_volume_20d"),
            ),
            rationale="RRG improving quadrant and accumulation volume",
        ),
        BlockSpec(
            name="survival",
            weight=30,
            kpis=(
                K(kpi="altman_z", weight=2),
                K(kpi="interest_coverage"),
                K(kpi="red_flag_8k_12m"),
            ),
            rationale="A turnaround needs the balance sheet to last",
        ),
    ),
    filters=(*_LIQUIDITY, Filter(kpi="drawdown_52w", op="<=", value=-0.2)),
    evidence="F-score (A), long-term reversal (A), practitioner turnaround patterns (B)",
)

DIVIDEND_QUALITY = CompositeSpec(
    name="dividend_quality",
    entity_type="equity",
    description="Yield that is covered by free cash flow and grows",
    blocks=(
        BlockSpec(
            name="income",
            weight=40,
            kpis=(K(kpi="dividend_yield", weight=2), K(kpi="dividend_growth_3y")),
            rationale="Yield and growth",
        ),
        BlockSpec(
            name="coverage",
            weight=40,
            kpis=(
                K(kpi="dividend_payout_fcf"),
                K(kpi="fcf_conversion"),
                K(kpi="fcf_positive_years_4"),
            ),
            rationale="Paid from cash, not debt",
        ),
        BlockSpec(
            name="balance_sheet",
            weight=20,
            kpis=(K(kpi="net_debt_to_ebitda"), K(kpi="interest_coverage")),
            rationale="Leverage cuts dividends",
        ),
    ),
    filters=(*_LIQUIDITY, Filter(kpi="dividend_yield", op=">", value=0.0)),
    evidence="Payout coverage (B), quality (A)",
)

QUALITY_ON_SALE = CompositeSpec(
    name="quality_on_sale",
    entity_type="equity",
    description="Sound businesses that are oversold on evidence the sell-off is temporary",
    blocks=(
        BlockSpec(
            name="fundamentals",
            weight=35,
            kpis=(
                K(kpi="roic"),
                K(kpi="fcf_conversion"),
                K(kpi="accruals_ratio"),
                K(kpi="piotroski_f"),
                K(kpi="net_debt_to_ebitda"),
                K(kpi="gross_margin"),
            ),
            rationale="Is it a good business?",
        ),
        BlockSpec(
            name="oversold",
            weight=30,
            kpis=(
                K(kpi="drawdown_52w", invert=True, weight=2),
                K(kpi="excess_drawdown_vs_sector", invert=True),
                K(kpi="rsi_14", invert=True),
                K(kpi="ret_1m", invert=True),
                K(kpi="capitulation_volume"),
            ),
            rationale="Short-term reversal (Jegadeesh 1990): the deeper and more company-specific the drop, the better",
        ),
        BlockSpec(
            name="temporary_evidence",
            weight=35,
            kpis=(
                K(kpi="insider_net_buy_3m"),
                K(kpi="insider_cluster_buy"),
                K(kpi="target_mean_change_3m"),
                K(kpi="eps_surprise_last"),
                K(kpi="short_interest_change_1m"),
                K(kpi="red_flag_8k_12m"),
                K(kpi="equity_offering_12m"),
            ),
            rationale="Insiders buying, estimates holding, no structural 8-K items and no dilution read as temporary",
        ),
    ),
    filters=(*_LIQUIDITY, Filter(kpi="drawdown_52w", op="<=", value=-0.15)),
    evidence="Reversal (A), F-score among beaten-down stocks (A), insider buying (A)",
)

EQUITY_PRESETS: dict[str, CompositeSpec] = {
    spec.name: spec
    for spec in (
        EMERGING_LEADER,
        QUALITY_COMPOUNDER,
        VALUE_WITH_CATALYST,
        TURNAROUND,
        DIVIDEND_QUALITY,
        QUALITY_ON_SALE,
    )
}
GROUP_COMPOSITES: dict[str, CompositeSpec] = {
    spec.name: spec
    for spec in (SECTOR_SCORE, INDUSTRY_SCORE, COUNTRY_SCORE, ETF_VEHICLE_SCORE)
}
ALL_COMPOSITES: dict[str, CompositeSpec] = EQUITY_PRESETS | GROUP_COMPOSITES
