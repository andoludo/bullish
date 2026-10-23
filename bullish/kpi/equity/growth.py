"""Growth KPIs from annual and quarterly statements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.equity._statements import (
    annual,
    free_cash_flow,
    growth,
    nth_last,
    quarterly,
    safe_div,
    ttm,
)
from bullish.kpi.registry import declare_many

BLOCK = "growth"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "revenue_growth_yoy",
            "A",
            "ratio",
            "Last fiscal year revenue over the prior year minus one",
            True,
        ),
        (
            "revenue_cagr_3y",
            "B",
            "ratio",
            "Compound annual revenue growth over the last three fiscal years",
            True,
        ),
        ("eps_growth_yoy", "A", "ratio", "Diluted EPS growth, last fiscal year", True),
        (
            "net_income_growth_yoy",
            "B",
            "ratio",
            "Net income growth, last fiscal year",
            True,
        ),
        (
            "revenue_ttm_growth",
            "A",
            "ratio",
            "Trailing-twelve-month revenue over the prior twelve months minus one",
            True,
        ),
        (
            "revenue_yoy_last_q",
            "A",
            "ratio",
            "Last quarter revenue over the same quarter a year earlier minus one",
            True,
        ),
        (
            "revenue_acceleration",
            "B",
            "ratio",
            "revenue_yoy_last_q minus the same measure one quarter earlier",
            True,
        ),
        (
            "eps_yoy_last_q",
            "A",
            "ratio",
            "Last quarter diluted EPS over the same quarter a year earlier minus one",
            True,
        ),
        (
            "rule_of_40",
            "B",
            "pct",
            "Trailing revenue growth plus free-cash-flow margin, in percent",
            True,
        ),
        (
            "operating_leverage",
            "B",
            "ratio",
            "Operating-income growth over revenue growth (trailing twelve months)",
            True,
        ),
        (
            "years_of_revenue_growth",
            "B",
            "count",
            "Consecutive fiscal years with rising revenue (max 5)",
            True,
        ),
        (
            "revenue_decelerating_2y",
            "B",
            "flag",
            "Revenue growth fell in each of the last two fiscal years",
            False,
        ),
        (
            "fcf_ttm_growth",
            "B",
            "ratio",
            "Trailing free cash flow over the prior twelve months minus one",
            True,
        ),
    ],
)


def _consecutive_growth_years(
    frame: pd.DataFrame, column: str, limit: int = 5
) -> pd.Series:
    def _count(values: pd.Series) -> float:
        series = values.dropna().iloc[-(limit + 1) :]
        count = 0
        for later, earlier in zip(
            series.iloc[::-1], series.iloc[::-1].shift(-1), strict=False
        ):
            if pd.isna(earlier) or later <= earlier:
                break
            count += 1
        return float(count)

    if frame.empty or column not in frame:
        return pd.Series(dtype=float)
    return frame.groupby("symbol")[column].agg(_count)


def compute(ctx: DataContext) -> pd.DataFrame:
    index = ctx.universe.index
    out = pd.DataFrame(index=index, columns=NAMES, dtype=float)
    annual_frame = annual(ctx)
    quarterly_frame = quarterly(ctx)
    if annual_frame.empty and quarterly_frame.empty:
        return out

    revenue_0, revenue_1 = nth_last(annual_frame, "total_revenue", 0), nth_last(
        annual_frame, "total_revenue", 1
    )
    revenue_3 = nth_last(annual_frame, "total_revenue", 3)
    out["revenue_growth_yoy"] = growth(revenue_0, revenue_1)
    cagr = (revenue_0 / revenue_3.replace(0.0, np.nan)).where(revenue_3 > 0) ** (
        1 / 3
    ) - 1
    out["revenue_cagr_3y"] = cagr
    out["eps_growth_yoy"] = growth(
        nth_last(annual_frame, "diluted_eps", 0),
        nth_last(annual_frame, "diluted_eps", 1),
    )
    out["net_income_growth_yoy"] = growth(
        nth_last(annual_frame, "net_income", 0), nth_last(annual_frame, "net_income", 1)
    )

    revenue_ttm, revenue_ttm_prev = ttm(ctx, "total_revenue"), ttm(
        ctx, "total_revenue", 4
    )
    out["revenue_ttm_growth"] = growth(revenue_ttm, revenue_ttm_prev)
    q0, q4, q1, q5 = (
        nth_last(quarterly_frame, "total_revenue", n) for n in (0, 4, 1, 5)
    )
    yoy_last = growth(q0, q4)
    yoy_prev = growth(q1, q5)
    out["revenue_yoy_last_q"] = yoy_last
    out["revenue_acceleration"] = yoy_last - yoy_prev
    out["eps_yoy_last_q"] = growth(
        nth_last(quarterly_frame, "diluted_eps", 0),
        nth_last(quarterly_frame, "diluted_eps", 4),
    )

    fcf_ttm = ttm(ctx, "operating_cash_flow") - ttm(ctx, "capital_expenditure").abs()
    fcf_margin = safe_div(fcf_ttm, revenue_ttm)
    out["rule_of_40"] = (out["revenue_ttm_growth"] + fcf_margin) * 100
    ebit_growth = growth(ttm(ctx, "operating_income"), ttm(ctx, "operating_income", 4))
    out["operating_leverage"] = safe_div(ebit_growth, out["revenue_ttm_growth"]).clip(
        -10, 10
    )
    out["years_of_revenue_growth"] = _consecutive_growth_years(
        annual_frame, "total_revenue"
    )
    g0 = growth(revenue_0, revenue_1)
    g1 = growth(revenue_1, nth_last(annual_frame, "total_revenue", 2))
    g2 = growth(nth_last(annual_frame, "total_revenue", 2), revenue_3)
    out["revenue_decelerating_2y"] = (
        ((g0 < g1) & (g1 < g2)).astype(float).where(g2.notna())
    )
    fcf_prev = (
        ttm(ctx, "operating_cash_flow", 4) - ttm(ctx, "capital_expenditure", 4).abs()
    )
    out["fcf_ttm_growth"] = growth(fcf_ttm, fcf_prev)
    _ = free_cash_flow  # kept for callers that need the per-row helper
    return out.replace([np.inf, -np.inf], np.nan)
