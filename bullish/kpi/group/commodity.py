"""Commodity KPIs from the futures prices and CFTC positioning."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext, returns_over
from bullish.kpi.registry import declare_many

BLOCK = "commodity"
MONTH, QUARTER, YEAR = 21, 63, 252
NAMES = declare_many(
    "commodity",
    BLOCK,
    [
        ("ret_1m", "A", "ratio", "Front-month price return over 1 month", True),
        ("ret_3m", "A", "ratio", "Front-month price return over 3 months", True),
        ("ret_12m", "A", "ratio", "Front-month price return over 12 months", True),
        ("vol_60d", "B", "ratio", "Annualised volatility, 60 days", False),
        (
            "drawdown_52w",
            "B",
            "ratio",
            "Last price over the 52-week high minus one",
            True,
        ),
        (
            "price_vs_sma200",
            "B",
            "ratio",
            "Last price over the 200-day average minus one",
            True,
        ),
        (
            "cot_net_spec_pct_oi",
            "B",
            "pct",
            "Net speculative positioning as a share of open interest, latest",
            None,
        ),
        (
            "cot_pct_1y",
            "B",
            "pct",
            "Percentile of net speculative positioning inside the last year (crowding)",
            False,
        ),
    ],
)


def compute(ctx: DataContext) -> pd.DataFrame:
    commodities = ctx.commodities
    close = ctx.commodity_close
    out = pd.DataFrame(
        index=pd.Index(commodities["code"].tolist(), name="commodity"),
        columns=NAMES,
        dtype=float,
    )
    if commodities.empty:
        return out
    mapping = commodities.dropna(subset=["yfinance_symbol"]).set_index("code")[
        "yfinance_symbol"
    ]
    if not close.empty:
        filled = close.ffill()
        by_code = {
            code: symbol for code, symbol in mapping.items() if symbol in filled.columns
        }
        panel = filled[list(by_code.values())].rename(
            columns={v: k for k, v in by_code.items()}
        )
        out.loc[panel.columns, "ret_1m"] = returns_over(panel, MONTH)
        out.loc[panel.columns, "ret_3m"] = returns_over(panel, QUARTER)
        out.loc[panel.columns, "ret_12m"] = returns_over(panel, YEAR)
        out.loc[panel.columns, "vol_60d"] = panel.pct_change().iloc[
            -60:
        ].std() * np.sqrt(YEAR)
        window = panel.iloc[-YEAR:]
        out.loc[panel.columns, "drawdown_52w"] = window.iloc[-1] / window.max() - 1
        out.loc[panel.columns, "price_vs_sma200"] = (
            panel.iloc[-1] / panel.rolling(200).mean().iloc[-1] - 1
        )
    for code in out.index:
        cot = ctx.macro_series("COT_NET_SPEC_PCT_OI", code)
        if len(cot):
            out.loc[code, "cot_net_spec_pct_oi"] = float(cot.iloc[-1])
            year = cot.iloc[-52:]
            out.loc[code, "cot_pct_1y"] = float((year < cot.iloc[-1]).mean() * 100)
    return out.replace([np.inf, -np.inf], np.nan).astype(float)
