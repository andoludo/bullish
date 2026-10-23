"""Smart-money and corporate-event KPIs: 13F flows, insiders, activists, short interest, 8-K items."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.registry import declare_many

BLOCK = "ownership"
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "inst_holders",
            "B",
            "count",
            "13F managers holding the stock in the latest quarter",
            True,
        ),
        (
            "inst_holders_change",
            "B",
            "count",
            "New minus closed 13F holders, latest quarter",
            True,
        ),
        (
            "inst_value_change",
            "B",
            "ratio",
            "Change of the 13F position value versus the prior quarter",
            True,
        ),
        ("inst_share_of_float", "B", "ratio", "13F shares over the float", None),
        (
            "insider_net_buy_3m",
            "A",
            "currency",
            "Insider open-market purchases minus sales, 3 months, local currency",
            True,
        ),
        (
            "insider_net_buy_12m",
            "A",
            "currency",
            "Insider open-market purchases minus sales, 12 months, local currency",
            True,
        ),
        (
            "insider_buyers_3m",
            "A",
            "count",
            "Distinct officers or directors buying in the open market, 3 months",
            True,
        ),
        (
            "insider_cluster_buy",
            "A",
            "flag",
            "At least three distinct insiders bought within 30 days",
            True,
        ),
        (
            "insider_sellers_3m",
            "B",
            "count",
            "Distinct insiders selling in the open market, 3 months",
            False,
        ),
        (
            "activist_13d",
            "B",
            "flag",
            "A Schedule 13D (active holder above 5 %) filed in the last 12 months",
            True,
        ),
        (
            "short_interest_pct_float",
            "A",
            "ratio",
            "Shares sold short over the float (latest snapshot)",
            False,
        ),
        (
            "short_ratio",
            "B",
            "days",
            "Days to cover: short shares over average volume",
            False,
        ),
        (
            "short_interest_change_1m",
            "B",
            "ratio",
            "Short shares over their value one month earlier minus one",
            False,
        ),
        (
            "filings_8k_3m",
            "C",
            "count",
            "8-K current reports filed in the last 3 months",
            None,
        ),
        (
            "red_flag_8k_12m",
            "A",
            "count",
            "Structural 8-K items in 12 months: restatement, auditor change, delisting, bankruptcy, debt acceleration",
            False,
        ),
        (
            "officer_departure_12m",
            "B",
            "count",
            "8-K item 5.02 (officer or director departure) filings in 12 months",
            False,
        ),
        (
            "equity_offering_12m",
            "B",
            "flag",
            "Proceeds from issuance of common stock above 2 % of market cap in the last year",
            False,
        ),
    ],
)
STRUCTURAL_ITEMS = {"4.01", "4.02", "3.01", "1.03", "2.04"}
OFFICER_ITEM = "5.02"
OFFERING_THRESHOLD = 0.02
CLUSTER_BUYERS = 3


def _days_window(
    frame: pd.DataFrame, column: str, as_of: pd.Timestamp, days: int
) -> pd.DataFrame:
    return frame[frame[column] > as_of - pd.Timedelta(days=days)]


def _insiders(ctx: DataContext, out: pd.DataFrame) -> None:
    frame = ctx.insiders
    if frame.empty:
        return
    frame = frame[
        (~frame["is_derivative"].astype(bool))
        & frame["transaction_code"].isin(["P", "S"])
    ].copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"])
    signed = frame["value"].fillna(frame["shares"] * frame["price"]).fillna(0.0)
    frame["_signed"] = np.where(frame["transaction_code"] == "P", signed, -signed)
    for name, days in (("insider_net_buy_3m", 91), ("insider_net_buy_12m", 365)):
        window = _days_window(frame, "transaction_date", ctx.as_of_ts, days)
        out[name] = window.groupby("symbol")["_signed"].sum().reindex(out.index)
    quarter = _days_window(frame, "transaction_date", ctx.as_of_ts, 91)
    buys = quarter[quarter["transaction_code"] == "P"]
    out["insider_buyers_3m"] = (
        buys.groupby("symbol")["reporting_owner"]
        .nunique()
        .reindex(out.index)
        .fillna(0.0)
    )
    out["insider_sellers_3m"] = (
        quarter[quarter["transaction_code"] == "S"]
        .groupby("symbol")["reporting_owner"]
        .nunique()
        .reindex(out.index)
        .fillna(0.0)
    )
    month_buys = _days_window(buys, "transaction_date", ctx.as_of_ts, 30)
    cluster = (
        month_buys.groupby("symbol")["reporting_owner"].nunique() >= CLUSTER_BUYERS
    )
    out["insider_cluster_buy"] = cluster.astype(float).reindex(out.index).fillna(0.0)


def _institutions(ctx: DataContext, out: pd.DataFrame) -> None:
    summary = ctx.holdings_summary
    if summary.empty:
        return
    latest = summary.sort_values(["symbol", "period"]).groupby("symbol").last()
    out["inst_holders"] = latest["holders"].reindex(out.index)
    out["inst_holders_change"] = (
        latest["new_holders"] - latest["closed_holders"]
    ).reindex(out.index)
    prior_value = latest["total_value"] - latest["value_change"]
    out["inst_value_change"] = (
        latest["value_change"] / prior_value.replace(0.0, np.nan)
    ).reindex(out.index)
    float_shares = ctx.universe["float_shares"].astype(float)
    out["inst_share_of_float"] = (
        latest["total_shares"].reindex(out.index) / float_shares.replace(0.0, np.nan)
    ).clip(upper=5)


def _short_interest(ctx: DataContext, out: pd.DataFrame) -> None:
    universe = ctx.universe
    float_shares = universe["float_shares"].astype(float).replace(0.0, np.nan)
    out["short_interest_pct_float"] = (
        universe["short_shares"].astype(float) / float_shares
    )
    out["short_ratio"] = universe["short_ratio"].astype(float)
    snapshots = ctx.equity_snapshots
    if snapshots.empty or "short_shares" not in snapshots:
        return
    snapshots = snapshots.dropna(subset=["short_shares"]).sort_values(
        ["symbol", "as_of"]
    )
    latest = snapshots.groupby("symbol")["short_shares"].last()
    earlier = (
        snapshots[snapshots["as_of"] <= ctx.as_of_ts - pd.Timedelta(days=30)]
        .groupby("symbol")["short_shares"]
        .last()
    )
    out["short_interest_change_1m"] = (
        latest / earlier.reindex(latest.index).replace(0.0, np.nan) - 1
    ).reindex(out.index)


def _filings(ctx: DataContext, out: pd.DataFrame) -> None:
    filings = ctx.filings
    if filings.empty:
        return
    eight_k = filings[filings["form_type"].str.startswith("8-K")].copy()
    if eight_k.empty:
        return
    items = (
        eight_k["items"]
        .fillna("")
        .map(lambda s: {code.strip() for code in s.split(",") if code.strip()})
    )
    eight_k["_structural"] = items.map(lambda codes: len(codes & STRUCTURAL_ITEMS))
    eight_k["_officer"] = items.map(lambda codes: float(OFFICER_ITEM in codes))
    year = _days_window(eight_k, "filed_date", ctx.as_of_ts, 365)
    quarter = _days_window(eight_k, "filed_date", ctx.as_of_ts, 91)
    out["filings_8k_3m"] = (
        quarter.groupby("symbol").size().reindex(out.index).fillna(0.0)
    )
    out["red_flag_8k_12m"] = (
        year.groupby("symbol")["_structural"].sum().reindex(out.index).fillna(0.0)
    )
    out["officer_departure_12m"] = (
        year.groupby("symbol")["_officer"].sum().reindex(out.index).fillna(0.0)
    )


def _offerings(ctx: DataContext, out: pd.DataFrame) -> None:
    cash_flow = (
        ctx.cash_flow_quarterly
        if not ctx.cash_flow_quarterly.empty
        else ctx.cash_flow_annual
    )
    if cash_flow.empty or "proceeds_from_issuance_of_common_stock" not in cash_flow:
        return
    year = cash_flow[cash_flow["date"] > ctx.as_of_ts - pd.Timedelta(days=365)]
    proceeds = (
        year.groupby("symbol")["proceeds_from_issuance_of_common_stock"].sum().abs()
    )
    market_cap = (
        ctx.universe["market_capitalization"].astype(float).replace(0.0, np.nan)
    )
    ratio = proceeds.reindex(out.index) / market_cap
    out["equity_offering_12m"] = (
        (ratio > OFFERING_THRESHOLD).astype(float).where(ratio.notna())
    )


def _activists(ctx: DataContext, out: pd.DataFrame) -> None:
    rows = ctx.ownership_13dg
    if rows.empty:
        return
    year = _days_window(rows, "filed_date", ctx.as_of_ts, 365)
    active = year[year["form_type"].str.contains("13D", na=False)]
    out["activist_13d"] = (
        active.groupby("symbol")
        .size()
        .gt(0)
        .astype(float)
        .reindex(out.index)
        .fillna(0.0)
    )


def compute(ctx: DataContext) -> pd.DataFrame:
    out = pd.DataFrame(index=ctx.universe.index, columns=NAMES, dtype=float)
    _institutions(ctx, out)
    _insiders(ctx, out)
    _short_interest(ctx, out)
    _filings(ctx, out)
    _offerings(ctx, out)
    _activists(ctx, out)
    return out.replace([np.inf, -np.inf], np.nan)
