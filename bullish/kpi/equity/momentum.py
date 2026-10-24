"""Price-based KPIs for every equity in the universe, computed on the close panel in one pass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext, returns_over
from bullish.kpi.registry import declare_many
from bullish.utils.typed import as_float

BLOCK = "momentum"
MONTH, QUARTER, HALF, YEAR = 21, 63, 126, 252
NAMES = declare_many(
    "equity",
    BLOCK,
    [
        (
            "ret_1m",
            "A",
            "ratio",
            "Return over the last 21 trading days (short-term reversal factor)",
            True,
        ),
        ("ret_3m", "A", "ratio", "Return over the last 63 trading days", True),
        ("ret_6m", "A", "ratio", "Return over the last 126 trading days", True),
        ("ret_12m", "A", "ratio", "Return over the last 252 trading days", True),
        (
            "mom_12_1",
            "A",
            "ratio",
            "12-month return skipping the last month (Jegadeesh-Titman momentum)",
            True,
        ),
        ("mom_6_1", "A", "ratio", "6-month return skipping the last month", True),
        (
            "rs_vs_market_3m",
            "A",
            "ratio",
            "3-month return minus the universe equal-weight return",
            True,
        ),
        (
            "rs_vs_sector_3m",
            "A",
            "ratio",
            "3-month return minus the sector median return",
            True,
        ),
        (
            "rs_ratio",
            "B",
            "index",
            "Relative-strength ratio vs the universe, 100 = in line (RRG-style)",
            True,
        ),
        (
            "rs_momentum",
            "B",
            "index",
            "Rate of change of rs_ratio, 100 = flat (RRG-style)",
            True,
        ),
        (
            "vol_20d",
            "B",
            "ratio",
            "Annualised volatility of daily returns, 20 days",
            False,
        ),
        (
            "vol_60d",
            "B",
            "ratio",
            "Annualised volatility of daily returns, 60 days",
            False,
        ),
        (
            "beta_1y",
            "B",
            "ratio",
            "Beta to the universe equal-weight return over one year",
            False,
        ),
        (
            "idio_vol_1y",
            "A",
            "ratio",
            "Annualised residual volatility after the market beta (low-vol anomaly)",
            False,
        ),
        (
            "max_drawdown_1y",
            "B",
            "ratio",
            "Worst peak-to-trough loss over one year (negative)",
            True,
        ),
        (
            "sharpe_1y",
            "B",
            "ratio",
            "Annualised return over volatility, one year",
            True,
        ),
        (
            "drawdown_52w",
            "B",
            "ratio",
            "Last close over the 52-week high minus one (negative)",
            True,
        ),
        (
            "excess_drawdown_vs_sector",
            "B",
            "ratio",
            "drawdown_52w minus the sector median drawdown",
            True,
        ),
        (
            "dist_52w_low",
            "B",
            "ratio",
            "Last close over the 52-week low minus one",
            True,
        ),
        (
            "price_vs_sma50",
            "B",
            "ratio",
            "Last close over the 50-day average minus one",
            True,
        ),
        (
            "price_vs_sma200",
            "B",
            "ratio",
            "Last close over the 200-day average minus one",
            True,
        ),
        (
            "sma50_vs_sma200",
            "B",
            "ratio",
            "50-day over 200-day average minus one (golden cross when positive)",
            True,
        ),
        ("rsi_14", "B", "index", "Wilder RSI, 14 days", None),
        (
            "rsi_turning_up",
            "B",
            "flag",
            "RSI was below 35 five days ago and is higher now",
            True,
        ),
        (
            "bollinger_pct_b",
            "B",
            "ratio",
            "Position inside the 20-day Bollinger band (0 = lower band)",
            None,
        ),
        (
            "above_20d_high",
            "B",
            "flag",
            "Last close at or above the previous 20-day high",
            True,
        ),
        (
            "volume_ratio_5d_60d",
            "B",
            "ratio",
            "Average volume of the last 5 days over the 60-day average",
            None,
        ),
        (
            "capitulation_volume",
            "B",
            "ratio",
            "Largest down-day volume of the last 10 days over the 60-day average",
            None,
        ),
        (
            "up_down_volume_20d",
            "B",
            "ratio",
            "Volume on up days over volume on down days, 20 days",
            True,
        ),
        (
            "avg_dollar_volume_60d",
            "B",
            "currency",
            "Average daily traded value, 60 days, in the base currency",
            True,
        ),
        (
            "amihud_60d",
            "B",
            "ratio",
            "Amihud illiquidity: mean |return| per million traded, 60 days",
            False,
        ),
        (
            "price_history_days",
            "C",
            "days",
            "Trading days of price history available",
            True,
        ),
    ],
)


def _rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _beta_and_residual(
    rets: pd.DataFrame, market: pd.Series
) -> tuple[pd.Series, pd.Series]:
    window = rets.iloc[-YEAR:]
    mkt = market.iloc[-YEAR:]
    centred = window - window.mean()
    mkt_centred = mkt - mkt.mean()
    var = float((mkt_centred**2).sum())
    if var == 0 or len(mkt) < QUARTER:
        nan = pd.Series(np.nan, index=rets.columns)
        return nan, nan
    beta = centred.mul(mkt_centred, axis=0).sum() / var
    residual = window - np.outer(mkt, beta)
    return beta, residual.std() * np.sqrt(YEAR)


def _group_median_excess(values: pd.Series, groups: pd.Series) -> pd.Series:
    medians = values.groupby(groups).transform("median")
    return values - medians


def compute(ctx: DataContext) -> pd.DataFrame:  # noqa: PLR0915
    close = ctx.close.ffill()
    symbols = list(ctx.universe.index)
    out = pd.DataFrame(
        index=pd.Index(symbols, name="symbol"), columns=NAMES, dtype=float
    )
    if close.empty or len(close) < 2:
        return out
    volume = ctx.volume.reindex(columns=close.columns)
    rets = close.pct_change()
    market = rets.mean(axis=1)
    last = close.iloc[-1]

    out["ret_1m"] = returns_over(close, MONTH)
    out["ret_3m"] = returns_over(close, QUARTER)
    out["ret_6m"] = returns_over(close, HALF)
    out["ret_12m"] = returns_over(close, YEAR)
    if len(close) > YEAR:
        out["mom_12_1"] = close.iloc[-1 - MONTH] / close.iloc[-1 - YEAR] - 1
    if len(close) > HALF:
        out["mom_6_1"] = close.iloc[-1 - MONTH] / close.iloc[-1 - HALF] - 1
    market_3m = (
        as_float(market.iloc[-QUARTER:].add(1).prod()) - 1
        if len(market) >= QUARTER
        else np.nan
    )
    out["rs_vs_market_3m"] = out["ret_3m"] - market_3m
    sector = ctx.universe["sector"].reindex(out.index)
    out["rs_vs_sector_3m"] = _group_median_excess(out["ret_3m"], sector)

    market_index = (1 + market.fillna(0.0)).cumprod()
    rs = close.div(market_index, axis=0)
    rs_ratio = 100 * rs / rs.rolling(QUARTER).mean()
    rs_momentum = 100 * rs_ratio / rs_ratio.shift(MONTH)
    out["rs_ratio"] = rs_ratio.iloc[-1]
    out["rs_momentum"] = rs_momentum.iloc[-1]

    out["vol_20d"] = rets.iloc[-20:].std() * np.sqrt(YEAR)
    out["vol_60d"] = rets.iloc[-60:].std() * np.sqrt(YEAR)
    beta, idio = _beta_and_residual(rets, market)
    out["beta_1y"] = beta
    out["idio_vol_1y"] = idio
    window = close.iloc[-YEAR:]
    out["max_drawdown_1y"] = (window / window.cummax() - 1).min()
    yearly = rets.iloc[-YEAR:]
    out["sharpe_1y"] = yearly.mean() / yearly.std().replace(0.0, np.nan) * np.sqrt(YEAR)
    out["drawdown_52w"] = last / window.max() - 1
    out["excess_drawdown_vs_sector"] = _group_median_excess(out["drawdown_52w"], sector)
    out["dist_52w_low"] = last / window.min() - 1

    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1]
    out["price_vs_sma50"] = last / sma50 - 1
    out["price_vs_sma200"] = last / sma200 - 1
    out["sma50_vs_sma200"] = sma50 / sma200 - 1

    rsi = _rsi(close)
    out["rsi_14"] = rsi.iloc[-1]
    if len(rsi) > 5:
        earlier = rsi.iloc[-6]
        out["rsi_turning_up"] = ((earlier < 35) & (rsi.iloc[-1] > earlier)).astype(
            float
        )
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    out["bollinger_pct_b"] = (
        (last - (sma20.iloc[-1] - 2 * std20.iloc[-1])) / (4 * std20.iloc[-1])
    ).replace([np.inf, -np.inf], np.nan)
    if len(close) > 20:
        out["above_20d_high"] = (last >= close.iloc[-21:-1].max()).astype(float)

    vol60 = volume.iloc[-60:].mean().replace(0.0, np.nan)
    out["volume_ratio_5d_60d"] = volume.iloc[-5:].mean() / vol60
    down_days = rets.iloc[-10:] < 0
    out["capitulation_volume"] = (volume.iloc[-10:].where(down_days) / vol60).max()
    recent = rets.iloc[-20:]
    up_volume = volume.iloc[-20:].where(recent > 0).sum()
    down_volume = volume.iloc[-20:].where(recent < 0).sum().replace(0.0, np.nan)
    out["up_down_volume_20d"] = up_volume / down_volume
    dollar = (close * volume).iloc[-60:]
    local_dollar_volume = dollar.mean()
    out["avg_dollar_volume_60d"] = ctx.to_base(
        local_dollar_volume, ctx.universe["currency"].reindex(out.index)
    )
    out["amihud_60d"] = (
        rets.iloc[-60:].abs() / dollar.replace(0.0, np.nan)
    ).mean() * 1e6
    out["price_history_days"] = close.notna().sum()
    return out.replace([np.inf, -np.inf], np.nan)
