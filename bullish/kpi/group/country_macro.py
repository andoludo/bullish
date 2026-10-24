"""Macro KPIs per country (ISO-3), added to the equity aggregations of the ``country`` entity."""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from bullish.data.panels import DataContext
from bullish.kpi.registry import declare_many

BLOCK = "macro"
NAMES = declare_many(
    "country",
    BLOCK,
    [
        (
            "cli_change_3m",
            "B",
            "index",
            "Change of the OECD composite leading indicator over 3 months",
            True,
        ),
        ("pmi", "B", "index", "Manufacturing PMI, latest", True),
        ("gdp_fcst", "B", "pct", "Real GDP growth forecast (IMF)", True),
        (
            "gdp_nowcast_minus_fcst",
            "B",
            "pp",
            "Current-quarter nowcast minus the annual forecast",
            True,
        ),
        ("ip_yoy", "B", "pct", "Industrial production, year on year", True),
        ("cpi_yoy", "B", "pct", "Headline inflation, latest", False),
        (
            "cpi_change_6m",
            "B",
            "pp",
            "Change of headline inflation over 6 months",
            False,
        ),
        ("policy_rate", "B", "pct", "Central bank policy rate", None),
        (
            "policy_rate_change_6m",
            "B",
            "pp",
            "Change of the policy rate over 6 months (negative = easing)",
            False,
        ),
        ("real_y10", "B", "pct", "10-year inflation-indexed yield", False),
        ("curve_10_2", "B", "pp", "10-year minus 2-year yield", True),
        ("credit_gdp_gap", "B", "pp", "BIS credit-to-GDP gap", False),
        (
            "house_price_yoy",
            "B",
            "pct",
            "Residential property prices, year on year",
            None,
        ),
        ("debt_gdp", "B", "pct", "General government gross debt over GDP", False),
        ("ca_gdp", "B", "pct", "Current account balance over GDP", True),
        (
            "reer_change_6m",
            "B",
            "pct",
            "Real effective exchange rate change over 6 months (positive = stronger)",
            None,
        ),
        ("cons_conf", "C", "index", "Consumer confidence", True),
        ("biz_conf", "C", "index", "Business confidence", True),
        ("epu", "C", "index", "Economic policy uncertainty", False),
        ("gpr", "C", "index", "Geopolitical risk index", False),
        (
            "fx_vs_base_3m",
            "C",
            "ratio",
            "Currency against the base currency over 3 months (positive = stronger)",
            True,
        ),
        (
            "terms_of_trade_tailwind_3m",
            "C",
            "score",
            "Sum of (export minus import share) x 3-month commodity return",
            True,
        ),
    ],
)
CURRENCY_BY_COUNTRY: dict[str, str] = {
    "USA": "USD",
    "DEU": "EUR",
    "FRA": "EUR",
    "BEL": "EUR",
    "NLD": "EUR",
    "ITA": "EUR",
    "ESP": "EUR",
    "AUT": "EUR",
    "FIN": "EUR",
    "IRL": "EUR",
    "PRT": "EUR",
    "GRC": "EUR",
    "GBR": "GBP",
    "JPN": "JPY",
    "CHE": "CHF",
    "CAN": "CAD",
    "AUS": "AUD",
    "CHN": "CNY",
    "HKG": "HKD",
    "SWE": "SEK",
    "NOR": "NOK",
    "DNK": "DKK",
    "IND": "INR",
    "KOR": "KRW",
    "BRA": "BRL",
    "POL": "PLN",
    "MEX": "MXN",
    "ZAF": "ZAR",
    "TWN": "TWD",
    "SGP": "SGD",
    "IDN": "IDR",
    "TUR": "TRY",
}
QUARTER = 63


def _latest(series: pd.Series) -> float:
    return float(series.iloc[-1]) if len(series) else np.nan


def _change(series: pd.Series, months: int) -> float:
    return (
        float(series.iloc[-1] - series.iloc[-1 - months])
        if len(series) > months
        else np.nan
    )


def _pct_change(series: pd.Series, months: int) -> float:
    return (
        float(series.iloc[-1] / series.iloc[-1 - months] - 1)
        if len(series) > months
        else np.nan
    )


def _fx_vs_base(ctx: DataContext, country: str) -> float:
    base = ctx.settings.base_currency.upper()
    currency = CURRENCY_BY_COUNTRY.get(country)
    if currency is None or ctx.fx.empty:
        return np.nan
    if currency == base:
        return 0.0
    fx = ctx.fx
    base_per_usd = fx[base] if base in fx else pd.Series(1.0, index=fx.index)
    ccy_per_usd = (
        fx[currency]
        if currency in fx
        else (pd.Series(1.0, index=fx.index) if currency == "USD" else None)
    )
    if ccy_per_usd is None:
        return np.nan
    cross = (base_per_usd / ccy_per_usd).dropna()
    return (
        float(cross.iloc[-1] / cross.iloc[-1 - QUARTER] - 1)
        if len(cross) > QUARTER
        else np.nan
    )


def _terms_of_trade(
    ctx: DataContext, country: str, commodity_returns: pd.Series
) -> float:
    exposures = ctx.country_exposures
    if exposures.empty or commodity_returns.empty:
        return np.nan
    rows = exposures[exposures["country"] == country]
    if rows.empty:
        return np.nan
    sign = rows["direction"].map({"export": 1.0, "import": -1.0})
    contribution = sign * rows["share"] * rows["commodity"].map(commodity_returns)
    return float(contribution.sum()) if contribution.notna().any() else np.nan


def _commodity_returns(ctx: DataContext) -> pd.Series:
    close, commodities = ctx.commodity_close, ctx.commodities
    if close.empty or commodities.empty:
        return pd.Series(dtype=float)
    filled = close.ffill()
    if len(filled) <= QUARTER:
        return pd.Series(dtype=float)
    returns = filled.iloc[-1] / filled.iloc[-1 - QUARTER] - 1
    mapping = commodities.dropna(subset=["yfinance_symbol"]).set_index(
        "yfinance_symbol"
    )["code"]
    return pd.Series(
        {code: returns.get(symbol, np.nan) for symbol, code in mapping.items()},
        dtype=float,
    ).dropna()


def compute(ctx: DataContext, countries: list[str]) -> pd.DataFrame:
    commodity_returns = _commodity_returns(ctx)
    rows = {}
    for country in countries:
        s = partial(ctx.macro_series, entity=country)
        curve = s("CURVE_10_2")
        rows[country] = {
            "cli_change_3m": _change(s("CLI"), 3),
            "pmi": _latest(s("PMI_MFG")),
            "gdp_fcst": _latest(s("GDP_FCST")),
            "gdp_nowcast_minus_fcst": _latest(s("GDP_NOWCAST"))
            - _latest(s("GDP_FCST")),
            "ip_yoy": _latest(s("IP_YOY")),
            "cpi_yoy": _latest(s("CPI_YOY")),
            "cpi_change_6m": _change(s("CPI_YOY"), 6),
            "policy_rate": _latest(s("POLICY_RATE")),
            "policy_rate_change_6m": _change(s("POLICY_RATE"), 6),
            "real_y10": _latest(s("REAL_Y10")),
            "curve_10_2": (
                _latest(curve) if len(curve) else _latest(s("Y10")) - _latest(s("Y2"))
            ),
            "credit_gdp_gap": _latest(s("CREDIT_GDP_GAP")),
            "house_price_yoy": _latest(s("HOUSE_PRICE_YOY")),
            "debt_gdp": _latest(s("DEBT_GDP")),
            "ca_gdp": _latest(s("CA_GDP")),
            "reer_change_6m": (
                _pct_change(s("REER"), 6) * 100 if len(s("REER")) > 6 else np.nan
            ),
            "cons_conf": _latest(s("CONS_CONF")),
            "biz_conf": _latest(s("BIZ_CONF")),
            "epu": _latest(s("EPU")),
            "gpr": _latest(s("GPR")),
            "fx_vs_base_3m": _fx_vs_base(ctx, country),
            "terms_of_trade_tailwind_3m": _terms_of_trade(
                ctx, country, commodity_returns
            ),
        }
    frame = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=NAMES)
    frame.index.name = "country"
    return frame.astype(float)
