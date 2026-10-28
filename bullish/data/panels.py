"""Read-only loaders over the bearish tables.

Everything the KPI engine needs is loaded here, once per run, into pandas
frames keyed by symbol so that the KPI blocks stay vectorised. Loaders are
point-in-time: prices stop at ``as_of`` and statements are only visible
once filed (``filed_date``; a configurable lag when the provider gave none).
"""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from bullish.config import BullishSettings

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb

logger = logging.getLogger(__name__)

SOURCE_PRIORITY = (
    "SEC",
    "Yfinance",
    "YahooQuery",
    "Tiingo",
    "FinanceDatabase",
    "investpy",
)
CHUNK = 500
TRADING_DAYS = 252


def _quote(values: Iterable[str]) -> str:
    return ", ".join("'" + value.replace("'", "''") + "'" for value in values)


def _chunks(values: Sequence[str], size: int = CHUNK) -> list[Sequence[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _source_rank(frame: pd.DataFrame) -> pd.Series:
    order = {source: rank for rank, source in enumerate(SOURCE_PRIORITY)}
    return frame["source"].map(lambda s: order.get(s, len(order))).astype(int)


def dedupe_by_source(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """One row per ``keys``: the best source wins, other sources fill its gaps."""
    if frame.empty:
        return frame
    ordered = frame.assign(_rank=_source_rank(frame)).sort_values([*keys, "_rank"])
    merged = ordered.groupby(keys, sort=False).first().reset_index()
    return merged.drop(columns=["_rank"], errors="ignore")


def _dates(frame: pd.DataFrame, column: str = "date") -> pd.DataFrame:
    if frame.empty:
        return frame
    frame[column] = (
        pd.to_datetime(frame[column], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    return frame


@dataclass
class DataContext:
    """Lazy, cached access to every frame a KPI run needs."""

    db: BullishDb
    settings: BullishSettings
    as_of: datetime.date
    countries: list[str] | None = None
    symbols: list[str] | None = None
    _price_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    # --- helpers ---------------------------------------------------------

    @property
    def as_of_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.as_of)

    @property
    def start_ts(self) -> pd.Timestamp:
        return self.as_of_ts - pd.DateOffset(years=self.settings.lookback_years)

    def query(self, sql: str) -> pd.DataFrame:
        frame: pd.DataFrame = self.db.read_query(sql)
        return frame

    def _symbol_filter(self, column: str = "symbol") -> str:
        return (
            f" AND {column} IN ({_quote(self.universe.index)})"
            if len(self.universe)
            else " AND 1 = 0"
        )

    # --- universe --------------------------------------------------------

    @cached_property
    def reference(self) -> pd.DataFrame:
        """One row per symbol of the equity table: reference data with canonical sector / industry / country.

        This is every equity bearish knows, whether or not it has data; the KPI
        run works on :attr:`universe`, the subset that has prices.
        """
        sql = """
            SELECT e.symbol, e.source, e.name, e.currency, e.exchange, e.market_capitalization,
                   e.shares_outstanding, e.float_shares, e.short_shares, e.short_ratio,
                   e.trailing_price_to_earnings, e.forward_price_to_earnings, e.trailing_peg_ratio,
                   e.price_to_book, e.trailing_price_to_sales, e.dividend_yield,
                   s.gics_sector AS sector, s.gics_industry AS industry, c.iso3 AS country
            FROM equity AS e
            LEFT JOIN equitysector AS s ON s.symbol = e.symbol AND s.source = e.source
            LEFT JOIN equitycountry AS c ON c.symbol = e.symbol AND c.source = e.source
        """
        frame = self.query(sql)
        if frame.empty:
            return frame.set_index("symbol") if "symbol" in frame else frame
        frame = dedupe_by_source(frame, ["symbol"])
        return frame.set_index("symbol").sort_index()

    def _priced_symbols(self, table: str) -> pd.Index:
        """Symbols with at least ``min_price_rows`` rows of ``table`` inside the lookback."""
        floor = self.settings.min_price_rows
        sql = (
            f'SELECT symbol FROM "{table}" '  # noqa: S608
            f"WHERE date >= '{self.start_ts.date()}' AND date <= '{self.as_of} 23:59:59' "
            f"GROUP BY symbol HAVING COUNT(*) >= {int(floor)}"
        )
        return pd.Index(self.query(sql)["symbol"].astype(str))

    @cached_property
    def universe(self) -> pd.DataFrame:
        """The equities the KPI run covers: :attr:`reference` rows that have prices, filtered by
        ``countries`` / ``symbols`` when given."""
        frame = self.reference
        if frame.empty:
            return frame
        if self.countries:
            frame = frame[frame["country"].isin(self.countries)]
        if self.symbols:
            frame = frame[frame.index.isin(self.symbols)]
        if self.settings.min_price_rows > 0:
            priced = self._priced_symbols(self.settings.price_table)
            dropped = len(frame) - int(frame.index.isin(priced).sum())
            if dropped:
                logger.info(
                    f"{dropped} equities skipped: fewer than "
                    f"{self.settings.min_price_rows} price rows in the lookback"
                )
            frame = frame[frame.index.isin(priced)]
        return frame

    def peer_group(self, name: str | None = None) -> pd.Series:
        column = name or self.settings.peer_group
        return self.universe[column]

    # --- prices ----------------------------------------------------------

    def _load_prices(self, table: str, symbols: Sequence[str]) -> pd.DataFrame:
        frames = []
        for chunk in _chunks(list(symbols)):
            sql = (
                f'SELECT symbol, source, date, open, high, low, close, volume, dividends, stock_splits FROM "{table}" '  # noqa: S608
                f"WHERE symbol IN ({_quote(chunk)}) AND date >= '{self.start_ts.date()}' "
                f"AND date <= '{self.as_of} 23:59:59'"
            )
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "source",
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "dividends",
                    "stock_splits",
                ]
            )
        return dedupe_by_source(_dates(frame), ["symbol", "date"])

    def price_panel(
        self, table: str, symbols: Sequence[str], column: str = "close"
    ) -> pd.DataFrame:
        """Wide ``date x symbol`` frame of ``column`` from ``table`` (cached per table)."""
        key = f"{table}:{','.join(sorted(symbols))}"
        if key not in self._price_cache:
            self._price_cache[key] = self._load_prices(table, symbols)
        long = self._price_cache[key]
        if long.empty:
            return pd.DataFrame()
        panel = long.pivot(  # noqa: PD010
            index="date", columns="symbol", values=column
        ).sort_index()
        panel.columns = panel.columns.astype(str)
        return panel

    @cached_property
    def close(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "close"
        )

    @cached_property
    def open(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "open"
        )

    @cached_property
    def high(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "high"
        )

    @cached_property
    def low(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "low"
        )

    @cached_property
    def volume(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "volume"
        )

    @cached_property
    def dividends(self) -> pd.DataFrame:
        return self.price_panel(
            self.settings.price_table, list(self.universe.index), "dividends"
        )

    @cached_property
    def last_price(self) -> pd.Series:
        return (
            self.close.ffill().iloc[-1] if len(self.close) else pd.Series(dtype=float)
        )

    @cached_property
    def index_close(self) -> pd.DataFrame:
        symbols = self.query("SELECT DISTINCT symbol FROM priceindex")[
            "symbol"
        ].tolist()
        return self.price_panel("priceindex", symbols) if symbols else pd.DataFrame()

    @cached_property
    def etf_close(self) -> pd.DataFrame:
        return (
            self.price_panel("priceetf", list(self.etf_universe.index), "close")
            if len(self.etf_universe)
            else pd.DataFrame()
        )

    @cached_property
    def etf_volume(self) -> pd.DataFrame:
        return (
            self.price_panel("priceetf", list(self.etf_universe.index), "volume")
            if len(self.etf_universe)
            else pd.DataFrame()
        )

    @cached_property
    def commodity_close(self) -> pd.DataFrame:
        symbols = self.query("SELECT DISTINCT symbol FROM pricecommodity")[
            "symbol"
        ].tolist()
        return (
            self.price_panel("pricecommodity", symbols) if symbols else pd.DataFrame()
        )

    # --- fx --------------------------------------------------------------

    @cached_property
    def fx(self) -> pd.DataFrame:
        """``date x quote`` units of ``quote`` per one USD."""
        frame = self.query(
            f"SELECT date, quote, rate FROM fxrate WHERE date <= '{self.as_of} 23:59:59'"  # noqa: S608
        )
        if frame.empty:
            return pd.DataFrame()
        frame = _dates(frame)
        return (
            frame.pivot_table(
                index="date", columns="quote", values="rate", aggfunc="last"
            )
            .sort_index()
            .ffill()
        )

    def usd_rate(self, currency: str | None) -> float:
        """Latest units of ``currency`` per USD (1.0 for USD or unknown)."""
        if (
            not currency
            or currency.upper() == "USD"
            or self.fx.empty
            or currency.upper() not in self.fx.columns
        ):
            return 1.0
        value = self.fx[currency.upper()].dropna()
        return float(value.iloc[-1]) if len(value) else 1.0

    def to_base(self, amount: pd.Series, currencies: pd.Series) -> pd.Series:
        """Convert ``amount`` (per symbol, in ``currencies``) into the base currency."""
        base_per_usd = self.usd_rate(self.settings.base_currency)
        rates = currencies.map(self.usd_rate).astype(float)
        return amount / rates * base_per_usd

    # --- statements ------------------------------------------------------

    def _statements(self, table: str) -> pd.DataFrame:
        frames = []
        for chunk in _chunks(list(self.universe.index)):
            sql = f'SELECT * FROM "{table}" WHERE symbol IN ({_quote(chunk)})'  # noqa: S608
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return frame
        frame = _dates(frame)
        frame = _dates(frame, "filed_date") if "filed_date" in frame else frame
        lag = pd.Timedelta(days=self.settings.statement_lag_days)
        visible = (
            frame["filed_date"].fillna(frame["date"] + lag)
            if "filed_date" in frame
            else frame["date"] + lag
        )
        frame = frame[visible <= self.as_of_ts]
        frame = dedupe_by_source(frame, ["symbol", "date"])
        return frame.sort_values(["symbol", "date"]).drop(
            columns=["created_at"], errors="ignore"
        )

    @cached_property
    def metrics_annual(self) -> pd.DataFrame:
        return self._statements("financialmetrics")

    @cached_property
    def balance_annual(self) -> pd.DataFrame:
        return self._statements("balancesheet")

    @cached_property
    def cash_flow_annual(self) -> pd.DataFrame:
        return self._statements("cashflow")

    @cached_property
    def metrics_quarterly(self) -> pd.DataFrame:
        return self._statements("quarterlyfinancialmetrics")

    @cached_property
    def balance_quarterly(self) -> pd.DataFrame:
        return self._statements("quarterlybalancesheet")

    @cached_property
    def cash_flow_quarterly(self) -> pd.DataFrame:
        return self._statements("quarterlycashflow")

    # --- expectations ----------------------------------------------------

    def _symbol_table(self, table: str, date_column: str = "date") -> pd.DataFrame:
        frames = []
        for chunk in _chunks(list(self.universe.index)):
            sql = (
                f'SELECT * FROM "{table}" WHERE symbol IN ({_quote(chunk)}) '  # noqa: S608
                f"AND {date_column} <= '{self.as_of} 23:59:59'"
            )
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return _dates(frame, date_column) if not frame.empty else frame

    @cached_property
    def earnings(self) -> pd.DataFrame:
        """Earnings dates up to ``as_of`` plus the next scheduled one per symbol."""
        past = self._symbol_table("earningsdate")
        frames = []
        for chunk in _chunks(list(self.universe.index)):
            frames.append(  # noqa: PERF401
                self.query(
                    f"SELECT * FROM earningsdate WHERE symbol IN ({_quote(chunk)}) "  # noqa: S608
                    f"AND date > '{self.as_of} 23:59:59'"
                )
            )
        future = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        future = _dates(future) if not future.empty else future
        return (
            pd.concat([past, future], ignore_index=True) if not future.empty else past
        )

    @cached_property
    def price_targets(self) -> pd.DataFrame:
        return self._symbol_table("pricetarget")

    @cached_property
    def recommendations(self) -> pd.DataFrame:
        return self._symbol_table("analystrecommendation")

    # --- ownership -------------------------------------------------------

    @cached_property
    def insiders(self) -> pd.DataFrame:
        return self._symbol_table("insidertransaction", "filed_date")

    @cached_property
    def holdings_summary(self) -> pd.DataFrame:
        frame = self._symbol_table("holding13fsummary", "latest_filed_date")
        return _dates(frame, "period") if not frame.empty else frame

    @cached_property
    def ownership_13dg(self) -> pd.DataFrame:
        return self._symbol_table("ownership13dg", "filed_date")

    @cached_property
    def filings(self) -> pd.DataFrame:
        return self._symbol_table("filing", "filed_date")

    @cached_property
    def equity_snapshots(self) -> pd.DataFrame:
        frame = self._symbol_table("equitysnapshot", "as_of")
        return (
            dedupe_by_source(frame, ["symbol", "as_of"]) if not frame.empty else frame
        )

    # --- etfs ------------------------------------------------------------

    @cached_property
    def etfs(self) -> pd.DataFrame:
        frame = self.query("SELECT * FROM etf")
        if frame.empty:
            return frame
        frame = dedupe_by_source(frame, ["symbol"])
        return frame.set_index("symbol").sort_index()

    @cached_property
    def etf_universe(self) -> pd.DataFrame:
        """The funds the KPI run covers: :attr:`etfs` rows with enough ``priceetf`` history."""
        if self.settings.min_price_rows <= 0:
            return self.etfs
        priced = self._priced_symbols("priceetf")
        if priced.empty:
            return pd.DataFrame(columns=[]).rename_axis("symbol")
        frames = []
        for chunk in _chunks(list(priced)):
            sql = f"SELECT * FROM etf WHERE symbol IN ({_quote(chunk)})"  # noqa: S608
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True)
        if frame.empty:
            return frame.rename_axis("symbol")
        frame = dedupe_by_source(frame, ["symbol"])
        return frame.set_index("symbol").sort_index()

    @cached_property
    def etf_snapshots(self) -> pd.DataFrame:
        frame = self.query(
            f"SELECT * FROM etfsnapshot WHERE as_of <= '{self.as_of}'"  # noqa: S608
        )
        return (
            dedupe_by_source(_dates(frame, "as_of"), ["symbol", "as_of"])
            if not frame.empty
            else frame
        )

    @cached_property
    def etf_exposures(self) -> pd.DataFrame:
        return self.query(
            "SELECT symbol, entity_type, entity, weight, method FROM etfexposure"
        )

    # --- links and macro -------------------------------------------------

    @cached_property
    def sector_sensitivities(self) -> pd.DataFrame:
        return self.query(
            "SELECT sector, commodity, direction, strength FROM sectorcommoditysensitivity"
        )

    @cached_property
    def commodities(self) -> pd.DataFrame:
        return self.query('SELECT code, name, "group", yfinance_symbol FROM commodity')

    @cached_property
    def country_exposures(self) -> pd.DataFrame:
        return self.query(
            "SELECT country, commodity, direction, share FROM countrycommodityexposure"
        )

    @cached_property
    def index_members(self) -> pd.DataFrame:
        frame = self.query(
            f"SELECT index_symbol, symbol, as_of FROM indexmember WHERE as_of <= '{self.as_of}'"  # noqa: S608
        )
        if frame.empty:
            return frame
        latest = frame.groupby("index_symbol")["as_of"].transform("max")
        return frame[frame["as_of"] == latest]

    @cached_property
    def macro(self) -> pd.DataFrame:
        """Latest vintage known at ``as_of`` of every macro series: long ``series_id, date, value``."""
        sql = (
            "SELECT series_id, date, value, as_of FROM macroobservation "  # noqa: S608
            f"WHERE as_of <= '{self.as_of}' AND date <= '{self.as_of}'"
        )
        frame = self.query(sql)
        if frame.empty:
            return pd.DataFrame(columns=["series_id", "date", "value"])
        frame = frame.sort_values(["series_id", "date", "as_of"]).drop_duplicates(
            ["series_id", "date"], keep="last"
        )
        return _dates(frame)[["series_id", "date", "value"]]

    def macro_series(self, concept: str, entity: str) -> pd.Series:
        rows = self.macro[self.macro["series_id"] == f"{concept}.{entity}"]
        return (
            rows.set_index("date")["value"].sort_index()
            if len(rows)
            else pd.Series(dtype=float)
        )

    @cached_property
    def coverage(self) -> pd.DataFrame:
        return self.query(
            "SELECT symbol, table_name, source, first_date, last_date, rows, as_of FROM coverage"
        )


def returns_over(close: pd.DataFrame, days: int) -> pd.Series:
    """Simple return over the last ``days`` trading rows of a wide close panel."""
    filled = close.ffill()
    if len(filled) <= days:
        return pd.Series(np.nan, index=close.columns)
    return filled.iloc[-1] / filled.iloc[-1 - days] - 1


def trailing_window(close: pd.DataFrame, days: int) -> pd.DataFrame:
    return close.ffill().iloc[-days:]
