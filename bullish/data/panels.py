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
from bullish.scoring.coverage import CoverageIndex

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
    parent: DataContext | None = field(default=None, repr=False)
    _price_cache: dict[str, pd.DataFrame] = field(default_factory=dict)
    _table_cache: dict[str, pd.DataFrame] = field(default_factory=dict)

    def derive(
        self, as_of: datetime.date, settings: BullishSettings | None = None
    ) -> DataContext:
        """A point-in-time context for an earlier ``as_of`` that slices this one's frames instead
        of querying again.

        ``settings`` are the run's own (notably its ``lookback_years``); this
        span context must have been loaded at or after ``as_of`` with a lookback
        reaching ``as_of - settings.lookback_years``.
        :func:`bullish.backtest.rank.build_history` builds one span per history.
        """
        if as_of > self.as_of:
            raise ValueError(f"{as_of} is after the span context's {self.as_of}")
        settings = settings or self.settings
        if (
            pd.Timestamp(as_of) - pd.DateOffset(years=settings.lookback_years)
            < self.start_ts
        ):
            raise ValueError(
                f"the span's lookback starts at {self.start_ts.date()}, after the "
                f"{settings.lookback_years}-year window of {as_of}"
            )
        return DataContext(
            db=self.db,
            settings=settings,
            as_of=as_of,
            countries=self.countries,
            symbols=self.symbols,
            parent=self,
        )

    def _up_to(self, frame: pd.DataFrame, column: str) -> pd.DataFrame:
        """Rows of a parent frame dated at or before ``as_of`` (``column`` already normalised)."""
        if frame.empty or column not in frame:
            return frame
        return frame[frame[column] <= self.as_of_ts]

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

    # --- universe --------------------------------------------------------

    @cached_property
    def reference(self) -> pd.DataFrame:
        """One row per symbol of the equity table: reference data with canonical sector / industry / country.

        This is every equity bearish knows, whether or not it has data; the KPI
        run works on :attr:`universe`, the subset that has prices.
        """
        if self.parent is not None:
            return self.parent.reference
        return self._reference_rows(None)

    def _reference_rows(self, symbols: Sequence[str] | None) -> pd.DataFrame:
        """Deduped equity rows joined to the canonical sector / country, for ``symbols`` or all."""
        base = """
            SELECT e.symbol, e.source, e.name, e.currency, e.exchange, e.market_capitalization,
                   e.shares_outstanding, e.float_shares, e.short_shares, e.short_ratio,
                   e.trailing_price_to_earnings, e.forward_price_to_earnings, e.trailing_peg_ratio,
                   e.price_to_book, e.trailing_price_to_sales, e.dividend_yield,
                   s.gics_sector AS sector, s.gics_industry AS industry, c.iso3 AS country
            FROM equity AS e
            LEFT JOIN equitysector AS s ON s.symbol = e.symbol AND s.source = e.source
            LEFT JOIN equitycountry AS c ON c.symbol = e.symbol AND c.source = e.source
        """
        if symbols is None:
            frame = self.query(base)
        else:
            frames = [
                self.query(f"{base} WHERE e.symbol IN ({_quote(chunk)})")
                for chunk in _chunks(list(symbols))
            ]
            frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return pd.DataFrame(columns=["symbol"]).set_index("symbol")
        frame = dedupe_by_source(frame, ["symbol"])
        return frame.set_index("symbol").sort_index()

    def _priced_symbols(self, table: str) -> pd.Index:
        """Symbols with at least ``min_price_rows`` rows of ``table`` inside the lookback."""
        floor = self.settings.min_price_rows
        if self.parent is not None and table in self.parent._price_cache:
            long = self.parent._price_cache[table]
            window = long[
                (long["date"] >= self.start_ts) & (long["date"] <= self.as_of_ts)
            ]
            counts = window.groupby("symbol").size()
            return pd.Index(counts[counts >= floor].index.to_numpy().astype(str))
        sql = (
            f'SELECT symbol FROM "{table}" '  # noqa: S608
            f"WHERE date >= '{self.start_ts.date()}' AND date <= '{self.as_of} 23:59:59' "
            f"GROUP BY symbol HAVING COUNT(*) >= {int(floor)}"
        )
        return pd.Index(self.query(sql)["symbol"].astype(str))

    @cached_property
    def universe(self) -> pd.DataFrame:
        """The equities the KPI run covers: equity rows that have prices, filtered by
        ``countries`` / ``symbols`` when given.

        Only the priced symbols' rows are read (the full table is :attr:`reference`).
        """
        priced: pd.Index | None = None
        if self.settings.min_price_rows > 0:
            priced = self._priced_symbols(self.settings.price_table)
            if self.symbols:
                priced = priced[priced.isin(self.symbols)]
            if priced.empty:
                logger.info(
                    f"No equity has {self.settings.min_price_rows} price rows in the lookback"
                )
                return pd.DataFrame(columns=["symbol"]).set_index("symbol")
        wanted = priced if priced is not None else self.symbols
        frame = (
            self.parent.universe
            if self.parent is not None
            else self._reference_rows(None if wanted is None else list(wanted))
        )
        if frame.empty:
            return frame
        if wanted is not None:
            frame = frame[frame.index.isin(wanted)]
        if self.countries:
            frame = frame[frame["country"].isin(self.countries)]
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

    def _long_prices(self, table: str, symbols: Sequence[str]) -> pd.DataFrame:
        """Long price rows of ``table`` for ``symbols`` inside the lookback, loaded once per table.

        A derived context slices its parent's rows (``symbol``, date window)
        instead of querying.
        """
        if self.parent is not None:
            long = self.parent._long_prices(table, symbols)
            if long.empty:
                return long
            return long[
                long["symbol"].isin(symbols)
                & (long["date"] >= self.start_ts)
                & (long["date"] <= self.as_of_ts)
            ]
        if table not in self._price_cache:
            self._price_cache[table] = self._load_prices(table, symbols)
        long = self._price_cache[table]
        if long.empty or set(symbols) <= set(long["symbol"].unique()):
            return long
        missing = sorted(set(symbols) - set(long["symbol"].unique()))
        extra = self._load_prices(table, missing)
        long = pd.concat([long, extra], ignore_index=True) if not extra.empty else long
        self._price_cache[table] = long
        return long

    def price_panel(
        self, table: str, symbols: Sequence[str], column: str = "close"
    ) -> pd.DataFrame:
        """Wide ``date x symbol`` frame of ``column`` from ``table`` for ``symbols``."""
        long = self._long_prices(table, symbols)
        if long.empty:
            return pd.DataFrame()
        long = long[long["symbol"].isin(symbols)]
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
        if self.parent is not None:
            fx = self.parent.fx
            return fx[fx.index <= self.as_of_ts] if not fx.empty else fx
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

    def _raw_statements(self, table: str) -> pd.DataFrame:
        """Every row of ``table`` for the universe, dates normalised, no visibility filter."""
        if self.parent is not None:
            return self.parent._raw_statements(table)
        if table in self._table_cache:
            return self._table_cache[table]
        frames = []
        for chunk in _chunks(list(self.universe.index)):
            sql = f'SELECT * FROM "{table}" WHERE symbol IN ({_quote(chunk)})'  # noqa: S608
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not frame.empty:
            frame = _dates(frame)
            frame = _dates(frame, "filed_date") if "filed_date" in frame else frame
        self._table_cache[table] = frame
        return frame

    def _statements(self, table: str) -> pd.DataFrame:
        frame = self._raw_statements(table)
        if frame.empty:
            return frame
        if self.parent is not None:
            frame = frame[frame["symbol"].isin(self.universe.index)]
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
        if self.parent is not None:
            frame = self.parent._symbol_table(table, date_column)
            if frame.empty:
                return frame
            return self._up_to(
                frame[frame["symbol"].isin(self.universe.index)], date_column
            )
        key = f"{table}:{date_column}"
        if key in self._table_cache:
            return self._table_cache[key]
        frames = []
        for chunk in _chunks(list(self.universe.index)):
            sql = (
                f'SELECT * FROM "{table}" WHERE symbol IN ({_quote(chunk)}) '  # noqa: S608
                f"AND {date_column} <= '{self.as_of} 23:59:59'"
            )
            frames.append(self.query(sql))
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        frame = _dates(frame, date_column) if not frame.empty else frame
        self._table_cache[key] = frame
        return frame

    @cached_property
    def earnings(self) -> pd.DataFrame:
        """Earnings dates up to ``as_of`` plus the next scheduled one per symbol."""
        if self.parent is not None:
            frame = self.parent.earnings
            return frame[frame["symbol"].isin(self.universe.index)]
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
        if self.parent is not None:
            return self.parent.etfs
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
        if self.parent is not None:
            priced = self._priced_symbols("priceetf")
            return self.parent.etf_universe.loc[
                self.parent.etf_universe.index.isin(priced)
            ]
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
        if self.parent is not None:
            return self._up_to(self.parent.etf_snapshots, "as_of")
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
        if self.parent is not None:
            return self.parent.etf_exposures
        return self.query(
            "SELECT symbol, entity_type, entity, weight, method FROM etfexposure"
        )

    # --- links and macro -------------------------------------------------

    @cached_property
    def sector_sensitivities(self) -> pd.DataFrame:
        if self.parent is not None:
            return self.parent.sector_sensitivities
        return self.query(
            "SELECT sector, commodity, direction, strength FROM sectorcommoditysensitivity"
        )

    @cached_property
    def commodities(self) -> pd.DataFrame:
        if self.parent is not None:
            return self.parent.commodities
        return self.query('SELECT code, name, "group", yfinance_symbol FROM commodity')

    @cached_property
    def country_exposures(self) -> pd.DataFrame:
        if self.parent is not None:
            return self.parent.country_exposures
        return self.query(
            "SELECT country, commodity, direction, share FROM countrycommodityexposure"
        )

    @cached_property
    def index_members(self) -> pd.DataFrame:
        if self.parent is not None:
            frame = self.parent._index_members_raw
            frame = (
                frame[frame["as_of"] <= str(self.as_of)] if not frame.empty else frame
            )
        else:
            frame = self._index_members_raw
        if frame.empty:
            return frame
        latest = frame.groupby("index_symbol")["as_of"].transform("max")
        return frame[frame["as_of"] == latest]

    @cached_property
    def _index_members_raw(self) -> pd.DataFrame:
        return self.query(
            f"SELECT index_symbol, symbol, as_of FROM indexmember WHERE as_of <= '{self.as_of}'"  # noqa: S608
        )

    @cached_property
    def _macro_raw(self) -> pd.DataFrame:
        """Every macro vintage known at ``as_of``: ``series_id, date, value, as_of``."""
        if self.parent is not None:
            frame = self.parent._macro_raw
            if frame.empty:
                return frame
            return frame[
                (frame["as_of"] <= str(self.as_of)) & (frame["date"] <= str(self.as_of))
            ]
        return self.query(
            "SELECT series_id, date, value, as_of FROM macroobservation "  # noqa: S608
            f"WHERE as_of <= '{self.as_of}' AND date <= '{self.as_of}'"
        )

    @cached_property
    def macro(self) -> pd.DataFrame:
        """Latest vintage known at ``as_of`` of every macro series: long ``series_id, date, value``."""
        frame = self._macro_raw
        if frame.empty:
            return pd.DataFrame(columns=["series_id", "date", "value"])
        frame = frame.sort_values(["series_id", "date", "as_of"]).drop_duplicates(
            ["series_id", "date"], keep="last"
        )
        return _dates(frame)[["series_id", "date", "value"]]

    @cached_property
    def _macro_by_series(self) -> dict[str, pd.Series]:
        """``series_id -> date-indexed values``, split once; every KPI block looks series up here."""
        frame = self.macro
        if frame.empty:
            return {}
        return {
            str(series_id): group.set_index("date")["value"].sort_index()
            for series_id, group in frame.groupby("series_id", sort=False)
        }

    def macro_series(self, concept: str, entity: str) -> pd.Series:
        return self._macro_by_series.get(f"{concept}.{entity}", pd.Series(dtype=float))

    @cached_property
    def coverage_index(self) -> CoverageIndex:
        """Per-symbol census aggregate, built once per context."""
        if self.parent is not None:
            return self.parent.coverage_index
        return CoverageIndex(self.coverage)

    @cached_property
    def coverage(self) -> pd.DataFrame:
        if self.parent is not None:
            return self.parent.coverage
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
