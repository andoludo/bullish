# Plan: making `bullish compute` fast

Status: proposal, 2026-10-24. Measured on the LFS test database
(`tests/data/bear.db`: 107,366 equity symbols, prices for 6 of them,
1,256 trading days of history) in a 4-core, 16 GB container, and on
`tests/data/prices_msft.csv` for the per-symbol indicator cost.

## 1. What was measured

### A full run on the test database does not finish

`bullish compute` on the unrestricted database was killed by the kernel
(exit 137, out of memory) in the momentum block, after ~90 s. Memory, not
CPU, is the first wall.

### Per-phase wall time, restricted universes

Universe = the 6 priced symbols plus N unpriced ones (`DataContext(symbols=...)`).

| phase | 6 symbols | 2,006 symbols | 10,006 symbols |
|---|---:|---:|---:|
| universe load (SQL join + dedupe) | 3.3 s | 2.7 s | 2.2 s |
| price panels (close, open, high, low, volume) | 0.1 s | 0.2 s | 1.9 s |
| statements, expectations, ownership tables | 0.1 s | 0.2 s | 0.6 s |
| equity.technical | 0.3 s | 2.8 s | 12.6 s |
| equity.momentum | 0.03 s | 1.9 s | 11.3 s |
| equity.valuation | 0.2 s | 0.6 s | 2.4 s |
| other equity blocks | 0.5 s | 0.6 s | 1.9 s |
| group KPIs (sector / industry / country / market / etf / commodity) | 3.1 s | 3.4 s | 6.4 s |
| composites (6 presets) | 0.3 s | 0.3 s | 0.4 s |
| persist equity frame | 0.2 s | 0.2 s | 0.7 s |
| **total** | **7.9 s** | **13.0 s** | **40.3 s** |
| peak RSS | 392 MB | 630 MB | 2,193 MB |

Only 6 symbols have any price data in every column of this table. Everything
that grows with N is pure overhead spent on symbols that have nothing to
compute. Memory grows ~180 MB per 1,000 symbols, i.e. ~19 GB at 107k, which
is the OOM above.

### Per-symbol indicator cost (real MSFT history)

`Indicators().compute(ohlcv)` costs **42-57 ms per symbol** regardless of
history length (500 to 2,047 rows). The TA-Lib calls themselves are 0.5-5 ms
per indicator; most of the time is the signal post-processing (`ta.cross`
from pandas-ta per signal, pandas Series construction per signal). On a real
database this is the dominant CPU cost: 5,000 priced symbols = ~4 min,
20,000 = ~15 min, single-threaded.

### Persistence and MCP reads (synthetic dense frames)

| run size | rows in `kpi` | write | `read_kpi_frame` (read + pivot) | DB growth |
|---|---:|---:|---:|---:|
| 5,000 symbols x 150 KPIs | 750k | 14.5 s | 3.6 s | +300 MB |
| 20,000 symbols x 150 KPIs | 3.0M | 66.5 s | 16.6 s | +360 MB |

`BullishService.frame()` is not cached: **every MCP tool call re-reads and
re-pivots the full long table for the run date** (17 s per call at 20k
symbols). A 36-month `bullish history` adds ~13 GB at that size.

### Constant per-run overheads

- Universe: a 197k-row three-way join over `equity`, `equitysector`,
  `equitycountry` (1.5 s) plus `dedupe_by_source` over 107k symbols
  (0.6 s). Repeated for every date in `bullish history`.
- Group KPIs floor of ~3-8 s even with 6 symbols: `etf.compute` loads
  `SELECT * FROM etf` (48k rows) and then builds two **48k-column empty
  DataFrames** (`priceetf` is empty), 2.6 s each, column by column.

### Quadratic coverage grading (not triggered on the test DB, which has no coverage rows)

`grade_universe` calls `grade_symbol` per symbol, and each dimension filters
the whole coverage frame with `coverage[(symbol == s) & (table_name == t)]`.
That is O(symbols x coverage rows): ~8 full-frame scans per symbol. On a real
database with, say, 20k symbols and 200k coverage rows, this alone is minutes.

## 2. Root causes

1. **The universe is every row of `equity`, not every symbol with data.** All
   panels are reindexed to the full universe (`price_panel(...).reindex(columns=symbols)`),
   so a 107k-column, 1,256-row float panel (1.1 GB) is built for each of
   close / open / high / low / volume / dividends, and the momentum block
   derives several more of the same shape (`pct_change`, rolling 20/50/200).
   Loops, allocations and `IN (...)` chunks are all sized by this.
2. **The technical block is a Python loop over symbols** that builds a
   DataFrame per symbol from five wide panels, instantiates pydantic
   `Indicators()` per symbol, runs pandas-ta `cross` per signal, and writes
   each result with a scalar `out.loc[symbol, name] = value`.
3. **Long-format persistence with three string B-trees per row** (primary
   key + two indexes on `entity_type, entity, name, date`), and no wide
   cache; reads pivot millions of rows and nothing caches the result.
4. **Constant costs repeated per run**: the universe join and dedupe, the
   ETF table, the giant empty frames, all re-done for every month of a
   history build.
5. **Coverage grading filters per symbol** instead of grouping once.

## 3. Plan

Ordered by payoff over effort. Each phase is independently shippable and
has a check that proves it.

### Phase 1: scope the run to symbols that have data (half a day)

- Universe = `equity` ∩ symbols with at least `min_price_rows` (default 60)
  rows in `price` inside the lookback window, resolved by one SQL
  `GROUP BY symbol HAVING COUNT(*) >= ...` (or from the `coverage` table
  when present). Symbols with no prices are reported by the coverage tool,
  not carried through the KPI engine.
- `price_panel` stops reindexing to the universe; frames are as wide as the
  data. The ETF block loads only funds that have `priceetf` rows and skips
  cleanly when the table is empty.
- CLI: `bullish compute --symbols` (already supported by `DataContext`) and
  a `BULLISH_MIN_PRICE_ROWS` setting.
- Check: the unrestricted run on the test DB finishes in seconds instead of
  being killed; peak RSS is proportional to priced symbols.

Expected effect: removes the OOM entirely and most of the N-proportional
overhead in the table above (technical 12.6 s and momentum 11.3 s at 10k
unpriced symbols become ~0).

### Phase 2: vectorise and parallelise the technical block (1-2 days)

- Compute the column-wise indicators once across the whole close panel
  (SMA 50/200, ROC, RSI, MACD, Bollinger, 20/200-day highs: all are
  `rolling` / `ewm` on the wide frame, already how the momentum block
  works). This covers roughly half of the 57 signals for free.
- For the per-series TA-Lib indicators (ADX, STOCH, MFI, ADOSC,
  candlesticks): pass numpy arrays, not per-symbol DataFrames; keep one
  `Indicators` instance; collect results into a dict of dicts and build the
  output frame once at the end (no per-cell `.loc`).
- Replace `pandas_ta.cross` in the signal processing with a numpy
  sign-change on the two arrays (it is the largest single cost per signal).
- Run the per-series part in a `ProcessPoolExecutor` over symbol chunks
  (4 cores here, ~3.5x), behind a `BULLISH_WORKERS` setting defaulting to
  `os.cpu_count() - 1`.
- Check: the golden indicator test (`tests/test_golden.py`) still passes
  bit-for-bit; per-symbol cost measured on `prices_msft.csv` drops from
  ~45 ms to under 10 ms; 20k priced symbols in about a minute.

### Phase 3: make persistence and MCP reads cheap (1 day)

- Cache `BullishService.frame()` per `(entity_type, run date)` with an
  `lru_cache`, invalidated when `latest_kpi_date` changes. This alone turns
  the 17 s per MCP call into a one-off.
- Store each run's wide frame once (per entity type and date) as a
  compressed numpy payload in a small `kpiwide` table, and have
  `read_kpi_frame` load that (~0.2 s) instead of pivoting millions of rows.
  The long `kpi` table stays for `read_kpi_series` and the backtest.
- Bulk-insert the long rows through a raw `sqlite3` `executemany` inside
  one transaction with `PRAGMA synchronous=OFF` for the duration; store
  values as `REAL` with float32 precision where that is enough.
- Check: write of 3M rows under 15 s; cold `read_kpi_frame` under 1 s;
  warm MCP call under 50 ms.

### Phase 4: remove the constant overheads, share loads across history (1 day)

- Universe query selects only the columns the engine uses and does the
  source-priority dedupe in SQL (`ROW_NUMBER() OVER (PARTITION BY symbol
  ORDER BY priority)`), so pandas never sees 197k rows. Target ~0.5 s.
- `bullish history`: load prices and statements once for the whole span and
  derive each monthly `DataContext` by truncating at `as_of` (prices by
  date, statements by `filed_date`), instead of re-querying everything per
  month. A 36-month build then costs 36 x compute, not 36 x (load +
  compute).
- Check: per-run overhead with an empty universe under 1 s; history of 12
  months on the test DB in under a minute.

### Phase 5: vectorise coverage grading (half a day)

- `grade_universe` groups the coverage frame once by `(symbol, table_name)`
  and derives all dimension grades with vectorised rules; `grade_symbol`
  stays as the single-symbol path for the MCP tool, implemented on the same
  grouped table.
- Check: 20k symbols x 200k coverage rows graded in under 2 s.

### Phase 6: guardrails so it stays fast (half a day)

- `bullish compute --profile` prints the per-phase table above (the engine
  times each phase and peak RSS; the benchmark script used for this plan
  becomes that feature).
- A CI benchmark on a synthetic 2,000-symbol database with asserted upper
  bounds for total time and memory, so a regression fails the build.

## 4. Expected end state

For a realistic database of ~20k priced symbols with five years of history:

| | today (estimated) | after the plan |
|---|---:|---:|
| `bullish compute`, one date | ~25-30 min, >4 GB, OOM beyond ~80k symbols | ~2-3 min, <1.5 GB |
| `bullish history`, 36 months | ~15 h | ~1.5 h |
| cold MCP call needing the equity frame | ~17 s | <1 s |
| warm MCP call | ~17 s | <50 ms |
| DB growth per run | ~360 MB | ~100 MB |

## 5. Decisions needed

1. Phase order is payoff-first; Phases 1 and 3 can ship within a day and
   remove the two user-visible pains (runs that never finish, slow MCP
   answers). Confirm that order, or ask for the technical block (Phase 2)
   first if run time matters more than memory.
2. Wide-frame storage: a `kpiwide` table inside the same SQLite file (no new
   dependency, one-file model, recommended) versus Parquet files next to the
   database (cleaner for external tools, adds `pyarrow`).
3. Symbols without prices: excluded from the KPI run entirely (recommended;
   the coverage tool still reports them from the `coverage` table) versus
   kept as all-NaN rows in the KPI frame.
