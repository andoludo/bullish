# First run on a real database: S&P 500, 2026-10-29

The engine, the MCP server and the backtest had only ever run on a 6-symbol
LFS fixture and on synthetic data. This is what happened on a bearish
database filled from the live providers for the current S&P 500
constituents (503 symbols), in a 4-core, 16 GB container.

## 1. Building the database (reproducible)

```bash
DB=sp500.db
bearish members   $DB ^GSPC                      # 503 constituents from Wikipedia, 5 s
SYMS=$(sqlite3 $DB "select group_concat(symbol) from (select distinct symbol from indexmember where index_symbol='^GSPC')")
bearish tickers    $DB US --filters "$SYMS"       # FinanceDatabase + investpy lists, then Yahoo details: 7 min
bearish prices     $DB US --filters "$SYMS"       # 4.4M daily rows back to 1962: 13 min
bearish financials $DB US --filters "$SYMS"       # SEC XBRL for 499 of 503, Yahoo for the rest: 8 min
bearish fx         $DB EUR GBP JPY CHF CAD        # 2 min
bearish macro      $DB USA                        # 140 series, 285k vintages: 2 min
bearish shorts     $DB US --filters "$SYMS"       # FINRA short interest into equitysnapshot: 42 s
# index prices, ~70 liquid US ETFs with details and prices: 6 min (python API, see below)
bearish link       $DB                            # sector / country aliases, ETF exposures: 1 min
bearish coverage   $DB                            # census: 4 s
```

The ETF step used the Python API because the CLI fetches prices for every
ETF row (43k listings) and has no details command:

```python
from bearish.main import Bearish
from bearish.models.base import Ticker
from bearish.models.query.query import AssetQuery, Symbols
from bearish.database.schemas import PriceEtfORM
b = Bearish(path=DB)
b.get_prices_index()
tickers = [Ticker(symbol=s) for s in ("SPY QQQ IWM XLK XLV XLF XLE ... GLD TLT EEM EFA").split()]
b.write_detailed_assets(AssetQuery(symbols=Symbols(etfs=tickers)))
b.write_many_series(tickers, "max", apply_filter=False, table=PriceEtfORM, track=False)
```

Total wall time for the fill: about 45 minutes, almost all of it waiting on
providers. The database is 1.3 GB.

What did not make it in this session:

- **SEC Form 4 insiders and EDGAR filings.** `bearish insiders` fetches one
  submissions page per company and then one XML per filing at five requests
  per second. For 503 large caps that is on the order of 100k requests, so
  it ran for two hours without a single row written before I stopped it.
  The SEC publishes quarterly bulk insider-transaction data sets; loading
  those is the practical route for a universe this size.
- **13F holdings** (`bearish holdings`, bulk data sets) and **Schedule 13D/G**
  were not attempted for the same reason.
- **Analyst expectations** (earnings dates, price targets, recommendations)
  only come from Yahoo, and the financials step stops at the first source
  that answers (SEC), so they had to be fetched separately with the Yahoo
  source forced (section 5).

Provider notes: Yahoo rate-limits the cookie bootstrap in bursts; two
ticker-detail runs died on it immediately and a third, minutes later, ran
clean. Every other endpoint (chart, info, fundamentals) answered at ~0.6 s
per symbol.

## 2. `bullish compute --profile`

| phase | first run | after the macro-lookup fix (PR #7) |
|---|---:|---:|
| universe | 3.25 s | 3.3 s |
| prices | 5.12 s | 5.1 s |
| equity.technical (3 workers) | 5.70 s | 5.7 s |
| equity.growth / quality / momentum | 4.25 s | 4.3 s |
| equity.valuation | **15.55 s** | 2.13 s |
| equity.macro_exposure | 1.23 s | 0.61 s |
| group.country / market | **6.94 s** | 0.37 s |
| group.etf / industry / sector | 2.81 s | 2.8 s |
| composites + coverage + persist | 1.68 s | 1.7 s |
| **engine total** | **46.6 s** | **27.7 s** |
| peak RSS | 789 MB | 774 MB |

501 equities, 182 KPIs, 68.6k values, 3.2k scores per run. The hotspot the
synthetic data could not show: `macro_series` filtered the 285k-row macro
frame on every call and the valuation block calls it twice per symbol. It is
a dict lookup now.

A 24-month monthly history (24 runs) took 392 s on a copy of the database,
16 s per run, with the span context loading the data once.

## 3. What the outputs look like

- Universe: 501 of 503 symbols (two have no price history yet); sector and
  industry resolved for all of them, country for all (477 USA, 10 IRL, ...).
- Coverage grades: 491 A, 7 B, 3 C.
- 108 of 182 KPIs are more than 80 % filled; 34 are empty because their
  tables are empty (insiders, 13F, 8-K, analyst targets, short-interest
  change, commodity tailwind).
- Sectors: Energy 74.9, Health Care 69.7, Information Technology 62.9 ...
  Consumer Discretionary 26.5. ETFs rank consistently with that: XLE, XOP,
  XLV, XBI, SOXX, XLK, SMH, OIH.
- Regime: risk_on, "growth up, inflation up (overheat)", momentum leadership.
- Verdicts: NVDA, AAPL, PFE "sound, not cheap"; INTC "avoid" (not sound,
  idiosyncratic drawdown, cause not identified in the data).
- MCP: 23 tools answered on the real database. Warm calls take 2 to 110 ms.
  The first call that touches the data context pays its lazy loads (4 s for
  a scorecard, 10 s for a portfolio); a warm-up on server start would hide
  that.

## 4. First calibration signal: rank backtest, 24 monthly runs

`bullish backtest <name> --horizon-months h` on the 24 stored runs
(2024-11-01 .. 2026-10-01). IC is the mean Spearman correlation between the
ranking and the forward return; hit rate is the share of dates with a
positive IC; top-bottom is the average forward return of the top quintile
minus the bottom quintile.

| ranking | horizon | dates | mean IC | t | hit rate | top - bottom |
|---|---:|---:|---:|---:|---:|---:|
| mom_12_1 | 6m | 18 | +0.060 | 2.8 | 0.78 | +10.4 % |
| price_vs_sma200 | 6m | 18 | +0.076 | 2.4 | 0.78 | +10.8 % |
| emerging_leader | 6m | 18 | +0.037 | 2.2 | 0.67 | +6.8 % |
| ret_3m | 6m | 18 | +0.052 | 1.3 | 0.67 | +9.2 % |
| rsi_14 | 6m | 18 | +0.027 | 0.7 | 0.78 | +4.8 % |
| dividend_quality | 6m | 18 | -0.001 | -0.1 | 0.50 | -1.6 % |
| revenue_ttm_growth | 6m | 18 | +0.014 | 0.5 | 0.67 | -0.5 % |
| turnaround | 6m | 18 | -0.030 | -1.5 | 0.44 | -6.6 % |
| quality_on_sale | 6m | 18 | -0.045 | -2.0 | 0.33 | -8.8 % |
| roic | 6m | 18 | -0.045 | -4.4 | 0.06 | -14.1 % |
| quality_compounder | 6m | 18 | -0.095 | -5.6 | 0.11 | -14.8 % |
| value_with_catalyst | 6m | 18 | -0.207 | -18.2 | 0.00 | -25.7 % |
| fcf_yield | 6m | 18 | -0.237 | -16.4 | 0.00 | -31.0 % |

The 3-month horizon tells the same story with weaker numbers.

Read it with two caveats before changing any weight:

1. **Survivorship bias.** The universe is today's S&P 500. Companies that
   were cheap on cash-flow yield in 2024 and then got dropped from the index
   are not in the sample; the ones that stayed are the ones that recovered
   or kept growing. That alone inflates momentum and penalises value. The
   `indexmember` table stores snapshots, so running `bearish members`
   regularly fixes this going forward; for the past, a historical
   constituent list is needed.
2. **One regime, 18 to 22 dates.** The window is a single growth-and-momentum
   led stretch. A t-statistic of 2 on 18 overlapping observations is
   suggestive, not conclusive.

What it does say, even so: in this universe and window, the trend and
momentum blocks carried the signal, and the value blocks (fcf_yield,
`value_with_catalyst`) ranked the wrong way strongly enough that the
weights of `quality_on_sale` and `value_with_catalyst` should not be
trusted as they stand. A longer history (5 years, 60 monthly runs, with a
point-in-time constituent list) is the prerequisite for recalibrating.

## 5. Analyst expectations fill (section updated after the second fetch)

See the end of this file.
