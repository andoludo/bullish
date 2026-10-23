# Bullish v2: decision KPIs on top of Bearish, exposed through MCP

Status: proposal (2026-10-23). Branch: `claude/bullish-repo-sector-etf-analysis-90s9nf`.

## 1. Verdict

The idea makes sense, and the timing is right: Bearish now stores far more than
prices and statements (point-in-time macro, commodities, 13F flows, 13D/G,
insiders, analyst history, SEC XBRL with filing dates, ETF exposures), while
Bullish still only computes **per-stock** indicators and boolean fundamental
checks into one wide `analysis` row. Nothing in Bullish answers questions
*across* entities ("which sector", "which ETF", "which country") or *through
time* (how a score evolved), and nothing is machine-readable for an LLM client.

Three honest caveats that shape the plan:

1. **Business-cycle sector rotation has weak evidence.** Peer-reviewed work
   (Cranfield 2023, Massey) finds the "cycle phase → sector" playbook earns
   little after costs and timing errors. What *does* have robust evidence is
   relative-strength / momentum (12-1 month, industry momentum), earnings
   revisions and surprise drift, quality (profitability, accruals), insider
   buying, and low-volatility. So the sector score is built on those, and the
   macro regime is shown as **context and a tilt**, never as a rule.
2. **"The next NVIDIA" is not forecastable as such.** What can be built is an
   *emerging-leader* screen whose ingredients have at least practitioner
   evidence (revenue acceleration, gross-margin expansion, operating leverage,
   revision breadth, institutional accumulation, momentum with a liquidity
   floor), with survivorship-free backtests to measure its hit rate. It is a
   candidate generator for your judgement, not a prediction.
3. **Every KPI must be point-in-time.** Bearish already stores `as_of`
   (macro) and `filed_date` (SEC). The new Bullish must store KPIs as dated
   series so that backtests and the MCP `as_of` parameter are honest. The
   current "one wide row per symbol" design cannot do that.

## 2. What Bearish provides today (inventory used by the plan)

| Domain | Tables | Decision use |
|---|---|---|
| Prices | `price`, `priceetf`, `priceindex`, `pricecommodity` (dividends, splits) | momentum, vol, breadth, relative strength |
| Reference | `equity` (sector, industry, mcap, float, short shares, snapshot ratios), `etf` (expense ratio, AUM, holdings %, sector weights, nav), `index`, `currency`, `crypto` | universes, ETF vehicle quality |
| Statements | annual + quarterly metrics / balance sheet / cash flow, `filed_date` from XBRL | growth, quality, valuation |
| Expectations | `earningsdate` (estimate vs reported, forward flag), `pricetarget` (dated), `analystrecommendation` (dated trend) | surprises, revisions, upside |
| Ownership | `holding13f`, `holding13fsummary` (holder flows), `ownership13dg`, `insidertransaction`, legacy `sec*` | smart-money accumulation |
| Macro | `macroseries`, `macroobservation` (point-in-time), concepts across growth / inflation / labour / rates / credit / fiscal / external / sentiment / positioning / factors | regime, country score, tailwinds |
| Links | `countryref`, `countryalias`, `sectorcommoditysensitivity`, `etfexposure`, `countrycommodityexposure`, `commodity` | sector ↔ commodity ↔ country ↔ ETF joins |

Gaps to close on the Bearish side (small, see §7): ETF metrics are a snapshot
(no AUM / NAV history → no flows, no premium/discount history), no FX table
for cross-country comparison, no index constituents, short interest is a
snapshot, sector vocabulary differs between `equity.sector` (Yahoo names such
as "Technology", "Consumer Cyclical") and `etf_exposure` (GICS names such as
"Information Technology", "Consumer Discretionary").

## 3. The KPI catalogue

Conventions used throughout:

* Every KPI is a dated series `(date, entity_type, entity, kpi, value, as_of)`.
* Cross-sectional KPIs are stored raw **and** as a percentile within a peer
  group (sector × region, or size bucket) after winsorising at 1/99.
* Each KPI carries an **evidence tier**: A = strong academic evidence,
  B = practitioner evidence / widely used, C = heuristic or context only.
* Composites expose their components, so the MCP can always explain a score.

### 3.1 Market regime (entity = region or country index)

| KPI | Definition | Tier |
|---|---|---|
| Trend | index vs 200d SMA, 50/200 slope, distance to 52w high | B |
| Breadth | % constituents above 50d / 200d SMA, % at 52w highs minus lows, advance/decline line, % with positive 3m return | B |
| Volatility regime | 20d / 60d realised vol percentile, VIX level and 1y percentile (`VIX` concept) | B |
| Credit / liquidity | `HY_SPREAD`, `IG_SPREAD` level and 3m change, `NFCI`, `M2_YOY` | A (spreads lead equity drawdowns) |
| Rates | `CURVE_10_2`, `CURVE_10_3M`, `REAL_Y10`, policy-rate direction (hiking / pausing / cutting) | B |
| Growth-inflation quadrant | sign of 3m change in `CLI` / `PMI_MFG` / `GDP_NOWCAST` × sign of change in `CPI_YOY` → four regimes (the "investment clock") | C, context only |
| Risk appetite ratios | cyclicals/defensives, small/large, equal-weight/cap-weight, copper/gold, high-beta/low-vol (built from sector and ETF prices) | B |
| Factor regime | rolling 3m / 12m returns of `FF_MOM`, `FF_HML`, `FF_SMB`, `FF_RMW`, `FF_CMA` → which style is paying | A |
| Stress | `RECESSION_PROB`, `EPU`, `GPR` percentile | C |

Output: a regime label (risk-on / neutral / risk-off), the quadrant, and the
factor leadership, each with the numbers behind it.

### 3.2 Sector and industry ranking ("which sector is most interesting")

| Block (weight) | KPIs | Tier |
|---|---|---|
| Relative strength (30) | excess return vs market 1 / 3 / 6 / 12m; 12-1 momentum; industry momentum (Moskowitz–Grinblatt); RRG-style quadrant from our own normalised RS-ratio and RS-momentum (10-week smoothed ratio, z-scored; JdK's smoothing is proprietary) | A |
| Breadth (15) | % constituents above 200d SMA, % golden cross, median 3m return, share of stocks with positive 12-1 momentum; dispersion (cross-sectional std of returns = stock-picking opportunity) | B |
| Fundamentals and revisions (25) | median revenue growth and **acceleration**, median EPS growth, aggregate earnings-surprise rate (last 4 quarters, from `earningsdate`), price-target revision ratio (share of names whose `target_mean` rose over 3m), recommendation-trend improvement (buy ratio change across `analystrecommendation` periods), median margin trend | A (revisions / surprise drift) |
| Valuation (10) | cap-weighted P/E, EV/EBITDA, FCF yield vs the sector's own 5y history (z-score) and vs the market; PEG | B |
| Flows and positioning (10) | net change in 13F holders and value (sector aggregate of `holding13fsummary`), insider net buying ratio, short interest trend | A (insiders), B (13F) |
| Macro tailwind (10) | Σ sensitivity × commodity 3m momentum (from `sectorcommoditysensitivity` first, then estimated rolling betas), estimated beta to Δ10y yield and to DXY × their recent moves | C |

Composite **Sector Attractiveness Score** = weighted percentile ranks. The
same engine runs at industry level (`industry`) and for user-defined groups.
Weights live in config and are validated by the rank backtest (§6).

### 3.3 ETF ranking ("which ETF is best")

Two questions, scored separately, then combined:

**(a) Which exposure?** Each ETF's exposures (`etfexposure`: sector, country,
region, commodity, theme) are mapped to the sector score (§3.2), country score
(§3.4) and commodity score (§3.6). Look-through exposure score = Σ weight ×
entity score.

**(b) Which vehicle inside that exposure?**

| KPI | Source / definition | Tier |
|---|---|---|
| Cost | `annual_report_expense_ratio` | A |
| Size and flows | `total_net_assets` and its change (needs the snapshot history from §7) | B |
| Liquidity | average daily dollar volume (60d), Amihud illiquidity | B |
| Tracking | tracking difference and tracking error vs the mapped index in `priceindex` where a mapping exists; otherwise vs the best-correlated index | A |
| Premium / discount | close vs `nav_price` (history needed, §7) | B |
| Concentration | top-10 weight and Herfindahl of `holding_percent`; overlap (Jaccard / weighted) with other ETFs in the same exposure | B |
| Risk-adjusted return | 1y / 3y return, vol, max drawdown, Sharpe, Sortino, Calmar, beta and up/down capture vs benchmark | B |
| Momentum | 12-1 momentum, distance to 52w high | A |
| Income and structure | dividend yield, turnover, age (`fund_inception_date`), `legal_type`, domicile / listing currency | C |
| Look-through fundamentals | weighted P/E, P/B, growth of underlying holdings from the ETF's own ratios | B |

Composite **ETF Score** = exposure attractiveness (50) + vehicle quality (30:
cost, tracking, liquidity, concentration) + risk-adjusted momentum (20).
`compare_etfs` also reports the pairwise overlap so a client does not buy the
same thing twice.

### 3.4 Country ranking

| Block | KPIs | Tier |
|---|---|---|
| Market | country index / country ETF momentum, breadth across the country's constituents, valuation vs own history | A/B |
| Growth | `CLI` change, `PMI_MFG`, `GDP_NOWCAST` vs `GDP_FCST`, `IP_YOY`, `RETAIL_YOY` | B |
| Inflation and policy | `CPI_YOY` trend vs target, policy-rate direction, `REAL_Y10`, curve slope | B |
| Financial stability | `CREDIT_GDP_GAP` (BIS), `HOUSE_PRICE_YOY`, `DEBT_GDP`, `CA_GDP`, `RESERVES` | B |
| External | `REER` 6m change, `FX_USD` momentum (and vs the user's base currency) | B |
| Sentiment and risk | `CONS_CONF`, `BIZ_CONF`, `ESI`, `EPU`, `GPR` | C |
| Terms-of-trade tailwind | `countrycommodityexposure` (export minus import share) × commodity momentum | C |

Composite **Country Score** with the same percentile machinery; feeds §3.3.

### 3.5 Single-stock scorecard (replaces the boolean-only fundamentals)

| Block | KPIs |
|---|---|
| Growth | revenue YoY / QoQ, **revenue acceleration** (Δ of YoY growth), EPS growth, 3y revenue and EPS CAGR, forward growth from `earningsdate` estimates, Rule of 40 (revenue growth + FCF margin) |
| Profitability and quality | gross / operating / FCF margin and their 4-quarter trend, **operating leverage** (ΔEBIT% / Δrevenue%), incremental margin, ROE, ROA, ROIC (NOPAT / (equity + debt − cash)), FCF conversion (FCF / net income), accruals ratio ((NI − CFO) / assets, Sloan), Piotroski F-score (9), Mohanram G-score (7-signal variant, no advertising data), Altman Z, net debt / EBITDA, interest coverage, cash runway for loss-makers, share-count change (dilution vs buybacks) |
| Valuation | trailing / forward P/E, PEG, EV/EBITDA, EV/Sales, P/B, P/FCF, FCF yield, earnings yield minus `Y10`; each vs own 5y history (z-score) and vs sector median |
| Momentum and risk | all existing TA-Lib signals (kept), 12-1 and 6-1 momentum, volatility-scaled momentum, 52w-high proximity, beta, idiosyncratic vol, max drawdown, 1y Sharpe, relative strength vs sector and vs market, average dollar volume, Amihud illiquidity |
| Expectations and revisions | target upside (`target_mean` vs price), Δ target mean 1m / 3m, Δ recommendation mean, buy-ratio change across recommendation periods, EPS and revenue surprise for the last 4 quarters and their consistency, days to next earnings (event-risk flag) |
| Ownership and smart money | 13F holders Δ, new vs closed holders, value Δ, institutional share of float, 13D (activist) presence, insider net purchases 3 / 6 / 12m (code P vs S, officers and directors weighted higher), cluster-buy flag, short interest % float and days-to-cover |
| Macro exposure | sector commodity tailwind, rate beta, USD beta, domicile country score, listing-currency FX momentum vs base currency |

Presets (composites, all explainable):

* **Emerging leader** (the "next NVIDIA" proxy): revenue acceleration +
  gross-margin expansion + operating leverage + revision momentum + 13F
  accumulation + 12-1 momentum, with a liquidity floor, a size band
  (configurable, e.g. 1–50 bn), and G-score ≥ 5 as a quality floor.
* **Quality compounder**: ROIC, FCF conversion, low accruals, stable growth,
  low leverage, moderate valuation.
* **Value with catalyst**: cheap vs history and sector + revisions turning up +
  insider buying.
* **Turnaround**: F-score ≥ 7 + RRG improving + surprise consistency.
* **Dividend quality**: yield, payout from FCF, dividend growth, coverage.

### 3.6 Single-stock due diligence: "is it fundamentally good, and is it only temporarily oversold?"

This is the question the scorecard in §3.5 must answer directly, not leave to
the reader. The engine produces a **verdict object** per symbol with five
parts, each backed by the KPIs that produced it.

**(1) Fundamental health** (is the business good?). A pass/fail checklist plus
a 0–100 score, computed from the last 3 fiscal years and the last 8 quarters:

| Check | Rule (defaults, configurable) | Source |
|---|---|---|
| Profitable and cash-generative | operating income > 0, FCF > 0 in ≥ 3 of the last 4 years, FCF conversion (FCF / net income) > 0.8 | metrics, cash flow |
| Returns on capital | ROIC > 10 % and ROE > 12 % (or > sector median) | metrics, balance sheet |
| Earnings quality | accruals ratio < 10 % of assets, CFO ≥ net income | metrics, cash flow |
| Growth | revenue 3y CAGR > 5 % and not decelerating two years in a row | metrics |
| Margins | gross and operating margin flat or rising over 8 quarters | quarterly metrics |
| Balance sheet | net debt / EBITDA < 3, interest coverage > 4, current ratio > 1, Altman Z > 2.6 | balance sheet, metrics |
| Capital discipline | share count flat or falling (no dilution), capex ratio stable, dividends covered by FCF | balance sheet, cash flow |
| Scores | Piotroski F ≥ 6, Mohanram G ≥ 4 | derived |
| Trusted statements | SEC XBRL source with `filed_date` for US names; Yahoo elsewhere flagged as lower confidence | `filing`, `financialmetrics.source` |

**(2) Oversold state** (is the price depressed?). Short-term reversal has
academic evidence (Jegadeesh 1990, Lehmann 1990), long-term reversal too
(De Bondt–Thaler 1985), and Piotroski showed high F-score names among beaten-
down stocks outperform. KPIs:

* RSI 14 < 30 and RSI turning up, Bollinger %B < 0, stochastic oversold
  (existing TA-Lib signals, kept);
* drawdown from 52-week high, distance below 50d / 200d SMA, z-score of price
  vs 200d SMA, proximity to 52-week low;
* **excess drawdown** = stock drawdown minus sector-ETF (or industry median)
  drawdown over the same window: separates a company-specific sell-off from a
  sector or market one;
* capitulation volume (volume / 30d average > 3 on down days), ADOSC turning
  positive after the drop (existing signal), 1-month return in the bottom
  decile of the universe (the reversal factor).

**(3) Temporary or structural?** The discriminating evidence, scored as
"temporary" (+), "structural" (−) or neutral:

| Evidence | Reads as temporary when… | Reads as structural when… | Source |
|---|---|---|---|
| Nature of the drawdown | sector / market fell with it (excess drawdown ≈ 0) | excess drawdown large and persistent | prices, `etfexposure` |
| Event type at the drop | drop on an earnings date with an EPS **beat** (overreaction), index exclusion, macro shock | drop on a **miss**, guidance cut, restatement | `earningsdate` (estimate vs reported), filings |
| Analyst reaction | target mean and recommendation trend flat or falling less than price (upside widening) | targets cut faster than price, buy ratio collapsing | `pricetarget`, `analystrecommendation` |
| Estimate trend | next-quarter EPS / revenue estimates stable | forward estimates falling | `earningsdate` forward rows |
| Insiders | officers and directors buy after the drop (cluster buy within 30 days) | insiders sell into the drop | `insidertransaction` |
| Institutions | 13F new holders > closed holders in the following quarter, 13D activist appears | holders leaving two quarters in a row | `holding13fsummary`, `ownership13dg` |
| Short interest | falling or flat | rising sharply | equity snapshots (§7) |
| SEC 8-K items | 2.02 results, 7.01 Reg FD, 1.01 ordinary agreements | 4.02 restatement, 4.01 auditor change, 3.01 delisting notice, 1.03 bankruptcy, 5.02 CEO / CFO departure, 2.04 debt acceleration | `filing` with 8-K items (§7) |
| Financing | no new issuance | share offering or convertible after the drop (`proceeds_from_issuance_of_common_stock` spike) | cash flow |
| Fundamentals vs price | **fundamental-price gap**: fundamental percentile minus price-momentum percentile is large and positive | gap small or negative (price was right) | derived |
| Valuation | P/E, EV/EBITDA, P/FCF at multi-year lows vs own history while margins hold | valuation low because earnings collapsed | derived |

**(4) Recovery evidence.** Signals that the market is turning: RSI bullish
crossover (existing), MACD zero-line cross, price back above 20d high, volume
on up days exceeding volume on down days, positive earnings surprise after the
drop, insider cluster buy, revisions turning up.

**(5) Verdict.** `fundamentally_sound` (bool, score, failed checks),
`oversold` (score and the signals firing), `drawdown_nature`
(idiosyncratic / sector / market), `likely_cause` (earnings miss, guidance,
offering, sector sell-off, event from 8-K, unknown), `structural_red_flags`
(list), `recovery_evidence` (list), `data_coverage` (§3.7), and a label:
**quality on sale** (sound + oversold + temporary), **falling knife** (oversold
+ structural), **sound, not cheap**, **avoid**. Each label is a preset in the
screener so "show me all quality-on-sale names in Europe" is one call.

### 3.7 Data coverage: does Bearish have enough to judge this stock?

Before any verdict the engine grades the inputs, per symbol, and the MCP
returns the grade with every scorecard:

| Dimension | Measured as | Needed for a grade A |
|---|---|---|
| Statements | fiscal years and quarters with non-null revenue, net income, CFO, total assets, equity; source (SEC XBRL vs Yahoo); `filed_date` present | ≥ 4 years and ≥ 8 quarters, SEC source for US names |
| Prices | first and last price date, gaps > 5 trading days, dividends / splits present | ≥ 3 years, no gaps, last price < 5 days old |
| Expectations | number of analysts, dated price targets, recommendation periods, earnings rows with both estimate and reported | ≥ 3 analysts, ≥ 4 earnings rows with surprises |
| Ownership | 13F summary quarters, insider rows, 13D/G rows | ≥ 4 quarters of 13F (US only) |
| Reference | sector, industry, shares outstanding, float, currency, ISIN / CUSIP mapping | all present |
| Links | sector mapped to the shared vocabulary, country mapped to ISO-3 | both resolved |

Output: grade A / B / C / insufficient per dimension and overall, the list of
missing items, and the Bearish command that would fill them (for example
`bearish financials ... --filters SYM`, `bearish insiders`, `bearish holdings`).
Grades are stored as KPIs so the screener can require a minimum grade, and the
universe-level summary ("how much of the Belgian market is grade A?") tells
you where Bearish needs another run.

### 3.8 Commodities

Price momentum 1 / 3 / 12m, COT positioning percentile (crowding, from
`COT_NET_SPEC_PCT_OI`), inventories / stocks-to-use trend, and the list of
sectors and countries each move helps or hurts (from the link tables).

### 3.9 Portfolio and decision support

Look-through exposure of a holdings list (sector, country, commodity, factor),
pairwise correlation and overlap, concentration, and what-if deltas. Keeps the
existing `myportfolio` optimiser as an optional step.

## 4. Target architecture

```
bullish/
  config.py              # pydantic-settings: db path, base currency, universes, weights
  data/                  # read-only access to the bearish db, FX normalisation, price panels
  kpi/
    registry.py          # @kpi(name, entity_type, tier, unit, description, higher_is_better)
    equity/              # growth, quality, valuation, momentum, revisions, ownership
    technical/           # existing TA-Lib indicators ported into the registry
    group/               # sector, industry, country, etf, market, commodity aggregations
    macro/               # regime, sensitivities (curated priors → estimated betas)
  scoring/               # winsorise, percentile within peers, composites, presets, explain
  backtest/              # rank-IC, decile portfolios, preset backtests (point-in-time)
  database/              # kpi (long), score, ranking snapshots, regime; alembic
  jobs/                  # huey: scheduled recompute after bearish updates
  mcp/                   # server, tools/, resources, prompts
  app/                   # streamlit viewer (optional, slimmed)
```

Design rules:

* **Long KPI table, not a wide row.** `SignalSeries` already has the right
  shape; generalise it to every entity type and keep a materialised "latest"
  view plus on-demand pivots. Indexes on `(entity_type, kpi, date)`.
* **Vectorised over panels**, not one `joblib` job per ticker: load a price
  panel per universe once, compute all momentum / vol / breadth KPIs in one
  pass. Statement KPIs are computed per symbol but from one bulk read.
* **Analytics engine**: keep SQLite as the store (Bearish owns it) and read it
  through DuckDB's SQLite scanner for the heavy joins and window functions.
  Optional, but it removes most of the custom pandas glue.
* **Base currency** (assumed EUR, configurable) applied to every cross-country
  comparison via an FX table (§7).
* **Vocabulary**: one sector enum (GICS-style) with a mapping from Yahoo
  names, shared with Bearish.
* Keep: TA-Lib indicators, huey jobs, `vectorbt` for the preset backtests,
  the Streamlit app as a thin viewer, OpenAI / tickermood news as an optional
  extra (they are not needed for the KPIs and their cost is unpredictable).
* Drop: `mysec` (Bearish covers 13F), `streamlit-pydantic`, the per-symbol
  wide `analysis` table (replaced by `kpi` + `score`).

## 5. MCP server

Built on the official `mcp` Python SDK (`mcp.server.fastmcp`), stdio for
Claude Desktop / Claude Code and Streamable HTTP for remote use. Strictly
read-only, every tool takes `as_of` (default: today) and returns Pydantic
structured output with units, currency, data freshness (last price date, last
filing date, last macro vintage) and the evidence tier of each component.

Tools:

| Tool | Purpose |
|---|---|
| `get_market_regime(region, as_of)` | §3.1 regime label and the numbers behind it |
| `rank_sectors(region, horizon, as_of, weights?)` / `rank_industries(sector, …)` | §3.2 ranked table with components |
| `rank_etfs(exposure?, listing_country?, currency?, criteria?, as_of, limit)` | §3.3 |
| `compare_etfs(symbols)` | cost, tracking, liquidity, overlap matrix |
| `rank_countries(as_of)` | §3.4 |
| `screen_equities(preset, filters, universe, limit, as_of)` | §3.5 presets or custom filters |
| `get_equity_scorecard(symbol, as_of)` | every KPI + composite + peers |
| `assess_stock(symbol, as_of)` | §3.6 verdict: fundamentally sound, oversold, temporary vs structural, red flags, recovery evidence |
| `get_data_coverage(symbol \| universe)` | §3.7 grade, missing items, Bearish commands to fill them |
| `get_filings(symbol, forms, since)` | 10-K / 10-Q / 8-K list with items and dates |
| `explain_score(entity, score, as_of)` | components, weights, tiers, caveats |
| `get_kpi_series(entity, kpi, start, end)` | time series for charts and trend questions |
| `get_macro_snapshot(entity, categories, as_of)` / `get_commodity_snapshot(code)` | macro and commodity context |
| `get_smart_money(symbol)` | 13F, 13D/G, insider detail |
| `get_earnings_calendar(window)` | upcoming events on the universe |
| `analyze_portfolio(holdings)` | §3.9 |
| `backtest_preset(preset, start, end, holding_period)` | hit rate and decile spread |
| `list_kpis()` / `describe_kpi(name)` | self-documenting catalogue |
| `query(sql)` (optional, off by default) | guarded read-only SQL with row limit |

Resources: `bullish://catalog/kpis`, `bullish://methodology/{score}`,
`bullish://universe/{country}`. Prompts: `weekly_market_review`,
`sector_briefing`, `etf_selection`, `stock_due_diligence`.

Testing: the SDK's in-memory client exercises every tool against the fixture
database in CI; a golden file pins the ranking output for a frozen date.

## 6. Validation

* Rank backtests per KPI and per composite: information coefficient (rank
  correlation of score vs forward 1 / 3 / 6 / 12m return), decile spread,
  turnover, by region. Point-in-time via `filed_date` / `as_of`.
* Preset backtests with the existing `vectorbt` engine (hit rate, median
  forward return, drawdown).
* Weights in config are only changed with a backtest result attached.
* Survivorship: use every symbol ever tracked, not only the current universe.
* §3.6 specifically: forward 3 / 6 / 12m excess return vs sector for each
  verdict label (quality on sale vs falling knife); the labels only ship if
  the spread between them is positive and stable across regions.

## 7. Bearish prerequisites (small PRs, do first)

Status (2026-10-23): all eight are implemented on the Bearish branch
`claude/bullish-repo-sector-etf-analysis-90s9nf` (tables `sectoralias` +
`equitysector` view, `filing.items`, statement detail columns,
`equitysnapshot`, `etfsnapshot`, `pricecurrency` + `fxrate` view,
`indexmember`, `coverage`; commands `bearish link`, `fx`, `members`,
`coverage`).

1. `etfsnapshot` table: `(symbol, as_of, total_net_assets, nav_price,
   expense_ratio, holdings_json)` written on every ETF fetch → flows,
   premium/discount, concentration history.
2. `fxrate` table (daily, via the existing `currency` asset and Yahoo pairs)
   for base-currency normalisation.
3. `shortinterestsnapshot` (`short_shares`, `short_ratio`, `float_shares`
   per fetch date).
4. Index constituents (`indexmember`) for breadth and tracking, from
   FinanceDatabase / Yahoo where available.
5. Shared sector vocabulary (GICS-style) and the Yahoo → GICS mapping.
6. `filing`: keep 8-K rows and their item codes (the EDGAR submissions
   payload already carries `items`; today only 10-K / 10-Q are kept) so the
   structural red flags in §3.6 can be read without text parsing.
7. XBRL tags missing for the quality checks: `InterestExpense`,
   `ResearchAndDevelopmentExpense`, `SellingGeneralAndAdministrativeExpense`,
   `ShareBasedCompensation`, `WeightedAverageNumberOfDilutedSharesOutstanding`,
   `PaymentsOfDividends`, `ProceedsFromIssuanceOfCommonStock`. Without R&D the
   G-score drops to 6 signals; without interest expense coverage is estimated.
8. A `coverage` view or table written by Bearish on each run (rows per
   symbol and table, first / last date, row count) so §3.7 is cheap.

## 8. Phases and effort (one developer)

Status (2026-10-23): phases 1-5 are implemented on this branch as an
additive layer (`bullish/config.py`, `bullish/data`, `bullish/kpi`,
`bullish/scoring`, `bullish/database/kpi_store.py`, `bullish/mcp`,
`bullish/backtest`; commands `bullish compute`, `history`, `backtest`,
`mcp`), tested on a synthetic bearish database (`tests/v2`). Phase 6 (slimming
the Streamlit app) is not started: the app is untouched and still works.

| Phase | Scope | Est. |
|---|---|---|
| 0 | Bearish prerequisites (§7, items 1–8) | 1–2 weeks |
| 1 | Core skeleton: config, data layer, KPI registry, long `kpi` table, migrations, TA indicators ported, equity KPI blocks | 2–3 weeks |
| 2 | Group KPIs: sector / industry / country / ETF / market regime / commodity | 2 weeks |
| 3 | Scoring: percentiles, composites, presets, explanations, the §3.6 verdict engine and §3.7 coverage grades | 2 weeks |
| 4 | MCP server, tests, Claude Desktop / Code config, docs | 1–2 weeks |
| 5 | Validation: rank backtests, weight calibration, golden tests | 2 weeks |
| 6 | Streamlit viewer slimmed to the new tables; scheduled jobs; cleanup of dropped modules | 1 week |

Phases 1–4 give a usable MCP within roughly six weeks; §5 is what turns
plausible KPIs into trusted ones.

## 9. Decisions to confirm (assumptions used above)

1. Base currency EUR, universes = countries already fetched by Bearish.
2. MCP-first; the Streamlit app stays as a viewer, not the main interface.
3. OpenAI / tickermood news kept as an optional extra, not a dependency.
4. DuckDB adopted for analytics over the SQLite file.
5. Python 3.12, `uv`, strict mypy and ruff as today.

## 10. Sources consulted

* Sector rotation evidence: Cranfield (2023) *Gauging the effectiveness of
  sector rotation strategies* — https://dspace.lib.cranfield.ac.uk/handle/1826/15495 ;
  Massey *The myth of business cycle sector rotation* — https://mro.massey.ac.nz/handle/10179/69590 ;
  Alexiou & Tygi (2020, J. Asset Management) via the same search.
* RRG construction: https://www.luxalgo.com/library/concept/relative-rotation-graphs.md ;
  https://sharpely.in/blogs/understanding-engine-rrg-complete-breakdown-rs-ratio-rs-momentum/
* ETF evaluation: Schwab — https://schwabassetmanagement.com/content/how-to-evaluate-etfs ;
  SSGA — https://www.ssga.com/in/en_gb/individual/insights/how-to-evaluate-etfs-6-things-that-matter-most ;
  etf.com efficiency — https://www.etf.com/sections/etf-basics/etf-efficiency-how-evaluate-etfs ;
  premiums / discounts — https://www.etf.com/sections/etf-basics/understanding-premiums-and-discounts
* Growth-stock scoring: Mohanram G-score — https://fffinstill.com/learning/concepts/mohanram-g-score ;
  AAII tech growth screen — https://www.aaii.com/latest/article/13046-aaii-stock-ideas-screening-for-technology-growth-stocks ;
  AAII glamour stocks — https://www.aaii.com/journal/article/13008-finding-future-winners-among-glamour-stocks
* Multi-bagger traits (margin expansion): https://stashaway.my/r/weekly-market-update-how-identify-multi-bagger ;
  revisions — https://articles.stockcharts.com/article/beyond-the-ai-hype-where-earnings-revisions-point-ahead-of-q3-reports/
* Reversal and F-score evidence: Jegadeesh (1990) and Lehmann (1990) one-month reversal;
  De Bondt & Thaler (1985) long-term reversal; Piotroski (2000) F-score on low P/B stocks.
* MCP Python SDK structured output and transports: https://pypi.org/project/MCP/ ;
  FastMCP practices — https://www.fast.io/resources/mcp-python-sdk.md ; https://dev.co/ai/mcp/fastmcp
