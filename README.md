# Bullish

**Bullish** turns a [bearish](https://github.com/andoludo/bearish) database
into decision KPIs and serves them to an LLM client over the
[Model Context Protocol](https://modelcontextprotocol.io): which sector,
country or ETF is attractive, which stocks fit a preset such as *quality on
sale*, whether a stock is fundamentally sound and only temporarily oversold,
and whether the data is good enough to judge. Technical analysis (TA-Lib
indicators and signals) is one block of those KPIs.

Nothing here is a forecast: every KPI carries an evidence tier (A academic,
B practitioner, C heuristic) and every answer the date of the run it is based
on.

## Prerequisites

- A bearish database: `bearish run ./bear.db Belgium` (prices, statements,
  analyst and ownership data), then `bearish macro`, `bearish link`,
  `bearish members`, `bearish fx` for the group-level KPIs.
- **TA-Lib**: the `ta-lib` wheel bundles the C library on supported
  platforms; otherwise see the [TA-Lib installation guide](https://ta-lib.org/).

```bash
pip install bullishpy
```

## What is computed

| Level | KPIs | Composite |
|---|---|---|
| Market regime (per region) | trend, breadth, volatility, credit spreads, curve, leading indicators, growth / inflation quadrant, factor leadership | rule-based risk-on / neutral / risk-off label |
| Sector and industry | cap-weighted relative strength, RRG quadrant, breadth, median growth and acceleration, revision ratio, beat rate, valuation vs history, 13F and insider flows, commodity tailwind | `sector_score`, `industry_score` |
| Country | the same market block plus CLI, PMI, forecasts, inflation and policy trend, real yields, BIS credit gap, debt, current account, confidence, uncertainty, FX vs base currency, terms of trade | `country_score` |
| ETF | expense ratio, AUM and flows, dollar volume, tracking difference / error vs the best-correlated index, premium to NAV, holdings concentration, risk-adjusted momentum, age | `etf_vehicle_score`; `etf_score` = 0.5 exposure attractiveness + 0.3 vehicle + 0.2 momentum |
| Equity | technical (every TA-Lib indicator and signal of `bullish.analysis.indicators`, dated signals as trading days since they last fired), momentum and risk, growth and acceleration, quality (ROIC, accruals, Piotroski F, Mohanram G, Altman Z, dilution), valuation vs peers and own history, revisions and surprises, 13F / insider / 13D / short interest / 8-K items, macro exposure | presets `emerging_leader`, `quality_compounder`, `value_with_catalyst`, `turnaround`, `dividend_quality`, `quality_on_sale` |
| Stock verdict | fundamental checklist, oversold signals, drawdown nature (company-specific / sector / market), temporary-vs-structural evidence, recovery evidence | label: quality on sale, falling knife, sound not cheap, oversold unclear, avoid |
| Data coverage | statements, prices, expectations, ownership and reference per symbol | grade A / B / C / insufficient with the bearish command that fills the gap |

KPIs are ranked 0-100 inside their peer group (winsorised, direction-aware),
blocks average their KPIs and composites average their blocks; weights are
declared in `bullish/scoring/presets.py` and every block carries its rationale.
Values are stored point-in-time in the long `kpi` table and the `score` table,
so `as_of` questions are answered with what was known then.

## Running it

```bash
# compute every KPI, score and grade for today (or a past date) and store them
bullish compute --db /path/to/bear.db [--as-of 2026-06-30] [--country USA --country DEU]

# a monthly grid of past runs, then a rank backtest of a KPI or a preset
bullish history 2024-01-31 2026-09-30 --db /path/to/bear.db --step-months 1
bullish backtest mom_12_1 --db /path/to/bear.db --horizon-months 3
bullish backtest quality_on_sale --db /path/to/bear.db

# serve the MCP server (stdio for Claude Desktop / Claude Code, or streamable HTTP)
bullish mcp --db /path/to/bear.db
bullish mcp --db /path/to/bear.db --transport streamable-http --port 8765
```

Claude Desktop / Claude Code configuration:

```json
{
  "mcpServers": {
    "bullish": {
      "command": "uv",
      "args": ["run", "bullish", "mcp", "--db", "/path/to/bear.db"]
    }
  }
}
```

Settings come from the environment or a `.env` file (`BULLISH_DATABASE_PATH`,
`BULLISH_BASE_CURRENCY`, `BULLISH_PEER_GROUP`, `BULLISH_MIN_DOLLAR_VOLUME`,
`BULLISH_MIN_MARKET_CAP`, `BULLISH_ALLOW_SQL`, ...), see `bullish/config.py`.

## MCP tools

`get_market_regime`, `rank_sectors`, `rank_industries`, `rank_countries`,
`rank_etfs`, `compare_etfs`, `screen_equities`, `explain_score`,
`get_equity_scorecard`, `assess_stock`, `get_kpi_series`, `get_smart_money`,
`get_technical_signals`, `get_filings`, `get_earnings_calendar`, `get_macro_snapshot`,
`get_commodity_snapshot`, `get_data_coverage`, `analyze_portfolio`,
`list_kpis`, `describe_kpi`, `list_composites`, `list_runs` and, when
`BULLISH_ALLOW_SQL=1`, a read-only `query`. Resources expose the KPI
catalogue and the methodology of each composite; prompts cover a weekly
market review, a sector briefing, an ETF selection and a stock due diligence.

The plan behind this layer, with the literature it rests on, is in
`docs/plans/2026-10-decision-kpis-and-mcp.md`.

---

## Development

Bullish uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --all-groups          # environment with the dev tools
uv run pytest                 # tests (tests/v2 run on a synthetic bearish database)
uv run nox -s linting         # black, mypy (strict), ruff
```

The repository layout:

| Package | Role |
|---|---|
| `bullish.analysis` | TA-Lib indicator functions and the signal definitions |
| `bullish.data` | point-in-time loaders over the bearish tables |
| `bullish.kpi` | KPI registry, equity and group blocks, the run engine |
| `bullish.scoring` | percentile ranking, composites and presets, stock verdict, coverage grades, regime |
| `bullish.backtest` | rank information coefficient across stored runs |
| `bullish.database` | bullish's tables (`kpi`, `score`, `kpirun`) and the migration bridge with bearish |
| `bullish.mcp` | the MCP server and its service layer |

## What Bullish is not

Not a trading platform and not real-time: it is a research layer for
decisions over days to months, on data you own locally.
