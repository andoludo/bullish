# Bullish

**Bullish** is a high-powered stock screener that helps you quickly identify the best stock or trading opportunities in the market.  
It can scan **thousands of equities** across multiple markets, exchanges, and countries to uncover strong *buy* candidates.  

Bullish uses the well-known **TA-Lib** library to calculate popular technical analysis indicators—such as **RSI**, **MACD**, and moving averages—then lets you filter and select the strongest stocks from your **local database**.

---

## Why Bullish?
The main goals behind Bullish are:
- **Full control over your data** — no dependency on third-party screeners  
- **Local analysis** — run any type of screening or backtesting on your own system  

Bullish is built on:
- **bearish** – a Python library that fetches equity data, statements, analyst and ownership data, macro series and ETF metadata from multiple sources  
- **tickermood** – retrieves recent, relevant news for screened tickers and uses LLMs to produce an investment recommendation.

---

## Prerequisites
### Install TA-Lib
Bullish depends on **TA-Lib** for technical analysis calculations.  
TA-Lib must be installed separately before using Bullish.  
See the [TA-Lib installation guide](https://ta-lib.org/) for instructions.

---

## Installation
```bash
pip install bullishpy
```

---

## Quick Start

### 1. Create a Bearish Database
A **bearish database** contains historical prices and fundamental data for all stocks in your chosen market.

Example: Create a database for the Belgian stock market:
```bash
bearish run ./bearish.db Belgium
```
You can replace `Belgium` with any supported country.  
**Note:** Building the database can take some time.

---

### 2. Run Bullish
Navigate to the folder containing your **bullish database** and run:
```bash
bullish
```
This launches a **local Streamlit app** where you can screen, filter, and analyze stocks interactively.

![img1.png](docs/img1.png)


![img.png](docs/img.png)

---

## Decision KPIs and the MCP server (bullish v2)

On top of the screener, bullish computes a **catalogue of decision KPIs** from
a bearish database and serves them to an LLM client over the
[Model Context Protocol](https://modelcontextprotocol.io). Nothing here is a
forecast: every KPI carries an evidence tier (A academic, B practitioner,
C heuristic) and every answer the date of the run it is based on.

### What is computed

| Level | KPIs | Composite |
|---|---|---|
| Market regime (per region) | trend, breadth, volatility, credit spreads, curve, leading indicators, growth / inflation quadrant, factor leadership | rule-based risk-on / neutral / risk-off label |
| Sector and industry | cap-weighted relative strength, RRG quadrant, breadth, median growth and acceleration, revision ratio, beat rate, valuation vs history, 13F and insider flows, commodity tailwind | `sector_score`, `industry_score` |
| Country | the same market block plus CLI, PMI, forecasts, inflation and policy trend, real yields, BIS credit gap, debt, current account, confidence, uncertainty, FX vs base currency, terms of trade | `country_score` |
| ETF | expense ratio, AUM and flows, dollar volume, tracking difference / error vs the best-correlated index, premium to NAV, holdings concentration, risk-adjusted momentum, age | `etf_vehicle_score`; `etf_score` = 0.5 exposure attractiveness + 0.3 vehicle + 0.2 momentum |
| Equity (124 KPIs) | momentum and risk, growth and acceleration, quality (ROIC, accruals, Piotroski F, Mohanram G, Altman Z, dilution), valuation vs peers and own history, revisions and surprises, 13F / insider / 13D / short interest / 8-K items, macro exposure | presets `emerging_leader`, `quality_compounder`, `value_with_catalyst`, `turnaround`, `dividend_quality`, `quality_on_sale` |
| Stock verdict | fundamental checklist, oversold signals, drawdown nature (company-specific / sector / market), temporary-vs-structural evidence, recovery evidence | label: quality on sale, falling knife, sound not cheap, oversold unclear, avoid |
| Data coverage | statements, prices, expectations, ownership and reference per symbol | grade A / B / C / insufficient with the bearish command that fills the gap |

KPIs are ranked 0-100 inside their peer group (winsorised, direction-aware),
blocks average their KPIs and composites average their blocks; weights are
declared in `bullish/scoring/presets.py` and every block carries its rationale.
Values are stored point-in-time in the long `kpi` table and the `score` table,
so `as_of` questions are answered with what was known then.

### Running it

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

### MCP tools

`get_market_regime`, `rank_sectors`, `rank_industries`, `rank_countries`,
`rank_etfs`, `compare_etfs`, `screen_equities`, `explain_score`,
`get_equity_scorecard`, `assess_stock`, `get_kpi_series`, `get_smart_money`,
`get_filings`, `get_earnings_calendar`, `get_macro_snapshot`,
`get_commodity_snapshot`, `get_data_coverage`, `analyze_portfolio`,
`list_kpis`, `describe_kpi`, `list_composites`, `list_runs` and, when
`BULLISH_ALLOW_SQL=1`, a read-only `query`. Resources expose the KPI
catalogue and the methodology of each composite; prompts cover a weekly
market review, a sector briefing, an ETF selection and a stock due diligence.

The plan behind this layer, with the literature it rests on, is in
`docs/plans/2026-10-decision-kpis-and-mcp.md`.

---

## Development

Bullish uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

```bash
# Install uv (https://docs.astral.sh/uv/getting-started/installation/)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create the virtual environment and install all dependencies (incl. dev tools)
uv sync --all-groups

# Run the test suite (integration tests excluded)
uv run pytest -m "not integration"

# Linting & type checking
uv run nox -s linting

# Run any command inside the project environment
uv run bullish
```

> **Note:** `ta-lib` is installed from a prebuilt wheel, so the TA-Lib C library
> no longer needs to be built separately for local development on supported
> platforms.

---

## What Bullish Is Not
Bullish is **not**:
- A real-time trading platform  
- A tool for intraday or high-frequency trading  

It is designed for **retail traders** and **swing traders** focusing on opportunities over days or weeks.