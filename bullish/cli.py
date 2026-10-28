from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

if TYPE_CHECKING:
    from bullish.config import BullishSettings

app = typer.Typer(
    add_completion=False,
    help="Decision KPIs and an MCP server on top of a bearish database.",
)


def _settings(db: Path | None, base_currency: str | None) -> BullishSettings:
    from bullish.config import BullishSettings

    overrides: dict[str, Any] = {}
    if db is not None:
        overrides["database_path"] = db
    if base_currency:
        overrides["base_currency"] = base_currency
    return BullishSettings(**overrides)


@app.command()
def compute(
    db: Path = typer.Option(  # noqa: B008
        None, help="bearish SQLite database (default: BULLISH_DATABASE_PATH)"
    ),
    as_of: str = typer.Option(None, help="Run date, ISO (default: today)"),
    country: list[str] = typer.Option(  # noqa: B008
        None, help="ISO-3 countries to include (default: all)"
    ),
    base_currency: str = typer.Option(
        None, help="Currency of every cross-country comparison (default: EUR)"
    ),
    symbol: list[str] = typer.Option(  # noqa: B008
        None, help="Restrict the run to these symbols (default: every priced equity)"
    ),
) -> None:
    """Compute every KPI, composite score and coverage grade for one date and store them."""
    import datetime

    from bullish.database.crud import BullishDb
    from bullish.kpi import engine

    settings = _settings(db, base_currency)
    database = BullishDb(database_path=settings.require_database())
    date = datetime.date.fromisoformat(as_of) if as_of else None
    result = engine.run(
        database,
        settings,
        as_of=date,
        countries=country or None,
        symbols=symbol or None,
    )
    if result.run is None:
        typer.secho("Nothing computed (empty universe).", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.echo(
        f"{result.run.as_of}: {result.run.entities} equities, {result.run.kpis} KPI values, "
        f"{result.run.scores} scores stored ({result.run.base_currency})."
    )


@app.command()
def history(  # noqa: PLR0913
    start: str = typer.Argument(..., help="First run date, ISO"),
    end: str = typer.Argument(..., help="Last run date, ISO"),
    db: Path = typer.Option(  # noqa: B008
        None, help="bearish SQLite database (default: BULLISH_DATABASE_PATH)"
    ),
    step_months: int = typer.Option(1, help="Months between runs"),
    country: list[str] = typer.Option(  # noqa: B008
        None, help="ISO-3 countries to include (default: all)"
    ),
    base_currency: str = typer.Option(None, help="Base currency (default: EUR)"),
) -> None:
    """Run the KPI engine on a grid of past dates, point-in-time, so series and backtests have history."""
    import datetime

    from bullish.backtest.rank import build_history
    from bullish.database.crud import BullishDb

    settings = _settings(db, base_currency)
    database = BullishDb(database_path=settings.require_database())
    dates = build_history(
        database,
        settings,
        datetime.date.fromisoformat(start),
        datetime.date.fromisoformat(end),
        step_months,
        country or None,
    )
    typer.echo(
        f"{len(dates)} runs stored: {dates[0]} .. {dates[-1]}"
        if dates
        else "No dates in range."
    )


@app.command()
def backtest(
    name: str = typer.Argument(..., help="Equity KPI name or preset name"),
    db: Path = typer.Option(  # noqa: B008
        None, help="bearish SQLite database (default: BULLISH_DATABASE_PATH)"
    ),
    horizon_months: int = typer.Option(3, help="Forward return horizon"),
    quantiles: int = typer.Option(5, help="Quantiles for the top-minus-bottom spread"),
) -> None:
    """Rank information coefficient and quantile spread of a KPI or preset across the stored runs."""
    from bullish.backtest.rank import backtest_kpi, backtest_score
    from bullish.database.crud import BullishDb
    from bullish.scoring.presets import EQUITY_PRESETS

    settings = _settings(db, None)
    database = BullishDb(database_path=settings.require_database())
    runner = backtest_score if name in EQUITY_PRESETS else backtest_kpi
    result = runner(database, settings, name, horizon_months, quantiles)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def mcp(  # noqa: PLR0913
    db: Path = typer.Option(  # noqa: B008
        None, help="bearish SQLite database (default: BULLISH_DATABASE_PATH)"
    ),
    transport: str = typer.Option(
        "stdio", help="stdio (Claude Desktop / Code) or streamable-http"
    ),
    host: str = typer.Option("127.0.0.1", help="Host for streamable-http"),
    port: int = typer.Option(8765, help="Port for streamable-http"),
    allow_sql: bool = typer.Option(False, help="Expose the read-only SQL tool"),
    base_currency: str = typer.Option(None, help="Base currency (default: EUR)"),
) -> None:
    """Serve the bullish MCP server over the stored KPI runs."""
    from bullish.mcp.server import serve

    settings = _settings(db, base_currency).model_copy(update={"allow_sql": allow_sql})
    settings.require_database()
    serve(settings, transport=transport, host=host, port=port)


if __name__ == "__main__":
    app()
