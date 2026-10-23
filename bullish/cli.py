from __future__ import annotations

import os
import subprocess
import signal
import sys
import time
import atexit
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

if TYPE_CHECKING:
    from bullish.config import BullishSettings

app = typer.Typer(add_completion=False)
STREAMLIT_FILE = Path(__file__).parent.joinpath("app", "app.py")


@app.command()
def serve(  # noqa: C901
    host: str = typer.Option("0.0.0.0", help="Streamlit host"),  # noqa: S104
    port: int = typer.Option(8501, help="Streamlit port"),
    env_vars: Path = typer.Option(  # noqa: B008
        None, help="Environment variables file"
    ),
    custom_filters: Path = typer.Option(  # noqa: B008
        None, help="Environment variables file"
    ),
) -> None:
    if env_vars:
        env_vars_path = Path(env_vars)
        if env_vars_path.exists():
            load_dotenv(dotenv_path=env_vars_path)
    if custom_filters:
        custom_filters_path = Path(custom_filters)
        if custom_filters_path.exists():
            os.environ["CUSTOM_FILTERS_PATH"] = str(custom_filters_path)
    children: list[subprocess.Popen] = []  # type: ignore

    def _shutdown(*_: Any) -> None:
        for p in children:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)

    try:
        children.append(
            subprocess.Popen(
                [  # noqa: S603
                    sys.executable,
                    "-m",
                    "huey.bin.huey_consumer",
                    "bullish.jobs.tasks.huey",
                    "--workers",
                    "2",
                ],
                env=os.environ.copy(),
            )
        )
        children.append(
            subprocess.Popen(
                [  # noqa: S603
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(STREAMLIT_FILE),
                    "--server.address",
                    host,
                    "--server.port",
                    str(port),
                    os.devnull,
                ],
                env=os.environ.copy(),
            )
        )
        while True:
            time.sleep(1)

    except Exception as exc:  # pragma: no cover
        typer.secho(
            f"❌ Failed to start services: {exc}", fg=typer.colors.RED, err=True
        )
        _shutdown()


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
) -> None:
    """Compute every KPI, composite score and coverage grade for one date and store them."""
    import datetime

    from bullish.database.crud import BullishDb
    from bullish.kpi import engine

    settings = _settings(db, base_currency)
    database = BullishDb(database_path=settings.require_database())
    date = datetime.date.fromisoformat(as_of) if as_of else None
    result = engine.run(database, settings, as_of=date, countries=country or None)
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
