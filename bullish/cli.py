from __future__ import annotations

import os
import subprocess
import signal
import sys
import time
import atexit
from pathlib import Path
from typing import Any

import typer
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd().joinpath(".env"))

app = typer.Typer(add_completion=False)
STREAMLIT_FILE = Path(__file__).parent.joinpath("app", "app.py")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Streamlit host"),  # noqa: S104
    port: int = typer.Option(8501, help="Streamlit port"),
    env_vars: Path = typer.Option(  # noqa: B008
        None, help="Environment variables file"
    ),
) -> None:
    if env_vars:
        env_vars_path = Path(env_vars)
        if env_vars_path.exists():
            load_dotenv(dotenv_path=env_vars_path)
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


if __name__ == "__main__":
    app()
