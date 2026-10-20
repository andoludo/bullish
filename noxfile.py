import nox
from nox import Session

nox.options.reuse_existing_virtualenvs = True
# uv manages the project virtual environment (.venv); nox only orchestrates the
# commands, so it does not need to create its own environments.
nox.options.default_venv_backend = "none"


@nox.session
def install(session: Session) -> None:
    session.run("uv", "sync", "--all-groups", "--frozen", external=True)


@nox.session
def linting(session: Session) -> None:
    session.run("uv", "run", "black", ".", external=True)
    session.run("uv", "run", "mypy", external=True)
    session.run(
        "uv",
        "run",
        "ruff",
        "check",
        "--fix",
        "--show-fixes",
        "--exit-non-zero-on-fix",
        external=True,
    )


@nox.session
def tests(session: Session) -> None:
    session.run("uv", "run", "pytest", "-m", "not integration", external=True)
