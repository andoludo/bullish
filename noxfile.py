from typing import List

import nox
from nox import Session

nox.options.reuse_existing_virtualenvs = True


def install(session: Session, groups: List[str], root: bool = True) -> None:
    if root:
        groups = ["main", *groups]

    session.run_always(
        "poetry",
        "install",
        "--no-root",
        "--sync",
        f"--only={','.join(groups)}",
        external=True,
    )
    if root:
        session.install(".")


@nox.session(python=["3.10"])
def tests(session: Session) -> None:
    session.run("poetry", "run", "pytest", "-m", "not integration")


@nox.session(python=["3.10"])
def linting(session: Session) -> None:
    session.run("poetry", "run", "black", ".")
    session.run("poetry", "run", "mypy")
    session.run(
        "poetry",
        "run",
        "ruff",
        "check",
        "--fix",
        "--show-fixes",
        "--exit-non-zero-on-fix",
    )
