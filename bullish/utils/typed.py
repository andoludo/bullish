"""Small helpers that give pandas results the static types the rest of the code relies on.

pandas-stubs cannot know that an index is unique or that a cell is numeric, so
``frame.loc[key]`` is typed ``Series | DataFrame`` and reductions come back as a
wide union of scalars. These helpers narrow those at the one place where the
assumption actually holds, instead of sprinkling casts and ``type: ignore``.
"""

from __future__ import annotations

import datetime
import math
from typing import SupportsFloat, cast

import pandas as pd


def as_float(value: object) -> float:
    """``float(value)`` for anything numeric, ``nan`` for ``None`` or non-numeric values."""
    if isinstance(value, SupportsFloat):
        return float(value)
    return math.nan


def row_of(frame: pd.DataFrame, key: str) -> pd.Series:
    """The single row labelled ``key``; the caller guarantees the index is unique."""
    return cast("pd.Series", frame.loc[key])


def iso_date(value: object) -> str:
    """ISO ``YYYY-MM-DD`` for dates, timestamps or anything that formats as one."""
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)
