"""Cross-sectional normalisation: winsorised percentile ranks inside a peer group."""

from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize(values: pd.Series, limit: float) -> pd.Series:
    """Clip both tails at the ``limit`` quantile (no-op when ``limit`` is 0 or the series is short)."""
    valid = values.dropna()
    if limit <= 0 or len(valid) < 10:
        return values
    low, high = valid.quantile(limit), valid.quantile(1 - limit)
    return values.clip(lower=low, upper=high)


def percentile_rank(
    values: pd.Series, groups: pd.Series | None = None, min_peers: int = 5
) -> pd.Series:
    """Percentile (0-100) of each value inside its group; groups smaller than ``min_peers`` rank in the whole set.

    Ranks are average-tied and scaled to 0 (worst) ... 100 (best); a lone value sits at 50.
    """
    if groups is None:
        return _rank(values)
    aligned = groups.reindex(values.index).fillna("_none")
    counts = values.notna().groupby(aligned).transform("sum")
    small = counts < min_peers
    inside = values.groupby(aligned).transform(_rank)
    universe_wide = _rank(values)
    return inside.where(~small, universe_wide).where(values.notna())


def _rank(values: pd.Series) -> pd.Series:
    n = values.notna().sum()
    if n == 0:
        return pd.Series(np.nan, index=values.index)
    if n == 1:
        return pd.Series(np.where(values.notna(), 50.0, np.nan), index=values.index)
    ranked: pd.Series = values.rank(method="average", pct=True)
    scaled: pd.Series = (ranked - 1 / n) / (1 - 1 / n) * 100.0
    return scaled


def directed_percentile(  # noqa: PLR0913
    values: pd.Series,
    higher_is_better: bool | None,
    groups: pd.Series | None,
    winsor_limit: float,
    min_peers: int,
    invert: bool = False,
) -> pd.Series:
    """Percentile where 100 is always "good": flips for lower-is-better KPIs and explicit inversions."""
    pct = percentile_rank(
        winsorize(values.astype(float), winsor_limit), groups, min_peers
    )
    better_high = True if higher_is_better is None else higher_is_better
    if invert:
        better_high = not better_high
    return pct if better_high else 100 - pct
