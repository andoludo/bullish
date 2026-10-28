"""A context derived from a span must equal a context queried fresh for the same date."""

import datetime

import pandas as pd
import pytest

from bullish.config import BullishSettings
from bullish.data.panels import DataContext
from bullish.database.crud import BullishDb
from bullish.kpi.equity import compute_equity_kpis
from bullish.kpi.group import compute_group_kpis
from tests.v2.synthetic import AS_OF

EARLIER = AS_OF - datetime.timedelta(days=200)


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["symbol", "date"]).reset_index(drop=True)


def test_derived_context_matches_fresh_context(
    bullish_db_v2: BullishDb, settings: BullishSettings
) -> None:
    span = DataContext(
        db=bullish_db_v2,
        settings=settings.model_copy(update={"lookback_years": 6}),
        as_of=AS_OF,
    )
    span.close  # noqa: B018
    derived = span.derive(EARLIER, settings)
    fresh = DataContext(db=bullish_db_v2, settings=settings, as_of=EARLIER)

    assert derived.start_ts == fresh.start_ts
    pd.testing.assert_frame_equal(derived.universe, fresh.universe)
    pd.testing.assert_frame_equal(derived.close, fresh.close)
    pd.testing.assert_frame_equal(
        derived.metrics_quarterly.reset_index(drop=True),
        fresh.metrics_quarterly.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(_ordered(derived.earnings), _ordered(fresh.earnings))
    pd.testing.assert_frame_equal(
        derived.macro.reset_index(drop=True), fresh.macro.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(derived.fx, fresh.fx)

    equity_derived = compute_equity_kpis(derived)
    equity_fresh = compute_equity_kpis(fresh)
    pd.testing.assert_frame_equal(equity_derived, equity_fresh)
    groups_derived = compute_group_kpis(derived, equity_derived)
    groups_fresh = compute_group_kpis(fresh, equity_fresh)
    assert groups_derived.keys() == groups_fresh.keys()
    for name in groups_fresh:
        pd.testing.assert_frame_equal(groups_derived[name], groups_fresh[name])


def test_derive_refuses_a_later_date(
    bullish_db_v2: BullishDb, settings: BullishSettings
) -> None:
    span = DataContext(db=bullish_db_v2, settings=settings, as_of=EARLIER)
    with pytest.raises(ValueError, match="after"):
        span.derive(AS_OF)


def test_derive_refuses_a_window_the_span_does_not_cover(
    bullish_db_v2: BullishDb, settings: BullishSettings
) -> None:
    span = DataContext(db=bullish_db_v2, settings=settings, as_of=AS_OF)
    with pytest.raises(ValueError, match="lookback"):
        span.derive(EARLIER, settings)
