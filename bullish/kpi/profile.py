"""Per-phase wall time and peak memory of a KPI run (``bullish compute --profile``)."""

from __future__ import annotations

import logging
import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager

from pydantic import BaseModel

logger = logging.getLogger(__name__)


def peak_rss_mb() -> float:
    """Peak resident set size of this process so far, in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class PhaseTiming(BaseModel):
    phase: str
    seconds: float
    peak_rss_mb: float


class PhaseTimer:
    """Collects :class:`PhaseTiming` rows; ``with timer.phase("name"):`` around each step."""

    def __init__(self) -> None:
        self.timings: list[PhaseTiming] = []

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            timing = PhaseTiming(
                phase=name,
                seconds=time.perf_counter() - start,
                peak_rss_mb=peak_rss_mb(),
            )
            self.timings.append(timing)
            logger.info(
                "%-40s %7.2fs  peak rss %6.0f MB",
                name,
                timing.seconds,
                timing.peak_rss_mb,
            )

    @property
    def total_seconds(self) -> float:
        return sum(t.seconds for t in self.timings)

    def table(self) -> str:
        total = self.total_seconds or 1.0
        lines = [f"{'phase':<40} {'seconds':>8} {'share':>6} {'peak RSS MB':>12}"]
        lines += [
            f"{t.phase:<40} {t.seconds:8.2f} {t.seconds / total * 100:5.1f}% {t.peak_rss_mb:12.0f}"
            for t in self.timings
        ]
        lines.append(f"{'total':<40} {self.total_seconds:8.2f}")
        return "\n".join(lines)
