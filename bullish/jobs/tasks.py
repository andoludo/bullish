import functools
import logging
from typing import Optional, Any, Callable, List

from bearish.main import Bearish  # type: ignore
from tickermood.main import get_news  # type: ignore
from tickermood.types import DatabaseConfig  # type: ignore

from .app import huey
from pathlib import Path
from huey.api import Task  # type: ignore

from .models import JobTrackerStatus
from ..analysis.analysis import run_analysis
from ..database.crud import BullishDb
from bullish.analysis.filter import FilterUpdate

logger = logging.getLogger(__name__)


def job_tracker(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(
        database_path: Path, *args: Any, task: Optional[Task] = None, **kwargs: Any
    ) -> None:
        bullish_db = BullishDb(database_path=database_path)
        if task is None:
            raise ValueError("Task must be provided for job tracking.")
        bullish_db.update_job_tracker_status(
            JobTrackerStatus(job_id=task.id, status="Running")
        )
        try:
            func(database_path, *args, task=task, **kwargs)
            bullish_db.update_job_tracker_status(
                JobTrackerStatus(job_id=task.id, status="Completed")
            )
        except Exception as e:
            logger.exception(f"Fail to complete job {func.__name__}: {e}")
            bullish_db.update_job_tracker_status(
                JobTrackerStatus(job_id=task.id, status="Failed")
            )

    return wrapper


@huey.task(context=True)  # type: ignore
@job_tracker
def update(
    database_path: Path,
    symbols: List[str],
    update_query: FilterUpdate,
    task: Optional[Task] = None,
) -> None:
    logger.debug(f"Running update task for {len(symbols)} tickers.")
    if not update_query.update_analysis_only:
        bearish = Bearish(path=database_path, auto_migration=False)
        bearish.update_prices(
            symbols,
            series_length=update_query.window_size,
            delay=update_query.data_age_in_days,
        )
        if update_query.update_financials:
            bearish.update_financials(symbols)
    bullish_db = BullishDb(database_path=database_path)
    run_analysis(bullish_db)


@huey.task(context=True)  # type: ignore
@job_tracker
def news(
    database_path: Path,
    symbols: List[str],
    headless: bool = True,
    task: Optional[Task] = None,
) -> None:
    database_config = DatabaseConfig(database_path=database_path, no_migration=True)
    get_news(symbols, database_config, headless=headless)
