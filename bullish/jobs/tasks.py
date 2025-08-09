import functools
import logging
from typing import Optional, Any, Callable, List

from bearish.main import Bearish  # type: ignore
from tickermood.main import get_news  # type: ignore
from tickermood.types import DatabaseConfig  # type: ignore

from .app import huey
from pathlib import Path
from huey.api import Task, crontab  # type: ignore

from .models import JobTrackerStatus, JobTracker, JobType
from ..analysis.analysis import run_analysis, run_signal_series_analysis
from ..analysis.backtest import run_many_tests, BackTestConfig
from ..analysis.industry_views import compute_industry_view
from ..analysis.predefined_filters import predefined_filters, NEXT_EARNINGS_DATE
from ..database.crud import BullishDb
from bullish.analysis.filter import FilterUpdate
from ..utils.checks import DataBaseSingleTon

logger = logging.getLogger(__name__)


def job_tracker(func: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(
        database_path: Path,
        job_type: JobType,
        *args: Any,
        task: Optional[Task] = None,
        **kwargs: Any,
    ) -> None:
        bullish_db = BullishDb(database_path=database_path)
        if task is None:
            raise ValueError("Task must be provided for job tracking.")
        if bullish_db.read_job_tracker(task.id) is None:
            bullish_db.write_job_tracker(JobTracker(job_id=str(task.id), type=job_type))
        bullish_db.update_job_tracker_status(
            JobTrackerStatus(job_id=task.id, status="Running")
        )
        try:
            func(database_path, job_type, *args, task=task, **kwargs)
            bullish_db.update_job_tracker_status(
                JobTrackerStatus(job_id=task.id, status="Completed")
            )
        except Exception as e:
            logger.exception(f"Fail to complete job {func.__name__}: {e}")
            bullish_db.update_job_tracker_status(
                JobTrackerStatus(job_id=task.id, status="Failed")
            )

    return wrapper


def _base_update(
    database_path: Path,
    job_type: JobType,
    symbols: Optional[List[str]],
    update_query: FilterUpdate,
    task: Optional[Task] = None,
) -> None:

    if not update_query.update_analysis_only:
        bearish = Bearish(path=database_path, auto_migration=False)
        bearish.update_prices(
            series_length=update_query.window_size,
            delay=update_query.data_age_in_days,
        )
        if update_query.update_financials:
            bearish.update_financials()
    bullish_db = BullishDb(database_path=database_path)
    run_analysis(bullish_db)
    compute_industry_view(bullish_db)


@huey.task(context=True)  # type: ignore
@job_tracker
def update(
    database_path: Path,
    job_type: JobType,
    symbols: Optional[List[str]],
    update_query: FilterUpdate,
    task: Optional[Task] = None,
) -> None:
    _base_update(database_path, job_type, symbols, update_query, task)


@huey.periodic_task(crontab(minute="0", hour="1"), context=True)  # type: ignore
def cron_update(
    task: Optional[Task] = None,
) -> None:
    database = DataBaseSingleTon()
    job_tracker(_base_update)(
        database.path, "Update data", [], FilterUpdate(), task=task
    )


@huey.task(context=True)  # type: ignore
@job_tracker
def analysis(
    database_path: Path,
    job_type: JobType,
    task: Optional[Task] = None,
) -> None:
    bullish_db = BullishDb(database_path=database_path)
    run_analysis(bullish_db)
    compute_industry_view(bullish_db)


@huey.task(context=True)  # type: ignore
@job_tracker
def backtest_signals(
    database_path: Path,
    job_type: JobType,
    task: Optional[Task] = None,
) -> None:
    bullish_db = BullishDb(database_path=database_path)
    run_signal_series_analysis(bullish_db)
    run_many_tests(bullish_db, predefined_filters(), BackTestConfig())


def base_news(
    database_path: Path,
    job_type: JobType,
    symbols: List[str],
    headless: bool = True,
    task: Optional[Task] = None,
) -> None:
    bullish_db = BullishDb(database_path=database_path)
    database_config = DatabaseConfig(database_path=database_path, no_migration=True)
    get_news(symbols, database_config, headless=headless, model_name="qwen3:4b")
    run_analysis(bullish_db)


@huey.task(context=True)  # type: ignore
@job_tracker
def news(
    database_path: Path,
    job_type: JobType,
    symbols: List[str],
    headless: bool = True,
    task: Optional[Task] = None,
) -> None:
    base_news(
        database_path=database_path,
        job_type=job_type,
        symbols=symbols,
        headless=headless,
        task=task,
    )


@huey.periodic_task(crontab(minute="0", hour="3"), context=True)  # type: ignore
def cron_news(
    task: Optional[Task] = None,
) -> None:
    earnings_date = NEXT_EARNINGS_DATE[1]
    database = DataBaseSingleTon()
    bullish_db = BullishDb(database_path=database.path)
    data = bullish_db.read_filter_query(earnings_date)
    job_tracker(base_news)(
        database_path=database.path,
        job_type="Fetching news",
        symbols=data["symbol"].unique().tolist(),
        headless=False,
        task=task,
    )
