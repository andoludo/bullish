import functools
import logging
from typing import Optional, Any, Callable, List

import pandas as pd
from bearish.main import Bearish  # type: ignore
from bearish.models.sec.sec import Secs  # type: ignore
from tickermood.main import get_news  # type: ignore
from tickermood.types import DatabaseConfig  # type: ignore

from .app import huey
from pathlib import Path
from huey.api import Task, crontab  # type: ignore

from .models import JobTrackerStatus, JobTracker, JobType
from ..analysis.analysis import run_analysis, run_signal_series_analysis
from ..analysis.backtest import run_many_tests, BackTestConfig
from ..analysis.industry_views import compute_industry_view
from ..analysis.openai import get_open_ai_news
from ..analysis.predefined_filters import predefined_filters, load_custom_filters
from ..database.crud import BullishDb
from bullish.analysis.filter import FilterUpdate

logger = logging.getLogger(__name__)


class DataBaseSingleTon:
    _instance = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, path: Optional[Path] = None) -> None:
        if not hasattr(self, "path"):  # Only set once
            self.path = path

    def valid(self) -> bool:
        return hasattr(self, "path") and self.path is not None


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
            batch_size=update_query.batch_size,
            pause=update_query.pause,
        )
        bearish.get_prices_index(series_length=update_query.window_size)
        Secs.upload(bearish._bearish_db)
        if update_query.update_financials:
            bearish.update_financials()
    bullish_db = BullishDb(database_path=database_path)
    run_analysis(bullish_db)
    compute_industry_view(bullish_db)


@huey.task(context=True)  # type: ignore
@job_tracker
def initialize(
    database_path: Path,
    job_type: JobType,
    task: Optional[Task] = None,
) -> None:
    database = DataBaseSingleTon(path=database_path)
    if not database.valid():
        raise ValueError("Database path is not valid.")


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
    if database.valid():
        job_tracker(_base_update)(
            database.path, "Update data", [], FilterUpdate(), task=task
        )


@huey.periodic_task(crontab(day_of_week=0, hour=9, minute=0), context=True)  # type: ignore
def cron_financial_update(
    task: Optional[Task] = None,
) -> None:
    database = DataBaseSingleTon()
    if database.valid():
        job_tracker(_base_update)(
            database.path,
            "Update data",
            [],
            FilterUpdate(update_financials=True),
            task=task,
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
    bullish_db = BullishDb(database_path=database_path)
    if get_open_ai_news(bullish_db, symbols):
        for symbol in symbols:
            subject = bullish_db.read_subject(symbol)
            if subject:
                logger.debug(
                    f"extracting news for {symbol} subject: {subject.model_dump()}"
                )
                try:
                    bullish_db.update_analysis(
                        symbol,
                        subject.model_dump(
                            exclude_none=True,
                            exclude_unset=True,
                            exclude_defaults=True,
                            exclude={"symbol"},
                        ),
                    )
                except Exception as e:
                    logger.error(f"failed to extract news for {symbol}: {e}")
                    print(f"failed to extract news for {symbol}: {e}")
                    continue


@huey.periodic_task(crontab(minute="0", hour="8"), context=True)  # type: ignore
def cron_news(
    task: Optional[Task] = None,
) -> None:
    filters = load_custom_filters()
    database = DataBaseSingleTon()
    if database.valid() and filters:
        bullish_db = BullishDb(database_path=database.path)
        data = pd.concat([bullish_db.read_filter_query(f) for f in filters])
        if not data.empty and "symbol" in data.columns:
            job_tracker(base_news)(
                database_path=database.path,
                job_type="Fetching news",
                symbols=data["symbol"].unique().tolist(),
                headless=False,
                task=task,
            )
