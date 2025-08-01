import logging
import shelve
import uuid
from pathlib import Path
from typing import Optional, List, Type, Dict, Any

import pandas as pd
import streamlit as st
import streamlit_pydantic as sp
from bearish.models.base import Ticker  # type: ignore
from bearish.models.price.prices import Prices  # type: ignore
from bearish.models.query.query import AssetQuery, Symbols  # type: ignore
from streamlit_file_browser import st_file_browser  # type: ignore

from bullish.analysis.backtest import BacktestResults
from bullish.analysis.industry_views import get_industry_comparison_data
from bullish.analysis.predefined_filters import PredefinedFilters
from bullish.database.crud import BullishDb
from bullish.figures.figures import plot
from bullish.analysis.filter import (
    FilterQuery,
    FilterUpdate,
    FilteredResults,
    FilterQueryStored,
    FundamentalAnalysisFilters,
    GROUP_MAPPING,
    GeneralFilter,
    TechnicalAnalysisFilters,
)
from bullish.jobs.tasks import update, news, analysis, backtest_signals
from pydantic import BaseModel

from bullish.utils.checks import (
    compatible_bearish_database,
    compatible_bullish_database,
    empty_analysis_table,
)

CACHE_SHELVE = "user_cache"
DB_KEY = "db_path"

st.set_page_config(layout="wide")
logger = logging.getLogger(__name__)


@st.cache_resource
def db_id() -> str:
    return f"{DB_KEY}_{uuid.uuid4()!s}"


@st.cache_resource
def bearish_db(database_path: Path) -> BullishDb:
    return BullishDb(database_path=database_path)


def store_db(db_path: Path) -> None:
    with shelve.open(CACHE_SHELVE) as storage:  # noqa:S301
        storage[db_id()] = str(db_path)


def load_db() -> Optional[str]:
    with shelve.open(CACHE_SHELVE) as db:  # noqa:S301
        db_path = db.get(db_id())
        return db_path


def assign_db_state() -> None:
    if "database_path" not in st.session_state:
        st.session_state.database_path = load_db()


@st.cache_data(hash_funcs={BullishDb: lambda obj: hash(obj.database_path)})
def load_analysis_data(bullish_db: BullishDb) -> pd.DataFrame:
    return bullish_db.read_analysis_data()


def on_table_select() -> None:

    row = st.session_state.selected_data["selection"]["rows"]

    db = bearish_db(st.session_state.database_path)
    if st.session_state.data.empty or (
        not st.session_state.data.iloc[row]["symbol"].to_numpy()
    ):
        return

    symbol = st.session_state.data.iloc[row]["symbol"].to_numpy()[0]
    country = st.session_state.data.iloc[row]["country"].to_numpy()[0]
    industry = st.session_state.data.iloc[row]["industry"].to_numpy()[0]
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol=symbol)]))
    prices = db.read_series(query, months=24)
    data = Prices(prices=prices).to_dataframe()
    dates = db.read_dates(symbol)
    industry_data = get_industry_comparison_data(db, data, "Mean", industry, country)

    fig = plot(data, symbol, dates=dates, industry_data=industry_data)

    st.session_state.ticker_figure = fig


@st.dialog("🔑  Provide database file to continue")
def dialog_pick_database() -> None:
    current_working_directory = Path.cwd()
    event = st_file_browser(
        path=current_working_directory, key="A", glob_patterns="**/*.db"
    )
    if event:
        db_path = Path(current_working_directory).joinpath(event["target"]["path"])
        if not (db_path.exists() and db_path.is_file()):
            st.stop()
        if not compatible_bearish_database(db_path):
            st.error(f"The database {db_path} is not compatible with this application.")
            st.stop()
        st.session_state.database_path = db_path
        store_db(db_path)
        compatible_bullish_db = compatible_bullish_database(db_path)
        if (not compatible_bullish_db) or (
            compatible_bullish_db and empty_analysis_table(db_path)
        ):
            st.warning(
                f"The database {db_path} has not the necessary data to run this application. "
                "A backround job will be started to update the data."
            )
            analysis(db_path)
        st.rerun()
    if event is None:
        st.stop()


@st.cache_resource
def symbols() -> List[str]:
    bearish_db_ = bearish_db(st.session_state.database_path)
    return bearish_db_.read_symbols()


def groups_mapping() -> Dict[str, List[str]]:
    GROUP_MAPPING["symbol"] = symbols()
    return GROUP_MAPPING


def build_filter(model: Type[BaseModel], data: Dict[str, Any]) -> Dict[str, Any]:

    for field, info in model.model_fields.items():
        name = info.description or info.alias or field
        default = info.default
        if data.get(field) and data[field] != info.default:
            default = data[field]
        if info.annotation == Optional[List[str]]:  # type: ignore
            data[field] = st.multiselect(
                name,
                groups_mapping()[field],
                default=default,
                key=hash((model.__name__, field)),
            )
        elif info.annotation == Optional[str]:  # type: ignore
            options = ["", *groups_mapping()[field]]
            data[field] = st.selectbox(
                name,
                options,
                index=0 if not default else options.index(default),
                key=hash((model.__name__, field)),
            )

        else:
            ge = next(
                (item.ge for item in info.metadata if hasattr(item, "ge")),
                info.default[0] if info.default and len(info.default) == 2 else None,
            )
            le = next(
                (item.le for item in info.metadata if hasattr(item, "le")),
                info.default[1] if info.default and len(info.default) == 2 else None,
            )
            if info.annotation == Optional[List[float]]:  # type: ignore
                ge = int(ge)  # type: ignore
                le = int(le)  # type: ignore
                default = [int(d) for d in default]
            try:
                data[field] = list(
                    st.slider(  # type: ignore
                        name, ge, le, tuple(default), key=hash((model.__name__, field))
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error building filter for {model.__name__}.{field} "
                    f"with the parameters {(info.annotation, name, ge, le, tuple(default))}: {e}"
                )
                raise e
    return data


@st.dialog("⏳  Jobs", width="large")
def jobs() -> None:
    with st.expander("Update data"):
        update_query = sp.pydantic_form(key="update", model=FilterUpdate)
        if (
            update_query
            and st.session_state.data is not None
            and not st.session_state.data.empty
        ):
            symbols = st.session_state.data["symbol"].unique().tolist()
            update(
                database_path=st.session_state.database_path,
                job_type="Update data",
                symbols=symbols,
                update_query=update_query,
            )  # enqueue & get result-handle

            st.success("Data update job has been enqueued.")
            st.rerun()
    with st.expander("Update analysis"):
        if st.button("Update analysis"):
            analysis(st.session_state.database_path, job_type="Update analysis")
            st.success("Data update job has been enqueued.")
            st.rerun()
    with st.expander("Compute backtest signals"):
        if st.button("Compute backtest signals"):
            backtest_signals(
                st.session_state.database_path, job_type="backtest signals"
            )
            st.rerun()


@st.dialog("📥  Load", width="large")
def load() -> None:
    bearish_db_ = bearish_db(st.session_state.database_path)
    existing_filtered_results = bearish_db_.read_list_filtered_results()
    option = st.selectbox("Select portfolio", ["", *existing_filtered_results])
    if option:
        filtered_results_ = bearish_db_.read_filtered_results(option)
        if filtered_results_:
            st.session_state.data = bearish_db_.read_analysis_data(
                symbols=filtered_results_.symbols
            )
            st.rerun()


@st.dialog("🔍  Filter", width="large")
def filter() -> None:
    with st.container():
        column_1, column_2 = st.columns(2)
        with column_1:
            # TODO: order here matters
            with st.expander("Predefined filters"):
                predefined_filter_names = (
                    PredefinedFilters().get_predefined_filter_names()
                )
                option = st.selectbox(
                    "Select a predefined filter",
                    ["", *predefined_filter_names],
                )
                if option:
                    data_ = PredefinedFilters().get_predefined_filter(option)
                    st.session_state.filter_query.update(data_)
            with st.expander("Technical Analysis"):
                for filter in TechnicalAnalysisFilters:
                    with st.expander(filter._description):  # type: ignore
                        build_filter(filter, st.session_state.filter_query)

        with column_2:
            with st.expander("Fundamental Analysis"):
                for filter in FundamentalAnalysisFilters:
                    with st.expander(filter._description):  # type: ignore
                        build_filter(filter, st.session_state.filter_query)
            with st.expander("General filter"):
                build_filter(GeneralFilter, st.session_state.filter_query)

    if st.button("🔍 Apply"):
        query = FilterQuery.model_validate(st.session_state.filter_query)
        if query.valid():
            st.session_state.data = bearish_db(
                st.session_state.database_path
            ).read_filter_query(query)
            st.session_state.ticker_figure = None
            st.session_state.filter_query = {}
            st.session_state.query = query
            st.rerun()


@st.dialog("📈  Price history and analysis", width="large")
def dialog_plot_figure() -> None:
    st.markdown(
        """
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        width: 90vw;
        height: 170vh;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.html("<span class='big-dialog'></span>")
    st.plotly_chart(st.session_state.ticker_figure, use_container_width=True)
    st.session_state.ticker_figure = None


@st.dialog("⭐ Save filtered results")
def save_filtered_results(bearish_db_: BullishDb) -> None:
    user_input = st.text_input("Selection name").strip()
    headless = st.checkbox("Headless mode", value=True)
    apply = st.button("Apply")
    if apply:
        if not user_input:
            st.error("This field is required.")
        else:
            symbols = st.session_state.data["symbol"].unique().tolist()
            filtered_results = FilteredResults(
                name=user_input,
                filter_query=FilterQueryStored.model_validate(
                    st.session_state.query.model_dump(
                        exclude_unset=True, exclude_defaults=True
                    )
                ),
                symbols=symbols,
            )

            bearish_db_.write_filtered_results(filtered_results)
            news(
                database_path=st.session_state.database_path,
                job_type="Fetching news",
                symbols=symbols,
                headless=headless,
            )
            st.session_state.filter_query = None
            st.session_state.query = None
            st.rerun()


def main() -> None:
    hide_elements = """
            <style>
                div[data-testid="stSliderTickBarMin"],
                div[data-testid="stSliderTickBarMax"] {
                    display: none;
                }
            </style>
    """

    st.markdown(hide_elements, unsafe_allow_html=True)
    assign_db_state()

    if st.session_state.database_path is None:
        dialog_pick_database()
    bearish_db_ = bearish_db(st.session_state.database_path)
    charts_tab, jobs_tab, backtests = st.tabs(["Charts", "Jobs", "Backtests"])
    if "data" not in st.session_state:
        st.session_state.data = load_analysis_data(bearish_db_)

    with charts_tab:
        with st.container():
            columns = st.columns(12)
            with columns[0]:
                if st.button(" 🔍 ", use_container_width=True):
                    st.session_state.filter_query = {}
                    filter()
            with columns[1]:
                if (
                    "query" in st.session_state
                    and st.session_state.query is not None
                    and st.session_state.query.valid()
                ):
                    favorite = st.button(" ⭐ ", use_container_width=True)
                    if favorite:
                        save_filtered_results(bearish_db_)
            with columns[-1]:
                if st.button(" 📥 ", use_container_width=True):
                    load()

        with st.container():
            st.dataframe(
                st.session_state.data,
                on_select=on_table_select,
                selection_mode="single-row",
                key="selected_data",
                use_container_width=True,
                height=600,
            )
            if (
                "ticker_figure" in st.session_state
                and st.session_state.ticker_figure is not None
            ):
                dialog_plot_figure()

    with jobs_tab:
        columns = st.columns(12)
        with columns[0]:
            if st.button(" ⏳ ", use_container_width=True):
                jobs()

        job_trackers = bearish_db_.read_job_trackers()
        st.dataframe(
            job_trackers,
            use_container_width=True,
            hide_index=True,
        )
    with backtests:
        results = bearish_db_.read_many_backtest_results()
        backtest_results = BacktestResults(results=results)
        with st.container():
            figure = backtest_results.figure()
            st.plotly_chart(figure)


if __name__ == "__main__":
    main()
