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

from bullish.database.crud import BullishDb
from bullish.figures.figures import plot
from bullish.analysis.filter import (
    FilterQuery,
    FilterUpdate,
    FilteredResults,
    FilterQueryStored,
    TechnicalAnalysisFilter,
    FundamentalAnalysisFilter,
    GROUP_MAPPING,
    GeneralFilter,
)
from bullish.jobs.models import JobTracker
from bullish.jobs.tasks import update, news
from pydantic import BaseModel

CACHE_SHELVE = "user_cache"
DB_KEY = "db_path"

st.set_page_config(layout="wide")


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
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol=symbol)]))
    prices = db.read_series(query, months=24)
    data = Prices(prices=prices).to_dataframe()

    fig = plot(data, symbol)

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
            st.error("Please choose a valid file.")
            st.stop()
        st.session_state.database_path = db_path
        store_db(db_path)
        st.rerun()
    if event is None:
        st.stop()


def build_filter(model: Type[BaseModel]) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for field, info in model.model_fields.items():
        name = info.alias or field
        if info.annotation == Optional[List[str]]:  # type: ignore
            data[field] = st.multiselect(name, GROUP_MAPPING[field], key=name)

        else:
            ge = next((item.ge for item in info.metadata if hasattr(item, "ge")), None)
            le = next((item.le for item in info.metadata if hasattr(item, "le")), None)
            data[field] = list(st.slider(name, le, ge, tuple(info.default)))  # type: ignore
    return data


@st.dialog("🔍  Filter", width="large")
def filter() -> None:

    st.markdown(
        """
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        width: 90vw;
        height: 110vh;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    with st.expander("Technical Analysis"):
        data_ = build_filter(TechnicalAnalysisFilter)
        st.session_state.filter_query.update(data_)
    with st.expander("Fundamental Analysis"):
        data_ = build_filter(FundamentalAnalysisFilter)
        st.session_state.filter_query.update(data_)
    with st.expander("General filter"):
        data_ = build_filter(GeneralFilter)
        st.session_state.filter_query.update(data_)
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
        height: 110vh;
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
            res = news(
                database_path=st.session_state.database_path,
                symbols=symbols,
                headless=headless,
            )
            bearish_db_.write_job_tracker(
                JobTracker(job_id=str(res.id), type="Fetching news")
            )
            st.session_state.filter_query = None
            st.session_state.query = None
            st.rerun()


def main() -> None:
    assign_db_state()

    if st.session_state.database_path is None:
        dialog_pick_database()
    bearish_db_ = bearish_db(st.session_state.database_path)
    charts_tab, jobs_tab = st.tabs(["Charts", "Jobs"])
    if "data" not in st.session_state:
        st.session_state.data = load_analysis_data(bearish_db_)
    with st.sidebar:
        with st.expander("Load"):
            existing_filtered_results = bearish_db_.read_list_filtered_results()
            option = st.selectbox("Saved results", ["", *existing_filtered_results])
            if option:
                filtered_results_ = bearish_db_.read_filtered_results(option)
                if filtered_results_:
                    st.session_state.data = bearish_db_.read_analysis_data(
                        symbols=filtered_results_.symbols
                    )

        with st.expander("Update"):
            update_query = sp.pydantic_form(key="update", model=FilterUpdate)
            if (
                update_query
                and st.session_state.data is not None
                and not st.session_state.data.empty
            ):
                symbols = st.session_state.data["symbol"].unique().tolist()
                res = update(
                    database_path=st.session_state.database_path,
                    symbols=symbols,
                    update_query=update_query,
                )  # enqueue & get result-handle
                bearish_db_.write_job_tracker(
                    JobTracker(job_id=str(res.id), type="Update data")
                )
                st.success("Data update job has been enqueued.")
                st.rerun()
    with charts_tab:
        with st.container():
            col1, col2, _ = st.columns([0.5, 0.5, 5])
            with col1:
                if st.button("🔍 Filter"):
                    st.session_state.filter_query = {}
                    filter()
            with col2:
                if (
                    "query" in st.session_state
                    and st.session_state.query is not None
                    and st.session_state.query.valid()
                ):
                    favorite = st.button(" ⭐ ")
                    if favorite:
                        save_filtered_results(bearish_db_)

        with st.container():
            st.header("✅ Data overview")

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
        job_trackers = bearish_db_.read_job_trackers()
        st.dataframe(
            job_trackers,
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
