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
from bullish.jobs.tasks import update, news, analysis, initialize
from pydantic import BaseModel

from bullish.utils.checks import (
    compatible_bearish_database,
    compatible_bullish_database,
    empty_analysis_table,
)
from mysec.services import sec  # type: ignore

CACHE_SHELVE = "user_cache"
DB_KEY = "db_path"

st.set_page_config(page_title="Bullish", page_icon="💰", layout="wide")
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
        not st.session_state.data.iloc[row]["symbol"].to_numpy().size > 0
    ):
        return

    symbol = st.session_state.data.iloc[row]["symbol"].to_numpy()[0]
    country = st.session_state.data.iloc[row]["country"].to_numpy()[0]
    industry = st.session_state.data.iloc[row]["industry"].to_numpy()[0]
    query = AssetQuery(symbols=Symbols(equities=[Ticker(symbol=symbol)]))
    prices = db.read_series(query, months=24)
    data = Prices(prices=prices).to_dataframe()
    dates = db.read_dates(symbol)
    subject = db.read_subject(symbol)
    industry_data = get_industry_comparison_data(db, data, "Mean", industry, country)

    fig = plot(data, symbol, dates=dates, industry_data=industry_data)

    st.session_state.ticker_figure = fig
    st.session_state.ticker_news = subject


@st.dialog("🔑  Select database file to continue")
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
            analysis(db_path, "Update analysis")
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
            mapping = groups_mapping().get(field)
            if not mapping:
                continue
            data[field] = st.multiselect(
                name,
                mapping,
                default=default,
                key=hash((model.__name__, field)),
            )
        elif info.annotation == Optional[str]:  # type: ignore
            mapping = groups_mapping().get(field)
            if not mapping:
                continue
            options = ["", *mapping]
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
                    f"with the parameters {(info.annotation, name, ge, le)}: {e}"
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
    with st.container(), st.expander("Predefined filters"):
        predefined_filter_names = PredefinedFilters().get_predefined_filter_names()
        option = st.selectbox(
            "Select a predefined filter",
            ["", *predefined_filter_names],
        )
        if option:
            data_ = PredefinedFilters().get_predefined_filter(option)
            st.session_state.filter_query.update(data_)
    with st.container():

        with st.expander("Technical Analysis"):
            for filter in TechnicalAnalysisFilters:
                with st.expander(filter._description):  # type: ignore
                    build_filter(filter, st.session_state.filter_query)

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
            st.session_state.ticker_news = None
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
    if st.session_state.ticker_news:
        st.markdown(
            f"""
            <div class="news-hover" >
              📰 <span class="label">News</span>
              <div class="tooltip">
                <h2>Date: {st.session_state.ticker_news.to_date()}</h2>
                <h2>Price targets</h2>
                <p>High price target: {st.session_state.ticker_news.high_price_target}</p>
                <p>Low price target: {st.session_state.ticker_news.low_price_target}</p>
                <p>OpenAI High price target: {st.session_state.ticker_news.oai_high_price_target}</p>
                <p>OpenAI Low price target: {st.session_state.ticker_news.oai_low_price_target}</p>
                <h2>Recommendation: {st.session_state.ticker_news.recommendation}</h2>
                <h2>OpenAI Recommendation: {st.session_state.ticker_news.oai_recommendation}</h2>
                <h2>Consensus: {st.session_state.ticker_news.consensus}</h2>
                <h2>Explanation & reasons</h2>
                <p>{st.session_state.ticker_news.explanation}</p>
                <p>{st.session_state.ticker_news.reason}</p>
                <p>{st.session_state.ticker_news.oai_explanation}</p>
                <h2>Recent news</h2>
                <p>{st.session_state.ticker_news.oai_recent_news}</p>
                <h2>News summaries</h2>
                {st.session_state.ticker_news.to_news()}
              </div>
            </div>    
            <style>
              /* Hover target (fixed top-left) */
              .news-hover {{
                position: absolute;
                left: 1rem;
                display: inline-flex;
                align-items: center;
                gap: .4rem;
                font-size: 1.7rem;       /* big label */
                font-weight: 600;
                color: #333;
                cursor: pointer;
                user-select: none;
                z-index: 1100;
              }}    
              /* Tooltip bubble */
              .news-hover .tooltip {{
                position: absolute;
                top: 110%;               /* below the label */
                left: 0;
                width: 840px;
                max-height: 620px;
                overflow-y: auto;
                background: #222;
                color: #fff;
                padding: 1.2rem;
                border-radius: 10px;
                font-size: .95rem;
                line-height: 1.45;
                box-shadow: 0 8px 20px rgba(0,0,0,.4);
                opacity: 0;
                visibility: hidden;
                transition: opacity .25s ease;
              }}
              .news-hover .tooltip hr {{
                border: none;
                border-top: 1px solid #444;
                margin: 1rem 0;
              }}    
              /* Show tooltip on hover or keyboard focus */
              .news-hover:hover .tooltip,
              .news-hover:focus-within .tooltip {{
                opacity: 1;
                visibility: visible;
              }}
              /* Little arrow under the bubble */
              .news-hover .tooltip::after {{
                content: "";
                position: absolute;
                top: -10px;
                left: 20px;
                border: 10px solid transparent;
                border-top-color: #222;
              }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.plotly_chart(st.session_state.ticker_figure, use_container_width=True)
    st.session_state.ticker_figure = None
    st.session_state.ticker_news = None


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
    if "initialized" not in st.session_state:
        initialize(
            database_path=st.session_state.database_path,
            job_type="Initialize",
        )
        st.session_state.initialized = True
    bearish_db_ = bearish_db(st.session_state.database_path)

    charts_tab, jobs_tab, sec_tab = st.tabs(["Charts", "Jobs", "Sec"])
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
    with sec_tab:
        st.plotly_chart(sec(bearish_db_), use_container_width=True)


if __name__ == "__main__":
    main()
