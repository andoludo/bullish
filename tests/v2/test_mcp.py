import json
from collections.abc import AsyncIterator

import pytest
from mcp import Client

from bullish.config import BullishSettings
from bullish.database.crud import BullishDb
from bullish.kpi import engine
from bullish.mcp.server import build_server
from bullish.mcp.service import BullishService
from tests.v2.synthetic import AS_OF, UNIVERSE

pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="module")
def served_db(bullish_db_v2: BullishDb, settings: BullishSettings) -> BullishDb:
    if bullish_db_v2.latest_kpi_date("equity") != AS_OF:
        engine.run(bullish_db_v2, settings, as_of=AS_OF)
    return bullish_db_v2


@pytest.fixture
async def client(
    served_db: BullishDb, settings: BullishSettings
) -> AsyncIterator[Client]:
    sql_settings = settings.model_copy(update={"allow_sql": True})
    server = build_server(sql_settings, BullishService(sql_settings, served_db))
    async with Client(server, raise_exceptions=True) as c:
        yield c


async def test_tools_resources_and_prompts_are_listed(client: Client) -> None:
    tools = {t.name for t in (await client.list_tools()).tools}
    expected = {
        "list_kpis",
        "describe_kpi",
        "list_composites",
        "list_runs",
        "get_market_regime",
        "rank_sectors",
        "rank_industries",
        "rank_countries",
        "rank_etfs",
        "compare_etfs",
        "screen_equities",
        "explain_score",
        "get_equity_scorecard",
        "assess_stock",
        "get_kpi_series",
        "get_smart_money",
        "get_filings",
        "get_earnings_calendar",
        "get_macro_snapshot",
        "get_commodity_snapshot",
        "get_data_coverage",
        "analyze_portfolio",
        "query",
    }
    assert expected <= tools
    for tool in (await client.list_tools()).tools:
        assert tool.annotations is not None and tool.annotations.read_only_hint is True
    templates = {
        t.uri_template
        for t in (await client.list_resource_templates()).resource_templates
    }
    assert "bullish://methodology/{composite}" in templates
    prompts = {p.name for p in (await client.list_prompts()).prompts}
    assert {
        "weekly_market_review",
        "sector_briefing",
        "etf_selection",
        "stock_due_diligence",
    } <= prompts


async def test_market_regime_and_rankings(client: Client) -> None:
    regime = (
        await client.call_tool("get_market_regime", {"region": "ALL"})
    ).structured_content
    assert regime["label"] in {"risk_on", "neutral", "risk_off"}
    assert regime["as_of"] == AS_OF.isoformat()
    sectors = (await client.call_tool("rank_sectors", {})).structured_content
    assert sectors["composite"] == "sector_score"
    assert sectors["rows"][0]["rank"] == 1
    assert {"mom_12_1", "rrg_quadrant", "members"} <= set(sectors["rows"][0]["kpis"])
    assert sectors["freshness"]["kpi_run"] == AS_OF.isoformat()
    countries = (await client.call_tool("rank_countries", {})).structured_content
    assert {r["entity"] for r in countries["rows"]} == {"USA", "DEU", "BEL"}
    etfs = (
        await client.call_tool("rank_etfs", {"exposure": "Health Care"})
    ).structured_content
    assert [r["entity"] for r in etfs["rows"]] == ["HLTHX", "WORLDX"] or {
        r["entity"] for r in etfs["rows"]
    } == {"HLTHX", "WORLDX"}


async def test_screen_scorecard_and_verdict(client: Client) -> None:
    screen = (
        await client.call_tool(
            "screen_equities", {"preset": "quality_on_sale", "limit": 5}
        )
    ).structured_content
    assert screen["rows"][0]["entity"] == "QSALE"
    us_only = (
        await client.call_tool(
            "screen_equities", {"preset": "quality_compounder", "country": "USA"}
        )
    ).structured_content
    assert {r["entity"] for r in us_only["rows"]} <= {
        "ACCEL",
        "STEADY",
        "QSALE",
        "KNIFE",
        "MEDCO",
        "PHARM",
        "OILCO",
        "DRILL",
    }
    card = (
        await client.call_tool("get_equity_scorecard", {"symbol": "QSALE"})
    ).structured_content
    assert card["reference"]["sector"] == "Information Technology"
    assert "momentum" in card["kpis"] and "drawdown_52w" in card["kpis"]["momentum"]
    assert card["preset_scores"]["quality_on_sale"]["rank"] == 1
    assert card["coverage"]["overall"] == "A"
    verdict = (
        await client.call_tool("assess_stock", {"symbol": "QSALE"})
    ).structured_content
    assert verdict["label"] == "quality_on_sale"
    assert verdict["data_coverage_grade"] == "A"
    knife = (
        await client.call_tool("assess_stock", {"symbol": "KNIFE"})
    ).structured_content
    assert knife["label"] == "falling_knife"
    explanation = (
        await client.call_tool(
            "explain_score", {"entity": "QSALE", "composite": "quality_on_sale"}
        )
    ).structured_content
    assert explanation["rank"] == 1
    assert any(
        k["kpi"] == "insider_cluster_buy" and k["percentile"] == 100.0
        for b in explanation["blocks"]
        for k in b["kpis"]
    )


async def test_raw_data_tools(client: Client) -> None:
    money = (
        await client.call_tool("get_smart_money", {"symbol": "QSALE"})
    ).structured_content
    assert len(money["insider_transactions_12m"]) == 3
    assert money["beneficial_owners_13dg"][0]["form_type"] == "SCHEDULE 13D"
    filings = (
        await client.call_tool("get_filings", {"symbol": "KNIFE", "forms": ["8-K"]})
    ).structured_content["result"]
    assert any("4.02" in f["item_codes"] for f in filings)
    calendar = (
        await client.call_tool("get_earnings_calendar", {"days": 30})
    ).structured_content["result"]
    assert len(calendar) >= 12 and {c["symbol"] for c in calendar} == set(UNIVERSE)
    macro = (
        await client.call_tool(
            "get_macro_snapshot", {"entity": "USA", "concepts": ["VIX", "CLI"]}
        )
    ).structured_content
    assert set(macro["series"]) == {"VIX", "CLI"}
    assert macro["series"]["CLI"]["change_3m"] > 0
    commodities = (
        await client.call_tool("get_commodity_snapshot", {"code": "BRENT"})
    ).structured_content["result"]
    assert (
        commodities[0]["commodity"] == "BRENT"
        and "Energy" in commodities[0]["benefits"]
    )
    coverage = (await client.call_tool("get_data_coverage", {})).structured_content
    assert coverage["grades"] == {"A": 12}
    series = (
        await client.call_tool("get_kpi_series", {"entity": "ACCEL", "kpi": "ret_3m"})
    ).structured_content
    assert series["points"] and series["points"][-1]["date"] == AS_OF.isoformat()


async def test_etf_comparison_and_portfolio(client: Client) -> None:
    comparison = (
        await client.call_tool(
            "compare_etfs", {"symbols": ["TECHX", "HLTHX", "WORLDX"]}
        )
    ).structured_content
    assert {v["symbol"] for v in comparison["vehicles"]} == {"TECHX", "HLTHX", "WORLDX"}
    assert comparison["overlap"]["TECHX"]["TECHX"] == pytest.approx(0.45)
    assert comparison["overlap"]["TECHX"]["HLTHX"] == 0.0
    portfolio = (
        await client.call_tool(
            "analyze_portfolio",
            {"holdings": {"ACCEL": 5000, "SAPDE": 3000, "TECHX": 2000, "ZZZ": 100}},
        )
    ).structured_content
    assert portfolio["unknown_symbols"] == ["ZZZ"]
    assert portfolio["sector_exposure"]["Information Technology"] == pytest.approx(
        1.0 - 100 / 10100, abs=0.01
    )
    assert set(portfolio["country_exposure"]) >= {"USA", "DEU"}
    assert portfolio["average_pairwise_correlation_1y"] is not None


async def test_catalogue_and_sql(client: Client) -> None:
    kpis = (
        await client.call_tool(
            "list_kpis", {"entity_type": "equity", "block": "momentum"}
        )
    ).structured_content["result"]
    assert any(k["name"] == "mom_12_1" and k["tier"] == "A" for k in kpis)
    described = (
        await client.call_tool("describe_kpi", {"name": "mom_12_1"})
    ).structured_content["result"]
    assert {d["entity_type"] for d in described} >= {"equity", "sector"}
    composites = (await client.call_tool("list_composites", {})).structured_content[
        "result"
    ]
    assert {c["name"] for c in composites} >= {"sector_score", "quality_on_sale"}
    runs = (await client.call_tool("list_runs", {})).structured_content["result"]
    assert runs[-1]["as_of"] == AS_OF.isoformat()
    rows = (
        await client.call_tool(
            "query", {"sql": "SELECT symbol FROM equity ORDER BY symbol"}
        )
    ).structured_content["result"]
    assert rows[0]["symbol"] == "ACCEL"
    content = await client.read_resource("bullish://methodology/quality_on_sale")
    payload = json.loads(content.contents[0].text)  # type: ignore[union-attr]
    assert payload["name"] == "quality_on_sale" and payload["blocks"]


async def test_errors_are_tool_errors(client: Client) -> None:
    result = await client.call_tool("assess_stock", {"symbol": "NOPE"})
    assert result.is_error
    bad = await client.call_tool("query", {"sql": "DELETE FROM equity"})
    assert bad.is_error
    bad_date = await client.call_tool("rank_sectors", {"as_of": "yesterday"})
    assert bad_date.is_error
