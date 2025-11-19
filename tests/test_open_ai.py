import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from bullish.analysis.analysis import run_analysis
from bullish.analysis.openai import OpenAINews, get_open_ai_news

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb


@pytest.mark.skip("Requires OpenAI API key")
def test_open_ai_news() -> None:
    os.environ["OPENAI_API_KEY"] = ""
    news = OpenAINews.from_ticker("VKTX")
    assert news.valid()


@pytest.mark.skip("Requires OpenAI API key")
def test_get_open_ai_news(bullish_db: "BullishDb"):
    os.environ["OPENAI_API_KEY"] = ""
    get_open_ai_news(bullish_db, ["VKTX", "UNH"])


def test_update():
    symbols = [
        "LIN.DE",
        "3CP.SG",
        "SE",
        "ICE",
        "USB-PH",
        "IC2.F",
        "RHM.DE",
        "2FE.DE",
        "APD",
        "FIV.F",
        "AP3.F",
        "AP3.DE",
        "LNG",
        "ONK.F",
        "OPC.DE",
        "HDMA.F",
        "CSGP",
        "VA7A.F",
        "RLG.F",
        "SSNC",
        "RBA",
        "GEN",
        "EYX.F",
        "AXJ.DE",
        "E3G1.F",
        "NC0E.F",
        "NEM.DE",
        "EW2.F",
        "PGV.F",
        "ADP.F",
        "CTP2.DE",
        "CYZB.F",
        "CIT.F",
        "PNP.F",
        "CCZ",
        "FAST",
        "VOL1.F",
        "FAS.F",
        "OXY",
        "0QN.F",
        "ONK.DE",
        "VG",
        "GRV.F",
        "ML.PA",
        "6076.F",
        "R9C.F",
        "17W.F",
        "KSPI",
        "38D.F",
        "OMV.F",
        "7NX.SG",
        "SYM.F",
        "ME9.F",
        "8TRA.DE",
        "NC0B.F",
        "KKS.F",
        "AXJ.F",
        "RN7.F",
        "HFF.SG",
        "WGSA.F",
    ]
    from bullish.database.crud import BullishDb

    bullish_db = BullishDb(database_path=Path("/media/aan/T7/bullish/updated.db"))
    for symbol in symbols:
        subject = bullish_db.read_subject(symbol)
        if subject:
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
                print(f"failed to extract news for {symbol}: {e}")
                continue


def test_database():
    from bullish.database.crud import BullishDb

    db_path = Path("/media/aan/T7/bearish_db/db_12_11_2025.db")
    bullish_db = BullishDb(database_path=db_path)
    run_analysis(bullish_db)
