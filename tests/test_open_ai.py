import os
from typing import TYPE_CHECKING

import pytest

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
