import json
import logging
import os
from datetime import date
from typing import Optional, List, TYPE_CHECKING

from pydantic import BaseModel, Field
from openai import OpenAI

if TYPE_CHECKING:
    from bullish.database.crud import BullishDb

logger = logging.getLogger(__name__)


def prompt(ticker: str) -> str:
    return f"""
You are a financial analysis assistant.

Using the latest reliable public data from the web — including analyst price targets from multiple reputable 
sources — analyze the stock ticker {ticker}.

Return ONLY valid JSON matching EXACTLY the schema below — no explanations, no preamble, no markdown, no code 
fences, no extra text:

{{
    "high_price_target": float,  // Analyst consensus high price target in USD (based on multiple sources)
    "low_price_target": float,   // Analyst consensus low price target in USD (based on multiple sources)
    "recent_news": str,          // Detailed, multi-sentence summary of recent news affecting the company; 
    include credible source names inline
    "recommendation": str,       // One of: "Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"
    "explanation": str           // Concise explanation for the recommendation above, covering key pros/cons 
    for investors
    "moat": bool           // Give as a boolean true or false if the company has a strong economic moat
}}

Formatting rules:
- Output must be a single valid JSON object with no surrounding text or formatting.
- Use plain numbers for high_price_target and low_price_target (no currency symbols, no commas).
- All text fields must be professional, investor-oriented, and reference credible named sources in `recent_news`.
- If exact data is unavailable, estimate based on web search results and note uncertainty in the relevant field.
"""


class OpenAINews(BaseModel):
    symbol: str
    news_date: date = Field(default_factory=date.today)
    high_price_target: Optional[float] = None
    low_price_target: Optional[float] = None
    recent_news: Optional[str] = None
    recommendation: Optional[str] = None
    explanation: Optional[str] = None
    moat: Optional[bool] = None

    def valid(self) -> bool:
        return bool(
            self.model_dump(
                exclude_none=True,
                exclude_unset=True,
                exclude_defaults=True,
                exclude={"symbol"},
            )
        )

    @classmethod
    def from_ticker(cls, ticker: str) -> "OpenAINews":
        if "OPENAI_API_KEY" not in os.environ:
            return cls(symbol=ticker)
        print(f"Fetching OpenAI news for {ticker}...")
        client = OpenAI()
        resp = client.responses.create(
            model="gpt-4o", input=prompt(ticker), tools=[{"type": "web_search"}]  # type: ignore
        )
        try:
            return cls.model_validate(json.loads(resp.output_text) | {"symbol": ticker})
        except Exception as e:
            logger.error(f"Failed to parse OpenAI response for {ticker}: {e}")
            return cls(symbol=ticker)

    @classmethod
    def from_tickers(cls, tickers: List[str]) -> List["OpenAINews"]:
        return [cls.from_ticker(t) for t in tickers]


def get_open_ai_news(bullish_db: "BullishDb", tickers: List[str]) -> bool:
    news = OpenAINews.from_tickers(tickers)
    valid_news = [n for n in news if n.valid()]
    if valid_news:
        bullish_db.write_many_openai_news(valid_news)
        return True
    return False
