from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NewsEventOutput(BaseModel):
    """
    Strict schema required by the Risk Governor. 
    The AI must absolutely not decide trade directions.
    """
    event_detected: bool = Field(description="True if a material market event was detected in the text.")
    event_type: Literal["macro", "regulation", "exchange", "security", "etf", "none"] = Field(
        description="The category of the event."
    )
    sentiment: float = Field(
        ge=-1.0, le=1.0, 
        description="Sentiment score between -1.0 (extremely negative) and 1.0 (extremely positive)."
    )
    event_importance: float = Field(
        ge=0.0, le=1.0, 
        description="How important the event is. 1.0 is a black swan or market-defining event."
    )
    risk_modifier: float = Field(
        ge=0.0, le=2.0, 
        description="Multiplier for risk. <1.0 reduces risk (e.g. 0.5 cuts position size in half). >1.0 increases risk."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence in the event extraction and sentiment analysis."
    )
    summary: str = Field(description="Short factual summary of the event. Do NOT include trading advice.")


class NewsExtractor:
    """
    Extracts events from raw headlines/news text and generates a strict Risk Modifier JSON using Gemini.
    """

    def __init__(self, api_key: str = "AQ.Ab8RN6IR5mS3eUS848rgh-Pg9qLSidV46YTHi1rCzz0WZ44oFw") -> None:
        self.client = genai.Client(api_key=api_key)

    def extract_event(self, symbol: str, raw_text: str, sources: list[str]) -> dict[str, object]:
        """
        Parses the raw_text using Gemini Structured Outputs to enforce the schema.
        """
        logger.info(f"Extracting news event for {symbol} using gemini-3.1-flash-lite")

        system_prompt = (
            "You are a strict quantitative risk extraction system. "
            "Your sole job is to evaluate news text and output a risk modifier. "
            "You MUST NEVER output BUY or SELL instructions. "
            "Extract the objective facts, determine the event category, and assess the risk impact. "
            "A high risk_modifier (>1.0) means the market is exceptionally safe/bullish. "
            "A low risk_modifier (<1.0) means the market is dangerous, volatile, or bearish."
        )

        prompt = f"{system_prompt}\n\nSymbol: {symbol}\nNews:\n{raw_text}"

        response = self.client.models.generate_content(
            model='gemini-3.1-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NewsEventOutput,
                temperature=0.0,
            )
        )

        # Gemini returns the strictly formatted JSON string
        parsed_dict = json.loads(response.text)

        # Construct final output ensuring the non-negotiable schema is met
        signal = {
            "news_id": f"news_{uuid.uuid4().hex[:8]}",
            "created_at": datetime.now(UTC).isoformat(),
            "symbol": symbol,
            "event_detected": parsed_dict.get("event_detected", False),
            "event_type": parsed_dict.get("event_type", "none"),
            "sentiment": float(parsed_dict.get("sentiment", 0.0)),
            "event_importance": float(parsed_dict.get("event_importance", 0.0)),
            "risk_modifier": float(parsed_dict.get("risk_modifier", 1.0)),
            "confidence": float(parsed_dict.get("confidence", 0.0)),
            "summary": parsed_dict.get("summary", ""),
            "sources": sources,
        }

        return signal

    def write_news_signal(self, signal: dict[str, object], output_dir: Path) -> Path:
        """Writes the news risk payload cleanly."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"latest_news_{signal['symbol']}.json"
        out_path.write_text(json.dumps(signal, indent=2), encoding="utf-8")
        return out_path
