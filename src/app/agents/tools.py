from __future__ import annotations

import logging
from functools import wraps
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from langchain_core.tools import tool
from tavily import TavilyClient

from app.config import get_settings


logger = logging.getLogger(__name__)


def log_tool_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        started_at = perf_counter()
        logger.info(
            "Deepagent tool call started tool=%s args=%s kwargs=%s",
            func.__name__,
            args,
            kwargs,
        )
        try:
            result = func(*args, **kwargs)
        except Exception:
            elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
            logger.exception(
                "Deepagent tool call failed tool=%s elapsed_ms=%s",
                func.__name__,
                elapsed_ms,
            )
            raise

        elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
        logger.info(
            "Deepagent tool call completed tool=%s elapsed_ms=%s result=%s",
            func.__name__,
            elapsed_ms,
            result,
        )
        return result

    return wrapper


@tool
@log_tool_call
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format in Colombia timezone.
    Use this tool always that you need to know the current date and time for your responses."""
    return datetime.now(ZoneInfo("America/Bogota")).isoformat()


@tool
@log_tool_call
def get_current_bitcoin_price() -> str:
    """Return the current Bitcoin price in USD."""
    import httpx
    resp = httpx.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=10)
    resp.raise_for_status()
    price = float(resp.json()["price"])
    return f"${price:,.2f}"


@tool
@log_tool_call
def web_search(query: str) -> str:
    """Search the internet for current or external information."""
    settings = get_settings()
    client = TavilyClient(api_key=settings.tavily_api_key)
    response = client.search(query=query)
    if not response.get("results"):
        return "No se encontraron resultados."
    lines = []
    for r in response["results"][:5]:
        lines.append(f"- {r['title']}\n  {r['url']}\n  {r['content']}")
    return "\n\n".join(lines)


DEFAULT_TOOLS = [get_current_datetime, get_current_bitcoin_price, web_search]
