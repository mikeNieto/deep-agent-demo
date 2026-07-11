from __future__ import annotations

import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import Settings, get_settings


SYSTEM_PROMPT = """You are SYLYS, a helpful conversational assistant.

Respond in the SAME LANGUAGE that the user uses. If the user writes in Spanish, respond in Spanish. If in English, respond in English. Auto-detect the language.
Be concise but complete.
If you lack context, ask one short clarifying question.
Use the get_current_datetime tool only when the current date or time matters.
Use the web_search tool to search the internet for current or external information when needed.
Never use markdown. Always answer in plain text.
Do not use emojis.
Do not include greetings or farewells."""


@tool
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format in Colombia timezone."""
    return datetime.now(ZoneInfo("America/Bogota")).isoformat()


@tool
async def get_current_bitcoin_price() -> str:
    """Return the current Bitcoin price in USD from Binance."""

    def _fetch() -> str:
        import httpx
        resp = httpx.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            timeout=10,
        )
        resp.raise_for_status()
        price = float(resp.json()["price"])
        return f"${price:,.2f}"

    return await asyncio.to_thread(_fetch)


@tool
async def web_search(query: str) -> str:
    """Search the internet for current or external information using Google Search."""
    from google import genai
    from google.genai import types

    from app.usage import get_tracker

    client = genai.Client(api_key=get_settings().google_api_key)
    tools = [types.Tool(google_search=types.GoogleSearch())]
    response = client.models.generate_content(
        model=get_settings().deepagent_model,
        contents=f"Search the web for: {query}\nReturn a detailed answer based on the search results.",
        config=types.GenerateContentConfig(tools=tools),
    )
    get_tracker().add_web_search()
    return response.text or ""


DEFAULT_TOOLS = [get_current_datetime, get_current_bitcoin_price, web_search]


def _memory_files(root_dir: str) -> list[str]:
    from pathlib import Path
    return [str(Path(root_dir) / "memory" / "AGENTS.md")]


def _skill_paths(root_dir: str) -> list[str]:
    from pathlib import Path
    return [str(Path(root_dir) / "skills")]


def create_chat_model(settings: Settings) -> ChatGoogleGenerativeAI:
    if not settings.google_api_key:
        msg = "GOOGLE_API_KEY must be set"
        raise ValueError(msg)
    kwargs: dict = {
        "model": settings.deepagent_model,
        "google_api_key": settings.google_api_key,
        "temperature": 0.2,
    }
    model_name = settings.deepagent_model.lower()
    if "gemini-3" in model_name and settings.thinking_level:
        kwargs["thinking_level"] = settings.thinking_level.lower()
    elif "gemini-2" in model_name and settings.thinking_budget:
        kwargs["thinking_budget"] = settings.thinking_budget
    return ChatGoogleGenerativeAI(**kwargs)


def create_agent_graph(settings: Settings):
    from app.config import ROOT_DIR

    model = create_chat_model(settings)
    backend = CompositeBackend(default=StateBackend(), routes={})
    return create_deep_agent(
        model=model,
        tools=DEFAULT_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        memory=_memory_files(str(ROOT_DIR)),
        skills=_skill_paths(str(ROOT_DIR)),
        backend=backend,
        debug=False,
        name="SYLYS",
    )
