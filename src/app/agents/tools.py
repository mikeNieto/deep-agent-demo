from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format in Colombia timezone."""
    return datetime.now(ZoneInfo("America/Bogota")).isoformat()


DEFAULT_TOOLS = [get_current_datetime]
