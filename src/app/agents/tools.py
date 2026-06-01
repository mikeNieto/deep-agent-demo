from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format in UTC."""
    return datetime.now(timezone.utc).isoformat()


DEFAULT_TOOLS = [get_current_datetime]
