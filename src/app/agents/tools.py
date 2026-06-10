from __future__ import annotations

import logging
from functools import wraps
from datetime import datetime
from time import perf_counter
from zoneinfo import ZoneInfo

from langchain_core.tools import tool


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
            "Deepagent tool call completed tool=%s elapsed_ms=%s",
            func.__name__,
            elapsed_ms,
        )
        return result

    return wrapper


@tool
@log_tool_call
def get_current_datetime() -> str:
    """Return the current date and time in ISO 8601 format in Colombia timezone."""
    return datetime.now(ZoneInfo("America/Bogota")).isoformat()


DEFAULT_TOOLS = [get_current_datetime]
