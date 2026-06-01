from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import Settings


@asynccontextmanager
async def checkpoint_saver(settings: Settings) -> AsyncIterator[AsyncSqliteSaver]:
    conn_string = str(settings.checkpoint_db_path)
    async with AsyncSqliteSaver.from_conn_string(conn_string) as saver:
        await saver.setup()
        yield saver
