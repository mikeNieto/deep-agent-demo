from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.live_service import GeminiLiveSession

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/live", tags=["live"])


@router.websocket("/ws")
async def gemini_live_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    logger.info("WebSocket client connected")

    from app.config import get_settings

    settings = get_settings()
    if not settings.google_api_key:
        await websocket.send_json({"type": "error", "message": "GOOGLE_API_KEY not configured"})
        await websocket.close()
        return

    session = GeminiLiveSession(settings)
    run_task: asyncio.Task[Any] | None = None

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "setup":
                session.configure(websocket, msg)
                run_task = asyncio.create_task(session.run())
            elif msg.get("type") == "audio_config":
                session.set_audio_rate(msg.get("sample_rate", 16000))
            elif msg.get("type") in ("audio_in", "text_in", "interrupt"):
                await session.send(msg)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        if run_task:
            run_task.cancel()
