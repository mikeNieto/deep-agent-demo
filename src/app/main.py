from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.agent import create_agent_graph
from app.audio import synthesize_speech, transcribe_audio
from app.config import get_settings

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


def _is_transcription_error(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in (
            "unable to",
            "cannot fulfill",
            "cannot process",
            "unable to process",
            "only 00:00:00",
        )
    )


async def _handle_websocket(ws: WebSocket) -> None:
    await ws.accept()
    settings = get_settings()
    agent_graph = create_agent_graph(settings)

    audio_buffer = bytearray()
    thread_id = "ws-default"

    try:
        while True:
            raw = await ws.receive()

            if "bytes" in raw:
                audio_buffer = bytearray(raw["bytes"])

            elif "text" in raw:
                data = json.loads(raw["text"])
                msg_type = data.get("type", "")

                if msg_type == "audio_end" and audio_buffer:
                    await ws.send_json({"type": "status", "state": "transcribing"})
                    t0 = perf_counter()
                    try:
                        transcription = await transcribe_audio(
                            settings,
                            bytes(audio_buffer),
                            mime_type="audio/wav",
                        )
                    except Exception as e:
                        logger.exception("STT failed")
                        await ws.send_json({"type": "error", "message": f"STT error: {e}"})
                        audio_buffer.clear()
                        continue
                    audio_buffer.clear()
                    logger.info(
                        "STT completed elapsed_ms=%s text=%s",
                        round((perf_counter() - t0) * 1000, 2),
                        transcription,
                    )

                    if transcription:
                        if _is_transcription_error(transcription):
                            await ws.send_json({"type": "error", "message": "Could not transcribe audio. Please try again."})
                        else:
                            await _process_text(ws, agent_graph, transcription, thread_id)
                    else:
                        await ws.send_json({"type": "error", "message": "Empty transcription"})

                elif msg_type == "text":
                    user_text = data.get("content", "").strip()
                    if user_text:
                        await _process_text(ws, agent_graph, user_text, thread_id)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")


async def _process_text(
    ws: WebSocket,
    agent_graph,
    text: str,
    thread_id: str,
) -> None:
    settings = get_settings()
    await ws.send_json({"type": "status", "state": "thinking"})

    t0 = perf_counter()
    result = await agent_graph.ainvoke(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    agent_elapsed = round((perf_counter() - t0) * 1000, 2)

    messages = result.get("messages", [])
    final_message = messages[-1] if messages else None
    content = getattr(final_message, "content", "") if final_message else ""
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )
    agent_text = str(content).strip()

    logger.info(
        "Agent completed elapsed_ms=%s output_chars=%s",
        agent_elapsed,
        len(agent_text),
    )

    await ws.send_json({"type": "text", "content": agent_text})

    if agent_text:
        await ws.send_json({"type": "status", "state": "speaking"})
        t0 = perf_counter()
        try:
            audio_bytes, mime_type = await synthesize_speech(settings, agent_text)
            tts_elapsed = round((perf_counter() - t0) * 1000, 2)
            if audio_bytes:
                await ws.send_json({"type": "audio_info", "mime_type": mime_type})
                await ws.send_bytes(audio_bytes)
                logger.info(
                    "TTS completed elapsed_ms=%s audio_bytes=%s mime=%s",
                    tts_elapsed,
                    len(audio_bytes),
                    mime_type,
                )
            else:
                logger.warning("TTS returned empty audio")
        except Exception as e:
            logger.exception("TTS failed")
            await ws.send_json({"type": "error", "message": f"TTS error: {e}"})

    await ws.send_json({"type": "done"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = get_settings()
    if settings.google_api_key:
        try:
            create_agent_graph(settings)
        except Exception as e:
            logger.warning("Could not create agent graph: %s", e)
    yield


app = FastAPI(title="SYLYS Agent", lifespan=lifespan)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text())
    return HTMLResponse("<h1>index.html not found</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await _handle_websocket(ws)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def run() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        factory=False,
    )
