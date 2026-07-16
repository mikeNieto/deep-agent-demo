from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.agent import create_agent_graph
from app.audio import (
    get_interim_audio,
    parse_wav_header,
    synthesize_speech,
    transcribe_audio,
)
from app.config import get_settings
from app.usage import get_tracker

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"
FIXED_USER_ID = "sylys-user"


def _new_thread_id() -> str:
    return f"thread-{uuid4().hex[:12]}"


def _default_response(language: str) -> str:
    defaults = {
        "es": "Hecho.",
        "en": "Done.",
        "fr": "C'est fait.",
        "pt": "Feito.",
        "zh": "完成了。",
    }
    return defaults.get(language, defaults["en"])


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


CHUNK_SIZE = 32768


async def _send_audio_chunked(ws: WebSocket, wav_bytes: bytes) -> None:
    sample_rate, channels, bits, pcm_data = parse_wav_header(wav_bytes)

    await ws.send_json({
        "type": "audio_start",
        "sample_rate": sample_rate,
        "channels": channels,
        "bits": bits,
    })

    offset = 0
    while offset < len(pcm_data):
        chunk = pcm_data[offset : offset + CHUNK_SIZE]
        await ws.send_bytes(bytes(chunk))
        offset += CHUNK_SIZE

    await ws.send_json({"type": "audio_end"})


async def _send_interim(ws: WebSocket, settings, language: str) -> None:
    try:
        result = await get_interim_audio(settings, language)
        if result:
            audio, _mime = result
            await _send_audio_chunked(ws, audio)
            get_tracker().add_tts(chars=15)
    except Exception as e:
        logger.warning("Interim audio failed: %s", e)


async def _handle_websocket(ws: WebSocket) -> None:
    await ws.accept()
    settings = get_settings()
    agent_graph = ws.app.state.agent_graph

    audio_buffer = bytearray()
    thread_id = _new_thread_id()
    logger.info("WebSocket connected thread_id=%s", thread_id)

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
                        transcription, language = await transcribe_audio(
                            settings,
                            bytes(audio_buffer),
                            mime_type="audio/wav",
                        )
                    except Exception as e:
                        logger.exception("STT failed")
                        await ws.send_json(
                            {"type": "error", "message": f"STT error: {e}"}
                        )
                        audio_buffer.clear()
                        continue
                    audio_dur = len(audio_buffer) / 32000.0
                    audio_buffer.clear()
                    get_tracker().add_stt(audio_dur)
                    logger.info(
                        "STT completed elapsed_ms=%s text=%s lang=%s",
                        round((perf_counter() - t0) * 1000, 2),
                        transcription,
                        language,
                    )

                    if transcription:
                        if _is_transcription_error(transcription):
                            await ws.send_json(
                                {
                                    "type": "error",
                                    "message": "Could not transcribe audio. Please try again.",
                                }
                            )
                        else:
                            await _process_text(
                                ws, agent_graph, transcription, thread_id, language
                            )
                    else:
                        await ws.send_json(
                            {"type": "error", "message": "Empty transcription"}
                        )

                elif msg_type == "text":
                    user_text = data.get("content", "").strip()
                    if user_text:
                        await _process_text(ws, agent_graph, user_text, thread_id, "en")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")


async def _process_text(
    ws: WebSocket,
    agent_graph,
    text: str,
    thread_id: str,
    language: str = "en",
) -> None:
    settings = get_settings()
    await ws.send_json({"type": "status", "state": "thinking"})

    t0 = perf_counter()
    full_text = ""
    tool_interim_sent = False
    agent_in_tokens = 0
    agent_out_tokens = 0
    last_tool_output = None

    def _extract_tool_text(output) -> str | None:
        if output is None:
            return None
        if hasattr(output, "content"):
            content = output.content
            if isinstance(content, list):
                return " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
            return str(content).strip()
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, (dict, list)):
            return json.dumps(output, ensure_ascii=False)
        return str(output).strip()

    async for event in agent_graph.astream_events(
        {"messages": [{"role": "user", "content": text}]},
        config={"configurable": {"thread_id": thread_id}},
        version="v2",
    ):
        kind = event.get("event")

        if kind == "on_tool_start":
            name = event.get("name", "unknown")
            data = event.get("data", {})
            inp = data.get("input", {})
            logger.info("Tool call started tool=%s input=%s", name, inp)
            if not tool_interim_sent:
                tool_interim_sent = True
                asyncio.create_task(_send_interim(ws, settings, language))

        if kind == "on_tool_end":
            name = event.get("name", "unknown")
            data = event.get("data", {})
            out = data.get("output", "")
            preview = str(out)[:300]
            logger.info("Tool call ended tool=%s output=%s", name, preview)
            extracted = _extract_tool_text(out)
            if extracted:
                last_tool_output = extracted

        if kind == "on_chat_model_end":
            output = event.get("data", {}).get("output", {})
            usage = getattr(output, "usage_metadata", None) or {}
            if isinstance(usage, dict):
                agent_in_tokens += usage.get("input_tokens", 0)
                agent_out_tokens += usage.get("output_tokens", 0)
            elif hasattr(usage, "get"):
                agent_in_tokens += usage.get("input_tokens", 0) or 0  # type: ignore[operator]
                agent_out_tokens += usage.get("output_tokens", 0) or 0  # type: ignore[operator]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            token = chunk.content if hasattr(chunk, "content") else ""
            if not token:
                continue
            if isinstance(token, list):
                token = "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in token
                )
            token = str(token)
            full_text += token
            await ws.send_json({"type": "token", "content": token})

    agent_elapsed = round((perf_counter() - t0) * 1000, 2)
    agent_text = full_text.strip()

    if not agent_text:
        if last_tool_output:
            agent_text = last_tool_output
        else:
            agent_text = _default_response(language)

    get_tracker().add_agent_tokens(
        input_tokens=agent_in_tokens, output_tokens=agent_out_tokens
    )

    logger.info(
        "Agent completed elapsed_ms=%s output_chars=%s",
        agent_elapsed,
        len(agent_text),
    )

    if agent_text:
        await ws.send_json({"type": "status", "state": "speaking"})
        t0 = perf_counter()
        try:
            audio_bytes, _mime_type = await synthesize_speech(settings, agent_text, language)
            get_tracker().add_tts(chars=len(agent_text))
            tts_elapsed = round((perf_counter() - t0) * 1000, 2)
            if audio_bytes:
                await _send_audio_chunked(ws, audio_bytes)
                logger.info(
                    "TTS completed elapsed_ms=%s audio_bytes=%s",
                    tts_elapsed,
                    len(audio_bytes),
                )
            else:
                logger.warning("TTS returned empty audio")
        except Exception as e:
            logger.exception("TTS failed")
            await ws.send_json({"type": "error", "message": f"TTS error: {e}"})

    await ws.send_json({"type": "text", "content": agent_text})
    await ws.send_json({"type": "done"})
    get_tracker().log_summary()


@asynccontextmanager
async def lifespan(app: FastAPI):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    settings = get_settings()
    db_path = str(settings.audio_temp_dir.parent / "sqlite" / "checkpoints.db")
    (settings.audio_temp_dir.parent / "sqlite").mkdir(parents=True, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        await saver.setup()
        agent_graph = None
        if settings.google_api_key:
            try:
                agent_graph = create_agent_graph(settings, checkpointer=saver)
                logger.info("Agent graph created with persistent checkpointer")
            except Exception as e:
                logger.warning("Could not create agent graph: %s", e)

        app.state.agent_graph = agent_graph
        app.state.settings = settings
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
