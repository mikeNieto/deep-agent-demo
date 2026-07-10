from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from google import genai
from google.genai.types import LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, Tool, GoogleSearch

from app.config import Settings

logger = logging.getLogger(__name__)


class GeminiLiveSession:
    def __init__(self, settings: Settings):
        self._client = genai.Client(
            api_key=settings.google_api_key,
        )
        self._model = settings.gemini_live_model
        self._config: dict[str, Any] = {}
        self._send: Callable[[dict[str, Any]], Coroutine[Any, Any, None]] = lambda _: _noop()
        self._out_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._audio_rate: int = 16000
        self._speaking = False

    def configure(self, ws: Any, config: dict[str, Any]) -> None:
        self._send = ws.send_json
        system_instruction = config.get("system_instruction", "")
        self._config = {
            "response_modalities": ["AUDIO"],
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "tools": [Tool(google_search=GoogleSearch())],
            "speech_config": SpeechConfig(
                voice_config=VoiceConfig(
                    prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Zephyr")
                )
            ),
        }
        if system_instruction:
            self._config["system_instruction"] = system_instruction

    def set_audio_rate(self, rate: int) -> None:
        self._audio_rate = rate

    async def run(self) -> None:
        logger.info("Connecting to Gemini Live model=%s", self._model)
        live_cfg = LiveConnectConfig(**self._config)

        async with self._client.aio.live.connect(
            model=self._model, config=live_cfg
        ) as session:
            logger.info("Gemini Live session started")
            await self._send({"type": "status", "state": "idle"})

            try:
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self._receive_loop(session))
                    tg.create_task(self._send_loop(session))
            except* Exception:
                logger.exception("Session error")

    async def _receive_loop(self, session: Any) -> None:
        while True:
            async for response in session.receive():
                if response.data:
                    if not self._speaking:
                        self._speaking = True
                        await self._send({"type": "status", "state": "speaking"})
                    await self._send({
                        "type": "audio_out",
                        "data": base64.b64encode(response.data).decode("ascii"),
                    })
                if response.text:
                    await self._send({
                        "type": "text",
                        "content": response.text,
                    })
                if response.server_content:
                    sc = response.server_content
                    if sc.interim_input_transcription and sc.interim_input_transcription.text:
                        await self._send({
                            "type": "text",
                            "role": "user",
                            "interim": True,
                            "content": sc.interim_input_transcription.text,
                        })
                    if sc.input_transcription and sc.input_transcription.text:
                        await self._send({
                            "type": "text",
                            "role": "user",
                            "content": sc.input_transcription.text,
                        })
                    if sc.output_transcription and sc.output_transcription.text:
                        await self._send({
                            "type": "text",
                            "role": "model",
                            "content": sc.output_transcription.text,
                        })
                    if sc.turn_complete:
                        self._speaking = False
                        await self._send({"type": "status", "state": "idle"})
            # Iterator exhausted after turn_complete; loop creates a fresh one.

    async def _send_loop(self, session: Any) -> None:
        while True:
            msg = await self._out_queue.get()

            if msg.get("type") == "audio_in":
                raw_data = base64.b64decode(msg["data"])
                await session.send_realtime_input(
                    audio={
                        "data": raw_data,
                        "mime_type": f"audio/pcm;rate={self._audio_rate}",
                    },
                )
            elif msg.get("type") in ("text_in", "interrupt"):
                await session.send_client_content(
                    turns={"role": "user", "parts": [{"text": msg["text"]}]},
                    turn_complete=True,
                )

    async def send(self, message: dict[str, Any]) -> None:
        await self._out_queue.put(message)


async def _noop() -> None:
    pass
