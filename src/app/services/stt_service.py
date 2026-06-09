from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import httpx
from google import genai
from google.genai import types

from app.config import Settings


logger = logging.getLogger(__name__)

DEFAULT_STT_PROMPT = (
    "You are an audio transcription and translation engine. "
    "Listen carefully to the user's speech and return only the exact English translation "
    "of what was said. Preserve the original meaning, tone, and intent. "
    "Do not add commentary, labels, quotes, explanations, or extra words. "
    "If the speech is already in English, return a clean English transcription only."
)

MIME_TYPE_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mp4": "audio/mp4",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
}


class STTServiceError(RuntimeError):
    pass


class STTService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: genai.Client | None = None

    @property
    def loaded(self) -> bool:
        return bool(self._settings.gemini_api_key)

    def _get_client(self) -> genai.Client:
        if not self._settings.gemini_api_key:
            raise STTServiceError("GEMINI_API_KEY is not configured")
        if self._client is None:
            logger.info("Initializing Gemini STT client with model %s", self._settings.stt_model)
            self._client = genai.Client(
                api_key=self._settings.gemini_api_key,
            )
        return self._client

    def transcribe(
        self,
        file_path: Path,
        mime_type: str | None = None,
    ) -> tuple[str, str | None, float | None]:
        client = self._get_client()
        resolved_mime_type = mime_type or self._guess_mime_type(file_path)
        audio_bytes = file_path.read_bytes()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=resolved_mime_type),
                    types.Part.from_text(text=DEFAULT_STT_PROMPT),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0,
        )

        try:
            text = self._collect_text(
                client.models.generate_content_stream(
                    model=self._settings.stt_model,
                    contents=contents,
                    config=config,
                )
            )
        except httpx.HTTPError as exc:
            logger.error("Gemini STT request failed: %s", exc)
            raise STTServiceError(f"Gemini STT request failed: {exc}") from exc
        except Exception as exc:
            logger.error("Gemini STT failed: %s", exc)
            raise STTServiceError(f"Gemini STT failed: {exc}") from exc

        return text.strip(), None, None

    @staticmethod
    def _collect_text(chunks: Iterable[object]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            text = getattr(chunk, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    @staticmethod
    def _guess_mime_type(file_path: Path) -> str:
        return MIME_TYPE_BY_SUFFIX.get(file_path.suffix.lower(), "audio/wav")
