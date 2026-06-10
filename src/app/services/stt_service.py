from __future__ import annotations

import base64
import logging
from pathlib import Path

import httpx

from app.config import Settings


logger = logging.getLogger(__name__)

OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/chat/completions"

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
        self._client = httpx.Client()

    @property
    def loaded(self) -> bool:
        return bool(self._settings.openrouter_api_key)

    def transcribe(
        self,
        file_path: Path,
        mime_type: str | None = None,
    ) -> tuple[str, str | None, float | None]:
        if not self._settings.openrouter_api_key:
            raise STTServiceError("OPENROUTER_API_KEY is not configured")

        resolved_mime_type = mime_type or self._guess_mime_type(file_path)
        audio_bytes = file_path.read_bytes()
        audio_format = self._guess_audio_format(file_path, resolved_mime_type)
        payload = {
            "model": self._settings.stt_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": DEFAULT_STT_PROMPT,
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_bytes).decode("ascii"),
                                "format": audio_format,
                            },
                        },
                    ],
                }
            ],
            "temperature": 0,
        }

        try:
            response = self._client.post(
                OPENROUTER_STT_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                    "X-OpenRouter-Title": "Deep Agent Demo",
                },
                json=payload,
            )
            response.raise_for_status()
            body = response.json()
            text = self._extract_text(body)
            if not text:
                raise STTServiceError("OpenRouter STT returned an empty transcription")
        except httpx.HTTPStatusError as exc:
            body = exc.response.text if exc.response is not None else str(exc)
            logger.error("OpenRouter STT request failed: %s", body)
            raise STTServiceError(f"OpenRouter STT request failed: {body}") from exc
        except httpx.HTTPError as exc:
            logger.error("OpenRouter STT request failed: %s", exc)
            raise STTServiceError(f"OpenRouter STT request failed: {exc}") from exc
        except Exception as exc:
            logger.error("OpenRouter STT failed: %s", exc)
            raise STTServiceError(f"OpenRouter STT failed: {exc}") from exc

        usage = body.get("usage") if isinstance(body, dict) else None
        duration = usage.get("seconds") if isinstance(usage, dict) else None
        if duration is None:
            logger.info(
                "OpenRouter STT response did not include usage.seconds model=%s",
                self._settings.stt_model,
            )
        return text.strip(), None, duration

    @staticmethod
    def _extract_text(body: object) -> str:
        if not isinstance(body, dict):
            return ""

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return ""

        content = message.get("content")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            return "".join(parts)

        return ""

    @staticmethod
    def _guess_mime_type(file_path: Path) -> str:
        return MIME_TYPE_BY_SUFFIX.get(file_path.suffix.lower(), "audio/wav")

    @staticmethod
    def _guess_audio_format(file_path: Path, mime_type: str) -> str:
        suffix = file_path.suffix.lower().lstrip(".")
        if suffix:
            if suffix == "mpeg":
                return "mp3"
            if suffix == "oga":
                return "ogg"
            if suffix == "mp4":
                return "m4a"
            return suffix

        subtype = mime_type.split("/", 1)[-1].lower()
        if subtype == "mpeg":
            return "mp3"
        if subtype == "mp4":
            return "m4a"
        return subtype or "wav"
