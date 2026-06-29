from __future__ import annotations

import base64
import json as json_module
import logging
from pathlib import Path

import httpx

from app.config import Settings


logger = logging.getLogger(__name__)

OPENROUTER_STT_URL = "https://openrouter.ai/api/v1/chat/completions"

STT_STRUCTURED_SCHEMA = {
    "name": "stt_transcription",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "original_text": {
                "type": "string",
                "description": "Full transcription in the original spoken language",
            },
            "translated_text": {
                "type": "string",
                "description": "English translation of the speech",
            },
            "detected_language": {
                "type": "string",
                "description": "ISO 639-1 language code of the original speech (e.g. 'es', 'en', 'fr')",
            },
        },
        "required": ["original_text", "translated_text", "detected_language"],
        "additionalProperties": False,
    },
}

DEFAULT_STT_PROMPT = (
    "You are an audio transcription and translation engine. "
    "Transcribe the audio in its original language, then provide an English translation. "
    "Detect the language of the speech automatically. "
    "Do not add commentary, labels, quotes, explanations, greetings, or extra words."
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
        self._client = httpx.Client(timeout=settings.stt_timeout)

    @property
    def loaded(self) -> bool:
        return bool(self._settings.openrouter_api_key)

    def transcribe(
        self,
        file_path: Path,
        mime_type: str | None = None,
    ) -> tuple[str, str, str, float | None]:
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
            "response_format": {
                "type": "json_schema",
                "json_schema": STT_STRUCTURED_SCHEMA,
            },
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
            result = self._parse_structured_response(body)
            if not result:
                raise STTServiceError("OpenRouter STT returned an empty transcription")
            original_text, translated_text, detected_language = result
        except httpx.HTTPStatusError as exc:
            raw = exc.response.text if exc.response is not None else str(exc)
            logger.error("OpenRouter STT request failed: %s", raw)
            if "response_format" in raw.lower() and "not supported" in raw.lower():
                raise STTServiceError(
                    f"STT model '{self._settings.stt_model}' does not support "
                    "structured outputs. Please use a compatible model."
                ) from exc
            raise STTServiceError(f"OpenRouter STT request failed: {raw}") from exc
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
        return original_text, translated_text, detected_language, duration

    @staticmethod
    def _parse_structured_response(body: object) -> tuple[str, str, str] | None:
        if not isinstance(body, dict):
            return None

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            return None

        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            return None

        raw_content = message.get("content")
        if not isinstance(raw_content, str) or not raw_content.strip():
            return None

        try:
            data = json_module.loads(raw_content)
        except json_module.JSONDecodeError:
            logger.warning("STT structured output is not valid JSON: %s", raw_content[:200])
            return None

        if not isinstance(data, dict):
            logger.warning("STT structured output is not a JSON object: %s", raw_content[:200])
            return None

        original = (data.get("original_text") or "").strip()
        translated = (data.get("translated_text") or "").strip()
        language = (data.get("detected_language") or "").strip()

        if not original or not translated:
            logger.warning(
                "STT structured output missing required fields: %s", raw_content[:200]
            )
            return None

        return original, translated, language

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
