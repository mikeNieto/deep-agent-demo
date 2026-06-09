from __future__ import annotations

import logging

import httpx

from app.agents.prompts import (
    TTS_ADAPTATION_SYSTEM_PROMPT,
    build_tts_adaptation_user_prompt,
)
from app.config import Settings


logger = logging.getLogger(__name__)

OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class TTSPreparationServiceError(RuntimeError):
    pass


class TTSPreparationService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.Client()

    def prepare_text(self, agent_response: str) -> str:
        text = agent_response.strip()
        if not text:
            return text
        if not self._settings.tts_preparation_model:
            raise TTSPreparationServiceError("TTS_PREPARATION_MODEL is not configured")

        try:
            response = self._client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self._settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._settings.tts_preparation_model,
                    "messages": [
                        {"role": "system", "content": TTS_ADAPTATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_tts_adaptation_user_prompt(text),
                        },
                    ],
                    "temperature": 0.2,
                },
                timeout=self._settings.tts_timeout,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            resp = getattr(exc, "response", None)
            body = ""
            status_code = "?"
            if resp is not None:
                status_code = getattr(resp, "status_code", "?")
                try:
                    body = resp.text
                except Exception:
                    body = "<could not read response body>"
            logger.error(
                "OpenRouter TTS preparation status error: %s %s",
                status_code,
                body,
            )
            raise TTSPreparationServiceError(
                f"Failed to prepare TTS text: status={status_code}, body={body}"
            ) from exc
        except httpx.HTTPError as exc:
            logger.error("OpenRouter TTS preparation request failed: %s", exc)
            raise TTSPreparationServiceError(
                f"Failed to prepare TTS text: {exc}"
            ) from exc

        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise TTSPreparationServiceError(
                "Failed to prepare TTS text: empty choices"
            )

        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )

        prepared_text = str(content).strip()
        if not prepared_text:
            raise TTSPreparationServiceError(
                "Failed to prepare TTS text: empty model response"
            )

        return prepared_text
