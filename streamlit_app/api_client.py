from __future__ import annotations

from io import BytesIO

import httpx


class ApiClient:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url.rstrip("/")

    def send_message(self, payload: dict) -> dict:
        response = httpx.post(
            f"{self._base_url}/api/chat/message", json=payload, timeout=120
        )
        response.raise_for_status()
        return response.json()

    def transcribe_audio(
        self, audio_bytes: bytes, filename: str = "recording.wav"
    ) -> dict:
        files = {"file": (filename, BytesIO(audio_bytes), "audio/wav")}
        response = httpx.post(
            f"{self._base_url}/api/audio/transcribe", files=files, timeout=120
        )
        response.raise_for_status()
        return response.json()

    def get_history(self, thread_id: str) -> dict:
        response = httpx.get(
            f"{self._base_url}/api/conversations/{thread_id}", timeout=30
        )
        response.raise_for_status()
        return response.json()

    def openrouter_audio_process(
        self,
        audio_bytes: bytes,
        filename: str = "recording.wav",
        prompt: str = "Analiza este audio y responde con tu propia voz.",
        voice: str | None = None,
        user_id: str = "default-user",
        thread_id: str = "default-thread",
    ) -> dict:
        """Send audio to the OpenRouter multimodal audio endpoint."""
        files = {"file": (filename, BytesIO(audio_bytes), "audio/wav")}
        data = {
            "prompt": prompt,
            "user_id": user_id,
            "thread_id": thread_id,
        }
        if voice:
            data["voice"] = voice
        response = httpx.post(
            f"{self._base_url}/api/audio/openrouter-audio",
            files=files,
            data=data,
            timeout=180,
        )
        response.raise_for_status()
        return response.json()
