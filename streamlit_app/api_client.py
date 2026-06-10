from __future__ import annotations

from io import BytesIO

import httpx


class ApiClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def send_message(self, payload: dict) -> dict:
        response = httpx.post(
            f"{self._base_url}/api/chat/message",
            json=payload,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def transcribe_audio(self, audio_bytes: bytes, filename: str = "recording.wav") -> dict:
        files = {"file": (filename, BytesIO(audio_bytes), "audio/wav")}
        response = httpx.post(
            f"{self._base_url}/api/audio/transcribe",
            files=files,
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_history(self, thread_id: str) -> dict:
        response = httpx.get(
            f"{self._base_url}/api/conversations/{thread_id}",
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
