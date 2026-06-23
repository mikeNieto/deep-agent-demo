from __future__ import annotations

from pydantic import BaseModel


class AudioSynthesisRequest(BaseModel):
    text: str
    voice: str | None = None


class AudioSynthesisResponse(BaseModel):
    audio_url: str
    audio_mime_type: str = "audio/mpeg"
    duration_seconds: float | None = None


class AudioTranscriptionResponse(BaseModel):
    original_text: str
    translated_text: str
    detected_language: str | None = None
    text: str = ""
    language: str | None = None
    duration_seconds: float | None = None
