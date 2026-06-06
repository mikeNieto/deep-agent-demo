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
    text: str
    language: str | None = None
    duration_seconds: float | None = None


class OpenRouterAudioRequest(BaseModel):
    prompt: str = "Analiza este audio y responde con tu propia voz."
    voice: str | None = None
    audio_format: str = "wav"


class OpenRouterAudioResponse(BaseModel):
    text: str
    audio_url: str | None = None
    audio_mime_type: str = "audio/mpeg"
