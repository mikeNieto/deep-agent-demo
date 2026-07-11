from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.audio import synthesize_speech, transcribe_audio
from app.config import Settings, get_settings


def _settings() -> Settings:
    get_settings.cache_clear()
    return get_settings.__wrapped__()  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_transcribe_audio_returns_string(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.text = "hola mundo"

    with patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_model):
        result = await transcribe_audio(settings, b"fake audio", "audio/wav")
        assert result == "hola mundo"


@pytest.mark.asyncio
async def test_transcribe_audio_with_list_content(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.text = [
        {"type": "text", "text": "hola"},
        {"type": "text", "text": "mundo"},
    ]

    with patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_model):
        result = await transcribe_audio(settings, b"fake", "audio/wav")
        assert result == "holamundo"


@pytest.mark.asyncio
async def test_synthesize_speech_returns_bytes(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()

    fake_audio = b"\x00\x01\x02\x03"
    mock_part = Mock()
    mock_part.inline_data.data = fake_audio
    mock_part.inline_data.mime_type = "audio/wav"

    mock_response = Mock()
    mock_response.parts = [mock_part]

    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        data, mime = await synthesize_speech(settings, "hello")
        assert data == fake_audio
        assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_synthesize_speech_handles_missing_audio(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()

    mock_response = Mock()
    mock_response.parts = []

    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("google.genai.Client", return_value=mock_client):
        data, mime = await synthesize_speech(settings, "hello")
        assert data == b""
        assert mime == "audio/wav"
