import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.audio import (
    get_interim_audio,
    synthesize_speech,
    transcribe_audio,
)
from app.config import Settings, get_settings


def _settings() -> Settings:
    get_settings.cache_clear()
    return get_settings.__wrapped__()  # type: ignore[return-value]


@pytest.mark.asyncio
async def test_transcribe_audio_returns_text_and_lang(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.text = json.dumps(
        {"text": "hola mundo", "language": "es"}
    )

    with patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_model):
        text, lang = await transcribe_audio(settings, b"fake audio", "audio/wav")
        assert text == "hola mundo"
        assert lang == "es"


@pytest.mark.asyncio
async def test_transcribe_audio_fallback_on_bad_json(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.text = "not valid json"

    with patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_model):
        text, lang = await transcribe_audio(settings, b"fake", "audio/wav")
        assert text == "not valid json"
        assert lang == "en"


@pytest.mark.asyncio
async def test_transcribe_audio_passes_any_lang(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()
    mock_model = AsyncMock()
    mock_model.ainvoke.return_value.text = json.dumps(
        {"text": "hallo", "language": "de"}
    )

    with patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_model):
        text, lang = await transcribe_audio(settings, b"fake", "audio/wav")
        assert text == "hallo"
        assert lang == "de"


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


@pytest.mark.asyncio
async def test_get_interim_audio_cached(monkeypatch) -> None:
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

    import app.audio as audio_mod
    audio_mod._MEMORY_CACHE.clear()

    with (
        patch("google.genai.Client", return_value=mock_client),
        patch("app.audio._load_from_disk", return_value=None),
    ):
        result = await get_interim_audio(settings, "en")
        assert result == (fake_audio, "audio/wav")
        assert "en" in audio_mod._MEMORY_CACHE

        result2 = await get_interim_audio(settings, "en")
        assert result2 == (fake_audio, "audio/wav")
        assert mock_client.models.generate_content.call_count == 1


@pytest.mark.asyncio
async def test_get_interim_audio_unknown_lang_generates(monkeypatch) -> None:
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

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.text = "Tarkistan..."

    import app.audio as audio_mod
    audio_mod._MEMORY_CACHE.clear()

    with (
        patch("google.genai.Client", return_value=mock_client),
        patch("app.audio._load_from_disk", return_value=None),
        patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_llm),
    ):
        result = await get_interim_audio(settings, "sw")
        assert result == (fake_audio, "audio/wav")
        mock_llm.ainvoke.assert_called_once()
