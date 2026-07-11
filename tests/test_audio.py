import base64
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.audio import (
    _clean_for_tts,
    _pcm_to_wav,
    get_interim_audio,
    synthesize_speech,
    transcribe_audio,
)
from app.config import Settings, get_settings


def _settings() -> Settings:
    get_settings.cache_clear()
    return get_settings.__wrapped__()  # type: ignore[return-value]


class _FakeHttpResponse:
    def __init__(self, json_data: dict) -> None:
        self._json = json_data
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json


class _FakeAsyncClient:
    def __init__(self, response_json: dict | None = None) -> None:
        self._response = _FakeHttpResponse(response_json or {})
        self.call_args: tuple | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url: str, **kwargs):
        self.call_args = (url, kwargs)
        return self._response


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
    audio_b64 = base64.b64encode(fake_audio).decode("ascii")

    fake_client = _FakeAsyncClient({"audioContent": audio_b64})

    with patch("app.audio.httpx.AsyncClient", return_value=fake_client):
        data, mime = await synthesize_speech(settings, "hello", "en")
        assert data == _pcm_to_wav(fake_audio)
        assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_synthesize_speech_handles_missing_audio(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()

    fake_client = _FakeAsyncClient({})

    with patch("app.audio.httpx.AsyncClient", return_value=fake_client):
        data, mime = await synthesize_speech(settings, "hello")
        assert data == b""
        assert mime == "audio/wav"


@pytest.mark.asyncio
async def test_synthesize_speech_uses_spanish_voice(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "cloud-tts-key")
    monkeypatch.setenv("TTS_VOICE_ES", "es-ES-Wavenet-D")
    settings = _settings()

    fake_audio = b"\x00\x01"
    audio_b64 = base64.b64encode(fake_audio).decode("ascii")

    fake_client = _FakeAsyncClient({"audioContent": audio_b64})

    with patch("app.audio.httpx.AsyncClient", return_value=fake_client):
        await synthesize_speech(settings, "hola", "es")

    assert fake_client.call_args is not None
    url = fake_client.call_args[0]
    body = fake_client.call_args[1]["json"]
    assert "key=cloud-tts-key" in url
    assert body["voice"]["name"] == "es-ES-Wavenet-D"
    assert body["voice"]["languageCode"] == "es-ES"


@pytest.mark.asyncio
async def test_synthesize_speech_falls_back_to_google_api_key(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "")
    settings = _settings()

    audio_b64 = base64.b64encode(b"\x00\x01").decode("ascii")
    fake_client = _FakeAsyncClient({"audioContent": audio_b64})

    with patch("app.audio.httpx.AsyncClient", return_value=fake_client):
        await synthesize_speech(settings, "hello", "en")

    assert fake_client.call_args is not None
    url = fake_client.call_args[0]
    assert "key=test-key" in url
    assert "cloud-tts-key" not in url


@pytest.mark.asyncio
async def test_get_interim_audio_cached(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()

    fake_audio = b"\x00\x01\x02\x03"
    audio_b64 = base64.b64encode(fake_audio).decode("ascii")

    fake_client = _FakeAsyncClient({"audioContent": audio_b64})

    import app.audio as audio_mod
    audio_mod._MEMORY_CACHE.clear()

    expected_audio = _pcm_to_wav(fake_audio)

    with (
        patch("app.audio.httpx.AsyncClient", return_value=fake_client),
        patch("app.audio._load_from_disk", return_value=None),
    ):
        result = await get_interim_audio(settings, "en")
        assert result == (expected_audio, "audio/wav")
        assert "en" in audio_mod._MEMORY_CACHE

        result2 = await get_interim_audio(settings, "en")
        assert result2 == (expected_audio, "audio/wav")
        assert fake_client._response._json == {"audioContent": audio_b64}


@pytest.mark.asyncio
async def test_get_interim_audio_unknown_lang_generates(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    settings = _settings()

    fake_audio = b"\x00\x01\x02\x03"
    audio_b64 = base64.b64encode(fake_audio).decode("ascii")

    fake_client = _FakeAsyncClient({"audioContent": audio_b64})

    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value.text = "Tarkistan..."

    import app.audio as audio_mod
    audio_mod._MEMORY_CACHE.clear()

    with (
        patch("app.audio.httpx.AsyncClient", return_value=fake_client),
        patch("app.audio._load_from_disk", return_value=None),
        patch("app.audio.ChatGoogleGenerativeAI", return_value=mock_llm),
    ):
        result = await get_interim_audio(settings, "sw")
        assert result == (_pcm_to_wav(fake_audio), "audio/wav")
        mock_llm.ainvoke.assert_called_once()


@pytest.mark.parametrize(
    "markdown,expected",
    [
        ("hola", "hola"),
        ("**negrita** y *cursiva*", "negrita y cursiva"),
        ("__subrayado__ y ~~tachado~~", "subrayado y tachado"),
        ("[click aqui](https://ejemplo.com)", "click aqui"),
            ("![foto](img.png) texto", "texto"),
        ("`codigo` en linea", "codigo en linea"),
        ("```python\nprint('hi')\n```", "print('hi')"),
        ("# Titulo\n## Subtitulo\nTexto", "Titulo\nSubtitulo\nTexto"),
        ("> quote\n> otra linea", "quote\notra linea"),
        ("- item 1\n- item 2", "item 1\nitem 2"),
        ("1. primero\n2. segundo", "primero\nsegundo"),
        ("---\ntexto\n***", "texto"),
        ("linea1\n\n\n\nlinea2", "linea1\n\nlinea2"),
        ("  leading spaces", "leading spaces"),
        ("**Hola**, ¿cómo estás? Tengo un [link](url).", "Hola ¿cómo estás? Tengo un link."),
        ("Mix: **bold**, *italic*, `code`, [link](url), # heading", "Mix: bold italic code link heading"),
        ("Real Madrid 2-1 Barcelona", "Real Madrid 2 1 Barcelona"),
        ("Today is Saturday, July 11, 2026 and the time is 5:59 PM", "Today is Saturday July 11 2026 and the time is 5:59 PM"),
        ("1-1 score", "1 1 score"),
        ("Hola \U0001F600 mundo", "Hola mundo"),
        ("\u2705 Done \u26a1 Fast \U0001f680 Go", "Done Fast Go"),
        ("texto \U0001F1E8\U0001F1F4 normal", "texto normal"),
    ],
)
def test_clean_for_tts(markdown: str, expected: str) -> None:
    assert _clean_for_tts(markdown) == expected
