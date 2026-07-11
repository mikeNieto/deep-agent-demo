from app.config import ROOT_DIR, Settings, get_settings
from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_index_returns_html() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "SYLYS" in response.text


def test_settings_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "env-key")
    monkeypatch.setenv("GOOGLE_CLOUD_API_KEY", "cloud-key")
    monkeypatch.setenv("DEEPAGENT_MODEL", "env-model")
    monkeypatch.setenv("STT_MODEL", "env-stt")
    monkeypatch.setenv("TTS_VOICE_EN", "env-voice-en")
    monkeypatch.setenv("TTS_VOICE_ES", "env-voice-es")
    monkeypatch.setenv("THINKING_LEVEL", "LOW")

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.google_api_key == "env-key"
    assert settings.google_cloud_api_key == "cloud-key"
    assert settings.tts_api_key == "cloud-key"
    assert settings.deepagent_model == "env-model"
    assert settings.stt_model == "env-stt"
    assert settings.tts_voice_en == "env-voice-en"
    assert settings.tts_voice_es == "env-voice-es"
    assert settings.thinking_level == "LOW"


def test_tts_api_key_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "only-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.tts_api_key == "only-key"
