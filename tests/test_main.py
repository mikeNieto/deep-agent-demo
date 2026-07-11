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
    monkeypatch.setenv("DEEPAGENT_MODEL", "env-model")
    monkeypatch.setenv("STT_MODEL", "env-stt")
    monkeypatch.setenv("TTS_MODEL", "env-tts")
    monkeypatch.setenv("THINKING_LEVEL", "LOW")

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    assert settings.google_api_key == "env-key"
    assert settings.deepagent_model == "env-model"
    assert settings.stt_model == "env-stt"
    assert settings.tts_model == "env-tts"
    assert settings.thinking_level == "LOW"
