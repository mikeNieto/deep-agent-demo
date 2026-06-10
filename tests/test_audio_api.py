from app.config import get_settings
from fastapi.testclient import TestClient

from app.main import app


def test_models_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/health/models")
        assert response.status_code == 200
        assert "agent" in response.json()


def test_transcribe_returns_503_when_openrouter_is_not_configured(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    get_settings.cache_clear()

    with TestClient(app) as client:
            response = client.post(
                "/api/audio/transcribe",
                files={"file": ("recording.wav", b"fake audio", "audio/wav")},
            )

    get_settings.cache_clear()

    assert response.status_code == 503
    assert response.json()["detail"] == "OPENROUTER_API_KEY is not configured"
