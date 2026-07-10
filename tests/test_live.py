from __future__ import annotations

import pytest


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_LIVE_MODEL", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)

    from app.config import Settings

    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        google_api_key="test-key",  # type: ignore[call-arg]
    )
    assert s.google_api_key == "test-key"
    assert s.gemini_live_model == "gemini-3.1-flash-live-preview"
    assert s.api_host == "0.0.0.0"
    assert s.api_port == 8000


def test_app_creates() -> None:
    from app.main import app

    assert app.title == "Gemini Live Test"
    assert len(app.routes) > 0
