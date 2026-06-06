from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = Field(default="development", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(
        default="openrouter:deepseek/deepseek-v4-flash",
        alias="OPENROUTER_MODEL",
    )

    checkpoint_db_path: Path = Field(
        default=DATA_DIR / "sqlite" / "checkpoints.db",
        alias="CHECKPOINT_DB_PATH",
    )
    audio_temp_dir: Path = Field(
        default=DATA_DIR / "audio",
        alias="AUDIO_TEMP_DIR",
    )
    models_dir: Path = Field(
        default=DATA_DIR / "models",
        alias="MODELS_DIR",
    )

    stt_model: str = Field(default="small", alias="STT_MODEL")
    stt_device: str = Field(default="cpu", alias="STT_DEVICE")
    stt_compute_type: str = Field(default="int8", alias="STT_COMPUTE_TYPE")

    tts_model: str = Field(default="hexgrad/kokoro-82m", alias="TTS_MODEL")
    tts_voice: str = Field(default="ef_dora", alias="TTS_VOICE")
    tts_language: str = Field(default="en-US", alias="TTS_LANGUAGE")
    tts_timeout: float = Field(default=180.0, alias="TTS_TIMEOUT")

    streamlit_api_base_url: str = Field(
        default="http://localhost:8000",
        alias="STREAMLIT_API_BASE_URL",
    )

    @property
    def app_root(self) -> Path:
        return ROOT_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.audio_temp_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
