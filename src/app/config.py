from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    google_cloud_api_key: str = Field(default="", alias="GOOGLE_CLOUD_API_KEY")

    @property
    def tts_api_key(self) -> str:
        return self.google_cloud_api_key or self.google_api_key

    deepagent_model: str = Field(
        default="gemini-2.5-flash",
        alias="DEEPAGENT_MODEL",
    )
    stt_model: str = Field(
        default="gemini-2.5-flash",
        alias="STT_MODEL",
    )
    tts_voice_en: str = Field(
        default="en-US-Wavenet-F",
        alias="TTS_VOICE_EN",
    )
    tts_voice_es: str = Field(
        default="es-ES-Wavenet-C",
        alias="TTS_VOICE_ES",
    )

    thinking_level: str = Field(default="", alias="THINKING_LEVEL")
    thinking_budget: int = Field(default=0, alias="THINKING_BUDGET")

    audio_temp_dir: Path = Field(
        default=DATA_DIR / "audio",
        alias="AUDIO_TEMP_DIR",
    )

    @field_validator("thinking_budget", mode="before")
    @classmethod
    def _empty_str_to_zero(cls, v: object) -> object:
        if v == "" or v is None:
            return 0
        return v

    @property
    def app_root(self) -> Path:
        return ROOT_DIR


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.audio_temp_dir.mkdir(parents=True, exist_ok=True)
    return settings
