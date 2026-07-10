from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    gemini_live_model: str = Field(
        default="gemini-3.1-flash-live-preview",
        alias="GEMINI_LIVE_MODEL",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
