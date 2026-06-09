"""Central configuration. ALL secrets live here, server-side only.

This is the fix for the client-side key leak: the browser never sees these
values. The frontend talks only to this backend; this backend holds the keys.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Commercial detector APIs (Layer 1) ---
    sightengine_api_user: str | None = None
    sightengine_api_secret: str | None = None
    hive_api_key: str | None = None
    illuminarty_api_key: str | None = None
    ai_or_not_api_key: str | None = None

    # --- Forensic / context services ---
    serpapi_key: str | None = None  # reverse image search

    # --- LLM explanation layer (Layer 3) ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"

    # --- Behavior ---
    signal_timeout_seconds: float = 6.0  # per-signal cap; supports p95 < 6s goal
    cors_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
