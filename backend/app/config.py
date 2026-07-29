"""Central configuration. ALL secrets live here, server-side only.

This is the fix for the client-side key leak: the browser never sees these
values. The frontend talks only to this backend; this backend holds the keys.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        extra="ignore",
    )

    # --- Commercial detector APIs (Layer 1) ---
    sightengine_api_user: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SIGHTENGINE_API_USER", "VITE_SIGHTENGINE_API_USER"),
    )
    sightengine_api_secret: str | None = Field(
        default=None,
        validation_alias=AliasChoices("SIGHTENGINE_API_SECRET", "VITE_SIGHTENGINE_API_SECRET"),
    )
    # `hive_api_key` is the V3 Playground Secret Key (self-serve accounts only
    # get V3; V2 Enterprise requires a sales-provisioned project). See
    # signals/hive.py for why this must NOT point at the v2 task endpoint.
    hive_api_key: str | None = None
    hive_access_key: str | None = None  # unused: V3 auth needs only the secret key
    hive_secret_key: str | None = None  # unused: kept for reference/future V2 upgrade
    hive_api_url: str = "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"
    illuminarty_api_key: str | None = None
    ai_or_not_api_key: str | None = None

    # --- Local trained detector ---
    local_model_checkpoint: str = str(REPO_ROOT / "detector-trainer" / "output" / "best_model.pt")
    local_model_name: str = "resnet50_dropout"

    # --- Forensic / context services ---
    serpapi_key: str | None = None  # reverse image search

    # --- LLM explanation layer (Layer 3) ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
    vlm_model_id: str = "hive/vision-language-model"
    vlm_max_new_tokens: int = 220

    # --- Behavior ---
    signal_timeout_seconds: float = 6.0  # per-signal cap; supports p95 < 6s goal
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
