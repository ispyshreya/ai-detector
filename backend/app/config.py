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
    hive_api_key: str | None = None
    hive_access_key: str | None = None
    hive_secret_key: str | None = None
    hive_api_url: str = "https://api.thehive.ai/api/v2/task/sync"
    illuminarty_api_key: str | None = None
    ai_or_not_api_key: str | None = None

    # --- Local trained detector ---
    # Serves the retrained frozen-CLIP + MLP head winner (3% real-photo FPR),
    # not the old overfit ResNet. Flip local_model_type back to "resnet" to serve
    # a resnet50_dropout checkpoint instead.
    local_model_type: str = "clip"
    local_model_checkpoint: str = str(
        REPO_ROOT / "detector-trainer" / "output" / "clip_mlp" / "clip_head_best.pt"
    )
    local_model_name: str = "clip_mlp"
    local_model_threshold: float = 0.57  # calibrated for ~2% real-photo FPR

    # --- Forensic / context services ---
    serpapi_key: str | None = None  # reverse image search

    # --- LLM explanation layer (Layer 3) ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
    vlm_model_id: str = "HuggingFaceTB/SmolVLM-500M-Instruct"
    vlm_max_new_tokens: int = 120

    # --- Face-swap / deepfake-face signal ---
    faceswap_enabled: bool = True
    faceswap_model_id: str = "dima806/deepfake_vs_real_image_detection"
    faceswap_threshold: float = 0.7  # conservative; tuned by the acceptance gate

    # --- Behavior ---
    signal_timeout_seconds: float = 6.0  # per-signal cap; supports p95 < 6s goal
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
