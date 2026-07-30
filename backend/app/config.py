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
    # Serves the retrained frozen-CLIP + MLP head winner (3% real-photo FPR),
    # not the old overfit ResNet. Flip local_model_type back to "resnet" to serve
    # a resnet50_dropout checkpoint via the legacy 32x32 patch-tiling pipeline
    # instead (see signals/local_model.py for why the two types run differently).
    local_model_type: str = "clip"
    local_model_checkpoint: str = str(
        REPO_ROOT / "detector-trainer" / "output" / "clip_mlp" / "clip_head_best.pt"
    )
    local_model_name: str = "clip_mlp"
    local_model_threshold: float = 0.57  # calibrated for ~2% real-photo FPR
    # Legacy resnet path only (local_model_type = "resnet"); unused by clip.
    local_model_resnet_checkpoint: str = str(
        REPO_ROOT / "detector-trainer" / "output" / "best_model.pt"
    )
    local_model_resnet_name: str = "resnet50_dropout"
    # When true, SignalResult.raw["debug"] on the `local` signal includes full
    # per-patch coordinates/scores and aggregate stats (mean/median/top-20%/
    # etc), for the resnet patch-tiling path only. Off by default so the normal
    # envelope stays lean -- this can get large for a big image's patch grid.
    local_model_debug: bool = False

    # --- Forensic / context services ---
    serpapi_key: str | None = None  # reverse image search

    # --- LLM explanation layer (Layer 3) ---
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-4-8"
    vlm_model_id: str = "hive/vision-language-model"
    vlm_max_new_tokens: int = 220

    # --- Face-swap / deepfake-face signal ---
    faceswap_enabled: bool = True
    faceswap_model_id: str = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    faceswap_threshold: float = 0.80  # locked by the acceptance gate (2026-07-29):
    # catches the Curry composite (0.873) while all reference genuine faces stay
    # below (max 0.744, the driver's-license photo). Thin margin on documents.

    # --- Document-tamper signal ---
    doctamper_enabled: bool = True
    doctamper_threshold: float = 0.60  # acceptance gate (2026-07-29): INDICATOR-ONLY.
    # ELA cannot separate real edits from ordinary recompression — a synthetic
    # field edit scored 0.044 vs 0.024-0.036 for genuine/scanned/recompressed
    # copies (all near the noise floor). 0.60 is far above any of these, so the
    # heatmap always shows on documents but the verdict never elevates in practice.
    doctamper_doc_gate_threshold: float = 0.55  # CLIP doc-vs-photo probability to treat as a document
    doctamper_backbone: str = "ViT-L-14"  # CLIP backbone for the document-gate (weights cached by the local signal)

    # --- Behavior ---
    signal_timeout_seconds: float = 6.0  # per-signal cap; supports p95 < 6s goal
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
