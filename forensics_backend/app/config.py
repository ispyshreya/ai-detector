from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    artifact_dir: Path
    serpapi_key: str | None
    public_base_url: str

    @property
    def artifact_url_prefix(self) -> str:
        return f"{self.public_base_url.rstrip('/')}/artifacts"


def get_settings() -> Settings:
    artifact_dir = Path(
        os.getenv("VEIL_FORENSICS_ARTIFACT_DIR", "./runtime/artifacts")
    ).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = (BACKEND_ROOT / artifact_dir).resolve()
    return Settings(
        host=os.getenv("VEIL_FORENSICS_HOST", "127.0.0.1"),
        port=int(os.getenv("VEIL_FORENSICS_PORT", "8010")),
        artifact_dir=artifact_dir,
        serpapi_key=os.getenv("VEIL_FORENSICS_SERPAPI_KEY") or None,
        public_base_url=os.getenv(
            "VEIL_FORENSICS_HTTPS_BASE_URL", "http://127.0.0.1:8010"
        ),
    )
