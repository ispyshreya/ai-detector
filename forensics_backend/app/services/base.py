from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisContext:
    analysis_id: str
    filename: str
    content_type: str | None
    media_bytes: bytes
    artifact_dir: Path
    artifact_url_prefix: str
