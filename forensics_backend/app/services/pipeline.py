from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from app.config import Settings
from app.schemas import AnalysisResponse, ForensicsEnvelope
from app.services.base import AnalysisContext
from app.services.c2pa_module import C2PAModule
from app.services.ela_module import ElaModule
from app.services.exif_module import ExifModule
from app.services.reverse_search_module import ReverseSearchModule


class ForensicsPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.modules = (
            ExifModule(),
            ElaModule(),
            ReverseSearchModule(settings.serpapi_key),
            C2PAModule(),
        )

    async def analyze(
        self, filename: str, content_type: str | None, media_bytes: bytes
    ) -> AnalysisResponse:
        analysis_id = uuid4().hex[:12]
        artifact_dir = self.settings.artifact_dir / analysis_id
        context = AnalysisContext(
            analysis_id=analysis_id,
            filename=filename,
            content_type=content_type,
            media_bytes=media_bytes,
            artifact_dir=artifact_dir,
            artifact_url_prefix=self.settings.artifact_url_prefix,
        )

        results = await asyncio.gather(
            *(module.analyze(context) for module in self.modules)
        )
        modules_by_name = {
            module.name: result for module, result in zip(self.modules, results, strict=True)
        }
        completed_modules = [
            name for name, result in modules_by_name.items() if result.status == "ok"
        ]
        partial_modules = [
            name
            for name, result in modules_by_name.items()
            if result.status in {"warning", "skipped"}
        ]
        failed_modules = [
            name for name, result in modules_by_name.items() if result.status == "error"
        ]

        return AnalysisResponse(
            analysis_id=analysis_id,
            filename=filename,
            content_type=content_type,
            sha256=hashlib.sha256(media_bytes).hexdigest(),
            analyzed_at=datetime.now(timezone.utc),
            ready_for_triangulation=bool(completed_modules or partial_modules),
            completed_modules=completed_modules,
            partial_modules=partial_modules,
            failed_modules=failed_modules,
            forensics=ForensicsEnvelope(
                exif=modules_by_name["exif"],
                ela=modules_by_name["ela"],
                reverse_search=modules_by_name["reverse_search"],
                c2pa=modules_by_name["c2pa"],
            ),
        )
