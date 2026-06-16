from __future__ import annotations

from time import perf_counter

from app.schemas import ReverseSearchResult
from app.services.base import AnalysisContext


class ReverseSearchModule:
    name = "reverse_search"

    def __init__(self, serpapi_key: str | None) -> None:
        self.serpapi_key = serpapi_key

    async def analyze(self, context: AnalysisContext) -> ReverseSearchResult:
        started = perf_counter()
        if not self.serpapi_key:
            return ReverseSearchResult(
                status="skipped",
                summary="SerpAPI key not configured.",
                warnings=[
                    "Set VEIL_FORENSICS_SERPAPI_KEY to enable reverse image search."
                ],
                errors=[],
                latency_ms=int((perf_counter() - started) * 1000),
                provider="SerpAPI",
                query_mode="not_configured",
                earliest_known_appearance=None,
                matches=[],
            )

        return ReverseSearchResult(
            status="skipped",
            summary="Reverse-search provider boundary is scaffolded but not wired yet.",
            warnings=[
                "Add the final SerpAPI request flow after the team settles the upload/privacy policy."
            ],
            errors=[],
            latency_ms=int((perf_counter() - started) * 1000),
            provider="SerpAPI",
            query_mode="integration_boundary",
            earliest_known_appearance=None,
            matches=[],
        )
