from __future__ import annotations

from shutil import which
from time import perf_counter

from app.schemas import C2PAResult
from app.services.base import AnalysisContext


class C2PAModule:
    name = "c2pa"

    async def analyze(self, context: AnalysisContext) -> C2PAResult:
        started = perf_counter()
        if not which("c2patool") and not which("c2pa"):
            return C2PAResult(
                status="skipped",
                summary="C2PA verification tool not configured.",
                warnings=[
                    "Install a C2PA CLI tool and wire its invocation into this module."
                ],
                errors=[],
                latency_ms=int((perf_counter() - started) * 1000),
                manifest_present=False,
                validation_status="not_configured",
                signer=None,
                issuer=None,
                assertions=[],
            )

        return C2PAResult(
            status="skipped",
            summary="C2PA tool detected, but verification command is intentionally not guessed in this scaffold.",
            warnings=[
                "Finalize the c2pa-rs CLI invocation and map its output into this response model."
            ],
            errors=[],
            latency_ms=int((perf_counter() - started) * 1000),
            manifest_present=False,
            validation_status="integration_boundary",
            signer=None,
            issuer=None,
            assertions=[],
        )
