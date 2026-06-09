"""FastAPI surface. POST /scan runs every available signal in parallel with a
per-signal timeout and isolates failures (graceful degradation: a failed signal
becomes status=error, the response still returns). Triangulation + LLM layers
plug in at the marked hooks.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import ScanResponse, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal
from app.signals.registry import available_signals

settings = get_settings()
app = FastAPI(title="Veil Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _run_signal(signal: Signal, image: ImageInput) -> SignalResult:
    """Run one signal with a timeout; convert any failure into a SignalResult."""
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            signal.analyze(image), timeout=settings.signal_timeout_seconds
        )
        if result.latency_ms is None:
            result.latency_ms = (time.perf_counter() - start) * 1000
        return result
    except asyncio.TimeoutError:
        return SignalResult(
            name=signal.name,
            signal_class=signal.signal_class,
            status=SignalStatus.error,
            error=f"timed out after {settings.signal_timeout_seconds}s",
            latency_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as exc:  # noqa: BLE001 - backstop; never break the response
        return SignalResult(
            name=signal.name,
            signal_class=signal.signal_class,
            status=SignalStatus.error,
            error=str(exc),
            latency_ms=(time.perf_counter() - start) * 1000,
        )


@app.get("/health")
async def health() -> dict:
    sigs = available_signals()
    return {"status": "ok", "available_signals": [s.name for s in sigs]}


@app.post("/scan", response_model=ScanResponse)
async def scan(media: UploadFile = File(...)) -> ScanResponse:
    data = await media.read()
    image = ImageInput(
        data=data, filename=media.filename, content_type=media.content_type
    )

    signals = available_signals()
    results = await asyncio.gather(*(_run_signal(s, image) for s in signals))

    response = ScanResponse(
        image_sha256=image.sha256,
        filename=image.filename,
        signals=list(results),
    )

    # --- Layer 2 hook: triangulation engine fills response.aggregate ---
    # from app.engine.triangulate import triangulate
    # response.aggregate = triangulate(response.signals)

    # --- Layer 3 hook: LLM explanation fills response.explanation ---
    # from app.explain.llm import explain
    # response.explanation = await explain(response)

    return response
