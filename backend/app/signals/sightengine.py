"""Sightengine detector signal (Layer 1).

Calls Sightengine's commercial check endpoint with the `genai` and `deepfake`
models. `genai` answers Axis 1 ("is this AI-generated?") and feeds ai_score;
`deepfake` is a face-manipulation signal of a real person, so it feeds the
Axis 2 manipulation_score. The API user/secret are server-side secrets held in
config — the browser never sees them.
"""

from __future__ import annotations

import time

import httpx

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

# Sightengine synchronous image check endpoint.
_ENDPOINT = "https://api.sightengine.com/1.0/check.json"
# Per-signal timeout; the runner caps signals at ~6s (see config), so stay under it.
_TIMEOUT_SECONDS = 6.0


class SightengineSignal(Signal):
    name = "sightengine"
    signal_class = SignalClass.detector

    def available(self) -> bool:
        """Configured only when both credentials are present."""
        settings = get_settings()
        return bool(settings.sightengine_api_user and settings.sightengine_api_secret)

    async def analyze(self, image: ImageInput) -> SignalResult:
        settings = get_settings()
        started = time.perf_counter()

        # Multipart form: text fields + the image bytes as the `media` part.
        data = {
            "api_user": settings.sightengine_api_user,
            "api_secret": settings.sightengine_api_secret,
            "models": "genai,deepfake",
        }
        files = {
            "media": (
                image.filename or "upload",
                image.data,
                image.content_type or "application/octet-stream",
            )
        }

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(_ENDPOINT, data=data, files=files)
        except httpx.HTTPError as exc:
            # Network/timeout failures are expected — report, don't raise.
            return self._error_result(f"request failed: {exc}", started)

        latency_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code != 200:
            return self._error_result(
                f"HTTP {response.status_code}: {response.text[:200]}",
                started,
                latency_ms=latency_ms,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            return self._error_result(
                f"invalid JSON response: {exc}", started, latency_ms=latency_ms
            )

        # Sightengine signals failure in-band with status=="failure".
        if payload.get("status") == "failure":
            message = payload.get("error", {}).get("message", "unknown error")
            return self._error_result(
                f"sightengine failure: {message}",
                started,
                latency_ms=latency_ms,
                raw=payload,
            )

        # Pull the two scores out of the `type` block (both floats in [0, 1]).
        type_block = payload.get("type", {}) or {}
        ai_score = _coerce_score(type_block.get("ai_generated"))
        manipulation_score = _coerce_score(type_block.get("deepfake"))

        notes: list[str] = []
        if ai_score is not None:
            notes.append(f"AI-generated likelihood {_pct(ai_score)}")
        if manipulation_score is not None:
            notes.append(f"Face-manipulation likelihood {_pct(manipulation_score)}")

        # Self-reported confidence: how decisive the AI verdict is (distance from 0.5).
        confidence = None
        if ai_score is not None:
            confidence = min(1.0, abs(ai_score - 0.5) * 2.0)

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=ai_score,
            manipulation_score=manipulation_score,
            confidence=confidence,
            latency_ms=latency_ms,
            notes=notes,
            raw=payload,
        )

    def _error_result(
        self,
        error: str,
        started: float,
        *,
        latency_ms: float | None = None,
        raw: dict | None = None,
    ) -> SignalResult:
        """Build a status=error result without raising."""
        if latency_ms is None:
            latency_ms = (time.perf_counter() - started) * 1000.0
        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=latency_ms,
            error=error,
            raw=raw,
        )


def _coerce_score(value: object) -> float | None:
    """Clamp an upstream score into [0, 1]; return None if absent/non-numeric."""
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _pct(score: float) -> str:
    return f"{round(score * 100)}%"
