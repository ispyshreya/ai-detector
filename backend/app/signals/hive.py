"""Hive AI-generated image/deepfake detector signal.

Hive returns a binary AI-generation score, optional source-generator scores,
and a deepfake score in one classification response. This adapter normalizes
those values into Veil's shared SignalResult contract.

IMPORTANT: this account is on Hive's self-serve V3 Playground tier, not a
sales-provisioned V2 Enterprise project. The two are genuinely different
products with different endpoints, auth schemes, and response shapes:

  * V3 (this signal): POST /api/v3/hive/ai-generated-and-deepfake-content-
    detection, `authorization: Bearer <secret key>`, response is a flat
    {"output": [{"classes": [...]}]}. Rate-limited to 100 requests/day on
    the free tier. https://docs.thehive.ai/reference/ai-generated-and-deepfake-content-detection
  * V2 Enterprise: POST /api/v2/task/sync, `authorization: Token <key>`,
    response is nested under status[0].response.output[0].classes. Requires
    contacting Hive sales for a project. Do NOT point this signal at it
    without also rewriting the parsing below.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

_TIMEOUT_SECONDS = 8.0
_NON_SOURCE_CLASSES = {
    "ai_generated",
    "not_ai_generated",
    "deepfake",
    "none",
    "inconclusive",
    "inconclusive_video",
    "ai_generated_audio",
    "not_ai_generated_audio",
}


class HiveSignal(Signal):
    name = "hive"
    signal_class = SignalClass.detector

    def available(self) -> bool:
        return bool(get_settings().hive_api_key)

    async def analyze(self, image: ImageInput) -> SignalResult:
        settings = get_settings()
        started = time.perf_counter()
        files = {
            "media": (
                image.filename or "upload",
                image.data,
                image.content_type or "application/octet-stream",
            )
        }
        headers = {"Authorization": f"Bearer {settings.hive_api_key}"}

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
                response = await client.post(settings.hive_api_url, headers=headers, files=files)
        except httpx.HTTPError as exc:
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
                f"invalid JSON response: {exc}",
                started,
                latency_ms=latency_ms,
            )

        classes = _classes_from_payload(payload)
        if not classes:
            return self._error_result(
                "hive response did not include classes",
                started,
                latency_ms=latency_ms,
                raw=payload,
            )

        # V3 class items are {"class": <label>, "value": <0..1 confidence>} —
        # NOT "score" (that was the V2 field name).
        class_scores = {
            item["class"]: item["value"]
            for item in classes
            if isinstance(item, dict)
            and isinstance(item.get("class"), str)
            and isinstance(item.get("value"), (int, float))
        }
        ai_score = _coerce_score(class_scores.get("ai_generated"))
        manipulation_score = _coerce_score(class_scores.get("deepfake"))
        source_name, source_score = _top_source(class_scores)

        notes: list[str] = []
        if ai_score is not None:
            notes.append(f"AI-generated likelihood {_pct(ai_score)}")
        if source_name and source_score is not None and source_score >= 0.10:
            notes.append(f"Most likely generator: {source_name} ({_pct(source_score)})")
        if manipulation_score is not None:
            notes.append(f"Deepfake likelihood {_pct(manipulation_score)}")

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


def _classes_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """V3 response shape: {"output": [{"classes": [...], "extra": [...]}]}.

    `output` has one entry per video frame; for a single image request there
    is exactly one entry, so we read output[0] (per Hive's own "Single Image"
    guidance).
    """
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, list) and output and isinstance(output[0], dict):
        classes = output[0].get("classes")
        if isinstance(classes, list):
            return classes
    return []


def _top_source(class_scores: dict[str, float]) -> tuple[str | None, float | None]:
    source_scores = {
        name: score for name, score in class_scores.items() if name not in _NON_SOURCE_CLASSES
    }
    if not source_scores:
        return None, None
    name, score = max(source_scores.items(), key=lambda item: item[1])
    return name, _coerce_score(score)


def _coerce_score(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def _pct(score: float) -> str:
    return f"{round(score * 100)}%"
