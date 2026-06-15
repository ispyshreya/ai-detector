"""Hive AI-generated image/deepfake detector signal.

Hive returns a binary AI-generation score, optional source-generator scores,
and a deepfake score in one classification response. This adapter normalizes
those values into Veil's shared SignalResult contract.
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
        headers = {"Authorization": f"Token {settings.hive_api_key}"}

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

        task = _first_task(payload)
        status = ((task.get("status") if isinstance(task, dict) else {}) or {})
        if isinstance(status, dict) and status.get("code") not in (None, "0", 0):
            message = status.get("message", "unknown error")
            return self._error_result(
                f"hive failure: {message}",
                started,
                latency_ms=latency_ms,
                raw=payload,
            )

        classes = _classes_from_task(task)
        if not classes:
            return self._error_result(
                "hive response did not include classes",
                started,
                latency_ms=latency_ms,
                raw=payload,
            )

        class_scores = {
            item["class"]: item["score"]
            for item in classes
            if isinstance(item, dict)
            and isinstance(item.get("class"), str)
            and isinstance(item.get("score"), (int, float))
        }
        ai_score = _coerce_score(class_scores.get("ai_generated"))
        manipulation_score = _coerce_score(class_scores.get("deepfake"))
        source_name, source_score = _top_source(class_scores)
        tags = _algorithmic_tags(task)

        notes: list[str] = []
        if ai_score is not None:
            notes.append(f"AI-generated likelihood {_pct(ai_score)}")
        if source_name and source_score is not None and source_score >= 0.10:
            notes.append(f"Most likely generator: {source_name} ({_pct(source_score)})")
        if manipulation_score is not None:
            notes.append(f"Deepfake likelihood {_pct(manipulation_score)}")
        if tags.get("c2pa"):
            c2pa = tags["c2pa"]
            generator = c2pa.get("claim_generator") or c2pa.get("actions_software_agent")
            if generator:
                notes.append(f"C2PA metadata references {generator}")

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


def _first_task(payload: dict[str, Any]) -> dict[str, Any]:
    status = payload.get("status")
    if isinstance(status, list) and status and isinstance(status[0], dict):
        return status[0]
    return payload


def _classes_from_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    response = task.get("response") if isinstance(task, dict) else None
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list) and output and isinstance(output[0], dict):
        classes = output[0].get("classes")
        if isinstance(classes, list):
            return classes
    return []


def _algorithmic_tags(task: dict[str, Any]) -> dict[str, Any]:
    response = task.get("response") if isinstance(task, dict) else None
    output = response.get("output") if isinstance(response, dict) else None
    if isinstance(output, list) and output and isinstance(output[0], dict):
        tags = output[0].get("algorithmic_tags")
        if isinstance(tags, dict):
            return tags
    return {}


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
