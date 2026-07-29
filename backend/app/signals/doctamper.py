"""Document-tamper signal.

For DOCUMENT images only (a CLIP zero-shot document-gate rejects ordinary
photos), runs Error Level Analysis to localize likely edits and returns a
`manipulation_score` + a base64 heatmap + the hottest-region bbox. Its own axis:
`ai_score` stays None. Silent (null score) on non-documents. A flag routes the
user to "verify with the issuer" — a lead, never proof.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

_DOC_GATE = None  # cached doc-gate callable


@dataclass
class TamperResult:
    score: float
    heatmap_png_b64: str
    bbox: list  # [x, y, w, h]


def _should_flag(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _load_doc_gate() -> Callable[[Image.Image], float]:
    """Return gate(pil) -> P(document). Implemented in Task 4."""
    raise NotImplementedError("document-gate wired in Task 4")


def _localize_tamper(pil: Image.Image) -> "TamperResult":
    """ELA tamper localization -> score + heatmap + bbox. Implemented in Task 5."""
    raise NotImplementedError("tamper localizer wired in Task 5")


class DocTamperSignal(Signal):
    name = "doctamper"
    signal_class = SignalClass.manipulation

    def available(self) -> bool:
        return bool(getattr(get_settings(), "doctamper_enabled", False))

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()
        settings = get_settings()
        try:
            pil = Image.open(BytesIO(image.data)).convert("RGB")
        except UnidentifiedImageError:
            return self._error("uploaded file is not a valid image", started)

        try:
            gate = _load_doc_gate()
            doc_prob = float(gate(pil))
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error(str(exc), started)

        gate_threshold = float(getattr(settings, "doctamper_doc_gate_threshold", 0.55))
        if doc_prob < gate_threshold:
            return SignalResult(
                name=self.name, signal_class=self.signal_class,
                status=SignalStatus.ok, ai_score=None, manipulation_score=None,
                confidence=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                notes=["not a document; tamper check not applicable"],
                raw={"doc_prob": doc_prob},
            )

        try:
            result = _localize_tamper(pil)
        except Exception as exc:  # noqa: BLE001
            return self._error(str(exc), started)

        threshold = float(getattr(settings, "doctamper_threshold", 0.6))
        flagged = _should_flag(result.score, threshold)
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.ok, ai_score=None,
            manipulation_score=max(0.0, min(1.0, result.score)),
            confidence=result.score if flagged else 1.0 - result.score,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=[
                ("Possible document edit detected — verify directly with the "
                 "issuing institution." if flagged
                 else "No strong sign of a localized document edit.")
                + f" ({round(result.score * 100)}%)",
            ],
            raw={"doc_prob": doc_prob, "heatmap_png_b64": result.heatmap_png_b64,
                 "bbox": result.bbox, "flagged": flagged},
        )

    def _error(self, error: str, started: float) -> SignalResult:
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=(time.perf_counter() - started) * 1000.0, error=error,
        )
