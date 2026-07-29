"""Face-swap / deepfake-face signal.

Runs a 3-step on-device pipeline (face-gate -> crop largest face -> classify) and
reports a `manipulation_score`. Silent (null score) when no face is present, so it
never affects the verdict for non-face images. Its own axis: `ai_score` stays None.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Callable

from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

_MODEL = None       # cached classifier callable
_DETECTOR = None    # cached detector callable


def _should_flag(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _load_face_detector() -> Callable[[Image.Image], list]:
    """Return detect(pil) -> list[PIL face crop] using OpenCV Haar cascade."""
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    import cv2
    import numpy as np

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    def detect(pil: Image.Image) -> list:
        arr = np.array(pil.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(48, 48))
        crops = []
        for (x, y, w, h) in boxes:
            m = int(0.25 * max(w, h))  # margin so context isn't cut off
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(pil.width, x + w + m), min(pil.height, y + h + m)
            crops.append(pil.crop((x0, y0, x1, y1)))
        return crops

    _DETECTOR = detect
    return _DETECTOR


_FAKE_LABELS = {"fake", "deepfake", "ai", "manipulated", "spoof", "1"}


def _p_fake(preds: list) -> float:
    """Map HF image-classification output ([{label, score}, ...]) to P(fake)."""
    for p in preds:
        if str(p["label"]).strip().lower() in _FAKE_LABELS:
            return float(p["score"])
    # Fallback: if only a 'real'-type label is present, invert it.
    top = max(preds, key=lambda p: p["score"])
    return 1.0 - float(top["score"]) if "real" in str(top["label"]).lower() else float(top["score"])


def _build_pipeline(model_id: str):
    from transformers import pipeline
    return pipeline("image-classification", model=model_id, top_k=None)


def _load_classifier() -> Callable[[Image.Image], float]:
    """Return classify(pil_face) -> p_fake in [0,1]. Implemented in Task 5."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    pipe = _build_pipeline(get_settings().faceswap_model_id)

    def classify(crop: Image.Image) -> float:
        return _p_fake(pipe(crop.convert("RGB")))

    _MODEL = classify
    return _MODEL


class FaceSwapSignal(Signal):
    name = "faceswap"
    signal_class = SignalClass.manipulation

    def available(self) -> bool:
        return bool(getattr(get_settings(), "faceswap_enabled", False))

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()
        settings = get_settings()
        try:
            pil = Image.open(BytesIO(image.data)).convert("RGB")
        except UnidentifiedImageError:
            return self._error("uploaded file is not a valid image", started)

        try:
            detect = _load_face_detector()
            faces = detect(pil)
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error(str(exc), started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not faces:
            return SignalResult(
                name=self.name, signal_class=self.signal_class,
                status=SignalStatus.ok, ai_score=None, manipulation_score=None,
                confidence=None, latency_ms=latency_ms,
                notes=["no face detected; face-swap check not applicable"],
                raw={"faces": 0},
            )

        try:
            classify = _load_classifier()
            crop = max(faces, key=lambda f: f.size[0] * f.size[1])
            score = float(classify(crop))
        except Exception as exc:  # noqa: BLE001
            return self._error(str(exc), started)

        threshold = float(getattr(settings, "faceswap_threshold", 0.7))
        flagged = _should_flag(score, threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.ok, ai_score=None,
            manipulation_score=max(0.0, min(1.0, score)),
            confidence=score if flagged else 1.0 - score,
            latency_ms=latency_ms,
            notes=[
                ("Possible face manipulation detected" if flagged
                 else "Face looks unmanipulated") + f" ({round(score * 100)}%)",
            ],
            raw={"faces": len(faces), "model": settings.faceswap_model_id,
                 "flagged": flagged},
        )

    def _error(self, error: str, started: float) -> SignalResult:
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=(time.perf_counter() - started) * 1000.0, error=error,
        )
