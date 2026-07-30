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

_DOC_PROMPTS = [
    # document class (index 0): P(document) = softmax prob over these
    "a photo of a document, ID card, passport, form, invoice, or bank statement",
    # non-document classes: ordinary photos that should be rejected
    "an ordinary photograph of a person, place, animal, or object",
    "a photo of scenery, nature, animals, food, sports, or everyday life",
]


@dataclass
class TamperResult:
    score: float
    heatmap_png_b64: str
    bbox: list  # [x, y, w, h]


def _should_flag(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _load_doc_gate() -> Callable[[Image.Image], float]:
    """Return gate(pil) -> P(document) via CLIP zero-shot over _DOC_PROMPTS.

    _DOC_PROMPTS[0] is the single document-class prompt; P(document) = softmax
    prob at index 0 (i.e. competing against all non-document prompts that follow).
    Cached in module-level _DOC_GATE after first call.
    """
    global _DOC_GATE
    if _DOC_GATE is not None:
        return _DOC_GATE
    import torch
    import open_clip

    backbone = getattr(get_settings(), "doctamper_backbone", "ViT-L-14")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer(backbone)
    text = tokenizer(_DOC_PROMPTS)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def gate(pil: Image.Image) -> float:
        with torch.no_grad():
            img = preprocess(pil.convert("RGB")).unsqueeze(0)
            feats = model.encode_image(img)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * feats @ text_features.T).softmax(dim=-1)
            return float(probs[0, 0])  # P(document) = prob of first prompt

    _DOC_GATE = gate
    return _DOC_GATE


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
