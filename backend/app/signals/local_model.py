"""Local trained image detector signal.

Wraps Shreya's ResNet checkpoint as a backend signal so the frontend can call
one `/scan` endpoint while still using the in-house model.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.config import get_settings
from app.config import REPO_ROOT
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

IMAGE_SIZE = 224
_MODEL = None
_DEVICE = None
_CHECKPOINT = None


def _load_build_model():
    train_path = REPO_ROOT / "detector-trainer" / "train.py"
    if not train_path.exists():
        raise FileNotFoundError("detector-trainer/train.py was not found")

    spec = importlib.util.spec_from_file_location("veil_detector_train", train_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {train_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_model


def _load_clip_builder():
    """Dynamically load ``build_clip_detector`` from the detector-trainer package.

    Mirrors :func:`_load_build_model`; kept separate so the CLIP path (frozen
    ViT-L/14 + trained head) can be swapped in without importing open_clip until
    a CLIP checkpoint is actually served.
    """
    clip_path = REPO_ROOT / "detector-trainer" / "models" / "clip_head.py"
    if not clip_path.exists():
        raise FileNotFoundError("detector-trainer/models/clip_head.py was not found")

    spec = importlib.util.spec_from_file_location("veil_clip_head", clip_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {clip_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_clip_detector


def _confidence(ai_score: float, threshold: float) -> float:
    """Confidence as normalized distance from the calibrated decision boundary.

    0.0 exactly at the threshold (maximally uncertain), rising to 1.0 at either
    extreme. Pivoting on the calibrated threshold — not a hardcoded 0.5 — means a
    real photo scoring just under the boundary reads as confidently REAL.
    """
    threshold = min(max(threshold, 1e-6), 1.0 - 1e-6)
    if ai_score >= threshold:
        return (ai_score - threshold) / (1.0 - threshold)
    return (threshold - ai_score) / threshold


def _eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load_model():
    global _MODEL, _DEVICE, _CHECKPOINT

    settings = get_settings()
    checkpoint = _resolve_checkpoint(settings.local_model_checkpoint)
    if _MODEL is not None and _CHECKPOINT == checkpoint:
        return _MODEL, _DEVICE, checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if getattr(settings, "local_model_type", "resnet") == "clip":
        # Frozen CLIP ViT-L/14 + trained head. The head checkpoint's sibling
        # config.json records {backbone_name, feat_dim, head_type, hidden}.
        build_clip_detector = _load_clip_builder()
        config_path = checkpoint.parent / "config.json"
        model = build_clip_detector(
            config=config_path, head_ckpt=checkpoint, device=str(device)
        )
    else:
        build_model = _load_build_model()
        model = build_model(settings.local_model_name, pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.to(device)
        model.eval()

    _MODEL = model
    _DEVICE = device
    _CHECKPOINT = checkpoint
    return model, device, checkpoint


class LocalModelSignal(Signal):
    name = "local"
    signal_class = SignalClass.detector

    def available(self) -> bool:
        return _resolve_checkpoint(get_settings().local_model_checkpoint).exists()

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()

        try:
            pil_image = Image.open(BytesIO(image.data)).convert("RGB")
        except UnidentifiedImageError:
            return self._error_result("uploaded file is not a valid image", started)

        try:
            model, device, checkpoint = _load_model()
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error_result(str(exc), started)

        tensor = _eval_transform()(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            ai_score = torch.sigmoid(model(tensor)).item()

        threshold = float(getattr(get_settings(), "local_model_threshold", 0.5))
        confidence = _confidence(ai_score, threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=max(0.0, min(1.0, float(ai_score))),
            manipulation_score=None,
            confidence=max(0.0, min(1.0, float(confidence))),
            latency_ms=latency_ms,
            notes=[
                f"Local {get_settings().local_model_name} fake/AI likelihood {round(ai_score * 100)}%",
                f"Checkpoint: {checkpoint}",
            ],
            raw={"checkpoint": str(checkpoint), "model": get_settings().local_model_name},
        )

    def _error_result(self, error: str, started: float) -> SignalResult:
        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            error=error,
        )


def _resolve_checkpoint(checkpoint: str):
    path = Path(checkpoint)
    return path if path.is_absolute() else REPO_ROOT / path
