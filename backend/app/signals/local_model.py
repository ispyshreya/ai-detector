"""Local trained image detector signal.

Wraps Shreya's ResNet checkpoint as a backend signal so the frontend can call
one `/scan` endpoint while still using the in-house model.

The checkpoint was trained only on CIFAKE, whose images are natively 32x32.
For any upload that isn't already 32x32 (i.e. almost every real-world photo),
we downsample to 32x32 before running inference, so the model sees roughly
the same level of fine detail it was trained on rather than a much sharper,
out-of-distribution image. This is a heuristic domain-matching trick, NOT a
validated technique — downsampling a 4000px camera photo introduces different
blur/aliasing than however CIFAKE's own 32x32 images were produced, so we
don't actually know this is well-calibrated (see the Quality Standard section
of VEIL_CLIENT_PROMPT.md: resizes are one of the transformations a detector
must be evaluated against before its probability is presented as meaningful —
that evaluation hasn't been done for this path yet). We flag it in `notes`
and knock down self-reported `confidence` accordingly so downstream fusion
(engine/triangulate.py) leans on it less than a native-resolution result.
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
TRAINING_NATIVE_SIZE = (32, 32)
# Extra confidence discount applied when we had to downsample the input to
# reach TRAINING_NATIVE_SIZE, on top of the triangulation engine's own
# blanket discount for this signal (see engine/triangulate.py).
_RESIZE_CONFIDENCE_FACTOR = 0.7
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

    build_model = _load_build_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        original_size = pil_image.size
        was_resized = original_size != TRAINING_NATIVE_SIZE
        if was_resized:
            pil_image = pil_image.resize(TRAINING_NATIVE_SIZE, Image.Resampling.LANCZOS)

        try:
            model, device, checkpoint = _load_model()
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error_result(str(exc), started)

        tensor = _eval_transform()(pil_image).unsqueeze(0).to(device)
        with torch.no_grad():
            ai_score = torch.sigmoid(model(tensor)).item()

        confidence = ai_score if ai_score >= 0.5 else 1.0 - ai_score
        if was_resized:
            confidence *= _RESIZE_CONFIDENCE_FACTOR
        latency_ms = (time.perf_counter() - started) * 1000.0

        notes = [
            f"Local {get_settings().local_model_name} fake/AI likelihood {round(ai_score * 100)}%",
            f"Checkpoint: {checkpoint}",
        ]
        if was_resized:
            notes.append(
                f"Input was {original_size[0]}x{original_size[1]}, downsampled to "
                f"{TRAINING_NATIVE_SIZE[0]}x{TRAINING_NATIVE_SIZE[1]} to match the checkpoint's "
                "CIFAKE training resolution. This is an unvalidated domain-matching heuristic, "
                "not a calibrated result — confidence is reduced accordingly."
            )

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=max(0.0, min(1.0, float(ai_score))),
            manipulation_score=None,
            confidence=max(0.0, min(1.0, float(confidence))),
            latency_ms=latency_ms,
            notes=notes,
            raw={
                "checkpoint": str(checkpoint),
                "model": get_settings().local_model_name,
                "was_resized": was_resized,
                "original_size": list(original_size),
                "training_native_size": list(TRAINING_NATIVE_SIZE),
            },
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
