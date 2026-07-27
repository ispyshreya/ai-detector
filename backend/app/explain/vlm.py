"""Local vision-language explanations for detector results.

The VLM is deliberately separate from scoring: it describes visible evidence,
but it never changes the detector score. Loading is lazy because the model is
large and many backend deployments only need the signal layer.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import torch
from PIL import Image

from app.config import get_settings


_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOAD_LOCK = threading.Lock()


@dataclass(frozen=True)
class Explanation:
    text: str
    model: str
    used_fallback: bool
    note: str


def build_prompt(score: float | None) -> str:
    score_context = (
        "No detector score is available."
        if score is None
        else f"The detector's AI-generation risk score is {score:.1%}."
    )
    return (
        f"{score_context} Inspect the uploaded image for visible authenticity "
        "evidence. The score is context, not proof, and you must disagree with it "
        "when the pixels do not support it. Return one to three short bullet "
        "points. Every bullet must name a concrete visible object or region and "
        "describe an observable detail. Check text, anatomy, object boundaries, "
        "lighting, shadows, reflections, and repeated patterns only when present. "
        "Never infer evidence from something absent or obscured. Do not mention "
        "the product name Veil, app chrome, the detector, or generic AI artifacts. "
        "Do not claim the image is authentic or fake with certainty. If there is "
        "no specific visible evidence, return exactly: "
        "'- No clear visual artifacts were found; verify the source before trusting it.'"
    )


def normalize_output(text: str) -> str:
    text = text.strip()
    for marker in ("Assistant:", "<end_of_utterance>"):
        text = text.replace(marker, "")

    lines: list[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw_line).strip()
        if not line:
            continue
        if any(
            phrase in line.lower()
            for phrase in (
                "the word veil",
                "the veil",
                "no hands",
                "no fingers",
                "not visible",
                "cannot see",
                "generic ai artifact",
            )
        ):
            continue
        lines.append(f"- {line}")
        if len(lines) == 3:
            break
    return "\n".join(lines)


def fallback_explanation(score: float | None) -> str:
    if score is not None and score >= 0.7:
        return (
            "- No clear visual artifacts were found; verify the source before trusting it.\n"
            "- The detector score is high, so seek independent proof for high-stakes decisions."
        )
    return (
        "- No clear visual artifacts were found; verify the source before trusting it.\n"
        "- Image appearance alone cannot verify the sender, source, or surrounding story."
    )


def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR, _DEVICE

    with _LOAD_LOCK:
        if _MODEL is not None:
            return _MODEL, _PROCESSOR, _DEVICE

        from transformers import AutoModelForImageTextToText, AutoProcessor

        settings = get_settings()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(settings.vlm_model_id)
        model = AutoModelForImageTextToText.from_pretrained(
            settings.vlm_model_id,
            torch_dtype=dtype,
            device_map={"": device.type},
        )
        model.eval()
        _MODEL, _PROCESSOR, _DEVICE = model, processor, device
        return model, processor, device


def explain_image(image: Image.Image, score: float | None) -> Explanation:
    settings = get_settings()
    model, processor, device = _load_model()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": build_prompt(score)},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.vlm_max_new_tokens,
            do_sample=False,
        )
    output = output[:, inputs["input_ids"].shape[-1]:]
    generated = processor.batch_decode(output, skip_special_tokens=True)[0]
    normalized = normalize_output(generated)
    used_fallback = len(normalized) < 24
    return Explanation(
        text=fallback_explanation(score) if used_fallback else normalized,
        model=settings.vlm_model_id,
        used_fallback=used_fallback,
        note=(
            "The visual model found no reliable image-specific evidence, so Veil showed cautious guidance."
            if used_fallback
            else "Visual explanations are AI-generated warning signs, not proof."
        ),
    )
