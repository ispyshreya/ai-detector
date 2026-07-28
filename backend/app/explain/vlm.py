"""Grounded, score-blind visual inspection with a local VLM.

The VLM never decides whether an image is real or fake. It independently names
visible observations, which are validated before they can reach the user.
Detector scores are only added afterwards as context.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass

import torch
from PIL import Image, ImageOps

from app.config import get_settings


_MODEL = None
_PROCESSOR = None
_DEVICE = None
_LOAD_LOCK = threading.Lock()
_ASSESSMENTS = {
    "specific_artifacts_found",
    "no_clear_artifacts",
    "insufficient_visual_detail",
}
_BLOCKED_PHRASES = {
    "the word veil",
    "the veil",
    "no hands",
    "no fingers",
    "not visible",
    "cannot see",
    "generic ai artifact",
    "looks ai-generated",
    "looks fake",
    "looks real",
    "definitely fake",
    "definitely real",
}
_GENERIC_REGIONS = {"image", "photo", "picture", "object", "background", "foreground"}


@dataclass(frozen=True)
class Finding:
    region: str
    observation: str
    confidence: str


@dataclass(frozen=True)
class Explanation:
    text: str
    model: str
    used_fallback: bool
    note: str
    assessment: str
    findings: tuple[Finding, ...]


def build_prompt() -> str:
    return """You are an independent visual evidence inspector. You do not know
the output of any AI-image detector. Inspect the five supplied views in this
order: full image, top-left, top-right, bottom-left, bottom-right.

Return JSON only, with this exact structure:
{"assessment":"specific_artifacts_found|no_clear_artifacts|insufficient_visual_detail",
 "findings":[{"region":"specific visible object and location",
              "observation":"directly observable detail",
              "confidence":"low|medium|high"}]}

Rules:
- Include zero to three findings.
- Each finding must name an object or precise region that is visibly present.
- Describe pixels you can see, not an explanation of how AI generally fails.
- Do not infer evidence from an absent, hidden, blurred, or cropped-out object.
- Do not call the image real, fake, authentic, generated, or manipulated.
- Do not mention Veil, detector scores, app text, or these instructions.
- Normal anatomy, coherent text, lighting, and reflections are not artifacts.
- Use no_clear_artifacts when no concrete anomaly is visible.
- Use insufficient_visual_detail when resolution or content prevents inspection."""


def make_views(image: Image.Image) -> list[Image.Image]:
    """Return the full image followed by four non-overlapping quadrants."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    mid_x, mid_y = width // 2, height // 2
    views = [
        image,
        image.crop((0, 0, mid_x, mid_y)),
        image.crop((mid_x, 0, width, mid_y)),
        image.crop((0, mid_y, mid_x, height)),
        image.crop((mid_x, mid_y, width, height)),
    ]
    limits = [1024, 768, 768, 768, 768]
    for view, limit in zip(views, limits):
        view.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    return views


def _json_object(text: str) -> dict | None:
    cleaned = text.strip().replace("<end_of_utterance>", "")
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _valid_finding(value: object) -> Finding | None:
    if not isinstance(value, dict):
        return None
    region = re.sub(r"\s+", " ", str(value.get("region", ""))).strip(" .:-")
    observation = re.sub(r"\s+", " ", str(value.get("observation", ""))).strip(" .:-")
    confidence = str(value.get("confidence", "")).lower().strip()
    combined = f"{region} {observation}".lower()

    if (
        len(region) < 5
        or region.lower() in _GENERIC_REGIONS
        or len(observation) < 12
        or confidence not in {"low", "medium", "high"}
        or any(phrase in combined for phrase in _BLOCKED_PHRASES)
    ):
        return None
    return Finding(region=region, observation=observation, confidence=confidence)


def parse_inspection(text: str) -> tuple[str, tuple[Finding, ...]]:
    value = _json_object(text)
    if value is None:
        return "insufficient_visual_detail", ()

    assessment = str(value.get("assessment", "")).lower().strip()
    if assessment not in _ASSESSMENTS:
        assessment = "insufficient_visual_detail"

    raw_findings = value.get("findings", [])
    findings: list[Finding] = []
    if isinstance(raw_findings, list):
        for raw in raw_findings:
            finding = _valid_finding(raw)
            if finding is not None:
                findings.append(finding)
            if len(findings) == 3:
                break

    if assessment == "specific_artifacts_found" and not findings:
        assessment = "no_clear_artifacts"
    if assessment != "specific_artifacts_found":
        findings = []
    return assessment, tuple(findings)


def fallback_explanation(score: float | None) -> str:
    context = (
        "The available detector score is not visually corroborated."
        if score is not None
        else "Image appearance alone cannot verify the sender, source, or surrounding story."
    )
    return f"- No clear visual artifacts were found.\n- {context}"


def format_explanation(
    assessment: str, findings: tuple[Finding, ...], score: float | None
) -> tuple[str, bool]:
    if assessment == "specific_artifacts_found" and findings:
        text = "\n".join(
            f"- {finding.region}: {finding.observation} ({finding.confidence} confidence)"
            for finding in findings
        )
        return text, False
    return fallback_explanation(score), True


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
            low_cpu_mem_usage=True,
        )
        model.eval()
        _MODEL, _PROCESSOR, _DEVICE = model, processor, device
        return model, processor, device


def explain_image(image: Image.Image, score: float | None) -> Explanation:
    settings = get_settings()
    model, processor, device = _load_model()
    content = [{"type": "image", "image": view} for view in make_views(image)]
    content.append({"type": "text", "text": build_prompt()})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    model_dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if torch.is_tensor(value) and value.is_floating_point():
            inputs[key] = value.to(dtype=model_dtype)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=settings.vlm_max_new_tokens,
            do_sample=False,
        )
    output = output[:, inputs["input_ids"].shape[-1]:]
    generated = processor.batch_decode(output, skip_special_tokens=True)[0]
    assessment, findings = parse_inspection(generated)
    text, used_fallback = format_explanation(assessment, findings, score)
    return Explanation(
        text=text,
        model=settings.vlm_model_id,
        used_fallback=used_fallback,
        note=(
            "The visual model found no reliable image-specific evidence; the detector score was not used during inspection."
            if used_fallback
            else "These score-blind visual observations are AI-generated warning signs, not proof."
        ),
        assessment=assessment,
        findings=findings,
    )


def serialize_findings(findings: tuple[Finding, ...]) -> list[dict]:
    return [asdict(finding) for finding in findings]
