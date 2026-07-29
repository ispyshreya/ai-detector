"""Grounded, score-blind visual inspection via Hive's Vision Language Model.

The VLM never decides whether an image is real or fake. It independently names
visible observations, which are validated before they can reach the user.
Detector scores are only added afterwards as context.

Uses Hive's hosted VLM (OpenAI-compatible /v3/chat/completions) rather than a
locally-loaded model, so there is no multi-GB weights download or cold-start
inference latency. Auth reuses the same V3 Secret Key as the `hive` detector
signal (see signals/hive.py) — that key's permission policy must also grant
`hive:CallApi` on the `hive/vision-language-model` resource, which is a
separate grant from the AI-generated/deepfake detection model's resource.
https://docs.thehive.ai/docs/hive-vision-language-model-vlm

ONE QUESTION PER CATEGORY, NOT ONE MEGA-PROMPT. Diagnosed empirically: this
model reliably answers a single, isolated, direct yes/no question (e.g. "is
this text misspelled?") but collapses to "everything normal" on any
multi-category or open-ended "list anything unusual" prompt — even when its
own transcription in the same response contains the anomaly. Verified even a
single-image, non-JSON "find issues" prompt fails, so it isn't the JSON
schema or the 5-view input; it's specifically batched/spontaneous judgment.
So each artifact category gets its own isolated API call, run concurrently,
and results are combined afterward — not left to the model to self-aggregate.
"""

from __future__ import annotations

import asyncio
import base64
import re
from dataclasses import asdict, dataclass
from io import BytesIO

import httpx
from PIL import Image, ImageOps

from app.config import get_settings

_ENDPOINT = "https://api.thehive.ai/api/v3/chat/completions"
_TIMEOUT_SECONDS = 30.0

# One isolated, direct yes/no question per category (see module docstring for
# why this must NOT be combined into a single multi-category prompt). Order
# matches VEIL_CLIENT_PROMPT.md's VLM inspection list.
_CATEGORY_QUESTIONS: dict[str, str] = {
    "Text/writing": (
        "Look closely at any visible text or writing in this image. Is every word "
        "spelled correctly and does it read as coherent, real language? Answer with "
        "YES if all text is normal, or NO if you see specific garbled, misspelled, or "
        "nonsensical text. If NO, state exactly which text and where it appears. If "
        "there is no text in the image, answer YES."
    ),
    "Hands/limbs/faces": (
        "Look closely at any hands, fingers, limbs, ears, teeth, or faces in this "
        "image. Do they have anatomically normal counts, shapes, and proportions? "
        "Answer with YES if everything looks anatomically normal, or NO if something "
        "is visibly wrong (e.g. wrong finger count, merged or warped digits, "
        "asymmetric features). If NO, state exactly what you see and where. If there "
        "are no people, hands, or faces visible, answer YES."
    ),
    "Perspective/geometry": (
        "Look closely at perspective, proportions, and geometry WITHIN the "
        "photographic scene in this image -- vanishing points, converging lines, "
        "relative scale between objects that are actually part of the same scene, "
        "and symmetry. If this image is a screenshot of a webpage or app, judge "
        "only the photo/content it displays, not its size or placement relative to "
        "surrounding UI chrome, text, or buttons -- that comparison is meaningless. "
        "Does the scene itself look spatially consistent, the way it would in a "
        "real photo? Answer with YES if consistent, or NO if something is visibly "
        "wrong (e.g. impossible perspective, two objects that should be the same "
        "size but aren't, warped or inconsistent geometry). If NO, state exactly "
        "what you see and where."
    ),
    "Reflections/lighting": (
        "Look closely at reflections, lighting, and shadows WITHIN the "
        "photographic scene in this image. Are they physically consistent with "
        "each other and the scene? Answer with YES if consistent, or NO if "
        "something is visibly wrong (e.g. a reflection that doesn't match what it "
        "should reflect, shadows going the wrong way, mismatched lighting "
        "direction). If NO, state exactly what you see and where."
    ),
    "Patterns/edges": (
        "You will see one photo repeated as a full view plus four zoomed-in corner "
        "crops of that SAME photo -- this is a viewing aid, not multiple different "
        "images, so never mention the crops/collage/grid itself. Within the "
        "photo's actual content, look at repeated patterns (e.g. fabric, tiles, "
        "foliage) and object edges/boundaries. Do they look natural, or do you see "
        "unnatural tiling, duplication, blurring, or melted-looking boundaries where "
        "objects meet? Answer with YES if everything looks structurally natural, or "
        "NO if something is visibly wrong. If NO, state exactly what you see and "
        "where in the photo's content (not which crop)."
    ),
}

# The model occasionally describes our own 5-view grid (full image + 4
# quadrants) as if it were the subject of the image ("a collage of multiple
# images") rather than describing the underlying photo. That's leakage of our
# own prompting technique, not evidence about the image — discard it.
_TECHNIQUE_LEAK_PHRASES = (
    "collage",
    "multiple images",
    "four quadrant",
    "zoomed-in view",
    "zoomed in view",
    "grid of images",
    "composite of images",
    "several images",
)
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

# The model sometimes answers NO to the wrong implicit question -- e.g.
# "NO, there is no text in the image" -- describing an ABSENCE of something
# rather than naming a genuine visible anomaly. An absence is not evidence;
# the category prompts already say "if there is no X, answer YES", so this
# is a formatting slip we must catch downstream, not real signal.
_ABSENCE_PHRASES = (
    "there is no",
    "there are no",
    "there's no",
    "no text",
    "no visible",
    "nothing unusual",
    "no unusual",
    "not present",
    "is absent",
    "are absent",
    "n/a",
)
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


def _parse_category_answer(text: str) -> str | None:
    """Extract the problem description from one category's raw answer, or
    None if the category reported normal (YES) or gave an unusable answer."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    lowered = cleaned.lower()

    if any(phrase in lowered for phrase in _TECHNIQUE_LEAK_PHRASES):
        return None
    if lowered.startswith("yes"):
        return None
    if not lowered.startswith("no"):
        # Didn't clearly answer yes/no -- treat as inconclusive, not a finding.
        return None

    observation = re.sub(r"^no[,.\-:]?\s*", "", cleaned, flags=re.IGNORECASE).strip(" .:-")
    if not observation:
        return None
    if any(phrase in observation.lower() for phrase in _ABSENCE_PHRASES):
        return None
    return observation


def _valid_finding(region: str, observation: str) -> Finding | None:
    region = re.sub(r"\s+", " ", region).strip(" .:-")
    observation = re.sub(r"\s+", " ", observation).strip(" .:-")
    combined = f"{region} {observation}".lower()

    if (
        len(region) < 5
        or region.lower() in _GENERIC_REGIONS
        or len(observation) < 12
        or any(phrase in combined for phrase in _BLOCKED_PHRASES)
    ):
        return None
    return Finding(region=region, observation=observation, confidence="medium")


def parse_category_findings(responses: dict[str, str]) -> tuple[str, tuple[Finding, ...]]:
    """Combine per-category yes/no answers into one (assessment, findings)."""
    findings: list[Finding] = []
    for category, raw in responses.items():
        observation = _parse_category_answer(raw)
        if observation is None:
            continue
        finding = _valid_finding(category, observation)
        if finding is not None:
            findings.append(finding)
        if len(findings) == 3:
            break

    assessment = "specific_artifacts_found" if findings else "no_clear_artifacts"
    return assessment, tuple(findings)


def fallback_explanation(score: float | None) -> str:
    """No visual findings never gets to say "looks authentic" when the fused
    detector score already says otherwise -- that reads as a flat
    contradiction next to a High Risk verdict. Only claim visual authenticity
    when the score itself is low; otherwise stay neutral and defer to the
    detector evidence, since a clean visual check doesn't clear a suspicious
    score (modern generators often leave nothing visible to find)."""
    if score is None:
        return (
            "No clear visual artifacts were found.\n"
            "- Image appearance alone cannot verify the sender, source, or surrounding story."
        )
    if score >= 0.4:
        return (
            "No specific visual artifacts were found in this check, but that does not mean "
            "the image is authentic.\n"
            "- Modern AI generators often leave no visible trace -- weigh the detector score "
            "above more heavily than this visual check for this image."
        )
    return (
        "This looks visually authentic: no unusual text, anatomy, perspective, reflections, "
        "or patterns were found.\n"
        "- This is consistent with the low-risk detector score above."
    )


def format_explanation(
    assessment: str, findings: tuple[Finding, ...], score: float | None
) -> tuple[str, bool]:
    """Render a plain-language, grounded lean -- "likely AI-generated because X"
    when something concrete was found, "looks authentic" when nothing was --
    not a neutral list of facts with no conclusion (that's what product wants:
    a reasoned opinion tied to specific visible evidence, never a bare score
    echo or a claim of certainty)."""
    if assessment == "specific_artifacts_found" and findings:
        headline = (
            "This looks likely AI-generated or manipulated, based on what's visible:"
        )
        bullets = "\n".join(
            f"- {finding.region}: {finding.observation}" for finding in findings
        )
        return f"{headline}\n{bullets}", False
    return fallback_explanation(score), True


def _data_url(view: Image.Image) -> str:
    buffer = BytesIO()
    view.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


async def _call_hive_vlm(views: list[Image.Image], prompt: str) -> str:
    settings = get_settings()
    if not settings.hive_api_key:
        raise RuntimeError("Hive API key is not configured (HIVE_API_KEY)")

    content = [{"type": "image_url", "image_url": {"url": _data_url(view)}} for view in views]
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": settings.vlm_model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": settings.vlm_max_new_tokens,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.hive_api_key}"}

    async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
        response = await client.post(_ENDPOINT, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Hive VLM HTTP {response.status_code}: {response.text[:300]}")

    body = response.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Hive VLM response missing choices[0].message.content: {body}") from exc


async def _inspect_categories(image: Image.Image) -> dict[str, str]:
    """Ask every category question concurrently against the same view set.

    A category that fails (network blip, etc.) contributes nothing rather
    than aborting the whole inspection -- consistent with how the rest of
    Veil treats unavailable evidence.
    """
    views = make_views(image)
    results = await asyncio.gather(
        *(_call_hive_vlm(views, question) for question in _CATEGORY_QUESTIONS.values()),
        return_exceptions=True,
    )

    responses: dict[str, str] = {}
    failures = 0
    for category, result in zip(_CATEGORY_QUESTIONS.keys(), results):
        if isinstance(result, BaseException):
            failures += 1
            continue
        responses[category] = result

    if failures == len(_CATEGORY_QUESTIONS):
        raise RuntimeError("all Hive VLM category checks failed")
    return responses


async def explain_image(image: Image.Image, score: float | None) -> Explanation:
    settings = get_settings()
    responses = await _inspect_categories(image)
    assessment, findings = parse_category_findings(responses)
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
