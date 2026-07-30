"""Local trained image detector signal.

Wraps Shreya's ResNet checkpoint as a backend signal so the frontend can call
one `/scan` endpoint while still using the in-house model.

INFERENCE CONTRACT (confirmed from detector-trainer/train.py,
detector-trainer/data/dataset.py, detector-trainer/output/test_metrics.json —
not assumed):
  * Network input: 224x224 RGB, ImageNet-normalized
    (mean [0.485,0.456,0.406], std [0.229,0.224,0.225]).
  * Output: one logit per image (`[B]` or `[B,1]`; we `.reshape(-1)` to be
    robust to either).
  * sigmoid(logit) = P(AI-generated). Confirmed by train.py's
    `_compute_pos_weight` ("BCE positives (fake)") AND by
    test_metrics.json's `positive_class: "FAKE"` /
    `score_meaning: "probability_fake_or_ai_generated"` for this exact
    checkpoint -- two independent corroborating sources, not a guess.

PATCH-BASED INFERENCE. The checkpoint was trained only on CIFAKE, whose
images are natively 32x32. Shrinking a whole real-world photo down to 32x32
(the original approach) throws away nearly all of its detail. Instead we:
  1. Downscale only if needed so the longest side is <=1024px (aspect
     preserved, never stretched, never upscaled).
  2. Slide a 32x32 window over the image with a 16px stride (50% overlap),
     always including a final edge-flush patch on the right/bottom even when
     the dimensions aren't stride-aligned.
  3. Drop near-blank patches by a conservative grayscale variance threshold
     (falls back to using every patch if that would filter all of them).
  4. Batch every surviving patch through the model in one pass per batch
     (model loaded once globally, never per-patch/per-request).
  5. Aggregate per-patch scores with an explicit, weighted formula instead of
     just the mean or the single worst patch.

This is STILL a heuristic, not a validated technique (see the Quality
Standard section of VEIL_CLIENT_PROMPT.md — crops are a transformation a
detector must be evaluated against before its probability is presented as
meaningful, and that evaluation hasn't been done for this path). A CIFAKE
image is a whole tiny *scene*; a patch is a fragment of a much bigger one.
We flag it in `notes` and knock down self-reported `confidence` accordingly
so downstream fusion (engine/triangulate.py) excludes it from the verdict
entirely and shows it for reference only.
"""

from __future__ import annotations

import asyncio
import importlib.util
import statistics
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from app.config import get_settings
from app.config import REPO_ROOT
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

IMAGE_SIZE = 224
TRAINING_NATIVE_SIZE = (32, 32)

# --- Patch geometry (tunable) -----------------------------------------
PATCH_SIZE = 32
PATCH_STRIDE = 16  # 50% overlap
MAX_LONGEST_SIDE = 1024  # cap before tiling; never upscale, never stretch

# --- Blank-patch filtering (tunable) -----------------------------------
# Grayscale pixel standard deviation (0-255 scale) below which a patch is
# treated as near-blank. Deliberately low/conservative: ordinary smooth
# regions (sky, walls, skin) almost always exceed this from noise/compression
# alone, so only genuinely flat/solid patches get dropped.
MIN_PATCH_STD = 2.0

# --- Compute bounds (tunable) -------------------------------------------
# Not part of the original spec -- a practical safety cap. A 1024px image at
# stride 16 produces ~63x63 ~= 3,969 overlapping patches; uncapped, that can
# take minutes on CPU and blow the per-signal timeout. We keep an evenly
# spaced subset (after blank-filtering) instead of every patch.
MAX_PATCHES = 512
PATCH_BATCH_SIZE = 128

# --- Score aggregation weights (tunable) --------------------------------
#
# INCIDENT NOTE (see diagnose_patch_pipeline.py for the measured numbers): the
# original top-20%-weighted formula below was found to inflate scores on real
# photos once patches overlap at 50% -- a single genuinely sharp/detailed
# region (a real eye, real skin pore texture, real jewelry) gets sampled by
# many overlapping windows, all landing in the top 20%, so top20_mean tracks
# "how textured is the sharpest region" more than "how much of the image
# looks synthetic". A known-real selfie went from ~0.38 (non-overlapping,
# stride 32) to ~0.75 (overlapping, stride 16) under this formula, on the
# SAME model and SAME normalization -- see the diagnostic script's report for
# the factorial breakdown isolating overlap vs. resize vs. formula. The
# legacy formula is kept only so `raw["debug"]` can show both values
# side-by-side; PRODUCTION now uses the conservative formula below instead.
TOP_PERCENTILE_20 = 0.20
TOP_PERCENTILE_10 = 0.10
LEGACY_WEIGHT_TOP20 = 0.50
LEGACY_WEIGHT_MEDIAN = 0.30
LEGACY_WEIGHT_MEAN = 0.20

# PRODUCTION formula: median + trimmed mean only -- deliberately excludes any
# top-percentile term, since that's exactly the component overlap inflates.
TRIM_FRACTION = 0.10  # drop the highest/lowest 10% before averaging
PROD_WEIGHT_MEDIAN = 0.60
PROD_WEIGHT_TRIMMED_MEAN = 0.40

# Diagnostics-only: histogram bucket count, and how many top patches to keep
# coordinates+scores for (so a heatmap can be built later without re-scoring).
HISTOGRAM_BUCKETS = 10
DEBUG_TOP_K = 20

# Non-max-suppression radius (pixels, in the resized/padded image's
# coordinate space) for `suppress_overlapping_patches`: two patches whose
# top-left corners are closer than this are treated as the same region, so
# one suspicious spot doesn't get counted as N independent pieces of
# evidence. PATCH_SIZE is a reasonable default -- roughly "one patch width"
# of separation before two windows count as genuinely distinct regions.
SUPPRESSION_RADIUS = PATCH_SIZE

# Extra confidence discount applied whenever the input wasn't already exactly
# 32x32 (i.e. we had to patch it), on top of the triangulation engine's own
# blanket discount for this signal (see engine/triangulate.py).
_PATCH_CONFIDENCE_FACTOR = 0.7

_MODEL = None
_DEVICE = None
_CHECKPOINT = None


@dataclass(frozen=True)
class Patch:
    """One sampled patch: its top-left corner and model score."""

    x: int
    y: int
    score: float


@dataclass(frozen=True)
class PatchAggregate:
    """Full diagnostic statistics over a set of patch scores, plus both the
    retired legacy blend and the production conservative blend so they can
    be compared side-by-side (see the INCIDENT NOTE above)."""

    n_patches: int
    mean: float
    median: float
    trimmed_mean: float
    top10_mean: float
    top20_mean: float
    max_score: float
    std: float
    pct_above_half: float
    pct_above_0_7: float
    histogram: dict[str, int]
    top_patches: tuple[Patch, ...]  # DEBUG_TOP_K highest-scoring, for a future heatmap
    legacy_score: float  # retired 0.5*top20 + 0.3*median + 0.2*mean
    final_score: float  # PRODUCTION: 0.6*median + 0.4*trimmed_mean


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
    """Byte-for-byte the training/eval contract (see module docstring)."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _load_model():
    global _MODEL, _DEVICE, _CHECKPOINT

    settings = get_settings()
    model_type = getattr(settings, "local_model_type", "resnet")
    checkpoint = _resolve_checkpoint(
        settings.local_model_checkpoint if model_type == "clip"
        else settings.local_model_resnet_checkpoint
    )
    if _MODEL is not None and _CHECKPOINT == checkpoint:
        return _MODEL, _DEVICE, checkpoint

    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "clip":
        # Frozen CLIP ViT-L/14 + trained head. The head checkpoint's sibling
        # config.json records {backbone_name, feat_dim, head_type, hidden}.
        build_clip_detector = _load_clip_builder()
        config_path = checkpoint.parent / "config.json"
        model = build_clip_detector(
            config=config_path, head_ckpt=checkpoint, device=str(device)
        )
    else:
        build_model = _load_build_model()
        model = build_model(settings.local_model_resnet_name, pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        model = model.to(device)
        model.eval()

    _MODEL = model
    _DEVICE = device
    _CHECKPOINT = checkpoint
    return model, device, checkpoint


# --------------------------------------------------------------------------
# Pure, independently-testable pipeline stages. None of these touch the
# network or the ImageInput/SignalResult envelope -- they operate on plain
# PIL images / coordinate lists / float lists so they're cheap to unit test.
# --------------------------------------------------------------------------

def resize_longest_side(image: Image.Image, max_side: int = MAX_LONGEST_SIDE) -> Image.Image:
    """Downscale so the longest side is <= max_side, preserving aspect ratio.

    Never upscales and never stretches -- if the image already fits, it's
    returned unchanged (aside from a defensive copy is not needed since PIL
    resize always returns a new image; we short-circuit to avoid a needless
    resample pass).
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def pad_to_min_size(image: Image.Image, min_size: int = PATCH_SIZE) -> Image.Image:
    """Edge-pad an image smaller than `min_size` in either dimension.

    Uses edge replication (not a solid color) so we don't invent a sharp
    artificial border the model could latch onto -- the point is to make at
    least one valid patch extractable, not to add new "content".
    """
    width, height = image.size
    if width >= min_size and height >= min_size:
        return image

    pad_w = max(0, min_size - width)
    pad_h = max(0, min_size - height)
    array = np.array(image)
    padded = np.pad(
        array,
        ((0, pad_h), (0, pad_w), (0, 0)),
        mode="edge",
    )
    return Image.fromarray(padded)


def extract_patch_coords(
    width: int, height: int, tile: int = PATCH_SIZE, stride: int = PATCH_STRIDE
) -> list[tuple[int, int]]:
    """Every (x, y) top-left patch corner at `stride`, plus a final coordinate
    flush with the right/bottom edge even when the dimensions aren't
    stride-aligned, so no strip along the edge is ever skipped.

    Requires width >= tile and height >= tile (call `pad_to_min_size` first).
    """
    if width < tile or height < tile:
        raise ValueError(f"image ({width}x{height}) is smaller than tile size {tile}")

    def axis_offsets(dimension: int) -> list[int]:
        last = dimension - tile
        offsets = list(range(0, last + 1, stride))
        if offsets[-1] != last:
            offsets.append(last)
        return offsets

    xs = axis_offsets(width)
    ys = axis_offsets(height)
    return [(x, y) for y in ys for x in xs]


def _grayscale_std(patch: Image.Image) -> float:
    return float(np.array(patch.convert("L"), dtype=np.float32).std())


def filter_blank_patches(
    image: Image.Image,
    coords: list[tuple[int, int]],
    tile: int = PATCH_SIZE,
    min_std: float = MIN_PATCH_STD,
) -> list[tuple[int, int]]:
    """Drop near-blank/flat patches by grayscale std deviation.

    Conservative by design (see MIN_PATCH_STD): only meant to catch solid
    padding/letterboxing, not ordinary smooth content. Falls back to every
    coordinate if the threshold would remove all of them.
    """
    kept = [
        (x, y)
        for x, y in coords
        if _grayscale_std(image.crop((x, y, x + tile, y + tile))) >= min_std
    ]
    return kept if kept else coords


def subsample_evenly(items: list, limit: int) -> list:
    """An evenly-spaced subset of `items` capped at `limit`, preserving
    spatial spread across the full list rather than truncating it."""
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def score_patches(
    image: Image.Image,
    coords: list[tuple[int, int]],
    model,
    device,
    tile: int = PATCH_SIZE,
    batch_size: int = PATCH_BATCH_SIZE,
) -> list[float]:
    """Crop every coordinate, batch through the model, return one sigmoid
    score per coordinate in the same order. `model.eval()` must already have
    been called by the caller (`_load_model` does this) -- BatchNorm running
    stats make this batch-size-invariant, which the tests verify."""
    transform = _eval_transform()
    scores: list[float] = []
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        tensors = torch.stack([
            transform(image.crop((x, y, x + tile, y + tile))) for x, y in chunk
        ]).to(device)
        with torch.no_grad():
            logits = model(tensors).reshape(-1)
            scores.extend(torch.sigmoid(logits).tolist())
    return scores


def trimmed_mean(scores: list[float], trim_fraction: float = TRIM_FRACTION) -> float:
    """Mean after dropping the highest and lowest `trim_fraction` of scores.

    Falls back to the ordinary mean if there are too few scores to trim
    safely (trimming both ends would leave nothing, or the list is tiny
    enough that a couple of outliers would dominate the trim count)."""
    if not scores:
        raise ValueError("trimmed_mean requires at least one score")

    ordered = sorted(scores)
    trim_n = int(len(ordered) * trim_fraction)
    if trim_n == 0 or len(ordered) - 2 * trim_n < 1:
        return statistics.fmean(scores)
    return statistics.fmean(ordered[trim_n: len(ordered) - trim_n])


def histogram_buckets(scores: list[float], buckets: int = HISTOGRAM_BUCKETS) -> dict[str, int]:
    """Count of scores falling in each [i/buckets, (i+1)/buckets) bucket
    (the final bucket is closed on both ends so a score of exactly 1.0 is
    counted)."""
    counts: dict[str, int] = {}
    width = 1.0 / buckets
    for i in range(buckets):
        lo, hi = i * width, (i + 1) * width
        label = f"{lo:.1f}-{hi:.1f}"
        if i == buckets - 1:
            counts[label] = sum(1 for s in scores if lo <= s <= hi)
        else:
            counts[label] = sum(1 for s in scores if lo <= s < hi)
    return counts


def suppress_overlapping_patches(
    patches: list[Patch], radius: float = SUPPRESSION_RADIUS
) -> list[Patch]:
    """Greedy non-max suppression over spatial position: sort by score
    descending, keep a patch, drop every remaining patch whose top-left
    corner is within `radius` pixels of a kept (higher-scoring) patch.

    This is what "one suspicious region -> N overlapping patches -> N pieces
    of evidence" looks like fixed: the returned list has (approximately) one
    entry per spatially distinct region instead of one per sampled window.
    Diagnostic/debug use only -- does not affect the production score."""
    remaining = sorted(patches, key=lambda p: p.score, reverse=True)
    kept: list[Patch] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [
            p for p in remaining
            if (p.x - best.x) ** 2 + (p.y - best.y) ** 2 > radius ** 2
        ]
    return kept


def aggregate_patch_scores(scores: list[float], patches: list[Patch] | None = None) -> PatchAggregate:
    """Full diagnostic statistics, the retired legacy blend, and the
    production conservative blend. `patches` (with coordinates) is optional
    and only used to populate `top_patches` for future heatmap use -- pass
    plain scores-only when coordinates aren't available/needed."""
    if not scores:
        raise ValueError("aggregate_patch_scores requires at least one score")

    ordered = sorted(scores, reverse=True)
    top20_n = max(1, round(len(ordered) * TOP_PERCENTILE_20))
    top10_n = max(1, round(len(ordered) * TOP_PERCENTILE_10))
    top20_mean = statistics.fmean(ordered[:top20_n])
    top10_mean = statistics.fmean(ordered[:top10_n])
    mean = statistics.fmean(scores)
    median = statistics.median(scores)
    trimmed = trimmed_mean(scores)
    std = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    pct_above_half = sum(1 for s in scores if s > 0.5) / len(scores)
    pct_above_0_7 = sum(1 for s in scores if s > 0.7) / len(scores)
    max_score = max(scores)

    legacy = LEGACY_WEIGHT_TOP20 * top20_mean + LEGACY_WEIGHT_MEDIAN * median + LEGACY_WEIGHT_MEAN * mean
    final = PROD_WEIGHT_MEDIAN * median + PROD_WEIGHT_TRIMMED_MEAN * trimmed

    if patches:
        top_patches = tuple(sorted(patches, key=lambda p: p.score, reverse=True)[:DEBUG_TOP_K])
    else:
        top_patches = ()

    return PatchAggregate(
        n_patches=len(scores),
        mean=mean,
        median=median,
        trimmed_mean=trimmed,
        top10_mean=top10_mean,
        top20_mean=top20_mean,
        max_score=max_score,
        std=std,
        pct_above_half=pct_above_half,
        pct_above_0_7=pct_above_0_7,
        histogram=histogram_buckets(scores),
        top_patches=top_patches,
        legacy_score=max(0.0, min(1.0, legacy)),
        final_score=max(0.0, min(1.0, final)),
    )


@dataclass(frozen=True)
class PatchPipelineResult:
    aggregate: PatchAggregate
    patches: list[Patch]
    resized_size: tuple[int, int]
    patches_extracted: int  # before blank-filtering/subsampling


def run_patch_pipeline(pil_image: Image.Image, model, device) -> PatchPipelineResult:
    """End-to-end: resize -> pad -> tile -> filter -> subsample -> score ->
    aggregate. Synchronous/CPU-bound -- callers on the event loop should run
    this via `asyncio.to_thread`."""
    resized = resize_longest_side(pil_image)
    padded = pad_to_min_size(resized)
    width, height = padded.size

    all_coords = extract_patch_coords(width, height)
    coords = filter_blank_patches(padded, all_coords)
    coords = subsample_evenly(coords, MAX_PATCHES)

    scores = score_patches(padded, coords, model, device)
    patches = [Patch(x=x, y=y, score=score) for (x, y), score in zip(coords, scores)]
    aggregate = aggregate_patch_scores(scores, patches)
    return PatchPipelineResult(
        aggregate=aggregate,
        patches=patches,
        resized_size=padded.size,
        patches_extracted=len(all_coords),
    )


class LocalModelSignal(Signal):
    name = "local"
    signal_class = SignalClass.detector

    def available(self) -> bool:
        settings = get_settings()
        checkpoint = (
            settings.local_model_checkpoint
            if getattr(settings, "local_model_type", "resnet") == "clip"
            else settings.local_model_resnet_checkpoint
        )
        return _resolve_checkpoint(checkpoint).exists()

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

        if getattr(get_settings(), "local_model_type", "resnet") == "clip":
            return await self._analyze_clip(pil_image, model, device, checkpoint, started)
        return await self._analyze_resnet_patches(pil_image, model, device, checkpoint, started)

    async def _analyze_clip(self, pil_image, model, device, checkpoint, started: float) -> SignalResult:
        """Single whole-image pass through the CLIP ViT-L/14 + trained head.

        Unlike the resnet path below, this model was trained and threshold-
        calibrated directly on full real-world photos (not CIFAKE's native
        32x32 scenes), so it needs no patch-tiling workaround -- and it is NOT
        an unvalidated heuristic, so `was_tiled` is always False here and this
        score DOES count toward the fused verdict (see `_is_unvalidated_resize`
        in engine/triangulate.py, which only ever excludes the resnet path).
        """
        settings = get_settings()
        try:
            tensor = await asyncio.to_thread(
                lambda: _eval_transform()(pil_image).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                ai_score = torch.sigmoid(model(tensor)).item()
        except Exception as exc:  # noqa: BLE001 - never let this signal crash the scan
            return self._error_result(f"clip inference failed: {exc}", started)

        threshold = float(getattr(settings, "local_model_threshold", 0.5))
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
                f"Local {settings.local_model_name} fake/AI likelihood {round(ai_score * 100)}%"
                f" (decision threshold {threshold:.2f}, calibrated for a low real-photo false-positive rate).",
                f"Checkpoint: {checkpoint}",
            ],
            raw={
                "checkpoint": str(checkpoint),
                "model": settings.local_model_name,
                "model_type": "clip",
                "threshold": threshold,
                "was_tiled": False,
            },
        )

    async def _analyze_resnet_patches(self, pil_image, model, device, checkpoint, started: float) -> SignalResult:
        """Legacy path: the CIFAKE-trained resnet, served via the overlapping
        32x32 patch pipeline (see module docstring). Kept for local_model_type
        = "resnet"; superseded by `_analyze_clip` in production."""
        original_size = pil_image.size
        was_patched = original_size != TRAINING_NATIVE_SIZE

        try:
            pipeline_result = await asyncio.to_thread(run_patch_pipeline, pil_image, model, device)
        except Exception as exc:  # noqa: BLE001 - never let this signal crash the scan
            return self._error_result(f"patch inference failed: {exc}", started)

        aggregate = pipeline_result.aggregate
        patches = pipeline_result.patches
        resized_size = pipeline_result.resized_size
        ai_score = aggregate.final_score
        confidence = ai_score if ai_score >= 0.5 else 1.0 - ai_score
        if was_patched:
            confidence *= _PATCH_CONFIDENCE_FACTOR
        latency_ms = (time.perf_counter() - started) * 1000.0

        notes = [
            f"Local {get_settings().local_model_resnet_name} fake/AI likelihood {round(ai_score * 100)}%"
            f" (blended from {len(patches)} patch{'es' if len(patches) != 1 else ''})",
            f"Checkpoint: {checkpoint}",
        ]
        if was_patched:
            notes.append(
                f"Input was {original_size[0]}x{original_size[1]}, resized to fit "
                f"{resized_size[0]}x{resized_size[1]} and diced into overlapping "
                f"{PATCH_SIZE}x{PATCH_SIZE} patches (stride {PATCH_STRIDE}) rather than shrunk "
                "whole. This is an unvalidated domain-matching heuristic, not a calibrated "
                "result — confidence is reduced accordingly."
            )

        raw: dict = {
            "checkpoint": str(checkpoint),
            "model": get_settings().local_model_resnet_name,
            "model_type": "resnet",
            "was_tiled": was_patched,
            "tile_count": len(patches),
            "max_tile_score": round(aggregate.max_score, 4),
            "original_size": list(original_size),
            "training_native_size": list(TRAINING_NATIVE_SIZE),
        }
        if get_settings().local_model_debug:
            suppressed = suppress_overlapping_patches(list(patches))
            raw["debug"] = {
                "original_size": list(original_size),
                "resized_size": list(resized_size),
                "patches_extracted": pipeline_result.patches_extracted,
                "patches_used": len(patches),
                "mean": round(aggregate.mean, 6),
                "median": round(aggregate.median, 6),
                "trimmed_mean_10pct": round(aggregate.trimmed_mean, 6),
                "top10_mean": round(aggregate.top10_mean, 6),
                "top20_mean": round(aggregate.top20_mean, 6),
                "max_score": round(aggregate.max_score, 6),
                "std": round(aggregate.std, 6),
                "pct_above_0.5": round(aggregate.pct_above_half, 6),
                "pct_above_0.7": round(aggregate.pct_above_0_7, 6),
                "histogram": aggregate.histogram,
                "top_20_patches": [
                    {"x": p.x, "y": p.y, "score": round(p.score, 6)} for p in aggregate.top_patches
                ],
                # Both scores, explicitly, so they can be compared -- production
                # uses final_score_conservative; legacy is retired (see the
                # INCIDENT NOTE by the weight constants above).
                "final_score_conservative_PRODUCTION": round(aggregate.final_score, 6),
                "final_score_legacy_top20_weighted_RETIRED": round(aggregate.legacy_score, 6),
                # Spatial de-duplication: how many independent regions survive
                # after suppressing overlapping high-scoring patches, and what
                # the conservative formula gives on just that de-duplicated
                # set. Diagnostic only -- not used for the production score.
                "suppressed_region_count": len(suppressed),
                "suppressed_conservative_score": round(
                    aggregate_patch_scores([p.score for p in suppressed]).final_score, 6
                ) if suppressed else None,
            }

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=max(0.0, min(1.0, float(ai_score))),
            manipulation_score=None,
            confidence=max(0.0, min(1.0, float(confidence))),
            latency_ms=latency_ms,
            notes=notes,
            raw=raw,
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
