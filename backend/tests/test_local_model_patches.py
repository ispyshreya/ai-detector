"""Tests for the patch-based local-model pipeline (see signals/local_model.py).

Split into two groups:
  * Pure unit tests on the geometry/filtering/aggregation functions -- no
    model, instant, cover the edge-case requirements directly.
  * End-to-end tests through LocalModelSignal.analyze() against the real
    checkpoint -- kept to small images so real inference stays fast while
    still exercising the full resize -> pad -> tile -> filter -> score ->
    aggregate path.
"""

import asyncio
from io import BytesIO

import pytest
from PIL import Image, ImageDraw

from app.schemas import SignalStatus
from app.signals.base import ImageInput
from app.signals.local_model import (
    LEGACY_WEIGHT_MEAN,
    LEGACY_WEIGHT_MEDIAN,
    LEGACY_WEIGHT_TOP20,
    PATCH_SIZE,
    PATCH_STRIDE,
    PROD_WEIGHT_MEDIAN,
    PROD_WEIGHT_TRIMMED_MEAN,
    TOP_PERCENTILE_20,
    LocalModelSignal,
    Patch,
    aggregate_patch_scores,
    extract_patch_coords,
    filter_blank_patches,
    pad_to_min_size,
    resize_longest_side,
    score_patches,
    subsample_evenly,
    suppress_overlapping_patches,
    trimmed_mean,
)


def _solid(width: int, height: int, color=(128, 128, 128)) -> Image.Image:
    return Image.new("RGB", (width, height), color)


def _textured(width: int, height: int) -> Image.Image:
    """A non-blank synthetic image with real variance (shapes, not noise)."""
    image = Image.new("RGB", (width, height), (30, 60, 90))
    draw = ImageDraw.Draw(image)
    for i in range(0, max(width, height), 11):
        draw.line([(i, 0), (0, i)], fill=(220, 180, 40), width=2)
    draw.ellipse([width // 4, height // 4, width // 2, height // 2], fill=(200, 50, 50))
    return image


def _jpeg_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


# --------------------------------------------------------------------- geometry

def test_resize_longest_side_downscales_landscape_preserving_aspect():
    resized = resize_longest_side(_solid(2000, 1000))
    assert max(resized.size) == 1024
    assert resized.size == (1024, 512)  # exact 2:1 ratio preserved


def test_resize_longest_side_downscales_portrait_preserving_aspect():
    resized = resize_longest_side(_solid(1000, 2000))
    assert resized.size == (512, 1024)


def test_resize_longest_side_never_upscales_or_stretches():
    small = _solid(200, 150)
    assert resize_longest_side(small).size == (200, 150)


def test_pad_to_min_size_pads_tiny_image_to_tile_size():
    padded = pad_to_min_size(_solid(20, 15), min_size=PATCH_SIZE)
    assert padded.size == (PATCH_SIZE, PATCH_SIZE)


def test_pad_to_min_size_leaves_large_enough_image_unchanged():
    image = _solid(100, 80)
    assert pad_to_min_size(image, min_size=PATCH_SIZE).size == (100, 80)


# ------------------------------------------------------------ patch coordinates

def test_extract_patch_coords_covers_bottom_and_right_edges_when_not_aligned():
    """773/513 aren't divisible by 16 or 32 -- the final coordinate on each
    axis must still be flush with the edge (width-32 / height-32)."""
    width, height = 773, 513
    coords = extract_patch_coords(width, height, tile=PATCH_SIZE, stride=PATCH_STRIDE)

    xs = {x for x, _ in coords}
    ys = {y for _, y in coords}
    assert max(xs) == width - PATCH_SIZE
    assert max(ys) == height - PATCH_SIZE
    assert min(xs) == 0 and min(ys) == 0
    # Every patch fits fully inside the image.
    assert all(x + PATCH_SIZE <= width and y + PATCH_SIZE <= height for x, y in coords)


def test_extract_patch_coords_50_percent_overlap_stride():
    coords = extract_patch_coords(96, 32, tile=PATCH_SIZE, stride=PATCH_STRIDE)
    xs = sorted({x for x, _ in coords})
    assert xs == [0, 16, 32, 48, 64]  # (96-32)/16 + 1 = 5 positions


def test_extract_patch_coords_rejects_input_smaller_than_tile():
    with pytest.raises(ValueError):
        extract_patch_coords(20, 40, tile=PATCH_SIZE)


# ---------------------------------------------------------------- blank filter

def test_filter_blank_patches_falls_back_to_all_when_everything_is_blank():
    image = _solid(64, 64)  # perfectly flat -- every patch would be filtered
    coords = extract_patch_coords(64, 64)
    kept = filter_blank_patches(image, coords)
    assert kept == coords  # fallback: nothing gets dropped


def test_filter_blank_patches_keeps_textured_regions():
    image = _textured(96, 96)
    coords = extract_patch_coords(96, 96)
    kept = filter_blank_patches(image, coords)
    assert len(kept) > 0


def test_subsample_evenly_caps_count_and_keeps_order():
    items = list(range(1000))
    subset = subsample_evenly(items, 50)
    assert len(subset) == 50
    assert subset == sorted(subset)  # spatial order preserved, no shuffling


# --------------------------------------------------------------- aggregation

def test_aggregate_patch_scores_output_always_in_unit_range():
    cases = [
        [0.0],
        [1.0],
        [0.0, 1.0],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [0.99] * 20,
        [0.01] * 20,
    ]
    for scores in cases:
        agg = aggregate_patch_scores(scores)
        for value in (
            agg.mean, agg.median, agg.trimmed_mean, agg.top10_mean, agg.top20_mean,
            agg.max_score, agg.pct_above_half, agg.pct_above_0_7, agg.legacy_score, agg.final_score,
        ):
            assert 0.0 <= value <= 1.0


def test_trimmed_mean_matches_documented_definition():
    scores = [0.9, 0.85, 0.8, 0.2, 0.1] * 4  # 20 scores
    result = trimmed_mean(scores, trim_fraction=0.10)
    ordered = sorted(scores)
    trim_n = int(len(ordered) * 0.10)  # 2
    expected = sum(ordered[trim_n:-trim_n]) / (len(ordered) - 2 * trim_n)
    assert result == pytest.approx(expected)


def test_trimmed_mean_falls_back_to_plain_mean_when_too_few_to_trim():
    scores = [0.1, 0.9, 0.5]  # trim_n = int(3*0.1) = 0 -> nothing to trim
    assert trimmed_mean(scores) == pytest.approx(sum(scores) / len(scores))


def test_aggregate_patch_scores_matches_the_documented_production_formula():
    scores = [0.9, 0.85, 0.8, 0.2, 0.1] * 4  # 20 scores
    agg = aggregate_patch_scores(scores)

    expected_median = sorted(scores)[9:11]
    expected_median = sum(expected_median) / 2
    expected_trimmed = trimmed_mean(scores)
    expected_final = PROD_WEIGHT_MEDIAN * expected_median + PROD_WEIGHT_TRIMMED_MEAN * expected_trimmed

    assert agg.median == pytest.approx(expected_median)
    assert agg.final_score == pytest.approx(expected_final)


def test_aggregate_patch_scores_also_reports_legacy_formula_for_comparison():
    scores = [0.9, 0.85, 0.8, 0.2, 0.1] * 4
    agg = aggregate_patch_scores(scores)

    ordered = sorted(scores, reverse=True)
    top_n = max(1, round(len(ordered) * TOP_PERCENTILE_20))
    expected_top20 = sum(ordered[:top_n]) / top_n
    expected_mean = sum(scores) / len(scores)
    expected_legacy = (
        LEGACY_WEIGHT_TOP20 * expected_top20
        + LEGACY_WEIGHT_MEDIAN * agg.median
        + LEGACY_WEIGHT_MEAN * expected_mean
    )
    assert agg.legacy_score == pytest.approx(expected_legacy)


def test_conservative_formula_resists_overlap_style_score_inflation():
    """This is the actual regression: heavy overlap turns one sharp/detailed
    real region into MANY near-duplicate high scores. Simulate that directly:
    a realistic MINORITY of patches (one region, ~20% of the frame) score
    high, the rest of the image is unremarkable. The legacy formula's
    top-20%-mean term is entirely inside that one duplicated cluster and
    crosses the risk threshold; median/trimmed-mean are not fooled by it."""
    scores = [0.10] * 80 + [0.95] * 20
    agg = aggregate_patch_scores(scores)

    assert agg.legacy_score > 0.5  # top-20% = the duplicated cluster verbatim
    assert agg.final_score < 0.3  # median/trimmed-mean stay anchored to the majority


def test_aggregate_patch_scores_not_dominated_by_single_max_patch():
    """One extreme outlier patch must not swing the final score to its value."""
    scores = [0.02] * 19 + [0.99]
    agg = aggregate_patch_scores(scores)
    assert agg.final_score < 0.5


def test_aggregate_patch_scores_requires_at_least_one_score():
    with pytest.raises(ValueError):
        aggregate_patch_scores([])


def test_histogram_buckets_sum_to_total_count():
    scores = [0.05, 0.15, 0.55, 0.55, 0.99, 1.0]
    agg = aggregate_patch_scores(scores)
    assert sum(agg.histogram.values()) == len(scores)


def test_suppress_overlapping_patches_collapses_one_region_to_one_entry():
    """Five overlapping windows all sampling the same suspicious spot should
    collapse to one representative patch, not count as five."""
    cluster = [Patch(x=x, y=0, score=0.9 - i * 0.01) for i, x in enumerate([0, 8, 16, 24, 32])]
    far_away = Patch(x=500, y=500, score=0.6)
    result = suppress_overlapping_patches(cluster + [far_away], radius=32)

    assert len(result) == 2  # one representative from the cluster + the distant one
    assert result[0].score == pytest.approx(0.9)  # kept the highest-scoring of the cluster


# ------------------------------------------------------------------ end-to-end

def test_landscape_image_end_to_end():
    image = ImageInput(
        data=_jpeg_bytes(_textured(300, 200)), filename="landscape.jpg", content_type="image/jpeg"
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["was_tiled"] is True
    assert result.raw["tile_count"] > 0


def test_portrait_image_end_to_end():
    image = ImageInput(
        data=_jpeg_bytes(_textured(200, 300)), filename="portrait.jpg", content_type="image/jpeg"
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["tile_count"] > 0


def test_image_smaller_than_patch_size_end_to_end():
    image = ImageInput(
        data=_jpeg_bytes(_solid(20, 15, (90, 140, 200))), filename="tiny.jpg", content_type="image/jpeg"
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["tile_count"] == 1  # padded up to exactly one tile


def test_nearly_blank_image_still_produces_a_score():
    image = ImageInput(
        data=_jpeg_bytes(_solid(200, 200)), filename="blank.jpg", content_type="image/jpeg"
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["tile_count"] > 0  # blank-filter fallback kicked in, not zero


def test_batched_scoring_matches_unbatched_within_tolerance():
    """BatchNorm in eval() mode uses running stats, not batch stats, so the
    result must be batch-size-invariant -- this is what makes batching safe."""
    from app.signals.local_model import _load_model

    image = _textured(96, 96)
    coords = extract_patch_coords(*image.size)
    model, device, _ = _load_model()

    unbatched = score_patches(image, coords, model, device, batch_size=1)
    batched = score_patches(image, coords, model, device, batch_size=128)

    assert len(unbatched) == len(batched) == len(coords)
    for a, b in zip(unbatched, batched):
        # BatchNorm eval-mode uses running stats regardless of batch size, so
        # results should match closely -- but matmul/conv accumulation order
        # still differs between batch sizes, so exact equality isn't
        # guaranteed. 1e-3 catches real bugs (e.g. misaligned indexing)
        # without failing on ordinary floating-point drift.
        assert a == pytest.approx(b, abs=1e-3)
