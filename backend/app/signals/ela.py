"""Error Level Analysis (ELA) forensic signal (Layer 1).

ELA re-saves an image as JPEG at a fixed quality and measures how much each
region changes under that recompression. Pixels that were *already* at (or near)
that JPEG quality barely change; regions that were edited, pasted in, or saved a
different number of times tend to show a noticeably different ("uneven") error
level. So ELA is a classic tool for spotting LOCAL RECOMPRESSION / SPLICING in a
*real* photograph that was EDITED.

TWO-AXIS NOTE (see schemas.py): ELA answers Axis 2 ("was a real image
manipulated/edited?"), NOT Axis 1 ("is this AI-generated?"). Fully synthetic
images have no "original camera" baseline to deviate from, so ELA says nothing
reliable about AI generation. We therefore fill `manipulation_score` and leave
`ai_score = None`.

ELA is also a *weak* heuristic: it is famously easy to misread, sensitive to the
chosen quality, and confounded by the image's own compression history. We keep
confidence low and treat it as a corroborating signal only.

This module is fully local (Pillow) — no network, no keys.
"""

from __future__ import annotations

import io
import time

from PIL import Image, ImageChops, UnidentifiedImageError

from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

# JPEG quality used for the recompression pass. 90 is a common ELA default:
# high enough that an unedited high-quality image changes little, low enough
# that re-saved/spliced regions stand out.
_QUALITY = 90

# Normalization anchors (see _to_manipulation_score). Difference values are in
# 0..255 per channel. Untouched, uniformly-compressed images typically show a
# small mean diff and a modest max; edited images push both up, especially the
# mean (broad, uneven error). These thresholds are deliberately rough — ELA is a
# soft heuristic, not a calibrated detector.
_MEAN_DIFF_FULL_SCALE = 18.0   # mean diff at/above this reads as strongly uneven
_MAX_DIFF_FULL_SCALE = 200.0   # max diff at/above this reads as a sharp edit edge


class ElaSignal(Signal):
    name = "ela"
    signal_class = SignalClass.forensic

    def available(self) -> bool:
        """Always available: runs locally with Pillow, needs no configuration."""
        return True

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()

        # Parse the upload. Anything Pillow can't open isn't an image we can
        # analyze — skip rather than error (it's "not applicable", not a fault).
        try:
            original = Image.open(io.BytesIO(image.data)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            return SignalResult(
                name=self.name,
                signal_class=self.signal_class,
                status=SignalStatus.skipped,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                notes=[f"Not a parseable image for ELA: {exc}"],
            )

        # Recompress to JPEG at a known quality, then reopen the result.
        buffer = io.BytesIO()
        original.save(buffer, format="JPEG", quality=_QUALITY)
        buffer.seek(0)
        recompressed = Image.open(buffer).convert("RGB")

        # Per-pixel absolute difference between the original and the recompressed
        # copy. The bands' extrema/mean summarize the "error level" surface.
        diff = ImageChops.difference(original, recompressed)

        # extrema() returns (min, max) per band; take the largest max across bands.
        max_diff = float(max(hi for (_lo, hi) in diff.getextrema()))

        # Mean absolute difference across all channels (a single 0..255 number).
        # diff.convert("L") would re-weight channels; averaging the per-band means
        # keeps every channel equal, which is what we want for raw error level.
        band_means = _band_means(diff)
        mean_diff = sum(band_means) / len(band_means)

        manipulation_score = _to_manipulation_score(mean_diff, max_diff)

        notes = [
            f"ELA at JPEG quality {_QUALITY}: mean diff {mean_diff:.2f}, "
            f"max diff {max_diff:.0f} (0-255 scale).",
            _interpret(manipulation_score),
            "ELA is a weak, easily-misread heuristic and only suggests local "
            "recompression/splicing in a real photo — it does not detect AI "
            "generation. Use it to corroborate, not to decide.",
        ]

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=None,  # Axis 1 left empty on purpose — see module docstring.
            manipulation_score=manipulation_score,
            # Deliberately low: ELA's verdicts are noisy and context-dependent.
            confidence=0.35,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=notes,
            # We expose only the statistics, not the (large) ELA heatmap bytes.
            # A dedicated /heatmap endpoint could render and return the diff image
            # on demand later if a visual is needed.
            raw={"quality": _QUALITY, "max_diff": max_diff, "mean_diff": mean_diff},
        )


def _band_means(diff: Image.Image) -> list[float]:
    """Mean value of each band of the difference image (each in 0..255)."""
    # Image.split() yields one single-channel image per band; ImageStat would
    # also work but split+histogram avoids importing it for one number.
    means: list[float] = []
    for band in diff.split():
        histogram = band.histogram()
        total = sum(histogram)
        if total == 0:
            means.append(0.0)
            continue
        weighted = sum(value * count for value, count in enumerate(histogram))
        means.append(weighted / total)
    return means


def _to_manipulation_score(mean_diff: float, max_diff: float) -> float:
    """Map ELA difference statistics to a manipulation likelihood in [0, 1].

    Higher and more uneven error levels -> higher manipulation_score. We blend
    two normalized components, both clamped to [0, 1]:

      * mean component  — broad elevated error across the frame (re-saving,
                          large edited regions). Weighted most heavily.
      * max component   — a sharp localized spike (a hard splice edge / pasted
                          object boundary).

    Each is a linear ramp from 0 up to a rough full-scale anchor; we weight the
    mean 0.7 and the max 0.3 because a single bright pixel is far weaker evidence
    than a generally raised error floor. The thresholds are intentionally coarse
    given ELA's low reliability.
    """
    mean_component = min(1.0, max(0.0, mean_diff / _MEAN_DIFF_FULL_SCALE))
    max_component = min(1.0, max(0.0, max_diff / _MAX_DIFF_FULL_SCALE))
    score = 0.7 * mean_component + 0.3 * max_component
    return max(0.0, min(1.0, score))


def _interpret(score: float) -> str:
    """Plain-language reading of the manipulation_score."""
    if score < 0.25:
        return (
            "Error levels look fairly uniform — no strong sign of local "
            "recompression or splicing."
        )
    if score < 0.55:
        return (
            "Error levels are somewhat uneven — possible editing or re-saving, "
            "but inconclusive."
        )
    return (
        "Error levels are notably uneven — consistent with local recompression "
        "or splicing, though benign causes (mixed compression history) are "
        "possible."
    )
