"""Diagnostic tool for the local-model patch pipeline.

NOT part of the live API. Run manually to compare patch-geometry
configurations and aggregation formulas side-by-side on a labeled set of
real/AI images, isolating which factor (overlap, resizing, blank-patch
filtering, aggregation formula) drives a score change.

Usage:
    python backend/scripts/diagnose_patch_pipeline.py img1.jpg:REAL img2.jpg:AI ...

Each argument is `path:LABEL` where LABEL is REAL or AI.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/

from PIL import Image  # noqa: E402

from app.signals.local_model import (  # noqa: E402
    MAX_PATCHES,
    Patch,
    _load_model,
    aggregate_patch_scores,
    extract_patch_coords,
    filter_blank_patches,
    pad_to_min_size,
    resize_longest_side,
    score_patches,
    subsample_evenly,
    suppress_overlapping_patches,
)

# name, patch size, stride, apply the 1024 resize cap?
CONFIGS = [
    ("A_nonoverlap_native  (32/32, no resize)", 32, 32, False),
    ("B_overlap_native     (32/16, no resize)", 32, 16, False),
    ("C_overlap_resized    (32/16, resize<=1024)", 32, 16, True),
]


def run_config(image, patch: int, stride: int, apply_resize: bool, model, device, filter_blanks: bool = True):
    img = resize_longest_side(image) if apply_resize else image
    img = pad_to_min_size(img, min_size=patch)
    width, height = img.size

    all_coords = extract_patch_coords(width, height, tile=patch, stride=stride)
    coords = filter_blank_patches(img, all_coords, tile=patch) if filter_blanks else all_coords
    coords = subsample_evenly(coords, MAX_PATCHES)

    scores = score_patches(img, coords, model, device, tile=patch)
    patches = [Patch(x=x, y=y, score=s) for (x, y), s in zip(coords, scores)]
    agg = aggregate_patch_scores(scores, patches)
    return img.size, len(all_coords), len(coords), agg


def print_row(label: str, size, extracted: int, used: int, agg) -> None:
    print(f"    [{label}] size={size} extracted={extracted} used={used}")
    print(
        f"        mean={agg.mean:.3f}  median={agg.median:.3f}  trimmed10%={agg.trimmed_mean:.3f}  "
        f"top10%={agg.top10_mean:.3f}  top20%={agg.top20_mean:.3f}  max={agg.max_score:.3f}  std={agg.std:.3f}"
    )
    print(f"        pct>0.5={agg.pct_above_half:.3f}  pct>0.7={agg.pct_above_0_7:.3f}")
    print(f"        LEGACY (0.5*top20+0.3*median+0.2*mean) = {agg.legacy_score:.4f}")
    print(f"        CONSERVATIVE (0.6*median+0.4*trimmed)  = {agg.final_score:.4f}")
    hist = "  ".join(f"{k}:{v}" for k, v in agg.histogram.items())
    print(f"        histogram: {hist}")
    top5 = list(agg.top_patches)[:5]
    print("        top-5 patches: " + ", ".join(f"({p.x},{p.y})={p.score:.3f}" for p in top5))


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    model, device, checkpoint = _load_model()
    print(f"checkpoint: {checkpoint}\n")

    summary: list[tuple[str, str, str, float, float]] = []  # image,label,config,legacy,conservative

    for arg in sys.argv[1:]:
        path, _, label = arg.rpartition(":")
        path = path or arg
        label = label or "?"
        image = Image.open(path).convert("RGB")
        name = Path(path).stem
        print(f"=== {name} [{label}]  original={image.size} ===")

        for config_label, patch, stride, apply_resize in CONFIGS:
            size, extracted, used, agg = run_config(image, patch, stride, apply_resize, model, device)
            print_row(config_label, size, extracted, used, agg)
            summary.append((name, label, config_label, agg.legacy_score, agg.final_score))

        # Ablation: blank-patch filtering on vs off, on config C's geometry.
        img = resize_longest_side(image)
        img = pad_to_min_size(img, min_size=32)
        all_coords = extract_patch_coords(*img.size, tile=32, stride=16)
        coords_filtered = subsample_evenly(filter_blank_patches(img, all_coords), MAX_PATCHES)
        coords_unfiltered = subsample_evenly(all_coords, MAX_PATCHES)
        scores_filtered = score_patches(img, coords_filtered, model, device)
        scores_unfiltered = score_patches(img, coords_unfiltered, model, device)
        agg_f = aggregate_patch_scores(scores_filtered)
        agg_u = aggregate_patch_scores(scores_unfiltered)
        print(
            f"    [blank-filter ablation] filtered: used={len(coords_filtered)} "
            f"conservative={agg_f.final_score:.4f}  |  unfiltered: used={len(coords_unfiltered)} "
            f"conservative={agg_u.final_score:.4f}"
        )

        # Spatial suppression on config C's patches.
        _, _, _, agg_c = run_config(image, 32, 16, True, model, device)
        patches_c = list(agg_c.top_patches)
        # Re-run to get the FULL patch list (top_patches is capped at 20) for suppression.
        img_c = pad_to_min_size(resize_longest_side(image), min_size=32)
        coords_c = subsample_evenly(
            filter_blank_patches(img_c, extract_patch_coords(*img_c.size, tile=32, stride=16)), MAX_PATCHES
        )
        scores_c = score_patches(img_c, coords_c, model, device)
        full_patches_c = [Patch(x=x, y=y, score=s) for (x, y), s in zip(coords_c, scores_c)]
        suppressed = suppress_overlapping_patches(full_patches_c)
        suppressed_agg = aggregate_patch_scores([p.score for p in suppressed])
        print(
            f"    [spatial suppression, config C] {len(full_patches_c)} patches -> "
            f"{len(suppressed)} independent regions  |  conservative-on-suppressed={suppressed_agg.final_score:.4f}"
        )
        print()

    print("\n=== SUMMARY: legacy vs conservative, by config ===")
    for config_label, _, _, _ in CONFIGS:
        real_legacy = [s[3] for s in summary if s[1] == "REAL" and s[2] == config_label]
        real_cons = [s[4] for s in summary if s[1] == "REAL" and s[2] == config_label]
        ai_legacy = [s[3] for s in summary if s[1] == "AI" and s[2] == config_label]
        ai_cons = [s[4] for s in summary if s[1] == "AI" and s[2] == config_label]
        if not real_legacy or not ai_legacy:
            continue
        rl, rc = sum(real_legacy) / len(real_legacy), sum(real_cons) / len(real_cons)
        al, ac = sum(ai_legacy) / len(ai_legacy), sum(ai_cons) / len(ai_cons)
        print(f"{config_label}")
        print(f"    REAL avg  legacy={rl:.3f}  conservative={rc:.3f}")
        print(f"    AI   avg  legacy={al:.3f}  conservative={ac:.3f}")
        print(f"    separation (AI-REAL)  legacy={al - rl:+.3f}  conservative={ac - rc:+.3f}")


if __name__ == "__main__":
    main()
