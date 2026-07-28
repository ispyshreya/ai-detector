#!/usr/bin/env python3
"""Tests for run_pipeline stage wiring (no GPU).

Focus: the manifest stage must forward ``--wild-real-sources`` so held-out
real-photo sources (phone / ID captures) land in ``test_wild`` instead of
train — the routing that makes the real-photo FPR metric honest.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import run_pipeline


def _make_noise_image(path: Path, seed: int = 0, size: int = 24) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _make_folder(root: Path, name: str, n: int, seed0: int) -> Path:
    """A dataset-root folder holding ``n`` images; its name is the source name."""
    d = root / name
    for i in range(n):
        _make_noise_image(d / f"{i}.jpg", seed=seed0 + i)
    return d


def test_manifest_stage_holds_out_wild_real_sources():
    tmp = Path(tempfile.mkdtemp(prefix="veil_pipeline_test_"))
    # Real-hinted folder names so classify_folder -> (label=0, generator=real);
    # each folder is its own --data-root so `source` == folder name.
    train_reals = _make_folder(tmp, "real_unsplash", 8, 0)
    train_fakes = _make_folder(tmp, "stable_diffusion", 8, 100)
    heldout = _make_folder(tmp, "real_phone_heldout", 6, 200)

    parser = run_pipeline.build_arg_parser()
    manifest_path = tmp / "manifest.csv"
    args = parser.parse_args([
        "--stages", "manifest",
        "--data-root", str(train_reals), str(train_fakes), str(heldout),
        "--manifest", str(manifest_path),
        "--wild-real-sources", "real_phone_heldout",
        "--no-dedup",
    ])
    run_pipeline.stage_manifest(args)

    df = pd.read_csv(manifest_path)
    held = df[df["source"] == "real_phone_heldout"]
    assert len(held) == 6, f"expected 6 held-out rows, got {len(held)}"
    # Every held-out real must be a REAL routed to test_wild, never to train.
    assert (held["label"] == 0).all(), "held-out phone photos must be labelled real"
    assert set(held["split"]) == {"test_wild"}, (
        f"held-out reals leaked out of test_wild: {set(held['split'])}"
    )
    # And the ordinary training reals must NOT be forced into test_wild.
    train_real_rows = df[df["source"] == "real_unsplash"]
    assert "train" in set(train_real_rows["split"]), (
        "ordinary reals should still populate the train split"
    )
    print("PASS test_manifest_stage_holds_out_wild_real_sources")


def test_manifest_stage_wild_generators_folds_flux_into_training():
    tmp = Path(tempfile.mkdtemp(prefix="veil_pipeline_gen_test_"))
    real = _make_folder(tmp, "real_coco", 8, 0)
    flux = _make_folder(tmp, "flux", 8, 100)
    mj = _make_folder(tmp, "midjourney", 8, 200)

    parser = run_pipeline.build_arg_parser()
    manifest_path = tmp / "manifest.csv"
    args = parser.parse_args([
        "--stages", "manifest",
        "--data-root", str(real), str(flux), str(mj),
        "--manifest", str(manifest_path),
        "--wild-generators", "midjourney",
        "--no-dedup",
    ])
    run_pipeline.stage_manifest(args)

    df = pd.read_csv(manifest_path)
    flux_rows = df[df["generator"] == "flux"]
    mj_rows = df[df["generator"] == "midjourney"]
    # flux is now trainable (holding out only midjourney).
    assert (flux_rows["label"] == 1).all(), "flux must be labelled fake"
    assert "test_wild" not in set(flux_rows["split"]), set(flux_rows["split"])
    assert "train" in set(flux_rows["split"])
    # midjourney stays held out for the honest cross-generator test.
    assert set(mj_rows["split"]) == {"test_wild"}
    print("PASS test_manifest_stage_wild_generators_folds_flux_into_training")


def _run_all() -> bool:
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
