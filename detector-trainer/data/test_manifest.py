"""Unit tests for the Veil detector data pipeline.

Run directly with any python on PATH::

    python detector-trainer/data/test_manifest.py

Tests generate tiny synthetic images (no dataset downloads, no training),
build a manifest, and assert the shared contract holds. The torch-dependent
``VeilDataset`` test is skipped gracefully if torch/torchvision are absent, but
all pure pandas/PIL tests always run.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

# Allow running as a bare script (python .../test_manifest.py).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import manifest as M  # noqa: E402

try:
    import torch  # noqa: F401,E402
    import torchvision  # noqa: F401,E402

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def _make_noise_image(path: Path, size: int = 24, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def _build_synthetic_tree(root: Path) -> list[dict]:
    """Create images across generators and return source specs.

    Layout: real (2 sources) + stable-diffusion (train fake) +
    midjourney/dalle3/flux (wild fakes). A deliberate duplicate is planted in
    stable-diffusion so dedup has something to remove.
    """
    specs = []
    layout = [
        ("real_a", 0, M.REAL_GENERATOR, "flickr", 6),
        ("real_b", 0, M.REAL_GENERATOR, "coco", 6),
        ("sd", 1, "stable-diffusion", "genimage", 6),
        ("mj", 1, "midjourney", "wild", 4),
        ("dalle", 1, "dalle3", "wild", 4),
        ("flux", 1, "flux", "wild", 4),
    ]
    seed = 0
    for folder, label, generator, source, n in layout:
        d = root / folder
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _make_noise_image(d / f"img_{i}.png", seed=seed)
            seed += 1
        specs.append(
            {"dir": str(d), "label": label, "generator": generator, "source": source}
        )

    # Plant an exact duplicate of an existing sd image -> dedup target.
    sd_dir = root / "sd"
    shutil.copy(sd_dir / "img_0.png", sd_dir / "img_0_dup.png")

    # Plant a corrupt file -> clean() target.
    (sd_dir / "corrupt.png").write_bytes(b"not really a png")

    return specs


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_schema_and_columns(df):
    assert list(df.columns) == M.MANIFEST_COLUMNS, df.columns.tolist()
    assert set(df["split"].unique()) <= set(M.SPLITS)
    assert df["label"].isin([0, 1]).all()
    print("PASS test_schema_and_columns")


def test_clean_drops_corrupt(df):
    # The corrupt file must not survive into the manifest.
    assert not df["path"].str.contains("corrupt.png").any()
    print("PASS test_clean_drops_corrupt")


def test_dedup_removes_duplicate(specs):
    # With dedup on, the planted exact duplicate is removed; with dedup off,
    # both copies survive. Difference proves dedup fired.
    with_dedup = M.build_manifest(specs, do_dedup=True)
    without_dedup = M.build_manifest(specs, do_dedup=False)
    assert len(with_dedup) < len(without_dedup), (
        len(with_dedup),
        len(without_dedup),
    )
    print(
        f"PASS test_dedup_removes_duplicate "
        f"(deduped={len(with_dedup)} vs raw={len(without_dedup)})"
    )


def test_splits_disjoint_by_path(df):
    groups = {s: set(df[df["split"] == s]["path"]) for s in M.SPLITS}
    seen = set()
    for s, paths in groups.items():
        overlap = seen & paths
        assert not overlap, f"split {s} overlaps earlier splits: {overlap}"
        seen |= paths
    print("PASS test_splits_disjoint_by_path")


def test_wild_generators_only_in_wild(df):
    for gen in M.WILD_GENERATORS:
        splits = set(df[df["generator"] == gen]["split"])
        assert splits <= {"test_wild"}, f"{gen} leaked into {splits}"
    # And test_wild must contain ONLY wild generators.
    wild_gens = set(df[df["split"] == "test_wild"]["generator"])
    assert wild_gens <= set(M.WILD_GENERATORS), wild_gens
    # Train/val/indist must never contain a wild generator.
    trainable = df[df["split"].isin(["train", "val", "test_indist"])]
    assert not trainable["generator"].isin(M.WILD_GENERATORS).any()
    print("PASS test_wild_generators_only_in_wild")


def test_reencode_uniform_jpeg(root, df):
    src = df["path"].iloc[0]
    out = root / "reencoded.jpg"
    result = M.reencode_uniform_jpeg(src, out, quality_range=(70, 90), seed=1)
    assert result.exists() and result.stat().st_size > 0
    with Image.open(result) as img:
        assert img.format == "JPEG"
    print("PASS test_reencode_uniform_jpeg")


def test_match_resolution(root, df):
    src = df["path"].iloc[0]
    out = root / "matched.png"
    M.match_resolution(src, out, target=(64, 64))
    with Image.open(out) as img:
        assert img.size == (64, 64)
    print("PASS test_match_resolution")


def test_veil_dataset(df):
    if not _HAS_TORCH:
        print("SKIP test_veil_dataset (torch/torchvision not installed)")
        return
    from dataset import VeilDataset, default_transform

    ds = VeilDataset(df, split="train", transform=default_transform())
    assert len(ds) > 0, "expected training rows"
    tensor, label = ds[0]
    assert tuple(tensor.shape) == (3, 224, 224), tensor.shape
    assert int(label) in (0, 1)

    wild = VeilDataset(df, split="test_wild")
    assert len(wild) > 0
    wt, wl = wild[0]
    assert tuple(wt.shape) == (3, 224, 224)
    print("PASS test_veil_dataset (tensors [3,224,224])")


def test_split_routes_wild_sources_to_test_wild():
    import pandas as pd
    import importlib.util as _ilu
    from pathlib import Path as _P

    _mpath = _P(__file__).resolve().parent / "manifest.py"
    _spec = _ilu.spec_from_file_location("veil_manifest_wildsrc", _mpath)
    m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(m)

    df = pd.DataFrame(
        {
            "path": [f"/x/{i}.jpg" for i in range(10)],
            "label": [0] * 10,
            "generator": ["real"] * 10,
            "source": ["held_out_phone"] * 5 + ["train_reals"] * 5,
            "split": [""] * 10,
        },
        columns=m.MANIFEST_COLUMNS,
    )
    out = m.split(df, wild_sources={"held_out_phone"})
    held = out[out["source"] == "held_out_phone"]
    trainable = out[out["source"] == "train_reals"]
    assert set(held["split"]) == {"test_wild"}
    assert "test_wild" not in set(trainable["split"])
    print("PASS test_split_routes_wild_sources_to_test_wild")


def test_split_wild_generators_override_makes_flux_trainable():
    import pandas as pd
    import importlib.util as _ilu
    from pathlib import Path as _P

    _mpath = _P(__file__).resolve().parent / "manifest.py"
    _spec = _ilu.spec_from_file_location("veil_manifest_wildgen", _mpath)
    m = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(m)

    # 10 flux + 10 midjourney fakes + 10 reals.
    df = pd.DataFrame(
        {
            "path": [f"/x/{i}.jpg" for i in range(30)],
            "label": [1] * 20 + [0] * 10,
            "generator": ["flux"] * 10 + ["midjourney"] * 10 + ["real"] * 10,
            "source": ["gen"] * 20 + ["reals"] * 10,
            "split": [""] * 30,
        },
        columns=m.MANIFEST_COLUMNS,
    )
    # Hold out ONLY midjourney; flux should now be trainable.
    out = m.split(df, wild_generators=("midjourney",))
    flux = out[out["generator"] == "flux"]
    mj = out[out["generator"] == "midjourney"]
    assert "test_wild" not in set(flux["split"]), set(flux["split"])
    assert set(flux["split"]) <= {"train", "val", "test_indist"}
    assert set(mj["split"]) == {"test_wild"}
    print("PASS test_split_wild_generators_override_makes_flux_trainable")


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="veil_manifest_test_"))
    try:
        specs = _build_synthetic_tree(tmp)
        df = M.build_manifest(specs, do_dedup=True)

        # Write + read round-trip to exercise IO and schema validation.
        out = tmp / "manifest.csv"
        M.write_manifest(df, out)
        df = M.read_manifest(out)

        test_schema_and_columns(df)
        test_clean_drops_corrupt(df)
        test_dedup_removes_duplicate(specs)
        test_splits_disjoint_by_path(df)
        test_wild_generators_only_in_wild(df)
        test_reencode_uniform_jpeg(tmp, df)
        test_match_resolution(tmp, df)
        test_veil_dataset(df)
        test_split_routes_wild_sources_to_test_wild()

        print("\nALL TESTS PASSED" + ("" if _HAS_TORCH else " (torch test skipped)"))
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
