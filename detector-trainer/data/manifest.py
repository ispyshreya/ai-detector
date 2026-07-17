"""Data pipeline: ingest → clean → dedup → split → manifest.csv.

The manifest is the single interface between the raw image folders and the
models (see the Veil detector design spec, section 3). No model touches raw
folders directly; every model reads rows from ``manifest.csv``.

Manifest schema (the shared contract other units build against):

    path:str, label:int(0=real,1=fake), generator:str, source:str,
    split:str in {train, val, test_indist, test_wild}

Group-aware split guarantees the wild-set generators (``midjourney``,
``dalle3``, ``flux``) appear ONLY in ``test_wild`` and never leak into
``train``/``val``/``test_indist``.
"""

from __future__ import annotations

import argparse
import io
import random
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

try:  # imagehash is only needed for dedup; keep the import soft for smoke use
    import imagehash
except ImportError:  # pragma: no cover - exercised only when dep is missing
    imagehash = None


# --------------------------------------------------------------------------- #
# Contract constants
# --------------------------------------------------------------------------- #

MANIFEST_COLUMNS = ["path", "label", "generator", "source", "split"]
SPLITS = ("train", "val", "test_indist", "test_wild")

#: Generators that are held out of training entirely and only ever appear in
#: the wild test set (design spec §3, §3a — the honesty check).
WILD_GENERATORS = ("midjourney", "dalle3", "flux")

#: Real images carry this generator token so grouping treats them uniformly.
REAL_GENERATOR = "real"

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff"}


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #

def ingest_directory(
    directory: str | Path,
    label: int,
    generator: str,
    source: str,
) -> pd.DataFrame:
    """Walk an image directory and produce raw manifest rows.

    Every image file below ``directory`` becomes one row labelled by the given
    ``label`` (0=real, 1=fake), ``generator`` and ``source``. No cleaning,
    dedup or splitting happens here — those are separate, testable stages.
    """
    directory = Path(directory)
    if label not in (0, 1):
        raise ValueError(f"label must be 0 (real) or 1 (fake), got {label}")

    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_EXTENSIONS:
            rows.append(
                {
                    "path": str(path.resolve()),
                    "label": int(label),
                    "generator": generator,
                    "source": source,
                    "split": "",  # assigned later by split()
                }
            )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def ingest_sources(sources: Iterable[dict]) -> pd.DataFrame:
    """Ingest a list of source specs into one raw manifest.

    Each spec is a dict with keys ``dir``, ``label``, ``generator``,
    ``source``. Returns the concatenated raw manifest.
    """
    frames = [
        ingest_directory(
            s["dir"],
            label=int(s["label"]),
            generator=s["generator"],
            source=s["source"],
        )
        for s in sources
    ]
    if not frames:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Clean
# --------------------------------------------------------------------------- #

def _is_valid_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()  # cheap integrity check without full decode
        return True
    except Exception:  # noqa: BLE001 - any decode error means "corrupt"
        return False


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose image file is missing or corrupt (PIL ``.verify()``)."""
    if df.empty:
        return df.copy()
    keep = df["path"].map(_is_valid_image)
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Dedup
# --------------------------------------------------------------------------- #

def _phash(path: str | Path):
    with Image.open(path) as img:
        return imagehash.phash(img.convert("RGB"))


def dedup(df: pd.DataFrame, max_distance: int = 0) -> pd.DataFrame:
    """Perceptual-hash dedup so near-duplicates never leak across splits.

    Images whose perceptual hashes are within ``max_distance`` Hamming bits of
    an already-kept image are dropped. ``max_distance=0`` removes exact
    perceptual duplicates; a small value (e.g. 4) also removes near-duplicates.

    Deduping BEFORE splitting is what guarantees a duplicated image cannot land
    on two different sides of the train/test boundary.
    """
    if imagehash is None:
        raise ImportError("dedup() requires the 'imagehash' package")
    if df.empty:
        return df.copy()

    hashes = df["path"].map(_phash)
    kept_indices: list[int] = []
    kept_hashes: list = []
    for idx, h in zip(df.index, hashes):
        is_dup = any((h - kh) <= max_distance for kh in kept_hashes)
        if not is_dup:
            kept_indices.append(idx)
            kept_hashes.append(h)
    return df.loc[kept_indices].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Split (group-aware)
# --------------------------------------------------------------------------- #

def split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_indist_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Group-aware split by generator/source.

    Rules (design spec §3):
    - Rows whose ``generator`` is in :data:`WILD_GENERATORS` go ONLY to
      ``test_wild`` and are never placed in train/val/test_indist.
    - Everything else (real images + training generators) is split into
      ``train`` / ``val`` / ``test_indist``.

    The split is performed per ``(generator, source)`` group so each pool stays
    balanced across generators, and it is deterministic under ``seed``.
    """
    if df.empty:
        return df.assign(split=pd.Series(dtype="object"))

    out = df.copy().reset_index(drop=True)
    out["split"] = ""

    is_wild = out["generator"].isin(WILD_GENERATORS)
    out.loc[is_wild, "split"] = "test_wild"

    rng = random.Random(seed)
    trainable = out[~is_wild]
    for _, group in trainable.groupby(["generator", "source"], sort=True):
        indices = list(group.index)
        rng.shuffle(indices)
        n = len(indices)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_indist_frac))
        # Guarantee at least one train row per group when possible.
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - 1 - n_val))

        val_idx = indices[:n_val]
        test_idx = indices[n_val : n_val + n_test]
        train_idx = indices[n_val + n_test :]

        out.loc[val_idx, "split"] = "val"
        out.loc[test_idx, "split"] = "test_indist"
        out.loc[train_idx, "split"] = "train"

    return out


# --------------------------------------------------------------------------- #
# Leakage-guard utilities (design spec §3a)
# --------------------------------------------------------------------------- #

def reencode_uniform_jpeg(
    path: str | Path,
    out_path: str | Path,
    quality_range: tuple[int, int] = (70, 95),
    seed: int | None = None,
) -> Path:
    """Re-encode an image through a uniform random JPEG quality.

    Both real and fake images are pushed through the SAME quality distribution
    so the detector cannot cheat by reading each source's native JPEG
    quantization fingerprint (design spec §3a).
    """
    lo, hi = quality_range
    if lo > hi:
        raise ValueError("quality_range must be (low, high)")
    rng = random.Random(seed if seed is not None else str(path))
    quality = rng.randint(lo, hi)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        img.convert("RGB").save(out_path, format="JPEG", quality=quality)
    return out_path


def reencode_bytes_uniform_jpeg(
    data: bytes,
    quality_range: tuple[int, int] = (70, 95),
    seed: int | None = None,
) -> bytes:
    """In-memory variant of :func:`reencode_uniform_jpeg` (used in augment)."""
    lo, hi = quality_range
    rng = random.Random(seed)
    quality = rng.randint(lo, hi)
    with Image.open(io.BytesIO(data)) as img:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def match_resolution(
    path: str | Path,
    out_path: str | Path,
    target: tuple[int, int],
) -> Path:
    """Resample an image to a common ``target`` resolution.

    Real and fake pools are matched on resolution before training so the model
    cannot exploit a resolution gap between sources (design spec §3a).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as img:
        resized = img.convert("RGB").resize(target, Image.BICUBIC)
        resized.save(out_path)
    return out_path


def resolution_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(label) median resolution, to expose real/fake mismatch.

    A large gap between the real and fake rows is a leakage red flag; the
    caller matches resolution via :func:`match_resolution` to close it.
    """
    widths, heights = [], []
    for path in df["path"]:
        try:
            with Image.open(path) as img:
                widths.append(img.width)
                heights.append(img.height)
        except Exception:  # noqa: BLE001
            widths.append(None)
            heights.append(None)
    stats = df.assign(_w=widths, _h=heights)
    return (
        stats.groupby("label")[["_w", "_h"]]
        .median()
        .rename(columns={"_w": "median_width", "_h": "median_height"})
        .reset_index()
    )


# --------------------------------------------------------------------------- #
# Build pipeline + IO
# --------------------------------------------------------------------------- #

def build_manifest(
    sources: Iterable[dict],
    dedup_distance: int = 0,
    val_frac: float = 0.15,
    test_indist_frac: float = 0.15,
    seed: int = 42,
    do_dedup: bool = True,
) -> pd.DataFrame:
    """Run the full pipeline: ingest → clean → dedup → split."""
    df = ingest_sources(sources)
    df = clean(df)
    if do_dedup:
        df = dedup(df, max_distance=dedup_distance)
    df = split(
        df,
        val_frac=val_frac,
        test_indist_frac=test_indist_frac,
        seed=seed,
    )
    return df[MANIFEST_COLUMNS].reset_index(drop=True)


def write_manifest(df: pd.DataFrame, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(MANIFEST_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_source_spec(spec: str) -> dict:
    """Parse one ``--sources`` entry.

    Format: ``dir=<path>,label=<0|1>,generator=<name>,source=<name>``
    """
    fields = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"bad source field '{part}' (expected key=value)"
            )
        key, value = part.split("=", 1)
        fields[key.strip()] = value.strip()

    required = {"dir", "label", "generator", "source"}
    missing = required - set(fields)
    if missing:
        raise argparse.ArgumentTypeError(
            f"source spec missing {sorted(missing)} in '{spec}'"
        )
    fields["label"] = int(fields["label"])
    return fields


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Veil detector data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build manifest.csv")
    build.add_argument(
        "--sources",
        nargs="+",
        required=True,
        type=_parse_source_spec,
        help=(
            "one or more source specs: "
            "'dir=<path>,label=<0|1>,generator=<name>,source=<name>'"
        ),
    )
    build.add_argument(
        "--out",
        default="detector-trainer/data/manifest.csv",
        help="output manifest path",
    )
    build.add_argument("--dedup-distance", type=int, default=0)
    build.add_argument("--val-frac", type=float, default=0.15)
    build.add_argument("--test-indist-frac", type=float, default=0.15)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument(
        "--no-dedup", action="store_true", help="skip perceptual dedup"
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        df = build_manifest(
            args.sources,
            dedup_distance=args.dedup_distance,
            val_frac=args.val_frac,
            test_indist_frac=args.test_indist_frac,
            seed=args.seed,
            do_dedup=not args.no_dedup,
        )
        out = write_manifest(df, args.out)
        counts = df["split"].value_counts().to_dict()
        print(f"Wrote {len(df)} rows to {out}")
        print(f"Split counts: {counts}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
