#!/usr/bin/env python3
"""Acquire diverse real-world photos to fix the detector's real-photo false positives.

The trained ResNet over-predicts FAKE on real captures because it learned to
separate GenImage's ImageNet-sourced reals from a diffusion dump, not real-vs-AI.
The fix is diverse, multi-source REAL training data plus a held-out real set that
looks like phone captures, so real-photo FPR can be measured honestly.

Sources (all no-API-key except FFHQ, which streams from HuggingFace if available):

    real_coco/            COCO val2017      — diverse everyday scenes (train reals)
    real_ffhq/            FFHQ              — human faces / selfie-adjacent (train reals)
    real_unsplash/        picsum.photos     — Unsplash-backed web photos (train reals)
    real_div2k_heldout/   DIV2K valid HR    — high-res camera-native photos, held out
                                              as the phone-capture proxy (WILD real)

Folder names carry a "real" hint so run_pipeline.discover_sources classifies them
as REAL, and each folder is its own dataset root so `source` == folder name. Pass
the *_heldout folders to `run_pipeline.py --wild-real-sources` (see USAGE) so they
route into test_wild instead of leaking into training.

Resumable: existing files are skipped, so re-running tops each source up to target.

USAGE
    # Acquire (tune counts to taste; DIV2K valid HR is only ~100 images total)
    python3 acquire_reals.py --out real_world \
        --coco 2000 --ffhq 1000 --unsplash 1000 --div2k 100

    # Then build the manifest on Kaggle (or locally) with the held-out real routed
    # to test_wild — each real folder is its own --data-root:
    python3 run_pipeline.py --stages manifest \
        --data-root <genimage roots...> \
                    real_world/real_coco real_world/real_ffhq \
                    real_world/real_unsplash real_world/real_div2k_heldout \
        --wild-real-sources real_div2k_heldout \
        --manifest data/manifest.csv

RAISE is not auto-downloaded (it requires a manual access request); drop any RAISE
images into an extra real_raise_heldout/ folder and add it to --wild-real-sources.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "veil-detector/acquire_reals (research; contact via repo)"
COCO_VAL_ZIP = "http://images.cocodataset.org/zips/val2017.zip"
DIV2K_VALID_HR_ZIP = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"


def _existing(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if p.is_file() and not p.name.startswith("."))


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #
def acquire_unsplash(out: Path, count: int) -> int:
    """picsum.photos — Unsplash-backed, infinite, no key. One request per image."""
    folder = out / "real_unsplash"
    folder.mkdir(parents=True, exist_ok=True)
    have = _existing(folder)
    got = 0
    for seed in range(have, count):
        dest = folder / f"unsplash_{seed:05d}.jpg"
        if dest.exists():
            continue
        # Deterministic per-seed image at a phone-ish resolution.
        url = f"https://picsum.photos/seed/veil{seed}/1024/768.jpg"
        try:
            dest.write_bytes(_http_get(url))
            got += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [unsplash] seed {seed} failed: {e}", file=sys.stderr)
        if got and got % 50 == 0:
            print(f"  [unsplash] {have + got}/{count}")
    return got


def _acquire_from_zip(url: str, folder: Path, count: int, label: str) -> int:
    """Stream a zip into memory, extract up to `count` images into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    have = _existing(folder)
    if have >= count:
        print(f"  [{label}] already have {have} >= {count}; skipping download")
        return 0
    print(f"  [{label}] downloading {url} (this can be large) ...")
    try:
        raw = _http_get(url, timeout=600)
    except Exception as e:  # noqa: BLE001
        print(f"  [{label}] download failed: {e}", file=sys.stderr)
        return 0
    got = 0
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        members = [
            m for m in zf.namelist()
            if m.lower().endswith((".jpg", ".jpeg", ".png")) and not m.endswith("/")
        ]
        members.sort()
        for m in members:
            if have + got >= count:
                break
            dest = folder / f"{label}_{Path(m).name}"
            if dest.exists():
                continue
            with zf.open(m) as src:
                dest.write_bytes(src.read())
            got += 1
            if got % 200 == 0:
                print(f"  [{label}] {have + got}/{count}")
    return got


def acquire_coco(out: Path, count: int) -> int:
    return _acquire_from_zip(COCO_VAL_ZIP, out / "real_coco", count, "coco")


def acquire_div2k(out: Path, count: int) -> int:
    # Held-out camera-native photos: the phone-capture proxy for the WILD real set.
    return _acquire_from_zip(
        DIV2K_VALID_HR_ZIP, out / "real_div2k_heldout", count, "div2k"
    )


def acquire_ffhq(out: Path, count: int) -> int:
    """FFHQ faces via HuggingFace streaming (optional dep). Skips if unavailable."""
    folder = out / "real_ffhq"
    folder.mkdir(parents=True, exist_ok=True)
    have = _existing(folder)
    if have >= count:
        print(f"  [ffhq] already have {have} >= {count}; skipping")
        return 0
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        print(
            "  [ffhq] `datasets` not installed; skipping FFHQ. Install with "
            "`pip install datasets` or drop face images into real_ffhq/ manually.",
            file=sys.stderr,
        )
        return 0
    got = 0
    try:
        ds = load_dataset("merkol/ffhq-256", split="train", streaming=True)
        for i, row in enumerate(ds):
            if have + got >= count:
                break
            dest = folder / f"ffhq_{have + got:05d}.png"
            if dest.exists():
                continue
            img = row.get("image")
            if img is None:
                continue
            img.convert("RGB").save(dest)
            got += 1
            if got % 200 == 0:
                print(f"  [ffhq] {have + got}/{count}")
    except Exception as e:  # noqa: BLE001
        print(f"  [ffhq] streaming failed: {e}", file=sys.stderr)
    return got


SOURCES = {
    "coco": acquire_coco,
    "ffhq": acquire_ffhq,
    "unsplash": acquire_unsplash,
    "div2k": acquire_div2k,
}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="real_world", help="output root for real folders.")
    p.add_argument("--coco", type=int, default=0, help="target COCO val2017 count.")
    p.add_argument("--ffhq", type=int, default=0, help="target FFHQ face count.")
    p.add_argument("--unsplash", type=int, default=0, help="target picsum/Unsplash count.")
    p.add_argument("--div2k", type=int, default=0,
                   help="target DIV2K held-out count (phone-capture proxy).")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = {"coco": args.coco, "ffhq": args.ffhq,
               "unsplash": args.unsplash, "div2k": args.div2k}
    if not any(targets.values()):
        print("Nothing to do: pass at least one of --coco/--ffhq/--unsplash/--div2k.")
        return 1
    summary = {}
    for name, target in targets.items():
        if target <= 0:
            continue
        print(f"[{name}] target {target} -> {out}/")
        summary[name] = SOURCES[name](out, target)
    print("\n=== acquired this run ===")
    for name, n in summary.items():
        folder = out / (f"real_{name}" if name != "div2k" else "real_div2k_heldout")
        print(f"  {name}: +{n}  (total on disk: {_existing(folder)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
