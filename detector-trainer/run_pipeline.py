"""End-to-end orchestrator for the Veil in-house image detector.

Wires the four already-built units — data (`data/manifest.py`, `data/dataset.py`),
ResNet baseline (`models/resnet.py` + `train.py`), CLIP contender
(`models/clip_head.py`) and the shared eval harness (`eval/harness.py`) — into
one runnable pipeline plus a Kaggle-friendly CLI (design spec §7).

Stages (selectable via ``--stages``; default runs all in order):

  manifest      build manifest.csv from a resilient, auto-discovering source map
  train_resnet  fine-tune ResNet-50 -> resnet50_best.pt + results.json (train.py)
  train_clip    cache CLIP features once, train BOTH linear + mlp heads
  evaluate      score both models on test splits -> report.md + plots + tables

Design principles honoured here (not re-implemented):
  * The manifest is the single interface between raw folders and models. This
    orchestrator only *discovers* folders and hands specs to the existing
    ``data.manifest.build_manifest`` (ingest -> clean -> dedup -> group-aware
    split). Wild generators (midjourney/dalle3/flux) are routed to ``test_wild``
    by the existing split logic — we never special-case them here.
  * Each stage calls the EXACT entrypoints the units already expose; see the
    per-stage docstrings for the signatures used.

Heavy imports (torch, open_clip, matplotlib) are deferred into the stage
functions so ``--smoke`` runs on a CPU box with no GPU and no model downloads.

Usage
-----
Real run (all stages)::

    python3 run_pipeline.py \
        --manifest data/manifest.csv \
        --data-root /kaggle/input \
        --out runs/veil \
        --resnet-epochs 10 --clip-epochs 200

Smoke (synthetic data, CPU, no downloads, verifies orchestration)::

    python3 run_pipeline.py --smoke --out runs/smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

# Make sibling packages (data/, models/, eval/) importable whether this file is
# run as a script from detector-trainer/ or imported. Mirrors train.py.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))


# =========================================================================== #
# Folder -> (label, generator) mapping — the RESILIENT discovery layer.
# =========================================================================== #
# Any folder name whose lowercased token contains one of these substrings is
# treated as REAL (label 0). Everything else is FAKE (label 1) with
# generator = folder-name lowercased. This is deliberately permissive so a new
# dataset with a "real"/"nature"/"imagenet" folder Just Works.
_REAL_FOLDER_HINTS = (
    "real",
    "nature",
    "imagenet",
    "genuine",
    "authentic",
    "natural",
)

# Normalize common folder spellings to the canonical wild-generator tokens the
# split logic (data.manifest.WILD_GENERATORS) recognises. Everything not listed
# keeps its lowercased folder name as the generator.
_GENERATOR_ALIASES = {
    "sd1.4": "sd1.4",
    "sd1.5": "sd1.5",
    "stable_diffusion_v_1_4": "sd1.4",
    "stable_diffusion_v_1_5": "sd1.5",
    "midjourney": "midjourney",
    "mj": "midjourney",
    "dalle3": "dalle3",
    "dalle-3": "dalle3",
    "dall-e-3": "dalle3",
    "dall_e_3": "dalle3",
    "flux": "flux",
}


def classify_folder(folder_name: str, overrides: Optional[dict] = None) -> tuple[int, str]:
    """Map a leaf folder name -> ``(label, generator)``.

    Precedence:
      1. explicit ``overrides`` (folder-name -> {"label", "generator"} or
         "real"/generator string), lowercased-key matched;
      2. REAL hints (:data:`_REAL_FOLDER_HINTS`) -> ``(0, "real")``;
      3. otherwise FAKE (1) with generator = alias-normalised folder name.

    Returns ``(label:int, generator:str)``. The generator token is what the
    existing group-aware ``split`` uses to route midjourney/dalle3/flux to
    ``test_wild`` — so naming a folder ``midjourney`` is all it takes.
    """
    key = folder_name.strip().lower()
    overrides = {k.lower(): v for k, v in (overrides or {}).items()}

    if key in overrides:
        spec = overrides[key]
        if isinstance(spec, str):
            if spec.lower() in ("real", "0"):
                return 0, "real"
            return 1, _GENERATOR_ALIASES.get(spec.lower(), spec.lower())
        label = int(spec["label"])
        generator = spec.get("generator") or ("real" if label == 0 else key)
        return label, _GENERATOR_ALIASES.get(generator.lower(), generator.lower())

    for hint in _REAL_FOLDER_HINTS:
        if hint in key:
            return 0, "real"

    return 1, _GENERATOR_ALIASES.get(key, key)


def discover_sources(
    data_roots: Iterable[str | Path],
    overrides: Optional[dict] = None,
    source_name_from_root: bool = True,
) -> list[dict]:
    """Auto-discover leaf image folders under each dataset root -> source specs.

    Walks each root's immediate subfolders (and one nested level, to catch the
    ``Ai_generated_dataset/{animals,city,...}`` layout) and emits one
    ``data.manifest`` source spec per folder that directly contains images::

        {"dir": <path>, "label": 0|1, "generator": <token>, "source": <name>}

    The ``source`` is the dataset-root name so the group-aware split keeps each
    dataset's generators grouped. Folder -> (label, generator) is decided by
    :func:`classify_folder`, giving the sane defaults for the two selected
    datasets (GenImage per-generator folders + ai-vs-real categories) while
    remaining resilient to unseen datasets.
    """
    from data.manifest import _IMAGE_EXTENSIONS  # reuse the canonical ext set

    def _has_images(d: Path) -> bool:
        return any(
            p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            for p in d.iterdir()
        )

    specs: list[dict] = []
    seen_dirs: set[str] = set()

    for root in data_roots:
        root = Path(root)
        if not root.exists():
            continue
        source = root.name if source_name_from_root else "dataset"

        # Candidate folders: the root's subtree, but we attach at the *shallowest*
        # folder that holds images so per-generator grouping stays coarse.
        stack = [root]
        while stack:
            current = stack.pop()
            if not current.is_dir():
                continue
            resolved = str(current.resolve())
            if resolved in seen_dirs:
                continue
            if _has_images(current):
                seen_dirs.add(resolved)
                # Use this folder's own name for classification; if it is the
                # root itself, fall back to the root name.
                folder_name = current.name if current != root else source
                label, generator = classify_folder(folder_name, overrides)
                specs.append(
                    {
                        "dir": resolved,
                        "label": label,
                        "generator": generator,
                        "source": source,
                    }
                )
                # Do not descend further into an image-bearing folder.
                continue
            # Otherwise descend into subfolders.
            for child in sorted(current.iterdir()):
                if child.is_dir():
                    stack.append(child)

    return specs


# =========================================================================== #
# Stage 1 — manifest
# =========================================================================== #
def stage_manifest(args) -> dict:
    """Build ``manifest.csv`` via ``data.manifest.build_manifest``.

    Entrypoints called:
      * ``discover_sources(data_roots, overrides)`` (local) -> source specs
      * ``data.manifest.build_manifest(sources, dedup_distance, val_frac,
        test_indist_frac, seed, do_dedup) -> pd.DataFrame``
      * ``data.manifest.write_manifest(df, out_path) -> Path``
    """
    from data.manifest import build_manifest, write_manifest

    overrides = _load_overrides(args.folder_map)
    sources = discover_sources(args.data_root, overrides=overrides)
    if not sources:
        raise SystemExit(
            f"No image folders discovered under {list(args.data_root)!r}. "
            "Pass --data-root pointing at the attached dataset roots."
        )

    print(f"[manifest] discovered {len(sources)} source folders:")
    for s in sources:
        print(f"    {s['source']}/{Path(s['dir']).name}: "
              f"label={s['label']} generator={s['generator']}")

    wild_real_sources = set(getattr(args, "wild_real_sources", []) or [])
    if wild_real_sources:
        discovered = {s["source"] for s in sources}
        missing = wild_real_sources - discovered
        if missing:
            print(f"[manifest] WARNING: --wild-real-sources {sorted(missing)} not "
                  f"among discovered sources {sorted(discovered)}; nothing held out "
                  "for those (check the folder/root names match).")
        print(f"[manifest] holding out real sources into test_wild: "
              f"{sorted(wild_real_sources & discovered)}")

    df = build_manifest(
        sources,
        dedup_distance=args.dedup_distance,
        val_frac=args.val_frac,
        test_indist_frac=args.test_indist_frac,
        seed=args.seed,
        do_dedup=not args.no_dedup,
        wild_sources=wild_real_sources,
    )
    out = write_manifest(df, args.manifest)
    counts = df["split"].value_counts().to_dict()
    print(f"[manifest] wrote {len(df)} rows -> {out}")
    print(f"[manifest] split counts: {counts}")
    return {"manifest": str(out), "n_rows": len(df), "split_counts": counts}


def _load_overrides(folder_map: Optional[str]) -> Optional[dict]:
    if not folder_map:
        return None
    p = Path(folder_map)
    if p.exists():
        return json.loads(p.read_text())
    # Also accept an inline JSON string.
    return json.loads(folder_map)


# =========================================================================== #
# Stage 2 — train_resnet
# =========================================================================== #
def stage_train_resnet(args) -> dict:
    """Fine-tune the ResNet-50 baseline via ``train.train``.

    Entrypoint called (train.py):
      * ``train(argparse.Namespace) -> dict`` — writes ``resnet50_best.pt``,
        per-epoch checkpoints and ``results.json`` into ``out``.

    We construct the exact Namespace ``train.build_arg_parser`` produces rather
    than shelling out, so smoke and real runs share one code path.
    """
    import train as resnet_train

    out_dir = Path(args.out) / "resnet50"
    ns = argparse.Namespace(
        manifest=args.manifest,
        out=str(out_dir),
        model="resnet50",
        epochs=args.resnet_epochs,
        batch_size=args.resnet_batch_size,
        lr=args.resnet_lr,
        weight_decay=1e-4,
        num_workers=args.num_workers,
        seed=args.seed,
        pretrained=args.pretrained,
        class_weight=args.class_weight,
    )
    print(f"[train_resnet] training resnet50 for {ns.epochs} epoch(s) -> {out_dir}")
    results = resnet_train.train(ns)
    return {
        "out_dir": str(out_dir),
        "best_checkpoint": results["best_checkpoint"],
        "best_val_auroc": results["best_val_auroc"],
    }


# =========================================================================== #
# Stage 3 — train_clip
# =========================================================================== #
def stage_train_clip(args, extractor=None) -> dict:
    """Cache CLIP features once, then train BOTH 'linear' and 'mlp' heads.

    Entrypoints called (models/clip_head.py):
      * ``cache_features(dataset, out_dir, backbone_name, split, batch_size,
        device, extractor) -> dict`` — one pass over each split, writes
        ``features_{split}.npy`` + ``labels_{split}.npy``.
      * ``load_cached_features(features_path, labels_path) -> (X, y)``
      * ``train_head(features, labels, head_type, out_dir, backbone_name,
        hidden, epochs, lr, ...) -> TrainHeadResult`` — for head in {linear, mlp}.

    ``extractor`` may be a pre-built / stubbed ``CLIPFeatureExtractor`` (smoke
    path passes a stub so no multi-GB weights download).
    """
    from models.clip_head import cache_features, load_cached_features, train_head
    from data.dataset import VeilDataset, default_transform

    feat_dir = Path(args.out) / "clip_features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    # Cache features once per split we will need (train for fitting, val/test for
    # eval + calibration). Splits with zero rows are skipped.
    splits_to_cache = ["train", "val", "test_indist", "test_wild"]
    cached: dict[str, dict] = {}
    for split in splits_to_cache:
        ds = VeilDataset(args.manifest, split=split, transform=default_transform())
        if len(ds) == 0:
            print(f"[train_clip] split {split!r} empty — skipping cache")
            continue
        meta = cache_features(
            ds,
            out_dir=feat_dir,
            backbone_name=args.clip_backbone,
            split=split,
            batch_size=args.clip_batch_size,
            extractor=extractor,
        )
        cached[split] = meta
        print(f"[train_clip] cached {meta['n']} {split} features "
              f"({meta['backbone_name']}, dim={meta['feat_dim']})")

    if "train" not in cached:
        raise SystemExit("[train_clip] no train features cached; cannot fit heads")

    Xtr, ytr = load_cached_features(
        cached["train"]["features_path"], cached["train"]["labels_path"]
    )

    head_results: dict[str, dict] = {}
    for head_type in ("linear", "mlp"):
        head_out = Path(args.out) / f"clip_{head_type}"
        result = train_head(
            Xtr,
            ytr,
            head_type=head_type,
            out_dir=head_out,
            backbone_name=args.clip_backbone,
            epochs=args.clip_epochs,
            lr=args.clip_lr,
            class_weight=args.class_weight,
            verbose=False,
        )
        head_results[head_type] = {
            "ckpt_path": result.ckpt_path,
            "config_path": result.config_path,
            "final_loss": result.final_loss,
        }
        print(f"[train_clip] trained {head_type} head "
              f"(final_loss={result.final_loss:.4f}) -> {result.ckpt_path}")

    return {"features_dir": str(feat_dir), "cached": cached, "heads": head_results}


# =========================================================================== #
# Stage 4 — evaluate
# =========================================================================== #
def _metadata_collate(batch):
    """Collate ``(tensor, label, meta_dict)`` items into the batch shape
    ``score_dataset`` accepts: ``(stacked_tensor, labels, meta_of_lists)``.
    """
    import torch

    tensors = torch.stack([b[0] for b in batch])
    labels = torch.tensor([int(b[1]) for b in batch])
    keys = ("path", "generator", "split")
    meta = {k: [b[2][k] for b in batch] for k in keys}
    return tensors, labels, meta


class _MetaDataset:
    """Wrap a manifest split so each item carries path/generator/split metadata.

    ``VeilDataset`` yields only ``(tensor, label)``; the harness's
    ``score_dataset`` wants per-row metadata to build the predictions schema
    (path/generator/split). This thin adapter re-reads the split's manifest rows
    (same filter VeilDataset applies) and zips the metadata back on.
    """

    def __init__(self, manifest, split, transform=None):
        import pandas as pd

        from data.dataset import VeilDataset, default_transform

        self.inner = VeilDataset(
            manifest, split=split, transform=transform or default_transform()
        )
        df = manifest if isinstance(manifest, pd.DataFrame) else pd.read_csv(manifest)
        self.rows = df[df["split"] == split].reset_index(drop=True)
        self.split = split

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, i):
        tensor, label = self.inner[i]
        row = self.rows.iloc[i]
        meta = {
            "path": str(row["path"]),
            "generator": str(row.get("generator", "unknown")),
            "split": str(row["split"]),
        }
        return tensor, label, meta


def _score_model_over_splits(model, manifest, splits, batch_size, num_workers):
    """Run ``harness.score_dataset`` over the given splits -> one predictions df."""
    import pandas as pd
    from torch.utils.data import DataLoader

    from eval.harness import score_dataset

    frames = []
    for split in splits:
        ds = _MetaDataset(manifest, split=split)
        if len(ds) == 0:
            continue
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_metadata_collate,
        )
        frames.append(score_dataset(model, loader))
    if not frames:
        from eval.harness import PREDICTION_COLUMNS

        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def stage_evaluate(args, clip_extractor=None) -> dict:
    """Score BOTH models over the test splits and write the 3-panel report.

    Entrypoints called (eval/harness.py):
      * ``score_dataset(model, dataloader) -> predictions_df`` (per model)
      * ``per_generator_table`` / robustness are invoked inside ``write_report``
      * ``write_report(predictions_by_model, out_dir, robustness_by_model,
        in_dist_split, threshold) -> Path``

    Models assembled via:
      * ResNet: ``train.build_model('resnet50')`` + ``load_state_dict``
      * CLIP:   ``clip_head.build_clip_detector(config, head_ckpt, extractor)``
        (best head by final train loss; ``extractor`` stubbed in smoke).

    Both models are scored over the in-distribution held-out split
    (``test_indist``) and the wild split (``test_wild`` — MJ/dalle3/flux),
    which the per-generator table splits into the headline cross-generator panel.
    """
    import torch

    import train as resnet_train
    from models.clip_head import build_clip_detector
    from eval.harness import robustness_eval, write_operating_point, write_report

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report_dir = Path(args.out) / "report"
    predictions_by_model: dict = {}

    test_splits = ["test_indist", "test_wild"]

    # --- ResNet baseline -------------------------------------------------- #
    resnet_ckpt = Path(args.out) / "resnet50" / "resnet50_best.pt"
    if resnet_ckpt.exists():
        model = resnet_train.build_model("resnet50", pretrained=False)
        model.load_state_dict(torch.load(resnet_ckpt, map_location=device))
        model.to(device).eval()
        preds = _score_model_over_splits(
            model, args.manifest, test_splits, args.eval_batch_size, args.num_workers
        )
        predictions_by_model["resnet50"] = preds
        print(f"[evaluate] scored resnet50 on {len(preds)} test rows")
    else:
        print(f"[evaluate] no resnet checkpoint at {resnet_ckpt} — skipping resnet")

    # --- CLIP contender (best of linear/mlp) ------------------------------- #
    best_head = _pick_best_clip_head(args.out)
    if best_head is not None:
        head_type, cfg_path, ckpt_path = best_head
        detector = build_clip_detector(
            cfg_path, ckpt_path, device=str(device), extractor=clip_extractor
        )
        detector.to(device).eval()
        preds = _score_model_over_splits(
            detector, args.manifest, test_splits, args.eval_batch_size, args.num_workers
        )
        predictions_by_model[f"clip_{head_type}"] = preds
        print(f"[evaluate] scored clip_{head_type} on {len(preds)} test rows")
    else:
        print("[evaluate] no CLIP head checkpoints found — skipping clip")

    if not predictions_by_model:
        raise SystemExit("[evaluate] no models to score; run train stages first")

    # --- Robustness panel: re-score wild reals+fakes under perturbation ---- #
    robustness_by_model = _build_robustness(
        args, predictions_by_model, clip_extractor, device
    )

    report_path = write_report(
        predictions_by_model,
        out_dir=report_dir,
        robustness_by_model=robustness_by_model or None,
        in_dist_split="test_indist",
    )
    print(f"[evaluate] wrote report -> {report_path}")

    # Operating point: pick the threshold that meets the real-photo FPR
    # target on the held-out wild reals, and persist it for the backend.
    _wild_preds = next(iter(predictions_by_model.values()))
    op_path = write_operating_point(_wild_preds, report_dir, target_fpr=0.02)
    print(f"[evaluate] wrote operating point -> {op_path}")

    # Persist raw predictions alongside the report for reproducibility.
    for name, preds in predictions_by_model.items():
        preds.to_csv(report_dir / f"predictions_{name}.csv", index=False)

    return {
        "report": str(report_path),
        "models": list(predictions_by_model),
        "robustness": bool(robustness_by_model),
    }


def _pick_best_clip_head(out_root) -> Optional[tuple[str, str, str]]:
    """Return ``(head_type, config_path, ckpt_path)`` for the lower-loss head.

    Reads each head's ``config.json`` + ``clip_head_best.pt``; picks the head
    whose training final loss was lower when both exist, else whichever exists.
    """
    out_root = Path(out_root)
    candidates = []
    for head_type in ("linear", "mlp"):
        d = out_root / f"clip_{head_type}"
        cfg = d / "config.json"
        ckpt = d / "clip_head_best.pt"
        if cfg.exists() and ckpt.exists():
            candidates.append((head_type, str(cfg), str(ckpt)))
    if not candidates:
        return None
    # Prefer mlp (the stronger contender) when both are present; simple + honest.
    for head_type, cfg, ckpt in candidates:
        if head_type == "mlp":
            return head_type, cfg, ckpt
    return candidates[0]


def _build_robustness(args, predictions_by_model, clip_extractor, device):
    """Robustness panel: re-score a sample of wild images under perturbations.

    Uses ``harness.robustness_eval`` with a per-model ``score_images`` callable
    that applies the eval transform + model forward + sigmoid. Returns
    ``{model_name: robustness_df}`` (empty if no wild images available).
    """
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    import train as resnet_train
    from models.clip_head import build_clip_detector
    from data.dataset import default_transform
    from eval.harness import build_perturbations, robustness_eval

    df = pd.read_csv(args.manifest)
    wild = df[df["split"] == "test_wild"]
    if wild.empty:
        return {}

    sample = wild.head(args.robustness_max)
    images, labels = [], []
    for _, row in sample.iterrows():
        try:
            images.append(Image.open(row["path"]).convert("RGB"))
            labels.append(int(row["label"]))
        except Exception:  # noqa: BLE001 - skip unreadable files
            continue
    if not images:
        return {}

    transform = default_transform()

    def _make_scorer(model):
        def score_images(pil_list):
            tensors = torch.stack([transform(img) for img in pil_list]).to(device)
            with torch.no_grad():
                logits = model(tensors).reshape(-1)
                return torch.sigmoid(logits).cpu().numpy()
        return score_images

    perts = build_perturbations()
    out: dict = {}

    resnet_ckpt = Path(args.out) / "resnet50" / "resnet50_best.pt"
    if "resnet50" in predictions_by_model and resnet_ckpt.exists():
        model = resnet_train.build_model("resnet50", pretrained=False)
        model.load_state_dict(torch.load(resnet_ckpt, map_location=device))
        model.to(device).eval()
        out["resnet50"] = robustness_eval(
            _make_scorer(model), images, labels, perturbations=perts
        )

    best_head = _pick_best_clip_head(args.out)
    if best_head is not None:
        head_type, cfg, ckpt = best_head
        key = f"clip_{head_type}"
        if key in predictions_by_model:
            detector = build_clip_detector(
                cfg, ckpt, device=str(device), extractor=clip_extractor
            )
            detector.to(device).eval()
            out[key] = robustness_eval(
                _make_scorer(detector), images, labels, perturbations=perts
            )

    return out


# =========================================================================== #
# Smoke path — synthetic data, stubbed CLIP backbone, CPU, no downloads.
# =========================================================================== #
def _make_smoke_data(root: Path, per_folder: int = 6) -> None:
    """Generate tiny random-noise JPEGs across real + several generators.

    Layout under ``root`` mimics the two selected datasets closely enough to
    exercise discovery + the group-aware split, INCLUDING the held-out wild
    generators (midjourney/dalle3/flux) so test_wild is populated:

        genimage/{real, ADM, sd1.4, midjourney}/*.jpg
        ai_vs_real/{nature, food, dalle3, flux}/*.jpg
    """
    import numpy as np
    from PIL import Image

    layout = {
        "genimage": ["real", "ADM", "sd1.4", "midjourney"],
        "ai_vs_real": ["nature", "food", "dalle3", "flux"],
    }
    rng = np.random.RandomState(0)
    for dataset, folders in layout.items():
        for folder in folders:
            d = root / dataset / folder
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per_folder):
                arr = rng.randint(0, 256, size=(48, 48, 3), dtype="uint8")
                Image.fromarray(arr).save(d / f"{folder}_{i}.jpg", quality=90)


class _StubCLIPExtractor:
    """Offline stand-in for ``CLIPFeatureExtractor`` (reuses the test approach).

    Produces deterministic pseudo-features so ``cache_features`` and
    ``build_clip_detector`` run with NO open_clip weights download. Features are
    a fixed random projection of the input pixels, so they are stable across the
    cache pass and the assembled-detector pass for the same image.
    """

    def __init__(self, feat_dim=768, device="cpu", backbone_name="ViT-L-14"):
        import torch

        self.feat_dim = feat_dim
        self.device = device
        self.backbone_name = backbone_name
        # A tiny fake "model" so any parameter-freezing logic has something to
        # iterate over (matches models/test_clip_head.py's _StubExtractor).
        import torch.nn as nn

        self.model = nn.Linear(4, 4)
        g = torch.Generator().manual_seed(0)
        self._proj = torch.randn(3 * 224 * 224, feat_dim, generator=g)

    def encode_tensor(self, tensor):
        import torch

        with torch.no_grad():
            flat = tensor.reshape(tensor.shape[0], -1).float()
            # Handle the tiny smoke images (already resized to 224 by transform).
            feats = flat @ self._proj
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def extract_features(self, image_tensors_or_paths, batch_size=64):
        """numpy [N, feat_dim] — the interface ``cache_features`` calls."""
        import torch

        tensor = image_tensors_or_paths
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        out = []
        for i in range(0, tensor.shape[0], batch_size):
            out.append(self.encode_tensor(tensor[i : i + batch_size]).cpu().numpy())
        import numpy as np

        return (
            np.concatenate(out, axis=0).astype("float32")
            if out
            else np.empty((0, self.feat_dim), dtype="float32")
        )


def _build_smoke_args(base_args) -> argparse.Namespace:
    """Override args for a fast, tiny, GPU-free run."""
    smoke_root = Path(base_args.out) / "smoke_data"
    _make_smoke_data(smoke_root)
    ns = argparse.Namespace(**vars(base_args))
    ns.data_root = [str(smoke_root / "genimage"), str(smoke_root / "ai_vs_real")]
    ns.manifest = str(Path(base_args.out) / "manifest.csv")
    ns.no_dedup = True          # imagehash optional; noise images dedup poorly
    ns.resnet_epochs = 1
    ns.resnet_batch_size = 4
    ns.eval_batch_size = 4
    ns.clip_epochs = 20
    ns.clip_batch_size = 4
    ns.clip_backbone = "ViT-L-14"
    ns.num_workers = 0
    ns.pretrained = False
    ns.robustness_max = 8
    return ns


# =========================================================================== #
# Driver
# =========================================================================== #
STAGES = ("manifest", "train_resnet", "train_clip", "evaluate")


def run(args) -> dict:
    stages = args.stages if args.stages != ["all"] else list(STAGES)
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stage(s): {unknown}; choose from {STAGES}")

    clip_extractor = None
    if args.smoke:
        args = _build_smoke_args(args)
        clip_extractor = _StubCLIPExtractor(feat_dim=768)
        print(f"[smoke] synthetic data + stubbed CLIP backbone; out={args.out}")

    summary: dict = {"stages": stages, "smoke": bool(args.smoke), "out": str(args.out)}
    for stage in stages:
        print(f"\n===== stage: {stage} =====")
        if stage == "manifest":
            summary["manifest"] = stage_manifest(args)
        elif stage == "train_resnet":
            summary["train_resnet"] = stage_train_resnet(args)
        elif stage == "train_clip":
            summary["train_clip"] = stage_train_clip(args, extractor=clip_extractor)
        elif stage == "evaluate":
            summary["evaluate"] = stage_evaluate(args, clip_extractor=clip_extractor)

    out_summary = Path(args.out) / "pipeline_summary.json"
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[pipeline] summary -> {out_summary}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Veil detector end-to-end pipeline orchestrator."
    )
    p.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        help=f"stages to run (default all): {STAGES} or 'all'",
    )
    p.add_argument("--smoke", action="store_true",
                   help="run the whole pipeline on tiny synthetic data (CPU, no downloads).")
    p.add_argument("--out", default="runs/veil", help="output root dir.")

    # Manifest stage.
    p.add_argument("--data-root", dest="data_root", nargs="+", default=[],
                   help="dataset root dir(s) to auto-discover (e.g. /kaggle/input/*).")
    p.add_argument("--manifest", default="data/manifest.csv",
                   help="manifest CSV path (written by manifest stage, read by others).")
    p.add_argument("--wild-real-sources", dest="wild_real_sources", nargs="*", default=[],
                   help="source names (dataset-root folder names) whose REAL images "
                        "are held out into test_wild instead of train — e.g. phone / "
                        "ID captures used to measure real-photo false positives.")
    p.add_argument("--folder-map", default=None,
                   help="optional JSON file or inline JSON: folder-name -> "
                        "{label,generator} overrides for classification.")
    p.add_argument("--dedup-distance", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-indist-frac", type=float, default=0.15)
    p.add_argument("--no-dedup", action="store_true", help="skip perceptual dedup.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True,
                   help="Inverse-frequency BCE weighting for both models (default on).")

    # ResNet stage.
    p.add_argument("--resnet-epochs", type=int, default=10)
    p.add_argument("--resnet-batch-size", type=int, default=64)
    p.add_argument("--resnet-lr", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true",
                   help="ImageNet-pretrained ResNet init (spec baseline).")

    # CLIP stage.
    p.add_argument("--clip-backbone", default="ViT-L-14",
                   help="CLIP backbone (ViT-L-14 default; ViT-B-16 fallback).")
    p.add_argument("--clip-epochs", type=int, default=200)
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--clip-lr", type=float, default=1e-3)

    # Eval stage.
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--robustness-max", type=int, default=200,
                   help="max wild images used for the robustness panel.")

    p.add_argument("--num-workers", type=int, default=4)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
