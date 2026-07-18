"""Training entry point for the Veil in-house image detector.

This module is the single interface between the detector-trainer and the backend:
``backend/app/signals/local_model.py`` imports ``build_model`` from *this file* to
reconstruct the architecture before loading a checkpoint. Keep ``build_model``'s
signature and behavior stable — it is a shared contract with the backend and with
the parallel data / eval / CLIP units.

Run training from the CLI, e.g.::

    python detector-trainer/train.py \
        --manifest detector-trainer/data/manifest.csv \
        --out runs/resnet50 \
        --model resnet50 --epochs 10

It writes ``resnet50_best.pt`` (plain state_dict, best val AUROC) plus a
per-epoch ``resnet50_epoch{N}.pt`` and a ``results.json`` metrics dict into
``--out``. Data is loaded through ``VeilDataset`` (data agent's module); its
import is GUARDED so this file, and the model unit tests, work before that module
lands.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# Model builders live under models/. Support running both as a script (bare
# "models.resnet") and as part of a package by fixing sys.path to this dir.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models.resnet import build_resnet50  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants (mirror local_model.py's transform contract exactly).
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model factory — the contract imported by the backend signal.
# ---------------------------------------------------------------------------
def build_model(name: str, pretrained: bool = False) -> nn.Module:
    """Return the detector architecture for ``name``.

    Dispatch:
      * ``"resnet50"`` / any ``"resnet*"`` -> the ResNet-50 baseline.
      * ``"clip"`` / ``"clip*"``           -> RESERVED extension point (see below).

    Called by ``backend/app/signals/local_model.py`` as
    ``build_model(settings.local_model_name, pretrained=False)`` before
    ``load_state_dict``, so ``pretrained`` defaults to False and the returned
    module's ``forward`` must map ``[B, 3, 224, 224]`` -> ``[B]``/``[B, 1]`` logits.

    Args:
        name: model identifier (case-insensitive).
        pretrained: initialize backbone from ImageNet weights when supported.

    Returns:
        An ``nn.Module`` conforming to the inference contract.
    """
    key = name.lower().strip()

    if key.startswith("resnet"):
        # Currently only resnet50 is implemented; the "resnet*" prefix leaves room
        # for e.g. resnet18 smoke-test variants without breaking the contract.
        return build_resnet50(pretrained=pretrained)

    # --- EXTENSION POINT: CLIP contender -----------------------------------
    # The spec (§6) says: "If CLIP wins, add a thin build_model branch assembling
    # 'frozen CLIP + head' behind the same model(tensor) -> logit interface."
    # That model is built by the *contender* unit (detector-trainer/models/
    # clip_head.py) and is intentionally NOT implemented here. When it lands,
    # wire it up as:
    #
    #     if key.startswith("clip"):
    #         from models.clip_head import build_clip_head
    #         return build_clip_head(pretrained=pretrained)
    #
    # Do not implement CLIP in this branch — keep the hook thin so the two units
    # stay isolated (spec §2 "each model is an isolated unit").
    # -----------------------------------------------------------------------

    raise ValueError(
        f"Unknown model name {name!r}. Supported: 'resnet50' (and 'resnet*'); "
        f"'clip*' is reserved for the CLIP contender (not yet implemented)."
    )


# ---------------------------------------------------------------------------
# Transforms (spec §4: hflip + JPEG recompression + resize jitter for train;
# plain resize/normalize for eval — the latter must match local_model.py).
# ---------------------------------------------------------------------------
class RandomJPEGRecompression:
    """Randomly re-encode a PIL image as JPEG at a random quality.

    Real-world images the detector sees in production have been through lossy
    JPEG pipelines. Training on pristine PNGs teaches the model to key on
    compression fingerprints (the leakage failure mode in spec §3a). Re-encoding
    both reals and fakes through the *same* random quality range breaks that
    shortcut and hardens the model against wild-set degradation.
    """

    def __init__(self, p: float = 0.5, quality_range: tuple[int, int] = (50, 95)) -> None:
        self.p = p
        self.quality_range = quality_range

    def __call__(self, img):
        # Imported lazily so importing train.py stays cheap and Pillow-optional
        # for the model unit tests (which never call the transforms).
        import io

        from PIL import Image

        if random.random() > self.p:
            return img
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


def build_train_transform() -> transforms.Compose:
    """Train-time augmentation: hflip + JPEG recompression + resize jitter.

    Resize jitter = ``RandomResizedCrop`` around the full frame (scale 0.8-1.0):
    a light scale/aspect wobble so the detector is robust to the rescaling that
    real uploads undergo, without cropping away enough to lose global cues.
    Normalization matches the eval transform / local_model.py.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            (IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        RandomJPEGRecompression(p=0.5, quality_range=(50, 95)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform() -> transforms.Compose:
    """Deterministic eval transform — byte-for-byte the local_model.py contract."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Data loading — guarded import of the data agent's VeilDataset.
# ---------------------------------------------------------------------------
def _load_veil_dataset(manifest: str, split: str, transform):
    """Instantiate ``VeilDataset`` from the data unit, guarded on availability.

    The dataset module (``detector-trainer/data/dataset.py``) is built in parallel
    by the data agent. We import it lazily and raise a clear, actionable error if
    it is missing, rather than failing at module import — that keeps train.py and
    the model unit tests usable before the data unit lands.
    """
    try:
        from data.dataset import VeilDataset
    except ImportError as exc:  # data unit not present yet
        raise ImportError(
            "detector-trainer/data/dataset.py (VeilDataset) is not available. "
            "It is provided by the data agent; ensure it exists before training. "
            f"Original error: {exc}"
        ) from exc

    return VeilDataset(manifest, split=split, transform=transform)


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------
@torch.no_grad()
def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """AUROC via the rank-sum (Mann-Whitney U) identity — no sklearn dependency.

    Returns 0.5 for degenerate batches (only one class present), which is the
    correct "no signal" value and avoids a divide-by-zero.
    """
    scores = scores.detach().float().flatten()
    labels = labels.detach().float().flatten()
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks (1-based) handle ties correctly.
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    sum_ranks_pos = ranks[labels == 1].sum().item()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# ---------------------------------------------------------------------------
# Train / eval loops.
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss, n = 0.0, 0
    all_scores, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).float()
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        n += images.size(0)
        all_scores.append(torch.sigmoid(logits).cpu())
        all_labels.append(labels.cpu())
    scores = torch.cat(all_scores) if all_scores else torch.zeros(0)
    labels = torch.cat(all_labels) if all_labels else torch.zeros(0)
    preds = (scores >= 0.5).float()
    acc = float((preds == labels).float().mean()) if len(labels) else 0.0
    return {
        "loss": total_loss / max(n, 1),
        "auroc": _auroc(scores, labels),
        "accuracy": acc,
        "n": n,
    }


def set_seed(seed: int) -> None:
    """Fixed seed for the shared, honest comparison (spec §4 conventions)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_pos_weight(manifest, split, device):
    """Inverse-frequency weight for BCE positives (fake), to counter imbalance.

    pos_weight = n_real / n_fake on the given split, so the majority class stops
    dominating the loss. Returns None if a class is empty.
    """
    import pandas as pd

    df = pd.read_csv(manifest)
    df = df[df["split"] == split]
    n_pos = float((df["label"] == 1).sum())
    n_neg = float((df["label"] == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    return torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)


def train(args: argparse.Namespace) -> dict:
    """Full training run: build model, fit ``args.epochs``, checkpoint, report."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args.model, pretrained=args.pretrained).to(device)
    pos_weight = _compute_pos_weight(args.manifest, "train", device) if args.class_weight else None
    if pos_weight is not None:
        print(f"class weighting on: pos_weight={pos_weight.item():.3f}", flush=True)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_ds = _load_veil_dataset(args.manifest, "train", build_train_transform())
    val_ds = _load_veil_dataset(args.manifest, "val", build_eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    best_auroc = -1.0
    history = []
    best_ckpt = out_dir / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  val_auroc={val_metrics['auroc']:.4f}  "
            f"val_acc={val_metrics['accuracy']:.4f}",
            flush=True,
        )

        # Per-epoch checkpoint (session-safe: Kaggle sessions can be killed at 12h).
        torch.save(model.state_dict(), out_dir / f"{args.model}_epoch{epoch}.pt")

        # Track best-on-val — this becomes the exported {model}_best.pt.
        if val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            torch.save(model.state_dict(), best_ckpt)

    results = {
        "model": args.model,
        "epochs": args.epochs,
        "seed": args.seed,
        "best_val_auroc": best_auroc,
        "best_checkpoint": str(best_ckpt),
        "history": history,
        "transform": {
            "image_size": IMAGE_SIZE,
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
        },
    }
    with open(out_dir / "results.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"wrote {best_ckpt} and {out_dir / 'results.json'}", flush=True)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the Veil in-house image detector.")
    p.add_argument("--manifest", default="detector-trainer/data/manifest.csv",
                   help="Path to the shared manifest CSV.")
    p.add_argument("--out", required=True, help="Output dir for checkpoints + results.json.")
    p.add_argument("--model", default="resnet50", help="Model name (resnet50; clip* reserved).")
    p.add_argument("--epochs", type=int, default=10, help="Training epochs (spec default 10).")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("VEIL_NUM_WORKERS", 4)))
    p.add_argument("--seed", type=int, default=42, help="Fixed seed for the honest comparison.")
    p.add_argument("--class-weight", action=argparse.BooleanOptionalAction, default=True,
                   help="Weight BCE by inverse class frequency to counter imbalance (default on).")
    p.add_argument("--pretrained", action="store_true",
                   help="Initialize backbone from ImageNet weights (spec baseline init).")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
