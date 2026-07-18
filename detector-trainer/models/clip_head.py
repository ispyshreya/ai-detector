"""CLIP contender for the Veil in-house image detector.

Implements the "CONTENDER" unit of the design spec (§4): a FROZEN CLIP
ViT-L/14 backbone whose image features are cached once, plus a cheap head
(linear probe or 2-layer MLP) that trains in minutes with zero GPU.

Pipeline
--------
Stage A (GPU, once):  extract_features / cache_features
    image -> frozen CLIP visual encoder -> feature vector [feat_dim]
    Cache the [N, feat_dim] matrix + labels to disk so head experiments
    never touch the GPU again.

Stage B (near-free):  train_head
    cached features -> LinearHead or MLPHead -> single logit
    Writes clip_head_best.pt (plain state_dict) + config.json.

Integration:  build_clip_detector
    Assembles frozen CLIP backbone + trained head into ONE nn.Module whose
    forward(tensor[B, 3, 224, 224]) -> logit[B].  ai_score = sigmoid(logit).

Shared contract
---------------
- forward() accepts the SAME ImageNet-normalized 224x224 tensor the rest of
  the system uses (see backend/app/signals/local_model.py _eval_transform).
  The assembled detector converts ImageNet normalization -> CLIP's own
  normalization internally, so callers do not need to know CLIP's stats.
- Checkpoints are plain state_dicts (.pt); configs are json.

Backbone
--------
Default: open_clip ViT-L/14 (OpenAI weights), feat_dim = 768.
Fallback (spec §4): ViT-B/16, feat_dim = 512 (~1% AUROC cost) when ViT-L is
too heavy / OOM.  Chosen backbone + feat_dim are recorded in config.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Guarded heavy imports.  open_clip is only needed for real feature extraction
# / assembling the detector; unit tests mock the backbone and never import it.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only when a real backbone is loaded
    import open_clip  # type: ignore

    _HAS_OPEN_CLIP = True
except Exception:  # pragma: no cover
    open_clip = None  # type: ignore
    _HAS_OPEN_CLIP = False

# Guarded import of the data agent's dataset (built in parallel).  Code against
# it but never hard-fail at import time if it is not there yet.
try:  # pragma: no cover - depends on the data agent
    from detector_trainer.data.dataset import VeilDataset  # type: ignore
except Exception:  # pragma: no cover
    try:
        from data.dataset import VeilDataset  # type: ignore
    except Exception:
        VeilDataset = None  # type: ignore


# ---------------------------------------------------------------------------
# Backbone registry.  Maps a short backbone name -> (open_clip model name,
# pretrained tag, feature dimension of the visual encoder output).
# ---------------------------------------------------------------------------
BACKBONES = {
    "ViT-L-14": {"model": "ViT-L-14", "pretrained": "openai", "feat_dim": 768},
    "ViT-B-16": {"model": "ViT-B-16", "pretrained": "openai", "feat_dim": 512},
}

DEFAULT_BACKBONE = "ViT-L-14"
FALLBACK_BACKBONE = "ViT-B-16"

IMAGE_SIZE = 224

# ImageNet normalization used by the rest of the Veil system
# (backend/app/signals/local_model.py).  Tensors arriving at the assembled
# detector are normalized with these stats.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CLIP's own normalization stats (OpenAI weights).  The visual encoder expects
# inputs normalized with these; we convert internally.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def feat_dim_for(backbone_name: str) -> int:
    """Feature dimension of a backbone's visual encoder output."""
    if backbone_name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {backbone_name!r}; known: {sorted(BACKBONES)}"
        )
    return BACKBONES[backbone_name]["feat_dim"]


# ===========================================================================
# 1. Feature extraction (frozen CLIP backbone)
# ===========================================================================
class CLIPFeatureExtractor:
    """Loads a FROZEN CLIP visual encoder and produces L2-normalized features.

    Inputs may be either:
      * a torch tensor batch [B, 3, 224, 224] already ImageNet-normalized
        (the system convention), or
      * a list of image file paths (loaded + preprocessed with CLIP's own
        transform).

    All parameters are frozen and the encoder runs under torch.no_grad().
    """

    def __init__(
        self,
        backbone_name: str = DEFAULT_BACKBONE,
        device: Optional[str] = None,
    ) -> None:
        if not _HAS_OPEN_CLIP:
            raise ImportError(
                "open_clip is required for feature extraction. "
                "Install with `pip install open_clip_torch`."
            )
        if backbone_name not in BACKBONES:
            raise ValueError(
                f"Unknown backbone {backbone_name!r}; known: {sorted(BACKBONES)}"
            )

        self.backbone_name = backbone_name
        self.feat_dim = BACKBONES[backbone_name]["feat_dim"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        cfg = BACKBONES[backbone_name]
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["model"], pretrained=cfg["pretrained"]
        )
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        # CLIP's own PIL->tensor transform (used for path inputs).
        self.preprocess = preprocess

        # Precompute the affine re-normalization from ImageNet stats to CLIP
        # stats so a tensor already ImageNet-normalized can be fed directly.
        # x_clip = (x_pixel - clip_mean) / clip_std
        #        = (x_imnet * imnet_std + imnet_mean - clip_mean) / clip_std
        imnet_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        imnet_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        clip_mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        clip_std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self._renorm_scale = (imnet_std / clip_std).to(self.device)
        self._renorm_shift = ((imnet_mean - clip_mean) / clip_std).to(self.device)

    def imagenet_to_clip(self, tensor: torch.Tensor) -> torch.Tensor:
        """Re-normalize an ImageNet-normalized batch into CLIP's normalization."""
        return tensor * self._renorm_scale + self._renorm_shift

    @torch.no_grad()
    def encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Encode an ImageNet-normalized batch -> L2-normalized features."""
        tensor = tensor.to(self.device)
        clip_input = self.imagenet_to_clip(tensor)
        feats = self.model.encode_image(clip_input)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def _encode_paths(self, paths: Sequence[Union[str, Path]]) -> torch.Tensor:
        from PIL import Image

        batch = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            batch.append(self.preprocess(img))
        tensor = torch.stack(batch).to(self.device)
        feats = self.model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    def extract_features(
        self,
        image_tensors_or_paths: Union[torch.Tensor, Sequence[Union[str, Path]]],
        batch_size: int = 64,
    ) -> np.ndarray:
        """Extract features -> np.ndarray[N, feat_dim].

        Accepts either a [N, 3, 224, 224] ImageNet-normalized tensor or a
        sequence of image paths.
        """
        out: list[np.ndarray] = []

        if isinstance(image_tensors_or_paths, torch.Tensor):
            tensor = image_tensors_or_paths
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            for i in range(0, tensor.shape[0], batch_size):
                chunk = tensor[i : i + batch_size]
                out.append(self.encode_tensor(chunk).float().cpu().numpy())
        else:
            paths = list(image_tensors_or_paths)
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                out.append(self._encode_paths(chunk).float().cpu().numpy())

        if not out:
            return np.empty((0, self.feat_dim), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32)


def cache_features(
    dataset,
    out_dir: Union[str, Path],
    backbone_name: str = DEFAULT_BACKBONE,
    split: Optional[str] = None,
    batch_size: int = 64,
    device: Optional[str] = None,
    extractor: Optional[CLIPFeatureExtractor] = None,
) -> dict:
    """Run every image in ``dataset`` through the frozen backbone once and save.

    ``dataset`` is expected to be a VeilDataset (or any indexable dataset)
    yielding ``(tensor[3, 224, 224], label)`` pairs.  Features + labels are
    written so heads can train later with ZERO GPU.

    Writes into ``out_dir``:
      * ``features_{split}.npy``  -> float32 [N, feat_dim]
      * ``labels_{split}.npy``    -> float32 [N]
      * ``features_meta.json``    -> {backbone_name, feat_dim, split, n}

    Returns the metadata dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if extractor is None:
        extractor = CLIPFeatureExtractor(backbone_name=backbone_name, device=device)
    feat_dim = extractor.feat_dim

    feats_all: list[np.ndarray] = []
    labels_all: list[float] = []

    n = len(dataset)
    for i in range(0, n, batch_size):
        tensors = []
        for j in range(i, min(i + batch_size, n)):
            tensor, label = dataset[j]
            tensors.append(tensor)
            labels_all.append(float(label))
        batch = torch.stack(tensors)
        feats_all.append(extractor.extract_features(batch, batch_size=batch_size))

    features = (
        np.concatenate(feats_all, axis=0).astype(np.float32)
        if feats_all
        else np.empty((0, feat_dim), dtype=np.float32)
    )
    labels = np.asarray(labels_all, dtype=np.float32)

    tag = split or "all"
    feat_path = out_dir / f"features_{tag}.npy"
    label_path = out_dir / f"labels_{tag}.npy"
    np.save(feat_path, features)
    np.save(label_path, labels)

    meta = {
        "backbone_name": backbone_name,
        "feat_dim": int(feat_dim),
        "split": tag,
        "n": int(features.shape[0]),
        "features_path": str(feat_path),
        "labels_path": str(label_path),
    }
    with open(out_dir / "features_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_cached_features(
    features_path: Union[str, Path],
    labels_path: Union[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached (features, labels) pair produced by ``cache_features``."""
    features = np.load(features_path).astype(np.float32)
    labels = np.load(labels_path).astype(np.float32)
    return features, labels


# ===========================================================================
# 2. Heads (single-logit classifiers over CLIP features)
# ===========================================================================
class LinearHead(nn.Module):
    """Linear probe: honest baseline head.  feat_dim -> 1 logit."""

    head_type = "linear"

    def __init__(self, feat_dim: int) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.fc = nn.Linear(feat_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, feat_dim] -> [B]
        return self.fc(x).squeeze(-1)


class MLPHead(nn.Module):
    """2-layer MLP head: stronger contender.  feat_dim -> hidden -> 1 logit."""

    head_type = "mlp"

    def __init__(self, feat_dim: int, hidden: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, feat_dim] -> [B]
        return self.net(x).squeeze(-1)


def build_head(head_type: str, feat_dim: int, hidden: int = 256) -> nn.Module:
    """Factory: build a head by name ('linear' or 'mlp')."""
    if head_type == "linear":
        return LinearHead(feat_dim)
    if head_type == "mlp":
        return MLPHead(feat_dim, hidden=hidden)
    raise ValueError(f"Unknown head_type {head_type!r}; use 'linear' or 'mlp'.")


# ===========================================================================
# 3. Cheap head training on cached features
# ===========================================================================
@dataclass
class TrainHeadResult:
    out_dir: str
    ckpt_path: str
    config_path: str
    config: dict
    losses: list = field(default_factory=list)
    final_loss: float = 0.0


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    head_type: str = "linear",
    out_dir: Union[str, Path] = "clip_head_out",
    backbone_name: str = DEFAULT_BACKBONE,
    hidden: int = 256,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: Optional[int] = None,
    device: Optional[str] = None,
    seed: int = 42,
    class_weight: bool = True,
    verbose: bool = False,
) -> TrainHeadResult:
    """Train a head on cached CLIP features.  Cheap: minutes, CPU-ok.

    Writes ``clip_head_best.pt`` (plain state_dict) + ``config.json`` recording
    ``{backbone_name, feat_dim, head_type}`` into ``out_dir``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = torch.as_tensor(np.asarray(features), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(labels), dtype=torch.float32).view(-1)
    feat_dim = X.shape[1]

    head = build_head(head_type, feat_dim, hidden=hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = None
    if class_weight:
        n_pos = float((y == 1).sum())
        n_neg = float((y == 0).sum())
        if n_pos > 0 and n_neg > 0:
            pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    n = X.shape[0]
    bs = batch_size or n
    X = X.to(device)
    y = y.to(device)

    losses: list[float] = []
    best_loss = float("inf")
    best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            opt.zero_grad()
            logits = head(X[idx])
            loss = loss_fn(logits, y[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * idx.numel()
        epoch_loss /= n
        losses.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"[train_head] epoch {epoch:4d}  loss={epoch_loss:.5f}")

    # Save the best head as a plain state_dict.
    ckpt_path = out_dir / "clip_head_best.pt"
    torch.save(best_state, ckpt_path)

    config = {
        "backbone_name": backbone_name,
        "feat_dim": int(feat_dim),
        "head_type": head_type,
        "hidden": int(hidden) if head_type == "mlp" else None,
        "image_size": IMAGE_SIZE,
        "input_normalization": "imagenet",  # forward() expects ImageNet-normed 224 tensors
    }
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return TrainHeadResult(
        out_dir=str(out_dir),
        ckpt_path=str(ckpt_path),
        config_path=str(config_path),
        config=config,
        losses=losses,
        final_loss=float(losses[-1]) if losses else 0.0,
    )


# ===========================================================================
# 4. Assembled detector: frozen CLIP backbone + trained head
# ===========================================================================
class CLIPDetector(nn.Module):
    """Frozen CLIP backbone + trained head as ONE module.

    forward(tensor[B, 3, 224, 224]) -> logit[B]

    ``tensor`` is expected to be ImageNet-normalized (system convention); the
    module re-normalizes to CLIP's stats internally.  ai_score = sigmoid(logit).
    """

    def __init__(self, extractor: CLIPFeatureExtractor, head: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.head = head
        # Ensure backbone stays frozen.
        for p in self.extractor.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            feats = self.extractor.encode_tensor(x)  # [B, feat_dim], L2-normed
        feats = feats.float()
        logit = self.head(feats)  # [B]
        return logit


def build_clip_detector(
    config: Union[str, Path, dict],
    head_ckpt: Union[str, Path],
    device: Optional[str] = None,
    extractor: Optional[CLIPFeatureExtractor] = None,
) -> nn.Module:
    """Assemble a frozen CLIP backbone + trained head into one detector.

    Parameters
    ----------
    config : path to config.json OR the config dict recording
             {backbone_name, feat_dim, head_type, hidden}.
    head_ckpt : path to clip_head_best.pt (plain state_dict).
    extractor : optional pre-built / mocked CLIPFeatureExtractor. If provided,
                the CLIP backbone is not re-loaded (used by tests and to avoid
                re-downloading multi-GB weights).

    Returns
    -------
    nn.Module with forward(tensor[B, 3, 224, 224]) -> logit[B], eval mode.
    """
    if isinstance(config, (str, Path)):
        with open(config) as f:
            config = json.load(f)

    backbone_name = config["backbone_name"]
    feat_dim = int(config["feat_dim"])
    head_type = config["head_type"]
    hidden = config.get("hidden") or 256

    if extractor is None:
        extractor = CLIPFeatureExtractor(backbone_name=backbone_name, device=device)

    head = build_head(head_type, feat_dim, hidden=hidden)
    state = torch.load(head_ckpt, map_location="cpu")
    head.load_state_dict(state)

    dev = device or extractor.device
    head = head.to(dev)

    detector = CLIPDetector(extractor, head)
    detector.eval()
    return detector
