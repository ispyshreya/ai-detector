"""Unit tests for clip_head.py — the CLIP contender.

Runs fully OFFLINE: the heavy CLIP backbone is mocked/stubbed so no multi-GB
weight download happens.  Real coverage:
  * heads (LinearHead, MLPHead) drive training loss down on a separable
    synthetic feature set, and write clip_head_best.pt + config.json;
  * build_clip_detector forward SHAPE with a stubbed backbone -> logit[B];
  * the ImageNet->CLIP re-normalization affine.

Run:  python -m pytest detector-trainer/models/test_clip_head.py -v
 or:  python detector-trainer/models/test_clip_head.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import clip_head  # noqa: E402
from clip_head import (  # noqa: E402
    LinearHead,
    MLPHead,
    build_clip_detector,
    build_head,
    train_head,
)

FEAT_DIM = 768


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def make_separable(n=64, feat_dim=FEAT_DIM, seed=0):
    """Linearly separable synthetic features + binary labels."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, feat_dim).astype(np.float64)
    w = rng.randn(feat_dim).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scores = X @ w
    X = X.astype(np.float32)
    labels = (scores > np.median(scores)).astype(np.float32)
    # Nudge features along w so the classes are cleanly separable.
    X = X + labels[:, None] * (w[None, :].astype(np.float32) * 3.0)
    return X, labels


# ---------------------------------------------------------------------------
# Head training tests
# ---------------------------------------------------------------------------
def test_linear_head_trains_and_writes(tmp_path):
    X, y = make_separable(64, FEAT_DIM, seed=1)
    out_dir = tmp_path / "linear_out"
    result = train_head(
        X, y, head_type="linear", out_dir=out_dir,
        backbone_name="ViT-L-14", epochs=150, lr=1e-2, verbose=False,
    )

    # Loss driven down on a separable set.
    assert result.losses[0] > result.losses[-1]
    assert result.final_loss < 0.15, f"final loss too high: {result.final_loss}"

    # Artifacts written.
    ckpt = Path(result.ckpt_path)
    cfg = Path(result.config_path)
    assert ckpt.exists() and ckpt.name == "clip_head_best.pt"
    assert cfg.exists() and cfg.name == "config.json"

    config = json.loads(cfg.read_text())
    assert config["backbone_name"] == "ViT-L-14"
    assert config["feat_dim"] == FEAT_DIM
    assert config["head_type"] == "linear"

    # Checkpoint is a plain state_dict loadable into a fresh head.
    head = build_head("linear", FEAT_DIM)
    head.load_state_dict(torch.load(ckpt, map_location="cpu"))
    print(f"[linear] loss {result.losses[0]:.4f} -> {result.final_loss:.4f}")


def test_mlp_head_trains_and_writes(tmp_path):
    X, y = make_separable(64, FEAT_DIM, seed=2)
    out_dir = tmp_path / "mlp_out"
    result = train_head(
        X, y, head_type="mlp", out_dir=out_dir,
        backbone_name="ViT-L-14", hidden=128, epochs=150, lr=1e-2, verbose=False,
    )

    assert result.losses[0] > result.losses[-1]
    assert result.final_loss < 0.15, f"final loss too high: {result.final_loss}"

    config = json.loads(Path(result.config_path).read_text())
    assert config["head_type"] == "mlp"
    assert config["hidden"] == 128

    head = build_head("mlp", FEAT_DIM, hidden=128)
    head.load_state_dict(torch.load(result.ckpt_path, map_location="cpu"))
    print(f"[mlp] loss {result.losses[0]:.4f} -> {result.final_loss:.4f}")


def test_head_output_shape():
    for head in (LinearHead(FEAT_DIM), MLPHead(FEAT_DIM, hidden=64)):
        x = torch.randn(8, FEAT_DIM)
        out = head(x)
        assert out.shape == (8,), f"{type(head).__name__} -> {tuple(out.shape)}"


# ---------------------------------------------------------------------------
# build_clip_detector forward-shape test with a STUB backbone (no download)
# ---------------------------------------------------------------------------
class _StubExtractor:
    """Mimics CLIPFeatureExtractor without loading any real CLIP weights.

    encode_tensor returns random L2-normalized features of feat_dim so the
    assembled detector's forward can be exercised offline.
    """

    def __init__(self, feat_dim=FEAT_DIM, device="cpu"):
        self.feat_dim = feat_dim
        self.device = device
        self.backbone_name = "ViT-L-14"
        # A tiny fake "model" with parameters so freezing logic has something
        # to iterate over.
        self.model = nn.Linear(4, 4)

    def encode_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        b = tensor.shape[0]
        feats = torch.randn(b, self.feat_dim)
        return feats / feats.norm(dim=-1, keepdim=True)


def test_build_clip_detector_forward_shape(tmp_path):
    # Train a real head on synthetic features, then assemble with a stub backbone.
    X, y = make_separable(64, FEAT_DIM, seed=3)
    result = train_head(
        X, y, head_type="mlp", out_dir=tmp_path / "det_out",
        backbone_name="ViT-L-14", hidden=128, epochs=50,
    )

    stub = _StubExtractor(feat_dim=FEAT_DIM)
    detector = build_clip_detector(
        result.config_path, result.ckpt_path, extractor=stub
    )

    # forward(tensor[B, 3, 224, 224]) -> logit[B]
    for B in (1, 5):
        x = torch.randn(B, 3, 224, 224)
        logit = detector(x)
        assert logit.shape == (B,), f"B={B} -> {tuple(logit.shape)}"
        ai_score = torch.sigmoid(logit)
        assert torch.all((ai_score >= 0) & (ai_score <= 1))

    # single unbatched image [3,224,224] is accepted -> [1]
    logit = detector(torch.randn(3, 224, 224))
    assert logit.shape == (1,)
    print("[detector] forward shapes OK: [B,3,224,224] -> [B]")


def test_build_clip_detector_with_dict_config(tmp_path):
    """config may be passed as a dict, not just a path."""
    X, y = make_separable(48, FEAT_DIM, seed=4)
    result = train_head(
        X, y, head_type="linear", out_dir=tmp_path / "dcfg",
        backbone_name="ViT-B-16", epochs=30,
    )
    stub = _StubExtractor(feat_dim=FEAT_DIM)
    detector = build_clip_detector(result.config, result.ckpt_path, extractor=stub)
    logit = detector(torch.randn(3, 3, 224, 224))
    assert logit.shape == (3,)


# ---------------------------------------------------------------------------
# Re-normalization affine correctness (ImageNet-normed -> CLIP-normed)
# ---------------------------------------------------------------------------
def test_imagenet_to_clip_renorm_math():
    """The affine must reproduce pixel-space round-trip within float tol."""
    imnet_mean = torch.tensor(clip_head.IMAGENET_MEAN).view(1, 3, 1, 1)
    imnet_std = torch.tensor(clip_head.IMAGENET_STD).view(1, 3, 1, 1)
    clip_mean = torch.tensor(clip_head.CLIP_MEAN).view(1, 3, 1, 1)
    clip_std = torch.tensor(clip_head.CLIP_STD).view(1, 3, 1, 1)

    # Random pixels in [0,1], normalize with ImageNet, then convert to CLIP.
    pixels = torch.rand(2, 3, 8, 8)
    x_imnet = (pixels - imnet_mean) / imnet_std

    scale = imnet_std / clip_std
    shift = (imnet_mean - clip_mean) / clip_std
    x_clip = x_imnet * scale + shift

    expected = (pixels - clip_mean) / clip_std
    assert torch.allclose(x_clip, expected, atol=1e-5)
    print("[renorm] ImageNet->CLIP affine matches pixel round-trip")


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------
def _run_all():
    import tempfile

    passed = 0
    tests = [
        ("test_head_output_shape", lambda d: test_head_output_shape()),
        ("test_linear_head_trains_and_writes", test_linear_head_trains_and_writes),
        ("test_mlp_head_trains_and_writes", test_mlp_head_trains_and_writes),
        ("test_build_clip_detector_forward_shape", test_build_clip_detector_forward_shape),
        ("test_build_clip_detector_with_dict_config", test_build_clip_detector_with_dict_config),
        ("test_imagenet_to_clip_renorm_math", lambda d: test_imagenet_to_clip_renorm_math()),
    ]
    for name, fn in tests:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
        print(f"PASS {name}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
