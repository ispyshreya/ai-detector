"""Unit tests for the ResNet-50 baseline and the build_model contract.

Fast and fully offline (pretrained=False -> no ImageNet download). These tests
assert the exact contract that ``backend/app/signals/local_model.py`` relies on:

    model = build_model("resnet50", pretrained=False)
    logit = model(tensor[B, 3, 224, 224])        # squeezes to [B]
    ai_score = torch.sigmoid(logit)              # in (0, 1)

plus a single real optimizer step to prove the graph trains (BCE + backward).

Run with pytest, or directly: ``python detector-trainer/models/test_resnet.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the trainer root importable whether run via pytest or as a script.
_TRAINER_ROOT = Path(__file__).resolve().parents[1]
if str(_TRAINER_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINER_ROOT))

from train import build_model  # noqa: E402


def test_build_model_forward_shape_and_sigmoid_range():
    """forward([2,3,224,224]) -> [2]; sigmoid gives probabilities in (0, 1)."""
    model = build_model("resnet50", pretrained=False)
    model.eval()

    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)

    # Contract: output squeezes to [B]. local_model.py squeezes too, so [B] or
    # [B, 1] both satisfy it; we standardize on [B].
    assert logits.squeeze().shape == (2,), f"expected [2], got {tuple(logits.shape)}"

    scores = torch.sigmoid(logits)
    assert torch.all(scores > 0.0) and torch.all(scores < 1.0), "sigmoid not in (0,1)"


def test_resnet_alias_dispatch():
    """Any 'resnet*' name dispatches to the ResNet builder (contract §build_model)."""
    model = build_model("resnet50", pretrained=False)
    assert isinstance(model, nn.Module)


def test_unknown_model_raises():
    """Unsupported names raise a clear ValueError (clip is reserved, not built)."""
    for bad in ("clip", "vit", "nonsense"):
        try:
            build_model(bad, pretrained=False)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for model name {bad!r}")


def test_single_optimizer_step_trains():
    """One AdamW step on a synthetic batch: loss finite, backward populates grads."""
    torch.manual_seed(0)
    model = build_model("resnet50", pretrained=False)
    model.train()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 2, (2,)).float()

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    # backward must produce a real gradient on the head (proves the graph is wired).
    head_grad = model.backbone.fc.weight.grad
    assert head_grad is not None and torch.isfinite(head_grad).all(), "no/invalid grad"

    optimizer.step()  # must not raise


def _run_all():
    """Minimal runner so the file works without pytest installed."""
    tests = [
        test_build_model_forward_shape_and_sigmoid_range,
        test_resnet_alias_dispatch,
        test_unknown_model_raises,
        test_single_optimizer_step_trains,
    ]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as exc:  # noqa: BLE001 - test harness reporting
            failures += 1
            print(f"FAIL {t.__name__}: {exc!r}")
    if failures:
        print(f"\n{failures}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    _run_all()
