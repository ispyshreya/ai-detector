"""ResNet-50 baseline for the Veil in-house image detector.

This is the BASELINE unit from the design spec (§2, §4): an ImageNet-pretrained
ResNet-50 fine-tuned end-to-end from pixels to a single "AI-ness" logit. It races
the CLIP contender on identical data and exports to the same inference contract.

Inference contract (must match ``backend/app/signals/local_model.py``):

    model(tensor[B, 3, 224, 224]) -> logit  # shape [B] or [B, 1]
    ai_score = torch.sigmoid(logit)

The signal loads a plain ``state_dict`` checkpoint via ``build_model(name,
pretrained=False)`` and calls ``load_state_dict`` on it, so the module here must
build the *same* architecture whether or not ImageNet weights are requested.
"""

from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class ResNet50Detector(nn.Module):
    """ResNet-50 backbone with a single-logit binary head.

    The torchvision ResNet-50's final 1000-way ``fc`` is replaced with a 1-unit
    linear layer producing one raw logit per image (positive -> "AI/fake"). We
    keep the head as a bare ``nn.Linear`` so BCE-with-logits can be applied
    directly and so ``torch.sigmoid`` on the output is a usable probability.

    ``forward`` returns a squeezed ``[B]`` tensor. The signal squeezes anyway and
    ``sigmoid`` is shape-agnostic, so ``[B]`` and ``[B, 1]`` are both valid under
    the contract; we standardize on ``[B]`` for clean BCE broadcasting against a
    ``[B]`` label vector.
    """

    def __init__(self, pretrained: bool) -> None:
        super().__init__()
        # pretrained=True pulls ImageNet weights (the spec's init for fine-tuning).
        # pretrained=False builds the identical graph with random weights — this is
        # the path local_model.py uses before load_state_dict restores our own
        # trained weights, and the path unit tests use to stay offline/fast.
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        # Swap the 1000-class classifier for a single-logit head. in_features is
        # 2048 for ResNet-50; reading it off the existing fc keeps this robust.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        # backbone -> [B, 1]; squeeze the trailing dim to [B] for BCE + the signal.
        logits = self.backbone(x)
        return logits.squeeze(-1)


def build_resnet50(pretrained: bool) -> nn.Module:
    """Construct the ResNet-50 detector.

    Args:
        pretrained: if True, initialize the backbone with ImageNet weights
            (requires a one-time download). If False, random init — used both by
            the backend signal (weights come from the checkpoint) and by tests.

    Returns:
        An ``nn.Module`` whose ``forward`` maps ``[B, 3, 224, 224]`` -> ``[B]``
        logits.
    """
    return ResNet50Detector(pretrained=pretrained)
