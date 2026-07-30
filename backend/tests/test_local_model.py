"""Tests for the local detector signal serving the CLIP winner.

The product must serve the retrained frozen-CLIP + MLP head (3% real-photo FPR),
not the old overfit ResNet, and report confidence relative to the calibrated
decision threshold rather than a hardcoded 0.5.
"""
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from app.signals import local_model


def _png_bytes(color=(120, 130, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, format="PNG")
    return buf.getvalue()


def _clip_settings(tmp_path, threshold=0.57):
    """Settings pointing the local signal at a CLIP head checkpoint."""
    ckpt = tmp_path / "clip_mlp" / "clip_head_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"stub")  # contents unused; the builder is stubbed
    (ckpt.parent / "config.json").write_text('{"backbone_name":"ViT-L-14"}')
    return SimpleNamespace(
        local_model_type="clip",
        local_model_checkpoint=str(ckpt),
        local_model_name="clip_mlp",
        local_model_threshold=threshold,
    )


def test_confidence_is_zero_at_threshold_and_grows_with_distance():
    assert local_model._confidence(0.57, 0.57) == pytest.approx(0.0)
    assert local_model._confidence(1.0, 0.57) == pytest.approx(1.0)
    assert local_model._confidence(0.0, 0.57) == pytest.approx(1.0)
    # Farther from the boundary on the FAKE side => more confident.
    assert local_model._confidence(0.9, 0.57) > local_model._confidence(0.65, 0.57)


def test_load_model_builds_clip_detector_from_head_and_sibling_config(tmp_path, monkeypatch):
    settings = _clip_settings(tmp_path)
    monkeypatch.setattr(local_model, "get_settings", lambda: settings)
    # Reset the module cache so our stub is (re)built.
    monkeypatch.setattr(local_model, "_MODEL", None)
    monkeypatch.setattr(local_model, "_CHECKPOINT", None)

    captured = {}

    def fake_builder(config, head_ckpt, device=None):
        captured["config"] = str(config)
        captured["head_ckpt"] = str(head_ckpt)
        import torch
        from torch import nn

        class Stub(nn.Module):
            def forward(self, x):  # -> logit[B]
                return torch.zeros(x.shape[0])

        return Stub().eval()

    monkeypatch.setattr(local_model, "_load_clip_builder", lambda: fake_builder)

    model, device, checkpoint = local_model._load_model()

    # The head checkpoint is the configured path; the config is its sibling.
    assert captured["head_ckpt"] == settings.local_model_checkpoint
    assert captured["config"].replace("\\", "/").endswith("clip_mlp/config.json")
    # The returned model is the built CLIP detector, not a ResNet.
    assert type(model).__name__ == "Stub"


async def test_analyze_scores_via_clip_and_pivots_confidence_on_threshold(tmp_path, monkeypatch):
    settings = _clip_settings(tmp_path, threshold=0.57)
    monkeypatch.setattr(local_model, "get_settings", lambda: settings)
    monkeypatch.setattr(local_model, "_MODEL", None)
    monkeypatch.setattr(local_model, "_CHECKPOINT", None)

    import torch
    from torch import nn

    class RealLeaning(nn.Module):
        # logit -> sigmoid ~= 0.12, i.e. a real photo well below the 0.57 boundary
        def forward(self, x):
            return torch.full((x.shape[0],), -2.0)

    monkeypatch.setattr(local_model, "_load_clip_builder", lambda: (lambda **kw: RealLeaning().eval()))

    signal = local_model.LocalModelSignal()
    image = local_model.ImageInput(data=_png_bytes(), filename="x.png", content_type="image/png")
    result = await signal.analyze(image)

    assert result.status.value == "ok"
    assert result.ai_score == pytest.approx(0.1192, abs=1e-3)   # sigmoid(-2)
    # Real-leaning and below threshold => confident it is REAL.
    assert result.confidence == pytest.approx(local_model._confidence(0.1192, 0.57), abs=1e-3)
    assert result.confidence > 0.5
