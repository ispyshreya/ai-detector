"""Unit tests for the shared evaluation harness.

Run: `python -m pytest detector-trainer/eval/test_harness.py`
or   `python detector-trainer/eval/test_harness.py` (falls back to a plain runner).

Everything here is synthetic — no datasets, no torch, no GPU.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

# Import the sibling harness by path so the test runs whether or not the package
# has an __init__.py (the orchestrator owns packaging).
_HARNESS_PATH = Path(__file__).resolve().parent / "harness.py"
_spec = importlib.util.spec_from_file_location("veil_eval_harness", _HARNESS_PATH)
harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(harness)


RNG = np.random.default_rng(20260717)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def test_auroc_well_separated_is_high():
    labels = np.array([0] * 200 + [1] * 200)
    # Reals cluster low, fakes cluster high, minimal overlap.
    scores = np.concatenate(
        [RNG.normal(0.15, 0.05, 200), RNG.normal(0.85, 0.05, 200)]
    ).clip(0, 1)
    auroc = harness.compute_auroc(scores, labels)
    assert auroc > 0.9, auroc


def test_auroc_random_is_near_half():
    labels = np.array([0] * 500 + [1] * 500)
    scores = RNG.uniform(0, 1, 1000)  # no signal
    auroc = harness.compute_auroc(scores, labels)
    assert 0.4 < auroc < 0.6, auroc


def test_accuracy_and_precision_recall():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.6, 0.3])
    labels = np.array([0, 0, 1, 1, 1, 0])
    assert harness.compute_accuracy(scores, labels, 0.5) == 1.0
    p, r = harness.precision_recall_at_threshold(scores, labels, 0.5)
    assert p == 1.0 and r == 1.0


def test_auroc_single_class_is_nan():
    assert np.isnan(harness.compute_auroc([0.2, 0.8], [1, 1]))


# --------------------------------------------------------------------------- #
# score_dataset (torch guarded; a tiny stub model + loader)
# --------------------------------------------------------------------------- #
def test_score_dataset_with_torch():
    try:
        import torch
    except ImportError:
        print("  [skip] torch not installed — score_dataset path not exercised")
        return

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(3 * 224 * 224, 1)

        def forward(self, x):
            return self.lin(x.reshape(x.shape[0], -1))

    model = Stub()
    batch = {
        "image": torch.randn(4, 3, 224, 224),
        "label": torch.tensor([0, 1, 0, 1]),
        "generator": ["real", "flux", "real", "midjourney"],
        "split": ["test"] * 4,
        "path": ["a", "b", "c", "d"],
    }
    df = harness.score_dataset(model, [batch])
    assert list(df.columns) == harness.PREDICTION_COLUMNS
    assert len(df) == 4
    assert df["score"].between(0, 1).all()
    assert df["generator"].tolist() == ["real", "flux", "real", "midjourney"]


# --------------------------------------------------------------------------- #
# Per-generator table
# --------------------------------------------------------------------------- #
def _synthetic_predictions() -> pd.DataFrame:
    """Predictions spanning real + 4 fake generators, with real signal."""
    generators = ["real", "stable-diffusion", "midjourney", "dalle3", "flux"]
    rows = []
    for gen in generators:
        n = 120
        label = 0 if gen == "real" else 1
        split = "test" if gen in ("real", "stable-diffusion") else "wild"
        center = 0.2 if label == 0 else 0.8
        for _ in range(n):
            rows.append(
                {
                    "path": f"{gen}/{RNG.integers(0, 1_000_000)}.jpg",
                    "generator": gen,
                    "split": split,
                    "label": label,
                    "score": float(np.clip(RNG.normal(center, 0.1), 0, 1)),
                }
            )
    return pd.DataFrame(rows, columns=harness.PREDICTION_COLUMNS)


def test_per_generator_table_shape_and_sanity():
    preds = _synthetic_predictions()
    tbl = harness.per_generator_table(preds)
    expected = {"real", "stable-diffusion", "midjourney", "dalle3", "flux"}
    assert set(tbl["generator"]) == expected
    assert len(tbl) == 5
    # Each all-fake generator should score well vs the real pool.
    for gen in ("midjourney", "dalle3", "flux"):
        auroc = float(tbl.loc[tbl["generator"] == gen, "auroc"].iloc[0])
        assert 0.85 <= auroc <= 1.0, (gen, auroc)
    # Accuracies are in [0, 1].
    assert tbl["accuracy"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_robustness_eval_produces_row_per_perturbation():
    from PIL import Image

    images = [
        Image.fromarray(RNG.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        for _ in range(20)
    ]
    labels = [i % 2 for i in range(20)]

    def score_images(imgs):
        # Deterministic-ish scorer that keys off mean brightness (just needs to run).
        return [float(np.asarray(im).mean() / 255.0) for im in imgs]

    perts = harness.build_perturbations(jpeg_qualities=(50,), downscale_factors=(0.5,))
    rob = harness.robustness_eval(score_images, images, labels, perturbations=perts)
    assert set(rob["perturbation"]) == {"clean", "jpeg_q50", "downscale_0.5"}
    assert "auroc" in rob.columns and "auroc_delta" in rob.columns
    assert float(rob.loc[rob["perturbation"] == "clean", "auroc_delta"].iloc[0]) == 0.0


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #
def test_temperature_scaling_reduces_ece():
    # Overconfident model: true logits scaled up by 3x => sigmoid too extreme.
    n = 2000
    true_logits = RNG.normal(0, 1.5, n)
    probs = harness._sigmoid(true_logits)
    labels = (RNG.uniform(0, 1, n) < probs).astype(int)
    miscalibrated_logits = true_logits * 3.0  # too confident

    split = n // 2
    result = harness.calibrate(
        miscalibrated_logits[:split],
        labels[:split],
        miscalibrated_logits[split:],
        labels[split:],
    )
    assert result["temperature"] > 1.0, result["temperature"]  # needs cooling
    assert result["ece_after"] < result["ece_before"], result


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #
def test_write_report_creates_markdown_and_plots(tmp_path=None):
    import tempfile

    out_dir = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
    preds = _synthetic_predictions()
    predictions_by_model = {"resnet50": preds, "clip_head": preds.copy()}

    # Also feed a robustness table for the model, so panel 3 is populated.
    from PIL import Image

    images = [
        Image.fromarray(RNG.integers(0, 255, (32, 32, 3), dtype=np.uint8))
        for _ in range(10)
    ]
    labels = [i % 2 for i in range(10)]
    rob = harness.robustness_eval(
        lambda imgs: [float(np.asarray(im).mean() / 255.0) for im in imgs],
        images,
        labels,
        perturbations=harness.build_perturbations((50,), (0.5,)),
    )

    report_path = harness.write_report(
        predictions_by_model, out_dir, robustness_by_model={"resnet50": rob}
    )
    assert report_path.exists()
    text = report_path.read_text()
    assert "Panel 1" in text and "Panel 2" in text and "Panel 3" in text
    assert "Verdict" in text
    plots = list((out_dir / "plots").glob("*.png"))
    assert len(plots) >= 1, "expected at least one plot file"
    print(f"  report + {len(plots)} plot(s) at {out_dir}")


# --------------------------------------------------------------------------- #
# Real-photo false-positive metrics
# --------------------------------------------------------------------------- #
def test_real_photo_fpr_and_specificity():
    preds = pd.DataFrame(
        {
            "path": [f"p{i}" for i in range(6)],
            "generator": ["real"] * 4 + ["flux", "flux"],
            "split": ["test_wild"] * 6,
            "label": [0, 0, 0, 0, 1, 1],
            # 2 of 4 reals wrongly scored >= 0.5
            "score": [0.1, 0.2, 0.8, 0.9, 0.7, 0.9],
        },
        columns=harness.PREDICTION_COLUMNS,
    )
    assert harness.real_photo_false_positive_rate(preds, 0.5) == 0.5
    assert harness.real_photo_specificity(preds, 0.5) == 0.5


def test_real_photo_fpr_no_reals_is_nan():
    preds = pd.DataFrame(
        {
            "path": ["a"],
            "generator": ["flux"],
            "split": ["test_wild"],
            "label": [1],
            "score": [0.9],
        },
        columns=harness.PREDICTION_COLUMNS,
    )
    assert np.isnan(harness.real_photo_false_positive_rate(preds, 0.5))


# --------------------------------------------------------------------------- #
# Plain runner (no pytest required)
# --------------------------------------------------------------------------- #
def _run_all():
    import tempfile
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            if fn.__name__ == "test_write_report_creates_markdown_and_plots":
                fn(tempfile.mkdtemp())
            else:
                fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys

    sys.exit(0 if _run_all() else 1)
