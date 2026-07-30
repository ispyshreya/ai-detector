"""Shared evaluation harness for the Veil in-house image detector.

One harness, run identically on both the ResNet baseline and the CLIP contender,
so the head-to-head comparison is honest (design spec §5).

The predictions contract every model produces and this module consumes:

    predictions.csv columns:
        path:str         image path (from the manifest)
        generator:str    e.g. real / stable-diffusion / midjourney / dalle3 / flux
        split:str        train / test / wild
        label:int        0 = real, 1 = fake
        score:float      sigmoid probability that the image is FAKE

Model interface: `forward(tensor[B, 3, 224, 224]) -> logit`; the score is
`sigmoid(logit)`.

Heavy deps (torch) are guarded so metric/report code runs on a plain CPU box
without a training stack installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Column contract for predictions.csv — the single interface between models and
# this harness. Do not reorder without updating every producer.
PREDICTION_COLUMNS = ["path", "generator", "split", "label", "score"]

# Decision threshold on the FAKE probability. 0.5 by default; temperature
# scaling (below) makes this a meaningful operating point for the fusion engine.
DEFAULT_THRESHOLD = 0.5


# --------------------------------------------------------------------------- #
# 1. Metrics from arrays
# --------------------------------------------------------------------------- #
def _as_arrays(scores, labels) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=float).ravel()
    labels = np.asarray(labels, dtype=int).ravel()
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} misaligned")
    return scores, labels


def compute_auroc(scores, labels) -> float:
    """AUROC of FAKE-probability vs binary label. NaN if only one class present."""
    scores, labels = _as_arrays(scores, labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_accuracy(scores, labels, threshold: float = DEFAULT_THRESHOLD) -> float:
    scores, labels = _as_arrays(scores, labels)
    if scores.size == 0:
        return float("nan")
    preds = (scores >= threshold).astype(int)
    return float((preds == labels).mean())


def precision_recall_at_threshold(
    scores, labels, threshold: float = DEFAULT_THRESHOLD
) -> tuple[float, float]:
    """Precision and recall for the positive (FAKE) class at a threshold."""
    scores, labels = _as_arrays(scores, labels)
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    return precision, recall


def expected_calibration_error(scores, labels, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) with equal-width confidence bins.

    Bins predictions by confidence in the predicted class, then averages the
    gap between mean confidence and empirical accuracy, weighted by bin size.
    0 = perfectly calibrated.
    """
    scores, labels = _as_arrays(scores, labels)
    if scores.size == 0:
        return float("nan")
    # Confidence in the predicted class and whether that prediction is correct.
    preds = (scores >= 0.5).astype(int)
    confidence = np.where(preds == 1, scores, 1.0 - scores)
    correct = (preds == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = scores.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Bins are (lo, hi]; the first bin also includes confidence == lo (0.5),
        # since a coin-flip prediction has confidence exactly 0.5.
        if lo == bins[0]:
            in_bin = (confidence >= lo) & (confidence <= hi)
        else:
            in_bin = (confidence > lo) & (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidence[in_bin].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def summary_metrics(
    scores, labels, threshold: float = DEFAULT_THRESHOLD, n_bins: int = 10
) -> dict:
    """Bundle of the headline metrics for a set of predictions."""
    precision, recall = precision_recall_at_threshold(scores, labels, threshold)
    return {
        "n": int(np.asarray(labels).size),
        "auroc": compute_auroc(scores, labels),
        "accuracy": compute_accuracy(scores, labels, threshold),
        "precision": precision,
        "recall": recall,
        "ece": expected_calibration_error(scores, labels, n_bins),
    }


# --------------------------------------------------------------------------- #
# 2. Score a model over a dataloader -> predictions DataFrame
# --------------------------------------------------------------------------- #
def score_dataset(model, dataloader) -> pd.DataFrame:
    """Run `model` over `dataloader`, applying sigmoid, into the predictions schema.

    Expects each batch to be either a mapping with keys among
    {image/tensor/x, label, generator, split, path} or a tuple
    (tensor, label[, meta_dict]). Missing metadata columns are filled with
    sensible defaults so callers can wire in a minimal loader for smoke tests.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError("score_dataset requires torch; install it to score models") from exc

    model.eval()
    device = next(model.parameters()).device

    rows: list[dict] = []
    with torch.no_grad():
        for batch in dataloader:
            tensor, labels, meta = _unpack_batch(batch)
            tensor = tensor.to(device)
            logits = model(tensor).reshape(-1)
            scores = torch.sigmoid(logits).cpu().numpy()
            labels_np = np.asarray(labels).reshape(-1)
            batch_n = scores.shape[0]
            for i in range(batch_n):
                rows.append(
                    {
                        "path": _meta_at(meta, "path", i, default=f"idx_{len(rows)}"),
                        "generator": _meta_at(meta, "generator", i, default="unknown"),
                        "split": _meta_at(meta, "split", i, default="test"),
                        "label": int(labels_np[i]),
                        "score": float(scores[i]),
                    }
                )
    return pd.DataFrame(rows, columns=PREDICTION_COLUMNS)


def _unpack_batch(batch):
    """Normalize a dataloader batch into (tensor, labels, meta_mapping)."""
    if isinstance(batch, Mapping):
        tensor = batch.get("image", batch.get("tensor", batch.get("x")))
        if tensor is None:
            raise KeyError("batch mapping has no image/tensor/x key")
        labels = batch["label"]
        meta = {k: batch[k] for k in ("path", "generator", "split") if k in batch}
        return tensor, labels, meta
    if isinstance(batch, (tuple, list)):
        if len(batch) == 2:
            tensor, labels = batch
            return tensor, labels, {}
        if len(batch) >= 3:
            tensor, labels, meta = batch[0], batch[1], batch[2]
            meta = meta if isinstance(meta, Mapping) else {}
            return tensor, labels, meta
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def _meta_at(meta: Mapping, key: str, i: int, default):
    if key not in meta:
        return default
    val = meta[key]
    try:
        item = val[i]
    except (TypeError, IndexError, KeyError):
        return default
    # torch tensors / numpy scalars -> python str/num
    if hasattr(item, "item"):
        try:
            item = item.item()
        except (ValueError, TypeError):
            pass
    return item


# --------------------------------------------------------------------------- #
# 3. Per-generator breakdown (panel 2 — the money table)
# --------------------------------------------------------------------------- #
def per_generator_table(
    predictions: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Per-generator AUROC + accuracy from a predictions DataFrame.

    AUROC needs both classes, but a single generator (e.g. midjourney) is all
    fakes. So for each generator we score its rows against a shared pool of
    REAL images — that mirrors the deployed decision (real-vs-this-generator).
    Accuracy is computed on the generator's own rows.
    """
    _validate_predictions(predictions)
    reals = predictions[predictions["label"] == 0]

    rows: list[dict] = []
    for generator, group in predictions.groupby("generator", sort=True):
        n = len(group)
        acc = compute_accuracy(group["score"], group["label"], threshold)

        if group["label"].nunique() >= 2:
            auroc = compute_auroc(group["score"], group["label"])
        elif int(group["label"].iloc[0]) == 1 and len(reals):
            # All-fake generator: pair against the shared real pool.
            paired = pd.concat([group, reals], ignore_index=True)
            auroc = compute_auroc(paired["score"], paired["label"])
        else:
            auroc = float("nan")

        rows.append(
            {
                "generator": generator,
                "n": n,
                "n_fake": int((group["label"] == 1).sum()),
                "auroc": auroc,
                "accuracy": acc,
            }
        )
    return pd.DataFrame(rows, columns=["generator", "n", "n_fake", "auroc", "accuracy"])


def _validate_predictions(predictions: pd.DataFrame) -> None:
    missing = [c for c in PREDICTION_COLUMNS if c not in predictions.columns]
    if missing:
        raise ValueError(f"predictions missing columns: {missing}")


def real_photo_false_positive_rate(
    predictions: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Fraction of REAL images wrongly scored FAKE (score >= threshold).

    The headline number for the real-photo false-positive bug: genuine selfies
    and phone photos that the detector flags as AI-generated. NaN if there are
    no real images in `predictions`.
    """
    _validate_predictions(predictions)
    reals = predictions[predictions["label"] == 0]
    if len(reals) == 0:
        return float("nan")
    fp = int((reals["score"] >= threshold).sum())
    return float(fp / len(reals))


def real_photo_specificity(
    predictions: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD
) -> float:
    """Fraction of REAL images correctly scored REAL (score < threshold). NaN if none."""
    fpr = real_photo_false_positive_rate(predictions, threshold)
    return float("nan") if np.isnan(fpr) else 1.0 - fpr


def select_threshold_for_target_fpr(
    predictions: pd.DataFrame,
    target_fpr: float = 0.02,
    grid: int = 199,
) -> float:
    """Lowest decision threshold whose real-photo FPR is <= `target_fpr`.

    Scans thresholds in (0, 1) low→high and returns the first that keeps real
    false positives at/below the target. Lower thresholds keep more FAKE recall,
    so the first qualifying threshold is the best trade. Falls back to the
    strictest candidate if none meets the target.
    """
    _validate_predictions(predictions)
    candidates = np.linspace(0.005, 0.995, grid)
    best = float(candidates[-1])
    for t in candidates:
        if real_photo_false_positive_rate(predictions, float(t)) <= target_fpr:
            best = float(t)
            break
    return best


def write_operating_point(
    predictions: pd.DataFrame, out_dir, target_fpr: float = 0.02
) -> Path:
    """Choose and persist the decision threshold that meets the real-FPR target."""
    import json

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    threshold = select_threshold_for_target_fpr(predictions, target_fpr=target_fpr)
    payload = {
        "threshold": threshold,
        "target_fpr": float(target_fpr),
        "real_fpr_at_threshold": real_photo_false_positive_rate(predictions, threshold),
    }
    path = out_dir / "operating_point.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


# --------------------------------------------------------------------------- #
# 4. Robustness eval (panel 3) — JPEG compression + downscaling
# --------------------------------------------------------------------------- #
def _perturb_jpeg(pil_image, quality: int):
    from io import BytesIO

    from PIL import Image

    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _perturb_downscale(pil_image, factor: float):
    from PIL import Image

    w, h = pil_image.size
    small = pil_image.resize(
        (max(1, int(w * factor)), max(1, int(h * factor))), Image.BILINEAR
    )
    # Upscale back so the model still receives a full-size image, now degraded.
    return small.resize((w, h), Image.BILINEAR)


def build_perturbations(
    jpeg_qualities: Iterable[int] = (90, 50, 30),
    downscale_factors: Iterable[float] = (0.5, 0.25),
) -> dict[str, Callable]:
    """Named image->image perturbations for the robustness panel."""
    perts: dict[str, Callable] = {"clean": lambda img: img.convert("RGB")}
    for q in jpeg_qualities:
        perts[f"jpeg_q{int(q)}"] = lambda img, q=q: _perturb_jpeg(img, q)
    for f in downscale_factors:
        perts[f"downscale_{f}"] = lambda img, f=f: _perturb_downscale(img, f)
    return perts


def robustness_eval(
    score_images: Callable[[list], np.ndarray],
    images: list,
    labels,
    perturbations: Mapping[str, Callable] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Re-score `images` under each perturbation and report degraded metrics.

    `score_images(list_of_PIL) -> array of FAKE probabilities` is the scoring
    callable (wraps a model's transform + forward + sigmoid); keeping it a plain
    callable means this panel needs no torch and is trivially unit-testable.
    """
    if perturbations is None:
        perturbations = build_perturbations()
    labels = np.asarray(labels, dtype=int).ravel()

    rows: list[dict] = []
    for name, fn in perturbations.items():
        perturbed = [fn(img) for img in images]
        scores = np.asarray(score_images(perturbed), dtype=float).ravel()
        metrics = summary_metrics(scores, labels, threshold)
        metrics = {"perturbation": name, **metrics}
        rows.append(metrics)
    df = pd.DataFrame(rows)
    # Delta AUROC vs the clean baseline, so degradation is readable at a glance.
    if "clean" in df["perturbation"].values:
        clean_auroc = float(df.loc[df["perturbation"] == "clean", "auroc"].iloc[0])
        df["auroc_delta"] = df["auroc"] - clean_auroc
    return df


# --------------------------------------------------------------------------- #
# 5. Calibration — temperature scaling (improvement #3)
# --------------------------------------------------------------------------- #
def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """Fit a single scalar temperature T on validation logits (minimise NLL).

    Uses torch's LBFGS when available, else a NumPy grid+bisection search so the
    calibration path runs without a training stack. Returns T > 0; test logits
    should be divided by T before sigmoid.
    """
    val_logits = np.asarray(val_logits, dtype=float).ravel()
    val_labels = np.asarray(val_labels, dtype=float).ravel()

    try:
        import torch

        logits = torch.tensor(val_logits, dtype=torch.float64)
        labels = torch.tensor(val_labels, dtype=torch.float64)
        log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(logits / log_t.exp(), labels)
            loss.backward()
            return loss.detach()

        optimizer.step(closure)
        return float(log_t.exp().item())
    except ImportError:
        return _fit_temperature_numpy(val_logits, val_labels)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _nll(logits, labels, t):
    p = np.clip(_sigmoid(logits / t), 1e-7, 1 - 1e-7)
    return float(-(labels * np.log(p) + (1 - labels) * np.log(1 - p)).mean())


def _fit_temperature_numpy(logits, labels) -> float:
    """Coarse grid + local refine on T (NumPy fallback, no torch)."""
    grid = np.geomspace(0.05, 20.0, 400)
    losses = [_nll(logits, labels, t) for t in grid]
    best = float(grid[int(np.argmin(losses))])
    # Local refine around the grid minimum.
    lo, hi = best * 0.5, best * 2.0
    fine = np.linspace(lo, hi, 200)
    losses = [_nll(logits, labels, t) for t in fine]
    return float(fine[int(np.argmin(losses))])


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Return calibrated FAKE probabilities = sigmoid(logits / T)."""
    logits = np.asarray(logits, dtype=float).ravel()
    return _sigmoid(logits / float(temperature))


def calibrate(
    val_logits, val_labels, test_logits, test_labels, n_bins: int = 10
) -> dict:
    """Fit T on val, apply to test; report ECE before/after and the temperature."""
    test_logits = np.asarray(test_logits, dtype=float).ravel()
    test_labels = np.asarray(test_labels, dtype=int).ravel()

    scores_before = _sigmoid(test_logits)
    temperature = fit_temperature(val_logits, val_labels)
    scores_after = apply_temperature(test_logits, temperature)

    return {
        "temperature": temperature,
        "ece_before": expected_calibration_error(scores_before, test_labels, n_bins),
        "ece_after": expected_calibration_error(scores_after, test_labels, n_bins),
        "scores_before": scores_before,
        "scores_after": scores_after,
    }


# --------------------------------------------------------------------------- #
# 6. Report generator — report.md + plots
# --------------------------------------------------------------------------- #
def _plot_roc_curves(predictions_by_model: Mapping[str, pd.DataFrame], out_path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for model_name, preds in predictions_by_model.items():
        if preds["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(preds["label"], preds["score"])
        auroc = compute_auroc(preds["score"], preds["label"])
        ax.plot(fpr, tpr, label=f"{model_name} (AUROC={auroc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves (all predictions)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_per_generator_bars(
    tables_by_model: Mapping[str, pd.DataFrame], out_path: Path
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    generators = sorted(
        {g for tbl in tables_by_model.values() for g in tbl["generator"]}
    )
    n_models = max(1, len(tables_by_model))
    x = np.arange(len(generators))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(generators)), 5))
    for i, (model_name, tbl) in enumerate(tables_by_model.items()):
        lookup = dict(zip(tbl["generator"], tbl["auroc"]))
        vals = [lookup.get(g, np.nan) for g in generators]
        ax.bar(x + i * width, vals, width, label=model_name)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(generators, rotation=30, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.axhline(0.85, color="red", ls="--", alpha=0.5, label="0.85 target")
    ax.set_title("Per-generator AUROC (cross-generator panel)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in df.columns) + " |")
    return "\n".join(lines)


def write_report(
    predictions_by_model: Mapping[str, pd.DataFrame],
    out_dir,
    robustness_by_model: Mapping[str, pd.DataFrame] | None = None,
    in_dist_split: str = "test",
    threshold: float = DEFAULT_THRESHOLD,
) -> Path:
    """Produce report.md (3 panels + verdict placeholder) and plots.

    Panels: (1) in-distribution summary, (2) cross-generator per-generator table
    — the headline, (3) robustness under perturbation. Saves ROC curves and a
    per-generator bar chart via matplotlib. Returns the path to report.md.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for name, preds in predictions_by_model.items():
        _validate_predictions(preds)

    # Plots.
    roc_path = plots_dir / "roc_curves.png"
    bars_path = plots_dir / "per_generator_auroc.png"
    _plot_roc_curves(predictions_by_model, roc_path)
    per_gen_tables = {
        name: per_generator_table(preds, threshold)
        for name, preds in predictions_by_model.items()
    }
    _plot_per_generator_bars(per_gen_tables, bars_path)

    lines: list[str] = []
    lines.append("# Veil Detector — Evaluation Report")
    lines.append("")
    lines.append(
        "Auto-generated by `detector-trainer/eval/harness.py`. "
        "Scores are the sigmoid probability of FAKE."
    )
    lines.append("")

    # Panel 1: in-distribution.
    lines.append("## Panel 1 — In-distribution")
    lines.append("")
    lines.append(f"Held-out `{in_dist_split}` split (same generators as train).")
    lines.append("")
    id_rows = []
    for name, preds in predictions_by_model.items():
        subset = preds[preds["split"] == in_dist_split]
        if subset.empty:
            subset = preds
        m = summary_metrics(subset["score"], subset["label"], threshold)
        id_rows.append({"model": name, **m})
    lines.append(_md_table(pd.DataFrame(id_rows)))
    lines.append("")

    # Real-photo false positives — the headline correctness number for the
    # false-positive bug. Reported on the in-dist reals and the held-out wild
    # reals separately so real-world generalization is visible.
    lines.append("### Real-photo false positives")
    lines.append("")
    lines.append("Fraction of REAL images wrongly scored FAKE (lower is better).")
    lines.append("")
    fpr_rows = []
    for name, preds in predictions_by_model.items():
        indist = preds[preds["split"] == in_dist_split]
        wild = preds[preds["split"] == "test_wild"]
        fpr_rows.append(
            {
                "model": name,
                "real_fpr_indist": real_photo_false_positive_rate(indist, threshold),
                "real_fpr_wild": real_photo_false_positive_rate(wild, threshold),
            }
        )
    lines.append(_md_table(pd.DataFrame(fpr_rows)))
    lines.append("")

    # Panel 2: cross-generator per-generator (headline).
    lines.append("## Panel 2 — Cross-generator (headline)")
    lines.append("")
    lines.append("Per-generator AUROC and accuracy. Unseen generators are the honest test.")
    lines.append("")
    for name, tbl in per_gen_tables.items():
        lines.append(f"**{name}**")
        lines.append("")
        lines.append(_md_table(tbl))
        lines.append("")
    lines.append(f"![Per-generator AUROC](plots/{bars_path.name})")
    lines.append("")
    lines.append(f"![ROC curves](plots/{roc_path.name})")
    lines.append("")

    # Panel 3: robustness.
    lines.append("## Panel 3 — Robustness")
    lines.append("")
    if robustness_by_model:
        lines.append("Metrics re-scored under JPEG compression + downscaling.")
        lines.append("")
        for name, rob in robustness_by_model.items():
            lines.append(f"**{name}**")
            lines.append("")
            lines.append(_md_table(rob))
            lines.append("")
    else:
        lines.append(
            "_No robustness results supplied. Pass `robustness_by_model` "
            "(see `robustness_eval`) to populate this panel._"
        )
        lines.append("")

    # Verdict placeholder.
    lines.append("## Verdict")
    lines.append("")
    lines.append(
        "_TODO: one-paragraph verdict — which model wins, where each fails, and "
        "the production recommendation for the Veil `local` signal. Fill in once "
        "both models have been scored on the wild set._"
    )
    lines.append("")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines))
    return report_path
