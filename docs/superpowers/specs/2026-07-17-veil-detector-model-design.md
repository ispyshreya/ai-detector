# Veil In-House Image Detector — Design Spec

**Date:** 2026-07-17
**Status:** Approved (brainstorming complete)
**Owner:** Sarthak Hans
**Context:** CAP 6951 graduate project. Veil is an image-authenticity dashboard that
fuses multiple detector signals into one score. This spec covers the **in-house
model** that feeds the `local` signal (`backend/app/signals/local_model.py`).

## 1. Goal

Deliver two things with one build:

1. **Research artifact** — an honest head-to-head study of two detection approaches
   with a defensible generalization evaluation (for the class grade / writeup).
2. **Real primary detector** — a genuinely competitive local signal that runs for
   free, privately, and feeds the Veil fusion score alongside SightEngine and Hive.

Target results: **>0.95 AUROC in-distribution** and **>0.85 AUROC cross-generator**
(on generators never seen in training), reported honestly per generator.

## 2. Architecture

Two models race on identical data, share one eval harness, and export to one
inference contract.

```
                Hybrid Dataset (shared manifest)
                 public benchmark + wild set
                            │  same splits
              ┌─────────────┴─────────────┐
              ▼                           ▼
   BASELINE                    CONTENDER
   ResNet-50 (fine-tuned)      CLIP ViT-L/14 (frozen) + head
   end-to-end from pixels      features cached once, head trains near-free
              └─────────────┬─────────────┘
                            ▼
                 Shared Eval Harness
        AUROC / acc / per-generator / robustness
                            ▼
             Winning checkpoint → backend/
           local_model.py → Veil fusion engine
```

**Principles:**
- One shared **data module** and one shared **eval module** — both models consume
  identical splits and identical metrics, so the comparison is honest.
- Each model is an isolated unit behind a common `predict(image) -> ai_score`
  interface; either can be understood, trained, or swapped without the other.
- CLIP features are **cached to disk once**, decoupling all head experiments from
  the GPU quota.
- The winner exports a checkpoint conforming to the **existing** `local_model.py`
  contract, so Veil integration is a drop-in.

## 3. Data Pipeline (hybrid)

Three logical pools, all described by a single manifest.

| Pool | Role | Source | Size |
|---|---|---|---|
| **Train** | Fit both models | Public benchmark, full-resolution (Kaggle: cashbowman / tristanzhang / GenImage). Balanced real vs fake | ~200–300k |
| **In-dist test** | "Fair" score | Held-out split of the same generators/sources as train | ~20k |
| **Wild test** | Honest real-world score | In-house: real photos + **Midjourney, DALL·E 3, Flux** (all excluded from training) | ~2–5k |

### Selected datasets (finalized 2026-07-17)

**Train + in-distribution test:**
- **`cartografia/unbiased-tiny-genimage`** (Kaggle, 2.5GB) — GenImage-derived,
  organized by generator (ADM, BigGAN, GLIDE, SD 1.4/1.5, VQDM, wukong + real
  ImageNet). Multi-generator core; its per-generator folders map directly onto the
  group-aware split. **Its Midjourney subset is held out** → becomes wild-test MJ.
- **`rhythmghai/ai-vs-real-images-dataset`** (Kaggle, 249MB, usability 1.0) —
  higher-resolution real+fake supplement (categories: animals/city/food/nature).
  Narrows the resolution gap to the 1024px wild set, fighting the resolution
  shortcut (§3a).

**Wild test (all unseen by training):**
- **Midjourney** — held-out GenImage MJ subset (free, already present).
- **DALL·E 3** — self-generate ~200 via OpenAI (~$8). Chosen over Kaggle because
  available DALL·E 3 sets are thin/mislabeled; DALL·E 3 is also architecturally
  distinct from the SD-family training generators, making it the strongest
  generalization probe.
- **Flux** — self-generate ~400 via Replicate (~$1–2); only single-class Flux sets
  exist on Kaggle.
- **Real wild** — a distinct real-photo source (NOT the GenImage reals) so real
  generalization is honestly tested.

Wild-set generation total ≈ **$10**, deferred to pipeline-run time; needs OpenAI +
Replicate keys and explicit spend go-ahead. Scripts written but never run without
confirmation.

**Manifest** is the single interface between data and models — a CSV/Parquet with
columns: `path, label(real/fake), generator, source, split`. No model touches raw
folders directly.

**Stages (each a small, testable unit):**
1. **Ingest** — attach Kaggle datasets (zero-upload) + pull wild-set images →
   normalize into the manifest.
2. **Clean & dedup** — drop corrupt files; perceptual-hash dedup so near-duplicates
   never leak across splits.
3. **Split** — **group-aware by generator/source** so wild-test generators
   (MJ / DALL·E 3 / Flux) are strictly unseen in training.

### 3a. Leakage guard (critical — improvement #2)

AI-detectors notoriously learn the *dataset's* fingerprint (JPEG quantization,
resolution, "reals from ImageNet vs fakes from a diffusion dump") instead of
AI-ness, yielding lab numbers that collapse in the wild. Mitigations, baked in:
- Match real/fake on resolution; **re-encode both through the same JPEG quality
  range** before training.
- Draw real images from **multiple sources**, not one.
- The wild test set is the honesty check.
- **Report the caveat** in the writeup regardless of numbers.

### 3b. CIFAKE demoted (improvement #1)

CIFAKE (32×32) is used **only as a fast end-to-end smoke test**. It never produces
headline numbers — CLIP expects 224px real-world images and 32px thumbnails would
inflate/mislead results.

### 3c. Wild-set sourcing fallback (improvement #4)

Midjourney is well-covered on Kaggle. DALL·E 3 and Flux may be thin. Fallback:
generate a few hundred each ourselves (DALL·E 3 via OpenAI API, Flux via
Replicate/HF) — a couple dollars, clean labels, guarantees the headline table.

## 4. Training

Two independent units, same manifest, same output contract.

**Baseline — ResNet-50 (fine-tuned end-to-end)**
- ImageNet-pretrained init → 1-logit head → fine-tune on pixels.
- Input 224×224, ImageNet normalization (matches `local_model.py` transform).
- Augmentation: horizontal flip + **JPEG recompression + resize jitter** (so the
  detector survives real-world degradation).
- BCE-with-logits, AdamW, ~10 epochs, checkpoint every epoch (session-safe).
- Output: `resnet50_best.pt` (plain `state_dict`, exactly what `local_model.py`
  loads).

**Contender — CLIP ViT-L/14 (frozen) + head**
- **Stage A (GPU, once):** every image → frozen CLIP → save feature vectors as a
  Kaggle Dataset (~1–2 GPU-hrs, never repeated).
- **Stage B (near-free):** train head on cached features — minutes, zero-GPU,
  repeatable. Two heads reported: **linear probe** (honest baseline) and **2-layer
  MLP** (stronger).
- Output: `clip_head_best.pt` + config recording the CLIP backbone used.
- Fallback if extraction is slow: drop to ViT-B/16 (~1% AUROC cost).

**Shared conventions:** fixed seed, identical splits, same metric-log format, both
write `results.json` + checkpoint to a uniform output folder.

## 5. Evaluation (the research artifact)

One harness, run identically on both models.

**Metrics:** AUROC (primary), accuracy, precision/recall at chosen threshold,
calibration note.

**Three-panel report:**
1. **In-distribution** — same-generator held-out test (expect 0.95–0.99 AUROC).
2. **Cross-generator (headline)** — wild set (MJ / DALL·E 3 / Flux), **per
   generator**. The money table.
3. **Robustness** — re-score wild set under JPEG compression + downscaling.

**Calibration (improvement #3):** temperature-scaling on val so the exported score
is a usable probability for the Veil fusion engine and the confidence rating.

**Deliverable:** `report.md` + plots (ROC curves, per-generator bars) auto-generated
by the harness, plus a one-paragraph verdict (which model wins, where each fails,
production recommendation).

## 6. Integration into Veil

Deliberately minimal — the contract already exists.
- Winner exports a checkpoint + `build_model()` matching what
  `backend/app/signals/local_model.py` already imports from
  `detector-trainer/train.py`.
- **If ResNet wins:** drop-in — point `local_model_checkpoint` at the new file.
  Zero code change.
- **If CLIP wins:** add a thin `build_model` branch assembling "frozen CLIP + head"
  behind the same `model(tensor) -> logit` interface. Signal, fusion, and frontend
  never change.

## 7. Execution — the Kaggle Loop

Repo layout:
```
detector-trainer/
  data/manifest.py      # ingest → clean → split → manifest.csv
  models/resnet.py      # build_model + train (baseline)
  models/clip_head.py   # feature extract + head train (contender)
  eval/harness.py       # shared metrics + report generator
  train.py              # entry: build_model() imported by local_model.py
  kernels/*.ipynb       # Kaggle notebooks wrapping the above
```

Agent-driven cycle (all runnable from the CLI via Bash):
1. Edit code here, commit.
2. `kaggle datasets version` pushes code + manifest as a Kaggle Dataset.
3. `kaggle kernels push` launches the training kernel on Kaggle's GPU.
4. `kaggle kernels output` pulls back checkpoints + `results.json` + plots.
5. Read metrics → decide next iteration → winner's checkpoint lands in `backend/`.

Credentials: `~/.kaggle/kaggle.json` (already configured, auth verified).
Compute: Kaggle free tier — P100 / 2×T4, 30 GPU-hrs/week, 12h sessions. Estimated
weekly burn ~10–15 GPU-hrs (half the quota). Feature caching + per-epoch
checkpoints keep us well inside the limits.

## 8. Non-goals (YAGNI)

- Not chasing the 1M+ image "beat-Hive-at-all-costs" tier (breadth goes in the
  *test* set, not train).
- No video / deepfake-face specialization — images only.
- No new frontend or fusion-engine changes beyond the drop-in signal.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Source-artifact leakage → inflated lab numbers | §3a leakage guard + wild test |
| DALL·E 3 / Flux data thin on Kaggle | §3c self-generate fallback |
| CLIP extraction slow / OOM | Cache once; fall back to ViT-B/16 |
| Kaggle session/quota limits | Per-epoch checkpoints; cached features |
| ResNet overfits dataset fingerprint | JPEG/resize aug; cross-generator eval |
