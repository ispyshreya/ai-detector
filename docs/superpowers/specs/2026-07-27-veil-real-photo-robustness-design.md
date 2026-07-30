# Veil Detector — Real-Photo Robustness & Consolidation Design Spec

**Date:** 2026-07-27
**Status:** Approved (brainstorming complete)
**Owner:** Sarthak Hans
**Context:** CAP 6951 graduate project. Builds directly on the approved detector
spec `docs/superpowers/specs/2026-07-17-veil-detector-model-design.md`. That spec
delivered a trained ResNet-50 baseline (now on `detector-model`) but the model
**over-predicts FAKE on real-world photos**. This spec covers the work to fix that
and to consolidate the additions that have since landed on the branch.

## 1. Problem

The trained ResNet-50 reports 97.86% accuracy on a held-out GenImage test set, but
that number is largely **dataset-artifact separation** (reals = ImageNet source,
fakes = a diffusion dump), not true real-vs-AI discrimination — the §3a leakage
risk from the prior spec, now confirmed in practice:

- A genuine selfie is flagged as AI-generated.
- An iPhone photo of a driver's license is flagged as AI-generated.

Real-world captures (phone photos, selfies, document scans) fall outside the narrow
"real" distribution the model trained on, so they trip the fake detector. This is
the single most important correctness defect in the product today.

## 2. Goal & success criteria

Ship a Veil that **works where existing detectors fail — on real-world photos.**
Grading is product-first; the depth that fixes the false positives is also the
differentiator versus off-the-shelf detectors. The research rigor and the product
fix are therefore the same work.

**Success criteria:**
1. **False-positive rate on real phone/selfie/document photos is a first-class,
   reported metric** — target low single-digit FPR on a held-out real-world set,
   not just high accuracy on in-distribution data.
2. **Cross-generator AUROC** on unseen generators (Midjourney / DALL·E 3 / Flux)
   reported per generator, targeting >0.85 (per prior spec §5).
3. A **working end-to-end demo**: upload → Veil score → plain-language verdict →
   visual explanation, in which real photos read as REAL.

## 3. Guiding principle

> Fix the false positives with diverse real-world data + an eval that measures
> real-photo FPR honestly. That fix is the differentiator.

Frozen **CLIP ViT-L/14** features generalize to real-world images far better than a
fine-tuned ResNet, making the spec's "contender" a direct candidate fix rather than
an academic comparison.

## 4. Scope — Keep / Remove / Build

### 4.1 Keep (works, aligned)

| Item | Rationale |
|---|---|
| Backend multi-signal architecture — `/scan`, `backend/app/signals/registry.py`, graceful per-signal degradation | Solid core; unchanged. |
| `backend/app/signals/local_model.py` wrapper contract | The drop-in socket; the retrained checkpoint plugs straight in. |
| Frontend `buildComparison` + disagreement/inconclusive logic in `app.jsx` | Good UX; matches the confidence-rating feature. Keep. |
| VLM explanation **concept** — `/explain` endpoint + frontend hook | Genuine differentiator ("why was this flagged"). Keep exactly one implementation. |
| `detector-trainer/train.py` `resnet50_dropout` builder + the Kaggle training loop | Proven infra; reused for retraining. |
| Trained ResNet **as the baseline** | Keep the pipeline; retrain the weights (see 4.3). |

### 4.2 Remove / Consolidate

| Item | Action |
|---|---|
| Two `/explain` implementations — `backend/app/explain/vlm.py` and `detector-trainer/api.py` (already diverged) | Collapse to one. Keep backend `vlm.py`; remove the duplicate explanation code from `detector-trainer/api.py`. |
| `detector-trainer/api.py` as a parallel model server | Redundant — the backend serves the model via `local_model.py`. Reduce to a dev-only tool or delete. |
| Orphaned Anthropic/OpenAI explain wiring in `backend/app/config.py` (`anthropic_model = "claude-opus-4-8"`) | Local-first per the README → make local SmolVLM the one explainer; keep the Anthropic key optional but drop the unused explain path. |
| `detector-trainer/split_data.py` naive random split | Replace with the group-aware-by-generator manifest split (prior spec §3) so wild generators stay unseen. |
| Windows dev paths baked into `detector-trainer/output/test_metrics.json` (`C:\Users\shrey\...`) | Regenerate from the new eval harness. |

### 4.3 Build / Build-upon (priority order)

1. **Diverse real-world training data.** Add multi-source reals — phone photos,
   selfies, documents/IDs, plus Unsplash/COCO/FFHQ-style images — matched on
   resolution and re-encoded through a uniform JPEG quality range (prior spec §3a).
   Re-add the `rhythmghai` reals with correct path-based labels.
2. **Wild eval harness with real-photo FPR as a headline metric.** Extend beyond
   `detector-trainer/test.py`'s accuracy+confusion to report: specificity
   (real → REAL) on a held-out real-world set, and per-generator AUROC on fakes
   (prior spec §5). Auto-generate the report table.
3. **Retrain ResNet + race the CLIP ViT-L/14 contender.** Retrain ResNet on the
   diverse reals with JPEG/resize augmentation; train the frozen-CLIP + head
   contender (prior spec §4). Score both on the same wild set; winner exports to the
   `local_model.py` contract.
4. **Calibration / threshold tuning.** The 0.5 threshold is mis-set for real photos;
   temperature-scale on a validation split so real captures land REAL and the
   exported score is a usable probability for the fusion engine.

## 5. Sequencing

`data (1) → eval harness (2) → retrain + CLIP race (3) → calibrate (4)`, then a
cleanup pass folds in the 4.2 Remove/Consolidate items. The false-positive fix
(steps 1–4) is both the product fix and the differentiator; the VLM demo is already
built and only needs the de-dupe.

## 6. Non-goals (YAGNI / parked)

- **Document/ID-forgery detection** as a distinct capability. For this build, a real
  photo of an ID should simply read REAL. ID/document authenticity is a roadmap item.
- **Shreya's in-flight VLM work.** Deferred coordination (see §8). We proceed with
  our plan; her additions get reconciled later. Not a blocker for this spec.
- No video / deepfake-face specialization (per prior spec §8).

## 7. Verification

End-to-end, the fix is verified when:

1. **Unit / eval:** the new harness runs on the wild set and emits a report with
   per-generator AUROC **and** real-photo FPR. FPR on the held-out real-world set is
   low single digits; cross-generator AUROC is reported per generator.
2. **Regression on the reported failures:** the two known false positives — a real
   selfie and an iPhone photo of a driver's license — are scored by the retrained
   winning checkpoint and both read REAL (score below the calibrated threshold).
3. **Integration:** `local_model_checkpoint` points at the new checkpoint; the
   backend `/scan` endpoint returns a REAL-leaning score for those same real photos,
   and `/explain` returns one coherent explanation (single implementation, no
   duplicate server).
4. **Demo:** upload each real photo in the React app and confirm the verdict reads
   authentic/low-risk, with a sensible visual explanation.

## 8. Open coordination items

- **CLIP ownership.** The CLIP contender (4.3 step 3) may overlap with Shreya's
  in-flight VLM work. Before training starts, confirm whether she is building the
  CLIP detector-contender, the SmolVLM explainer, or a VLM-as-detector, and divide
  labor so CLIP is trained once. Tracked, not blocking.

## 9. Risks

| Risk | Mitigation |
|---|---|
| Even with diverse reals, cross-generator AUROC lands below 0.85 | That is an honest finding; report per generator and iterate on data. |
| Retrained model trades fake recall for real specificity | Report both; tune threshold via calibration (4.3 step 4). |
| Duplicated `/explain` drifts further before consolidation | De-dupe early in the cleanup pass; single owner for `vlm.py`. |
| Wild real-world set too small to trust FPR | Source reals from multiple distinct sources; report set size and CI. |
