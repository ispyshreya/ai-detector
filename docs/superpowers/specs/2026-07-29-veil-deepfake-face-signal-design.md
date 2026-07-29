# Veil — Deepfake-Face (Face-Swap) Signal Design Spec

**Date:** 2026-07-29
**Status:** Approved (brainstorming complete)
**Owner:** Sarthak Hans
**Context:** CAP 6951 graduate project. Follows the detector spec
(`docs/superpowers/specs/2026-07-17-veil-detector-model-design.md`) and the
real-photo robustness spec (`docs/superpowers/specs/2026-07-27-veil-real-photo-robustness-design.md`).
The in-house CLIP detector reliably separates real photos from full-image AI
generation, but **misses face-swaps / composites** — e.g. a real-looking portrait
with a celebrity's face swapped/composited in scored 0.13–0.40 (read authentic).
A modern-generator retrain moved that case toward fake but did not catch it and
regressed real-photo FPR (3%→10%), confirming face-swap detection is a *different
problem* that needs its own signal rather than more training data.

## 1. Problem

Face-swaps and face composites are a distinct manipulation family from
full-image diffusion generation. The general AI-image detector treats a
photorealistic portrait of a real-looking person as authentic, so a swapped or
composited face slips through. This is the one confirmed miss on the reference
image set and a common real-world scam vector (fake profiles, impersonation).

## 2. Goal & success criteria

Add a specialist signal that flags face-swaps/composites **without reintroducing
false positives on genuine faces** — the project's core theme.

**Success criteria:**
1. The known composite (Steph Curry face-swap portrait) is scored **high** and
   raises the overall verdict to "needs review / suspicious."
2. The genuine faces in the reference set — the owner's headshot, the iPhone
   photo of a driver's license, and the students-at-an-event photo — are **not**
   flagged (low face-manipulation score, verdict unchanged).
3. Images with **no face** (landscape, bicycle, object) produce **no** face-swap
   score and do not affect the verdict.
4. The signal degrades gracefully: a missing model/dependency disables only this
   signal, never the `/scan` response.

## 3. Guiding principle

> Catch face-swaps as a separate axis, conservatively. When the model is
> uncertain, stay silent. Never let this signal flag a real person's face — a
> face-swap detector that trips on genuine selfies would undo the real-photo work.

## 4. Scope — Build / Non-goals

### 4.1 Build

| Item | Detail |
|---|---|
| `FaceSwapSignal` backend signal | `name="faceswap"`, `signal_class=manipulation` (new enum value). Implements `available()` + `async analyze()`; registered in `registry.py` behind a try/except like the others. |
| 3-step `analyze()` pipeline | face-gate → crop largest face → classify. |
| On-device face detector | Lightweight (leaning `facenet-pytorch` MTCNN; OpenCV Haar as a lighter fallback). Decided in implementation by reliability on the reference set. |
| On-device face-forgery classifier | A HuggingFace ViT-style deepfake/face-forgery model via `transformers`, chosen in implementation by the §7 acceptance gate. Config-driven (`faceswap_model_id`, `faceswap_threshold`). |
| Frontend verdict integration | `mapEnvelope` reads `faceswap.manipulation_score`; `buildComparison` elevates the verdict to "needs review / suspicious" with a "Possible face manipulation detected" reason when the score is a strong positive. |

### 4.2 Non-goals (YAGNI / parked)

- Fully-synthetic-face detection (StyleGAN / "thispersondoesnotexist"). Different model family.
- Deepfake **video** / multi-frame analysis.
- Training our own face-swap model. If a pretrained model can't clear the
  acceptance gate, fall back to "indicator only" (show a flag, do not change the
  main verdict) rather than shipping a real-face-flagging model.
- Document/ID forgery detection (still parked from the prior spec).

## 5. Architecture & data flow

`FaceSwapSignal.analyze(image)`:

1. **Face-gate.** Detect faces. **No face → `SignalResult(status=ok,
   manipulation_score=None, notes=["no face detected; face-swap check not
   applicable"])`.** The signal contributes nothing to the verdict for non-face
   images — this is the primary guard against false positives on objects/scenes.
2. **Crop.** Take the largest detected face, expand by a margin, crop.
3. **Classify.** Run the crop through the face-forgery classifier →
   `p_fake ∈ [0,1]`. Return `SignalResult(status=ok,
   manipulation_score=p_fake, ai_score=None, confidence=…,
   notes=[…], raw={"faces": n, "model": …})`.

Model + detector are **lazy-loaded and cached** module-level (mirroring
`local_model.py`). Load failure → `available()` returns False (signal disabled),
consistent with the registry's graceful degradation.

`manipulation_score` (not `ai_score`) is used deliberately so the face-swap
signal is its own axis and never enters the general AI-detector agreement logic.

## 6. Verdict integration (frontend `app.jsx`)

- `mapEnvelope` maps `faceswap` → `{ manipulation_score, status, … }`.
- `buildComparison` adds a rule: if `faceswap.manipulation_score >= faceswapThreshold`
  (a strong positive), set a `faceManipulation` flag that:
  - elevates the overall verdict to at least "needs review / suspicious," and
  - pushes a reason line: "Possible face manipulation detected — the face may be
    swapped or composited. Verify the source."
- If the score is below threshold or `None`, the verdict is unchanged and no
  face-manipulation reason is shown. The signal is silent when unsure.

## 7. Verification — the acceptance gate

The pretrained model is only shipped if it clears this bar on the reference set
(`~/Downloads/photos`), which is the go/no-go test:

| image | true | required behavior |
|---|---|---|
| Curry face-swap portrait | composite | **high** score → verdict raised |
| owner's headshot | real face | **low** → not flagged |
| driver's license (iPhone photo) | real face | **low** → not flagged |
| students at event | real faces | **low** → not flagged |
| bicycle / non-face | no face | **no score** → verdict unchanged |

If no candidate pretrained model clears this (specifically: does not flag the
three genuine faces), we fall back to **indicator-only** (§4.2) and do not change
the main verdict.

**Automated tests:**
- Unit: model-unavailable → signal `available()` False (graceful); no-face image →
  `manipulation_score is None`; threshold/elevation logic in isolation.
- Acceptance: a script scoring the reference set, asserting the table above.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Face-swap model flags genuine faces (headshot/license) | Conservative threshold; hard acceptance gate; indicator-only fallback; never ship a model that fails §7. |
| Pretrained model weak on still-image composites (trained on video deepfakes) | Evaluate multiple candidates against §7; config-swappable model id; self-trained model is a parked follow-up. |
| Extra dependency weight (face detector + transformers model) | Lazy-load + cache; graceful-degrade if absent; prefer the lighter detector that passes §7. |
| No face detected on a genuinely swapped face (detector miss) | Acceptable failure mode — signal stays silent, general detector still runs; documented, not blocking. |

## 9. Open items (resolved in implementation, not blocking)

- Exact face-detector library (MTCNN vs Haar) and classifier model id — picked by
  the §7 gate.
- Default `faceswap_threshold` — tuned so the three genuine faces pass.
