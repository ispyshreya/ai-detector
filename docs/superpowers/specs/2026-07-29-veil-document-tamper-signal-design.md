# Veil — Document-Tamper Signal Design Spec

**Date:** 2026-07-29
**Status:** Approved (brainstorming complete)
**Owner:** Sarthak Hans
**Context:** CAP 6951 graduate project. Follows the detector spec
(`docs/superpowers/specs/2026-07-17-veil-detector-model-design.md`), the
real-photo robustness spec (`2026-07-27`), and the deepfake-face signal
(`2026-07-29-veil-deepfake-face-signal-design.md`). Document/ID-forgery detection
was explicitly **parked** in the prior specs ("a real photo of an ID should just
read REAL for now"). This spec un-parks it as the next specialist signal, using
the same pattern that shipped face-swap: a gated on-device model, an acceptance
gate on genuine data, and a plain-language explanation. Target users: journalists
verifying leaked documents, and parents/consumers checking suspicious documents.

## 1. Problem

Veil's general detector answers "was this synthesized by AI?" An **edited real
document** — a genuine bank statement, ID, or invoice where a number, name, date,
or balance was altered — is a real photo/scan, so the CLIP detector reads it REAL
and says nothing about the edit. Detecting a *localized alteration of a genuine
document* is a distinct problem (tamper localization), not generation detection —
the same way face-swap was distinct from AI-generation.

## 2. Goal & success criteria

Add a specialist signal that flags a localized document edit and **shows where**,
without crying forgery on ordinary photos or on genuine-but-processed documents.

**Success criteria:**
1. A known edited document (a genuine statement/ID with a Photoshopped field)
   scores **high**, and the returned heatmap localizes the edited region.
2. **Genuine-but-processed documents** — a real document that has been scanned,
   screenshotted, and/or JPEG-recompressed — score **low** (not flagged). This is
   the hard bar: re-compression is exactly what trips pixel tamper models.
3. **Non-document images** (selfie, landscape, object) produce **no** tamper score
   and never affect the verdict.
4. On a flag: the verdict rises to "needs review," a heatmap of the suspected
   region is shown, and the guidance **always routes to "verify with the issuing
   institution"** — a lead, never proof.
5. Graceful degradation: a missing model/dependency disables only this signal,
   never breaks `/scan`.

## 3. Guiding principle

> Localize edits conservatively, show the region, and defer to the issuer. A false
> "this document is forged" is a serious harm (a journalist mis-publishing, a
> parent wrongly accusing), so when the model is unsure — or the model can't keep
> genuine scanned/compressed documents safe — stay an indicator, not a verdict.

## 4. Scope — Build / Non-goals

### 4.1 Build

| Item | Detail |
|---|---|
| `DocTamperSignal` backend signal | `name="doctamper"`, `signal_class=manipulation`. `available()` + `async analyze()`; registered behind a try/except like the others. |
| 3-step `analyze()` pipeline | document-gate → tamper-localize → package (score + heatmap + hottest-region bbox). |
| Document-gate (reuses loaded CLIP) | Zero-shot classify "scanned document / form / statement / ID" vs "an ordinary photograph" using the ViT-L/14 CLIP backbone already loaded for the `local` signal. Not document-like → `manipulation_score=None` (silent). Config threshold. |
| On-device tamper-localization model | A pretrained pixel manipulation-localization model (TruFor / CAT-Net / MVSS-Net-style) producing a per-pixel heatmap + scalar P(edited). Config-driven (`doctamper_model_id`, `doctamper_threshold`); chosen by the §7 acceptance gate. |
| Heatmap delivery to UI | Signal returns a base64-PNG tamper heatmap + hottest-region bbox in `raw`; the frontend renders it alongside the upload so the region is visible. |
| Frontend verdict integration | `app.jsx` reads the doctamper score; a confident score elevates the verdict to "needs review" with a "possible document edit — verify with the issuer" reason and displays the heatmap. |
| VLM explanation reuse | Reuse the existing SmolVLM `/explain` layer to describe the flagged region in plain language. |

### 4.2 Non-goals (YAGNI / parked)

- **Structural/semantic checks** (OCR + checksum/routing/MRZ validation, arithmetic
  consistency, font/alignment forensics). Powerful and low-false-positive, but a
  separate follow-on signal — deferred.
- **Fabricated-document detection** (wholesale fakes, template generation). Different
  cues (provenance, format mismatch); separate signal.
- **AI-generated document images** — already covered by the `local` CLIP detector.
- No video / multi-page PDF parsing. Single still image in.
- No pixel-perfect editing UI; heatmap is view-only.

## 5. Architecture & data flow

`DocTamperSignal.analyze(image)`:

1. **Document-gate.** Encode the image with the already-loaded CLIP backbone and
   score it against the text prompts "a scanned document / form / bank statement /
   ID card" vs "an ordinary photograph." Below the document threshold →
   `SignalResult(status=ok, manipulation_score=None, notes=["not a document; tamper
   check not applicable"])`. This is the primary guard against flagging normal photos.
2. **Tamper-localize.** Run the tamper model → heatmap `H` (float per pixel) and a
   scalar `p_tampered` (e.g. max/mean of `H` over a calibrated aggregation).
3. **Package.** Return `SignalResult(status=ok, ai_score=None,
   manipulation_score=p_tampered, confidence=…, notes=[…],
   raw={"heatmap_png_b64": …, "bbox": [x,y,w,h], "model": …})`.

Model + backbone are **lazy-loaded and cached** module-level (mirroring
`faceswap.py` / `local_model.py`). Load failure → `available()`/analyze report
error; the registry try/except keeps `/scan` alive. `manipulation_score` (not
`ai_score`) keeps this signal on its own axis, out of the general AI-detector
agreement logic.

The CLIP document-gate should share the extractor the `local` signal already
builds where feasible (avoid loading ViT-L/14 twice); if sharing is impractical
across signal modules, gate with the same backbone name so weights are cached once.

## 6. Verdict integration (frontend `app.jsx`)

- `mapEnvelope` maps `doctamper` → `{ manipulation_score, raw, status }` (the `raw`
  carries the heatmap + bbox).
- `buildComparison` adds: if `doctamperScore >= docTamperThreshold`, set a
  `documentEdit` flag that (a) elevates the returned verdict to at least "needs
  review," and (b) pushes a reason: "Possible document edit detected — a region may
  have been altered. Verify directly with the issuing institution." Below threshold
  or null → verdict unchanged, no reason.
- The render displays the heatmap image (from `raw.heatmap_png_b64`) beside the
  uploaded image when `documentEdit` is set.
- Threshold constant kept in sync with the backend `doctamper_threshold`.

## 7. Verification — the acceptance gate

The pretrained model ships only if it clears this bar; this is the go/no-go and it
enforces the false-positive rule:

| case | required behavior |
|---|---|
| Known edited document (Photoshopped field) | **high** score; heatmap over the edit |
| Genuine document, clean scan | **low** → not flagged |
| Genuine document, screenshotted | **low** → not flagged |
| Genuine document, JPEG-recompressed | **low** → not flagged |
| Non-document photo (selfie/landscape) | **no score** (document-gate rejects) |

If no candidate model keeps the three genuine-but-processed documents below
threshold, **do not elevate the verdict** — ship **indicator-only** (heatmap shown,
main verdict unchanged), or disable the signal, per §3.

**Automated tests:** unit — model-unavailable → graceful; non-document → null score
(document-gate); threshold/elevation logic; heatmap present in `raw` on a flag.
**Acceptance:** a script scoring a small reference set (built during implementation)
asserting the table above.

## 8. Risks

| Risk | Mitigation |
|---|---|
| Tamper model flags genuine scanned/compressed documents (the core risk) | Conservative threshold; hard acceptance gate on processed genuine docs; indicator-only fallback; never ship a model that fails §7. |
| Signal runs on ordinary photos and false-positives | CLIP document-gate returns null for non-documents before the tamper model runs. |
| Pretrained tamper models are hard to run on-device (custom weights/preprocessing) | Evaluate multiple candidates; config-swappable model id; if none is viable on-device, indicator-only or defer — do not add an external API dependency without a follow-up decision. |
| Users treat a heatmap as proof | Verdict copy always routes to "verify with the issuing institution"; framed as a lead. |
| Loading ViT-L/14 twice (gate + local signal) wastes memory | Share the extractor / same backbone name so weights cache once (§5). |

## 9. Open items (resolved in implementation, not blocking)

- Exact tamper model (TruFor vs CAT-Net vs MVSS-Net vs an HF manipulation-detector)
  and `doctamper_threshold` — picked by the §7 gate.
- Document-gate prompt set and threshold — tuned so genuine documents pass and
  ordinary photos are rejected.
- Heatmap encoding detail (overlay vs side-by-side) — chosen during frontend work;
  base64 PNG in `raw` is the contract.
