# Deepfake-Face (Face-Swap) Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `FaceSwapSignal` backend signal that flags face-swaps/composites as a separate "face manipulation" axis and can raise the `/scan` verdict, without flagging genuine faces.

**Architecture:** A new signal implements the existing `Signal` interface (`available()` + `async analyze()`), registered in `registry.py` behind a try/except like every other signal. `analyze()` runs a 3-step pipeline — **face-gate → crop largest face → classify with an on-device model** — and returns `manipulation_score` (not `ai_score`) so it stays out of the general AI-detector agreement logic. Face detector and classifier are lazy-loaded via module-level functions (`_load_face_detector`, `_load_classifier`) that tests monkeypatch, mirroring `local_model.py`'s `_load_clip_builder` pattern. The frontend maps the new score and elevates the verdict when it is a strong positive.

**Tech Stack:** Python, FastAPI, PyTorch + `transformers` (already backend deps), OpenCV Haar cascade for face detection (`opencv-python-headless`), Vite/React frontend (`app.jsx`).

## Global Constraints

- Signals MUST NOT raise for expected failures — return `SignalResult(status=error, error=…)`; the runner is a backstop. (`backend/app/signals/base.py`)
- A missing model/dependency MUST disable only this signal (registry try/except + `available()` returning False), never break `/scan`.
- The signal outputs `manipulation_score` ∈ [0,1]; `ai_score` stays `None`. It never enters the general AI-detector (`local`/`sightengine`/`hive`) agreement logic.
- **Real-face safety is the gate:** the shipped model MUST NOT flag the reference genuine faces (owner headshot, iPhone driver's-license photo, students-at-event). If no candidate passes, fall back to indicator-only (do not change the main verdict). See Task 8.
- Reference image set for the acceptance gate: `~/Downloads/photos` (`Untitled design.jpg` = Curry composite/AI = positive; `sarthakhans_photo.jpg`, `WhatsApp Image 2026-06-14 at 12.05.14.jpeg`, `55195110328_b139fc3758_b.jpg` = genuine faces = negative; `real_ai_demo_1_bicycle.jpg` = no face).
- Backend tests run from `backend/` with `.venv/bin/python -m pytest` (async enabled via `backend/pytest.ini` `asyncio_mode=auto`).
- Config settings mirror the existing `vlm_model_id` pattern in `backend/app/config.py`.

---

## File Structure

- Create: `backend/app/signals/faceswap.py` — `FaceSwapSignal` + `_load_face_detector` / `_load_classifier` + `_should_flag` helper.
- Modify: `backend/app/schemas.py` — add `SignalClass.manipulation`.
- Modify: `backend/app/config.py` — `faceswap_enabled`, `faceswap_model_id`, `faceswap_threshold`.
- Modify: `backend/app/signals/registry.py` — register `FaceSwapSignal`.
- Modify: `backend/requirements.txt` — add `opencv-python-headless`.
- Create: `backend/tests/test_faceswap.py` — unit tests (fakes; no model download).
- Create: `backend/scripts/faceswap_gate.py` — acceptance eval over the reference set + model selection.
- Modify: `app.jsx` — `mapEnvelope` + `buildComparison` verdict integration.

---

## Task 1: Add `manipulation` signal class + face-swap config

**Files:**
- Modify: `backend/app/schemas.py` (SignalClass enum, ~line 19)
- Modify: `backend/app/config.py` (Settings, after the VLM block ~line 47)
- Test: `backend/tests/test_faceswap.py`

**Interfaces:**
- Produces: `SignalClass.manipulation` (str enum value `"manipulation"`); `Settings.faceswap_enabled: bool`, `Settings.faceswap_model_id: str`, `Settings.faceswap_threshold: float`.

- [ ] **Step 1: Write the failing test** — create `backend/tests/test_faceswap.py`:

```python
from app.schemas import SignalClass
from app.config import get_settings


def test_manipulation_signal_class_exists():
    assert SignalClass.manipulation.value == "manipulation"


def test_faceswap_settings_defaults():
    s = get_settings()
    assert isinstance(s.faceswap_enabled, bool)
    assert isinstance(s.faceswap_model_id, str) and s.faceswap_model_id
    assert 0.0 < s.faceswap_threshold < 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: FAIL — `AttributeError: manipulation` / missing settings.

- [ ] **Step 3: Add the enum value** — in `backend/app/schemas.py`, add to `SignalClass`:

```python
    manipulation = "manipulation"  # learned face-swap / composite detector
```

- [ ] **Step 4: Add the settings** — in `backend/app/config.py`, after the VLM block:

```python
    # --- Face-swap / deepfake-face signal ---
    faceswap_enabled: bool = True
    faceswap_model_id: str = "dima806/deepfake_vs_real_image_detection"
    faceswap_threshold: float = 0.7  # conservative; tuned by the acceptance gate
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas.py backend/app/config.py backend/tests/test_faceswap.py
git commit -m "feat(schema): add manipulation signal class + face-swap config"
```

---

## Task 2: FaceSwapSignal core logic (face-gate, threshold, graceful degrade)

**Files:**
- Create: `backend/app/signals/faceswap.py`
- Test: `backend/tests/test_faceswap.py`

**Interfaces:**
- Consumes: `Signal`, `ImageInput` (`backend/app/signals/base.py`); `SignalResult`, `SignalClass`, `SignalStatus` (`backend/app/schemas.py`); `get_settings` (`backend/app/config.py`).
- Produces:
  - `_load_face_detector() -> Callable[[PIL.Image], list]` — returns a detector callable that maps a PIL image to a list of face crops (PIL images); empty list = no face. Monkeypatched in tests.
  - `_load_classifier() -> Callable[[PIL.Image], float]` — returns a callable mapping a face crop to `p_fake ∈ [0,1]`. Monkeypatched in tests.
  - `_should_flag(score: float | None, threshold: float) -> bool`.
  - `class FaceSwapSignal(Signal)` with `name="faceswap"`, `signal_class=SignalClass.manipulation`.

- [ ] **Step 1: Write the failing tests** — append to `backend/tests/test_faceswap.py`:

```python
import io
from types import SimpleNamespace
from PIL import Image
from app.signals import faceswap as fs


def _png_bytes(color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def _settings(threshold=0.7):
    return SimpleNamespace(
        faceswap_enabled=True, faceswap_model_id="stub", faceswap_threshold=threshold
    )


def _img():
    return fs.ImageInput(data=_png_bytes(), filename="x.png", content_type="image/png")


def test_should_flag_threshold():
    assert fs._should_flag(0.9, 0.7) is True
    assert fs._should_flag(0.5, 0.7) is False
    assert fs._should_flag(None, 0.7) is False


async def test_no_face_yields_null_score(monkeypatch):
    monkeypatch.setattr(fs, "get_settings", lambda: _settings())
    monkeypatch.setattr(fs, "_MODEL", None)
    monkeypatch.setattr(fs, "_load_face_detector", lambda: (lambda pil: []))          # no faces
    monkeypatch.setattr(fs, "_load_classifier", lambda: (lambda crop: 0.99))
    result = await fs.FaceSwapSignal().analyze(_img())
    assert result.status.value == "ok"
    assert result.manipulation_score is None
    assert result.ai_score is None


async def test_face_produces_score(monkeypatch):
    monkeypatch.setattr(fs, "get_settings", lambda: _settings())
    monkeypatch.setattr(fs, "_MODEL", None)
    face = Image.new("RGB", (32, 32), (200, 180, 170))
    monkeypatch.setattr(fs, "_load_face_detector", lambda: (lambda pil: [face]))
    monkeypatch.setattr(fs, "_load_classifier", lambda: (lambda crop: 0.88))
    result = await fs.FaceSwapSignal().analyze(_img())
    assert result.status.value == "ok"
    assert result.manipulation_score == 0.88
    assert result.ai_score is None


async def test_loader_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(fs, "get_settings", lambda: _settings())
    monkeypatch.setattr(fs, "_MODEL", None)
    def boom():
        raise RuntimeError("model download failed")
    monkeypatch.setattr(fs, "_load_face_detector", boom)
    result = await fs.FaceSwapSignal().analyze(_img())
    assert result.status.value == "error"
    assert "model download failed" in (result.error or "")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: FAIL — `ModuleNotFoundError: app.signals.faceswap`.

- [ ] **Step 3: Implement** — create `backend/app/signals/faceswap.py`:

```python
"""Face-swap / deepfake-face signal.

Runs a 3-step on-device pipeline (face-gate -> crop largest face -> classify) and
reports a `manipulation_score`. Silent (null score) when no face is present, so it
never affects the verdict for non-face images. Its own axis: `ai_score` stays None.
"""
from __future__ import annotations

import time
from io import BytesIO
from typing import Callable

from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

_MODEL = None       # cached classifier callable
_DETECTOR = None    # cached detector callable


def _should_flag(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _load_face_detector() -> Callable[[Image.Image], list]:
    """Return detect(pil) -> list[PIL face crop]. Implemented in Task 4."""
    raise NotImplementedError("face detector wired in Task 4")


def _load_classifier() -> Callable[[Image.Image], float]:
    """Return classify(pil_face) -> p_fake in [0,1]. Implemented in Task 5."""
    raise NotImplementedError("classifier wired in Task 5")


class FaceSwapSignal(Signal):
    name = "faceswap"
    signal_class = SignalClass.manipulation

    def available(self) -> bool:
        return bool(getattr(get_settings(), "faceswap_enabled", False))

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()
        settings = get_settings()
        try:
            pil = Image.open(BytesIO(image.data)).convert("RGB")
        except UnidentifiedImageError:
            return self._error("uploaded file is not a valid image", started)

        try:
            detect = _load_face_detector()
            faces = detect(pil)
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error(str(exc), started)

        latency_ms = (time.perf_counter() - started) * 1000.0
        if not faces:
            return SignalResult(
                name=self.name, signal_class=self.signal_class,
                status=SignalStatus.ok, ai_score=None, manipulation_score=None,
                confidence=None, latency_ms=latency_ms,
                notes=["no face detected; face-swap check not applicable"],
                raw={"faces": 0},
            )

        try:
            classify = _load_classifier()
            crop = max(faces, key=lambda f: f.size[0] * f.size[1])
            score = float(classify(crop))
        except Exception as exc:  # noqa: BLE001
            return self._error(str(exc), started)

        threshold = float(getattr(settings, "faceswap_threshold", 0.7))
        flagged = _should_flag(score, threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.ok, ai_score=None,
            manipulation_score=max(0.0, min(1.0, score)),
            confidence=score if flagged else 1.0 - score,
            latency_ms=latency_ms,
            notes=[
                ("Possible face manipulation detected" if flagged
                 else "Face looks unmanipulated") + f" ({round(score * 100)}%)",
            ],
            raw={"faces": len(faces), "model": settings.faceswap_model_id,
                 "flagged": flagged},
        )

    def _error(self, error: str, started: float) -> SignalResult:
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=(time.perf_counter() - started) * 1000.0, error=error,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/faceswap.py backend/tests/test_faceswap.py
git commit -m "feat(faceswap): signal core logic with face-gate + graceful degrade"
```

---

## Task 3: Register the signal

**Files:**
- Modify: `backend/app/signals/registry.py`
- Test: `backend/tests/test_faceswap.py`

**Interfaces:**
- Consumes: `FaceSwapSignal` (Task 2); `all_signals()` (`registry.py`).

- [ ] **Step 1: Write the failing test** — append to `backend/tests/test_faceswap.py`:

```python
def test_faceswap_registered():
    from app.signals.registry import all_signals
    assert any(s.name == "faceswap" for s in all_signals())
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py::test_faceswap_registered -q`
Expected: FAIL — no signal named `faceswap`.

- [ ] **Step 3: Register it** — in `backend/app/signals/registry.py`, add a block mirroring the others (after the `local_model` block):

```python
    try:
        from app.signals.faceswap import FaceSwapSignal
        signals.append(FaceSwapSignal())
    except Exception:  # noqa: BLE001 - never let one signal break the registry
        pass
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/registry.py backend/tests/test_faceswap.py
git commit -m "feat(faceswap): register signal in the registry"
```

---

## Task 4: Real face detector (OpenCV Haar)

**Files:**
- Modify: `backend/app/signals/faceswap.py` (`_load_face_detector`)
- Modify: `backend/requirements.txt`
- Test: `backend/tests/test_faceswap.py`

**Interfaces:**
- Produces: `_load_face_detector()` returns `detect(pil) -> list[PIL crop]` using OpenCV's bundled `haarcascade_frontalface_default.xml`. Cached module-level in `_DETECTOR`.

- [ ] **Step 1: Add the dependency**

Edit `backend/requirements.txt`: add `opencv-python-headless>=4.9` under the vision deps. Then:
Run: `cd backend && .venv/bin/pip install "opencv-python-headless>=4.9"`

- [ ] **Step 2: Write the failing integration test** — append to `backend/tests/test_faceswap.py` (guarded so it skips if the reference image is absent):

```python
import os
import pytest

_REF = os.path.expanduser("~/Downloads/photos")
_has_ref = os.path.isdir(_REF)


@pytest.mark.skipif(not _has_ref, reason="reference images not present")
def test_real_detector_finds_face_and_skips_nonface():
    detect = fs._load_face_detector()
    headshot = Image.open(os.path.join(_REF, "sarthakhans_photo.jpg")).convert("RGB")
    bicycle = Image.open(os.path.join(_REF, "real_ai_demo_1_bicycle.jpg")).convert("RGB")
    assert len(detect(headshot)) >= 1        # a real face is found
    assert len(detect(bicycle)) == 0          # no face on the bicycle
```

- [ ] **Step 3: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -k real_detector -q`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 4: Implement** — replace `_load_face_detector` in `backend/app/signals/faceswap.py`:

```python
def _load_face_detector() -> Callable[[Image.Image], list]:
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    import cv2
    import numpy as np

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)

    def detect(pil: Image.Image) -> list:
        arr = np.array(pil.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                         minSize=(48, 48))
        crops = []
        for (x, y, w, h) in boxes:
            m = int(0.25 * max(w, h))  # margin so context isn't cut off
            x0, y0 = max(0, x - m), max(0, y - m)
            x1, y1 = min(pil.width, x + w + m), min(pil.height, y + h + m)
            crops.append(pil.crop((x0, y0, x1, y1)))
        return crops

    _DETECTOR = detect
    return _DETECTOR
```

- [ ] **Step 5: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -k real_detector -q`
Expected: PASS. If Haar misses the headshot, note it and switch the default to `facenet-pytorch` MTCNN (add `facenet-pytorch` to requirements, replace the detector body with an MTCNN detector returning crops) — the interface is unchanged.

- [ ] **Step 6: Commit**

```bash
git add backend/app/signals/faceswap.py backend/requirements.txt backend/tests/test_faceswap.py
git commit -m "feat(faceswap): OpenCV Haar face detector + face-gate integration test"
```

---

## Task 5: Real classifier (transformers)

**Files:**
- Modify: `backend/app/signals/faceswap.py` (`_load_classifier`)
- Test: `backend/tests/test_faceswap.py`

**Interfaces:**
- Produces: `_load_classifier()` returns `classify(pil_face) -> float` = P(fake), using a `transformers` image-classification pipeline over `settings.faceswap_model_id`. Cached in `_MODEL`. Maps the model's fake/real label to a probability in [0,1].

- [ ] **Step 1: Write the failing (import-level) test** — append to `backend/tests/test_faceswap.py`:

```python
def test_classifier_maps_label_to_probability(monkeypatch):
    # Fake the transformers pipeline: return HF-style label/score dicts.
    class FakePipe:
        def __call__(self, pil):
            return [{"label": "fake", "score": 0.82}, {"label": "real", "score": 0.18}]
    monkeypatch.setattr(fs, "_build_pipeline", lambda model_id: FakePipe())
    monkeypatch.setattr(fs, "_MODEL", None)
    monkeypatch.setattr(fs, "get_settings", lambda: _settings())
    classify = fs._load_classifier()
    p = classify(Image.new("RGB", (32, 32)))
    assert abs(p - 0.82) < 1e-6
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -k classifier_maps -q`
Expected: FAIL — `_build_pipeline` / `_load_classifier` NotImplemented.

- [ ] **Step 3: Implement** — replace `_load_classifier` and add `_build_pipeline` and a label-mapping helper in `backend/app/signals/faceswap.py`:

```python
_FAKE_LABELS = {"fake", "deepfake", "ai", "manipulated", "spoof", "1"}


def _p_fake(preds: list) -> float:
    """Map HF image-classification output ([{label, score}, ...]) to P(fake)."""
    for p in preds:
        if str(p["label"]).strip().lower() in _FAKE_LABELS:
            return float(p["score"])
    # Fallback: if only a 'real'-type label is present, invert it.
    top = max(preds, key=lambda p: p["score"])
    return 1.0 - float(top["score"]) if "real" in str(top["label"]).lower() else float(top["score"])


def _build_pipeline(model_id: str):
    from transformers import pipeline
    return pipeline("image-classification", model=model_id, top_k=None)


def _load_classifier() -> Callable[[Image.Image], float]:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    pipe = _build_pipeline(get_settings().faceswap_model_id)

    def classify(crop: Image.Image) -> float:
        return _p_fake(pipe(crop.convert("RGB")))

    _MODEL = classify
    return _MODEL
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_faceswap.py -q`
Expected: PASS (all tests; the label-mapping test uses the fake pipeline — no download).

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/faceswap.py backend/tests/test_faceswap.py
git commit -m "feat(faceswap): transformers classifier + label->P(fake) mapping"
```

---

## Task 6: Frontend verdict integration

**Files:**
- Modify: `app.jsx` (`mapEnvelope` ~line 99; `buildComparison` ~line 119)

**Interfaces:**
- Consumes: `/scan` `signals[].manipulation_score` for `name=="faceswap"`.
- Produces: a `faceManipulation` flag + reason that elevates the verdict.

- [ ] **Step 1: Map the signal** — in `app.jsx` `mapEnvelope` return object, add:

```javascript
    faceswap: toDetector(find("faceswap")),
```

- [ ] **Step 2: Read the score in `buildComparison`** — near the top of `buildComparison`, after `manipulationScores` is built, add the face-swap score and flag (threshold mirrors the backend default; keep them in sync):

```javascript
  const faceSwapScore = detectors.faceswap?.deepfake ?? null;   // manipulation_score
  const FACE_SWAP_THRESHOLD = 0.7;
  const faceManipulation = faceSwapScore != null && faceSwapScore >= FACE_SWAP_THRESHOLD;
```

- [ ] **Step 3: Elevate the verdict** — where the return object is assembled, force at least "needs review" and add a reason when `faceManipulation`:

```javascript
  if (faceManipulation) {
    userSummary.unshift(
      "Possible face manipulation detected — the face may be swapped or composited. Verify the source."
    );
  }
  // ...in the returned object, raise the effective score/verdict:
  const effectiveScore = faceManipulation
    ? Math.max(overallScore ?? 0, 0.6)   // >=0.4 => "Needs review" per verdictLabel
    : overallScore;
```

Use `effectiveScore` where the returned `overallScore`/verdict is computed (replace the returned `overallScore` with `effectiveScore`, keeping the raw score available in `rawScores`). Add `rawScores.faceswap = faceSwapScore` alongside the existing `rawScores`.

- [ ] **Step 4: Manual verification (no automated frontend harness)**

Run the backend + `npm run dev`; confirm via the running app in Task 8. Confirm the JSX compiles (Vite HMR shows `hmr update /app.jsx` with no error in the dev-server log).

- [ ] **Step 5: Commit**

```bash
git add app.jsx
git commit -m "feat(frontend): surface face-swap signal and elevate verdict"
```

---

## Task 7: Acceptance gate — model selection over the reference set

> Operational task (gated on model behavior, not a unit test). Selects the shipped model + threshold and enforces the real-face-safety rule.

**Files:**
- Create: `backend/scripts/faceswap_gate.py`

- [ ] **Step 1: Write the gate script** — `backend/scripts/faceswap_gate.py`:

```python
"""Acceptance gate for the face-swap signal (spec 2026-07-29 §7).

Scores the reference set with a candidate model and asserts: the Curry composite
is flagged while the three genuine faces are NOT. Run per candidate model id.
"""
import argparse, os, asyncio
from types import SimpleNamespace
from app.signals import faceswap as fs

REF = os.path.expanduser("~/Downloads/photos")
POSITIVE = {"Untitled design.jpg"}                     # composite -> must flag
NEGATIVE = {"sarthakhans_photo.jpg",                   # genuine faces -> must NOT flag
            "WhatsApp Image 2026-06-14 at 12.05.14.jpeg",
            "55195110328_b139fc3758_b.jpg"}


async def main(model_id, threshold):
    fs.get_settings = lambda: SimpleNamespace(
        faceswap_enabled=True, faceswap_model_id=model_id, faceswap_threshold=threshold)
    fs._MODEL = None; fs._DETECTOR = None
    sig = fs.FaceSwapSignal()
    rows = {}
    for name in POSITIVE | NEGATIVE:
        fp = os.path.join(REF, name)
        r = await sig.analyze(fs.ImageInput(data=open(fp, "rb").read(),
                                            filename=fp, content_type=None))
        rows[name] = r.manipulation_score
        print(f"  {r.manipulation_score if r.manipulation_score is not None else 'n/a':>6}  {name}")
    ok_pos = all((rows[n] or 0) >= threshold for n in POSITIVE)
    ok_neg = all((rows[n] or 0) < threshold for n in NEGATIVE)
    print(f"PASS={ok_pos and ok_neg}  (positives flagged={ok_pos}, genuine-faces-safe={ok_neg})")
    return ok_pos and ok_neg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dima806/deepfake_vs_real_image_detection")
    ap.add_argument("--threshold", type=float, default=0.7)
    args = ap.parse_args()
    asyncio.run(main(args.model, args.threshold))
```

- [ ] **Step 2: Evaluate candidate models** — run the gate for each candidate (first downloads the model):

```bash
cd backend
for M in \
  "dima806/deepfake_vs_real_image_detection" \
  "prithivMLmods/Deep-Fake-Detector-v2-Model" \
  "prithivMLmods/deepfake-detector-model-v1"; do
  echo "=== $M ==="; PYTHONPATH=$PWD .venv/bin/python scripts/faceswap_gate.py --model "$M"; done
```

- [ ] **Step 3: Pick the winner + tune threshold.** Choose the model where the three genuine faces score below threshold AND the composite scores above it. If multiple pass, prefer the largest margin. Adjust `faceswap_threshold` so all three genuine faces are safely below it. Update `backend/app/config.py` defaults (`faceswap_model_id`, `faceswap_threshold`) and the frontend `FACE_SWAP_THRESHOLD` to match.

- [ ] **Step 4: If NO model passes the genuine-face rule** — set `faceswap_enabled` default to keep the signal running BUT do not elevate the verdict: in `app.jsx` set `FACE_SWAP_THRESHOLD` high enough that only the composite flags, or if even that flags genuine faces, ship **indicator-only** — revert the Task 6 Step 3 verdict elevation, keep only the `rawScores.faceswap` display. Record the decision in the commit message.

- [ ] **Step 5: Commit the locked configuration**

```bash
git add backend/app/config.py app.jsx backend/scripts/faceswap_gate.py
git commit -m "feat(faceswap): lock model + threshold via acceptance gate"
```

---

## Task 8: End-to-end demo verification

**Files:** none (verification only)

- [ ] **Step 1: Restart the backend** so it picks up the new signal:

```bash
cd backend && PYTHONPATH=$PWD .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 2: Confirm the signal is available**

Run: `curl -s http://127.0.0.1:8000/health | python3 -m json.tool`
Expected: `available_signals` includes `faceswap`.

- [ ] **Step 3: Verify the acceptance behavior end-to-end via `/scan`**

```bash
cd /Users/sarthakhans/ai-detector-repo
for f in "Untitled design.jpg" "sarthakhans_photo.jpg" "WhatsApp Image 2026-06-14 at 12.05.14.jpeg" "real_ai_demo_1_bicycle.jpg"; do
  echo "=== $f ==="
  curl -s -X POST http://127.0.0.1:8000/scan -F "media=@$HOME/Downloads/photos/$f" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);f=[s for s in d['signals'] if s['name']=='faceswap'][0];print('  faceswap manipulation_score=',f.get('manipulation_score'),'status=',f['status'])"
done
```

Expected: Curry composite `manipulation_score` ≥ threshold; the two genuine faces `< threshold`; the bicycle `None` (no face).

- [ ] **Step 4: Verify in the UI** — with `npm run dev` running, upload the Curry composite and confirm the verdict reads "needs review / suspicious" with the face-manipulation reason; upload the headshot and confirm it still reads authentic.

- [ ] **Step 5: Final commit (docs/notes if any)** — no code change expected; if thresholds were nudged during verification, commit them.

---

## Self-Review

- **Spec coverage:** §4.1 signal → Tasks 2–5; new `manipulation` class → Task 1; face-gate → Task 2/4; on-device classifier → Task 5; verdict integration → Task 6; acceptance gate (§7) → Tasks 7–8; graceful degrade (§2.4) → Task 2/3; real-face-safety + indicator-only fallback (§8) → Task 7 Step 4. All covered.
- **Placeholder scan:** none — every code step has concrete content.
- **Type consistency:** `_load_face_detector`/`_load_classifier`/`_should_flag`/`_build_pipeline`/`_p_fake` names are consistent across Tasks 2/4/5/7; `manipulation_score` used consistently; frontend reads `deepfake` (mapped from `manipulation_score`, matching the existing `toDetector` mapping in `app.jsx`).
