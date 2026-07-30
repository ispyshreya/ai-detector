# Document-Tamper Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `DocTamperSignal` that, only for document images, produces a tamper-localization heatmap + score from Error Level Analysis, and (if the acceptance gate allows) elevates the `/scan` verdict to "verify with the issuer" — without flagging ordinary photos or genuine-but-processed documents.

**Architecture:** A new backend signal implements the existing `Signal` interface. `analyze()` runs a 3-step pipeline — **document-gate (CLIP zero-shot) → ELA tamper-localize → package (score + base64 heatmap + hottest-region bbox)** — returning `manipulation_score` (its own axis). Document-gate and localizer are lazy-loaded via monkeypatchable module functions (mirroring `faceswap.py`). The v1 localizer reuses the ELA logic already in `ela.py` (per the design's option A: no plug-and-play pixel model exists on-device); a `doctamper_model_id` config seam is left for a future deep model. The frontend shows the heatmap and elevates the verdict only when the score clears `doctamper_threshold` — which the acceptance gate may set high enough to be effectively indicator-only.

**Tech Stack:** Python, FastAPI, PyTorch + open_clip (already backend deps, ViT-L/14 weights cached by the `local` signal), Pillow (ELA), Vite/React (`app.jsx`).

## Global Constraints

- Signals MUST NOT raise for expected failures — return `SignalResult(status=error, error=…)`; the runner is a backstop. (`backend/app/signals/base.py`)
- A missing model/dependency disables ONLY this signal (registry try/except + `available()`), never breaks `/scan`.
- The signal outputs `manipulation_score` ∈ [0,1]; `ai_score` stays None. It never enters the general AI-detector (local/sightengine/hive) agreement logic.
- **Non-document image → `manipulation_score=None`** (document-gate rejects it): the signal is silent on selfies/landscapes/objects.
- **Real-face/real-document safety is the gate:** genuine-but-processed documents (clean scan, screenshot, JPEG-recompressed) must NOT elevate the verdict. If ELA can't keep them below threshold, ship **indicator-only** (heatmap shown, verdict unchanged) by setting `doctamper_threshold` above any achievable score. See Task 7.
- On a flag, the verdict copy MUST route to "verify with the issuing institution" — a lead, never proof.
- `SignalClass.manipulation` already exists (added for `faceswap`) — reuse it; no schema change.
- `doctamper_threshold` in `app.jsx` (`DOC_TAMPER_THRESHOLD`) must match backend `doctamper_threshold`.
- Backend tests run from `backend/` with `.venv/bin/python -m pytest` (`backend/pytest.ini` sets `asyncio_mode=auto`).
- Reference set for the gate: `~/Downloads/photos` plus documents assembled during Task 7 (a known-edited document as positive; genuine scanned/screenshot/recompressed documents as negatives).

---

## File Structure

- Create: `backend/app/signals/doctamper.py` — `DocTamperSignal` + `_load_doc_gate` / `_localize_tamper` / `_should_flag` / heatmap packaging.
- Modify: `backend/app/signals/ela.py` — extract a public `compute_ela(pil, quality=90) -> ElaResult` (diff image + mean/max diff) and have `ElaSignal` use it, so `doctamper` reuses the same ELA math (DRY).
- Modify: `backend/app/config.py` — `doctamper_enabled`, `doctamper_threshold`, `doctamper_doc_gate_threshold`, `doctamper_backbone`.
- Modify: `backend/app/signals/registry.py` — register `DocTamperSignal`.
- Create: `backend/tests/test_doctamper.py` — unit tests (fakes; no downloads).
- Create: `backend/scripts/doctamper_gate.py` — acceptance gate + posture decision.
- Modify: `app.jsx` — `mapEnvelope` + `buildComparison` verdict elevation + heatmap display.

---

## Task 1: Config + shared ELA helper

**Files:**
- Modify: `backend/app/config.py` (after the faceswap block)
- Modify: `backend/app/signals/ela.py`
- Test: `backend/tests/test_doctamper.py`

**Interfaces:**
- Produces: `Settings.doctamper_enabled: bool`, `Settings.doctamper_threshold: float`, `Settings.doctamper_doc_gate_threshold: float`, `Settings.doctamper_backbone: str`.
- Produces (ela.py): `compute_ela(pil: PIL.Image, quality: int = 90) -> ElaResult` where `ElaResult` is a dataclass `{diff: PIL.Image, mean_diff: float, max_diff: float}`.

- [ ] **Step 1: Write the failing test** — create `backend/tests/test_doctamper.py`:

```python
from app.config import get_settings


def test_doctamper_settings_defaults():
    s = get_settings()
    assert isinstance(s.doctamper_enabled, bool)
    assert 0.0 < s.doctamper_doc_gate_threshold < 1.0
    assert s.doctamper_threshold > 0.0
    assert isinstance(s.doctamper_backbone, str) and s.doctamper_backbone


def test_compute_ela_returns_diff_and_stats():
    import io
    from PIL import Image
    from app.signals.ela import compute_ela

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (120, 120, 120)).save(buf, format="PNG")
    pil = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    res = compute_ela(pil, quality=90)
    assert res.diff.size == (64, 64)
    assert res.mean_diff >= 0.0
    assert res.max_diff >= 0.0
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -q`
Expected: FAIL — missing settings / `compute_ela` not importable.

- [ ] **Step 3a: Add settings** — in `backend/app/config.py`, after the faceswap block:

```python
    # --- Document-tamper signal ---
    doctamper_enabled: bool = True
    doctamper_threshold: float = 0.60  # tuned by the acceptance gate; set >1.0 for indicator-only
    doctamper_doc_gate_threshold: float = 0.55  # CLIP doc-vs-photo probability to treat as a document
    doctamper_backbone: str = "ViT-L-14"  # CLIP backbone for the document-gate (weights cached by the local signal)
```

- [ ] **Step 3b: Extract the ELA helper** — in `backend/app/signals/ela.py`, add a public dataclass + function and refactor `ElaSignal.analyze` to use them:

```python
from dataclasses import dataclass

@dataclass
class ElaResult:
    diff: "Image.Image"
    mean_diff: float
    max_diff: float


def compute_ela(image: "Image.Image", quality: int = _QUALITY) -> ElaResult:
    """Recompress at `quality`, return the per-pixel diff image + mean/max diff."""
    original = image.convert("RGB")
    buffer = io.BytesIO()
    original.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")
    diff = ImageChops.difference(original, recompressed)
    max_diff = float(max(hi for (_lo, hi) in diff.getextrema()))
    band_means = _band_means(diff)
    mean_diff = sum(band_means) / len(band_means)
    return ElaResult(diff=diff, mean_diff=mean_diff, max_diff=max_diff)
```

Then in `ElaSignal.analyze`, replace the inline recompress/diff/mean/max block with:
```python
        res = compute_ela(original, _QUALITY)
        max_diff, mean_diff = res.max_diff, res.mean_diff
```
(keep the rest — `_to_manipulation_score(mean_diff, max_diff)`, notes, return — unchanged).

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py tests/ -q`
Expected: PASS (new tests + existing ELA behavior unchanged).

- [ ] **Step 5: Commit**

```bash
git add backend/app/config.py backend/app/signals/ela.py backend/tests/test_doctamper.py
git commit -m "feat(doctamper): config + shared compute_ela helper"
```

---

## Task 2: DocTamperSignal core logic (document-gate, threshold, heatmap packaging, graceful degrade)

**Files:**
- Create: `backend/app/signals/doctamper.py`
- Test: `backend/tests/test_doctamper.py`

**Interfaces:**
- Consumes: `Signal`, `ImageInput`; `SignalResult`, `SignalClass`, `SignalStatus`; `get_settings`.
- Produces:
  - `_load_doc_gate() -> Callable[[PIL.Image], float]` — returns `gate(pil) -> doc_probability ∈ [0,1]`. Monkeypatched in tests. (Real impl: Task 4.)
  - `_localize_tamper(pil: PIL.Image) -> TamperResult` where `TamperResult` is a dataclass `{score: float, heatmap_png_b64: str, bbox: list[int]}`. Monkeypatched/real in Task 5.
  - `_should_flag(score: float | None, threshold: float) -> bool`.
  - `class DocTamperSignal(Signal)` — `name="doctamper"`, `signal_class=SignalClass.manipulation`.

- [ ] **Step 1: Write the failing tests** — append to `backend/tests/test_doctamper.py`:

```python
import io
from types import SimpleNamespace
from PIL import Image
from app.signals import doctamper as dt


def _png_bytes(color=(128, 128, 128)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (48, 48), color).save(buf, format="PNG")
    return buf.getvalue()


def _settings(threshold=0.6, gate=0.55):
    return SimpleNamespace(
        doctamper_enabled=True, doctamper_threshold=threshold,
        doctamper_doc_gate_threshold=gate, doctamper_backbone="ViT-L-14",
    )


def _img():
    return dt.ImageInput(data=_png_bytes(), filename="d.png", content_type="image/png")


def _tamper(score):
    return dt.TamperResult(score=score, heatmap_png_b64="ZmFrZQ==", bbox=[1, 2, 3, 4])


def test_should_flag():
    assert dt._should_flag(0.9, 0.6) is True
    assert dt._should_flag(0.4, 0.6) is False
    assert dt._should_flag(None, 0.6) is False


async def test_non_document_yields_null_score(monkeypatch):
    monkeypatch.setattr(dt, "get_settings", lambda: _settings())
    monkeypatch.setattr(dt, "_DOC_GATE", None)
    monkeypatch.setattr(dt, "_load_doc_gate", lambda: (lambda pil: 0.10))  # not a document
    monkeypatch.setattr(dt, "_localize_tamper", lambda pil: _tamper(0.99))
    r = await dt.DocTamperSignal().analyze(_img())
    assert r.status.value == "ok"
    assert r.manipulation_score is None
    assert r.ai_score is None


async def test_document_produces_score_and_heatmap(monkeypatch):
    monkeypatch.setattr(dt, "get_settings", lambda: _settings())
    monkeypatch.setattr(dt, "_DOC_GATE", None)
    monkeypatch.setattr(dt, "_load_doc_gate", lambda: (lambda pil: 0.92))  # a document
    monkeypatch.setattr(dt, "_localize_tamper", lambda pil: _tamper(0.83))
    r = await dt.DocTamperSignal().analyze(_img())
    assert r.status.value == "ok"
    assert r.manipulation_score == 0.83
    assert r.ai_score is None
    assert r.raw["heatmap_png_b64"] == "ZmFrZQ=="
    assert r.raw["bbox"] == [1, 2, 3, 4]


async def test_loader_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(dt, "get_settings", lambda: _settings())
    monkeypatch.setattr(dt, "_DOC_GATE", None)
    def boom():
        raise RuntimeError("clip load failed")
    monkeypatch.setattr(dt, "_load_doc_gate", boom)
    r = await dt.DocTamperSignal().analyze(_img())
    assert r.status.value == "error"
    assert "clip load failed" in (r.error or "")
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -q`
Expected: FAIL — `ModuleNotFoundError: app.signals.doctamper`.

- [ ] **Step 3: Implement** — create `backend/app/signals/doctamper.py`:

```python
"""Document-tamper signal.

For DOCUMENT images only (a CLIP zero-shot document-gate rejects ordinary
photos), runs Error Level Analysis to localize likely edits and returns a
`manipulation_score` + a base64 heatmap + the hottest-region bbox. Its own axis:
`ai_score` stays None. Silent (null score) on non-documents. A flag routes the
user to "verify with the issuer" — a lead, never proof.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Callable

from PIL import Image, UnidentifiedImageError

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

_DOC_GATE = None  # cached doc-gate callable


@dataclass
class TamperResult:
    score: float
    heatmap_png_b64: str
    bbox: list  # [x, y, w, h]


def _should_flag(score: float | None, threshold: float) -> bool:
    return score is not None and score >= threshold


def _load_doc_gate() -> Callable[[Image.Image], float]:
    """Return gate(pil) -> P(document). Implemented in Task 4."""
    raise NotImplementedError("document-gate wired in Task 4")


def _localize_tamper(pil: Image.Image) -> "TamperResult":
    """ELA tamper localization -> score + heatmap + bbox. Implemented in Task 5."""
    raise NotImplementedError("tamper localizer wired in Task 5")


class DocTamperSignal(Signal):
    name = "doctamper"
    signal_class = SignalClass.manipulation

    def available(self) -> bool:
        return bool(getattr(get_settings(), "doctamper_enabled", False))

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()
        settings = get_settings()
        try:
            pil = Image.open(BytesIO(image.data)).convert("RGB")
        except UnidentifiedImageError:
            return self._error("uploaded file is not a valid image", started)

        try:
            gate = _load_doc_gate()
            doc_prob = float(gate(pil))
        except Exception as exc:  # noqa: BLE001 - report as signal failure
            return self._error(str(exc), started)

        gate_threshold = float(getattr(settings, "doctamper_doc_gate_threshold", 0.55))
        if doc_prob < gate_threshold:
            return SignalResult(
                name=self.name, signal_class=self.signal_class,
                status=SignalStatus.ok, ai_score=None, manipulation_score=None,
                confidence=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                notes=["not a document; tamper check not applicable"],
                raw={"doc_prob": doc_prob},
            )

        try:
            result = _localize_tamper(pil)
        except Exception as exc:  # noqa: BLE001
            return self._error(str(exc), started)

        threshold = float(getattr(settings, "doctamper_threshold", 0.6))
        flagged = _should_flag(result.score, threshold)
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.ok, ai_score=None,
            manipulation_score=max(0.0, min(1.0, result.score)),
            confidence=result.score if flagged else 1.0 - result.score,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=[
                ("Possible document edit detected — verify directly with the "
                 "issuing institution." if flagged
                 else "No strong sign of a localized document edit.")
                + f" ({round(result.score * 100)}%)",
            ],
            raw={"doc_prob": doc_prob, "heatmap_png_b64": result.heatmap_png_b64,
                 "bbox": result.bbox, "flagged": flagged},
        )

    def _error(self, error: str, started: float) -> SignalResult:
        return SignalResult(
            name=self.name, signal_class=self.signal_class,
            status=SignalStatus.error,
            latency_ms=(time.perf_counter() - started) * 1000.0, error=error,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/doctamper.py backend/tests/test_doctamper.py
git commit -m "feat(doctamper): signal core with document-gate + heatmap packaging"
```

---

## Task 3: Register the signal

**Files:**
- Modify: `backend/app/signals/registry.py`
- Test: `backend/tests/test_doctamper.py`

- [ ] **Step 1: Write the failing test** — append to `backend/tests/test_doctamper.py`:

```python
def test_doctamper_registered():
    from app.signals.registry import all_signals
    assert any(s.name == "doctamper" for s in all_signals())
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py::test_doctamper_registered -q`
Expected: FAIL — no signal named `doctamper`.

- [ ] **Step 3: Register it** — in `backend/app/signals/registry.py`, add a block mirroring the others (after the `faceswap` block):

```python
    try:
        from app.signals.doctamper import DocTamperSignal
        signals.append(DocTamperSignal())
    except Exception:  # noqa: BLE001 - never let one signal break the registry
        pass
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/registry.py backend/tests/test_doctamper.py
git commit -m "feat(doctamper): register signal in the registry"
```

---

## Task 4: Real document-gate (CLIP zero-shot)

**Files:**
- Modify: `backend/app/signals/doctamper.py` (`_load_doc_gate`)
- Test: `backend/tests/test_doctamper.py`

**Interfaces:**
- Produces: `_load_doc_gate()` returns `gate(pil) -> P(document)`, using open_clip zero-shot over the configured backbone. Cached in `_DOC_GATE`. Weights are cached on disk from the `local` signal's first load.

- [ ] **Step 1: Write the failing integration test** — append (skipif-guarded on the reference images):

```python
import os
import pytest

_REF = os.path.expanduser("~/Downloads/photos")
_has_ref = os.path.isdir(_REF)


@pytest.mark.skipif(not _has_ref, reason="reference images not present")
def test_doc_gate_ranks_document_over_photo():
    gate = dt._load_doc_gate()
    # The driver's-license capture is document-like; the bicycle is a plain photo.
    doc = Image.open(os.path.join(_REF, "WhatsApp Image 2026-06-14 at 12.05.14.jpeg")).convert("RGB")
    photo = Image.open(os.path.join(_REF, "real_ai_demo_1_bicycle.jpg")).convert("RGB")
    assert gate(doc) > gate(photo)
    assert gate(photo) < 0.55   # a plain photo is rejected by the default gate threshold
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -k doc_gate -q`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement** — replace `_load_doc_gate` in `doctamper.py`:

```python
_DOC_PROMPTS = [
    "a scanned document, a form, a bank statement, an invoice, or an ID card",
    "an ordinary photograph of a person, place, animal, or object",
]


def _load_doc_gate() -> Callable[[Image.Image], float]:
    global _DOC_GATE
    if _DOC_GATE is not None:
        return _DOC_GATE
    import torch
    import open_clip

    backbone = getattr(get_settings(), "doctamper_backbone", "ViT-L-14")
    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained="openai")
    model.eval()
    tokenizer = open_clip.get_tokenizer(backbone)
    text = tokenizer(_DOC_PROMPTS)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def gate(pil: Image.Image) -> float:
        with torch.no_grad():
            img = preprocess(pil.convert("RGB")).unsqueeze(0)
            feats = model.encode_image(img)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * feats @ text_features.T).softmax(dim=-1)
            return float(probs[0, 0])  # P(document)

    _DOC_GATE = gate
    return _DOC_GATE
```

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -k doc_gate -q`
Expected: PASS (first run downloads nothing new — ViT-L-14 weights are cached). If the license does not rank above the bicycle, widen `_DOC_PROMPTS` (add "a close-up photo of a plastic ID card or paper document") and re-run; report which prompts were used.

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/doctamper.py backend/tests/test_doctamper.py
git commit -m "feat(doctamper): CLIP zero-shot document-gate"
```

---

## Task 5: Real tamper localizer (ELA heatmap + score + bbox)

**Files:**
- Modify: `backend/app/signals/doctamper.py` (`_localize_tamper`)
- Test: `backend/tests/test_doctamper.py`

**Interfaces:**
- Consumes: `compute_ela` from `app.signals.ela` (Task 1).
- Produces: `_localize_tamper(pil) -> TamperResult` — `score` from ELA stats, `heatmap_png_b64` a base64 PNG of a colorized, downscaled ELA diff, `bbox` the bounding box of the hottest region.

- [ ] **Step 1: Write the failing test** — append (uses a synthetic splice so it needs no network):

```python
def test_localizer_flags_a_pasted_region_and_returns_heatmap():
    import base64
    # A flat JPEG-compressed image with a sharp bright square pasted in — the
    # square is a "splice": it recompresses very differently from the flat field.
    from PIL import Image as I
    import io as _io
    base = I.new("RGB", (128, 128), (110, 110, 110))
    buf = _io.BytesIO(); base.save(buf, format="JPEG", quality=90)
    doc = I.open(_io.BytesIO(buf.getvalue())).convert("RGB")
    doc.paste((240, 20, 20), (80, 80, 120, 120))  # pasted bright square, not recompressed
    res = dt._localize_tamper(doc)
    assert 0.0 <= res.score <= 1.0
    assert res.score > 0.0
    # heatmap is a decodable PNG
    raw = base64.b64decode(res.heatmap_png_b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    # bbox is 4 ints within the image
    assert len(res.bbox) == 4 and all(isinstance(v, int) for v in res.bbox)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -k localizer -q`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement** — replace `_localize_tamper` in `doctamper.py` and add helpers:

```python
import base64

_HEATMAP_MAX = 512  # downscale the heatmap so the base64 payload stays small


def _localize_tamper(pil: Image.Image) -> "TamperResult":
    from app.signals.ela import compute_ela
    from PIL import ImageOps

    res = compute_ela(pil, quality=90)
    # Score: reuse ELA's calibrated blend via its statistics.
    from app.signals.ela import _to_manipulation_score
    score = _to_manipulation_score(res.mean_diff, res.max_diff)

    # Heatmap: grayscale magnitude of the diff, autocontrast, downscaled PNG.
    gray = res.diff.convert("L")
    heat = ImageOps.autocontrast(gray)
    heat.thumbnail((_HEATMAP_MAX, _HEATMAP_MAX))
    buf = BytesIO()
    heat.save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Bbox of the hottest region: threshold the grayscale diff at a high percentile
    # and take the bounding box of the bright mask.
    bbox = _hot_bbox(gray)
    return TamperResult(score=score, heatmap_png_b64=heatmap_b64, bbox=bbox)


def _hot_bbox(gray: Image.Image) -> list:
    """Bounding box [x,y,w,h] of the brightest region of the diff, or whole image."""
    hi = gray.point(lambda p: 255 if p >= 200 else 0)
    box = hi.getbbox()
    if box is None:
        return [0, 0, gray.width, gray.height]
    x0, y0, x1, y1 = box
    return [x0, y0, x1 - x0, y1 - y0]
```

Note: `_to_manipulation_score` is imported from `ela.py`; if it is not module-public there, promote it (remove the leading underscore or add a thin public wrapper) as part of this task and update `ela.py`'s own call site.

- [ ] **Step 4: Run to verify pass**

Run: `cd backend && .venv/bin/python -m pytest tests/test_doctamper.py -q`
Expected: PASS (all doctamper tests).

- [ ] **Step 5: Commit**

```bash
git add backend/app/signals/doctamper.py backend/app/signals/ela.py backend/tests/test_doctamper.py
git commit -m "feat(doctamper): ELA tamper localizer (heatmap + score + bbox)"
```

---

## Task 6: Frontend verdict integration + heatmap display

**Files:**
- Modify: `app.jsx`

**Interfaces:**
- Consumes: `/scan` `signals[]` where `name=="doctamper"` carries `manipulation_score` and `raw.{heatmap_png_b64, bbox}`.

- [ ] **Step 1: Map the signal** — in `mapEnvelope`'s returned object, add:

```javascript
    doctamper: toDetector(find("doctamper")),
```
(`toDetector` already exposes `deepfake` = manipulation_score and `raw` = signal.raw.)

- [ ] **Step 2: Read score + heatmap in `buildComparison`** — near where the face-swap block reads its score, add:

```javascript
  const docTamperScore = detectors.doctamper?.deepfake ?? null;   // manipulation_score
  // Keep DOC_TAMPER_THRESHOLD in sync with backend doctamper_threshold (config.py).
  const DOC_TAMPER_THRESHOLD = 0.60;
  const documentEdit = docTamperScore != null && docTamperScore >= DOC_TAMPER_THRESHOLD;
  const docHeatmap = detectors.doctamper?.raw?.heatmap_png_b64 ?? null;
```

- [ ] **Step 3: Elevate the verdict + expose the heatmap** — mirror the face-swap elevation. Where `effectiveScore` and the returned object are built, fold in `documentEdit`:

```javascript
  if (documentEdit) {
    userSummary.unshift(
      "Possible document edit detected — a region may have been altered. Verify directly with the issuing institution."
    );
  }
  // combine with any existing face-swap elevation:
  const effectiveScore = (faceManipulation || documentEdit)
    ? Math.max(overallScore ?? 0, 0.6)
    : overallScore;
```
Add `rawScores.doctamper = docTamperScore` to the returned `rawScores`, and add `docHeatmap` (the base64 PNG, or null) to the returned object so the render can show it. Apply the same treatment in the zero-usable-scores branches (mirror the face-swap render-complete objects: include `docHeatmap: null` and `rawScores.doctamper`).

- [ ] **Step 4: Render the heatmap** — in the results view, when `scan.comparison.docHeatmap` is set, render it next to the uploaded image:

```jsx
{scan.comparison.docHeatmap && (
  <figure className="doc-heatmap">
    <img alt="Tamper heatmap — brighter regions changed most under recompression"
         src={`data:image/png;base64,${scan.comparison.docHeatmap}`} />
    <figcaption>Tamper heatmap — brighter = more likely edited. Verify with the issuer.</figcaption>
  </figure>
)}
```

- [ ] **Step 5: Verify it parses**

Run: `cd /Users/sarthakhans/ai-detector-repo && npx --yes esbuild app.jsx --bundle --loader:.jsx=jsx --format=esm --outfile=/dev/null 2>&1 | tail -20`
Expected: no syntax error (import-resolution warnings for 'react' are fine). Behavioral check is Task 8.

- [ ] **Step 6: Commit**

```bash
git add app.jsx
git commit -m "feat(frontend): surface document-tamper heatmap and elevate verdict"
```

---

## Task 7: Acceptance gate — posture decision (elevate vs indicator-only)

> Operational task. Decides whether ELA can safely elevate the verdict, and sets `doctamper_threshold` accordingly.

**Files:**
- Create: `backend/scripts/doctamper_gate.py`

- [ ] **Step 1: Write the gate script** — `backend/scripts/doctamper_gate.py`:

```python
"""Acceptance gate for the document-tamper signal (spec 2026-07-29 §7).

Scores a reference set and reports whether a threshold exists that flags the
edited document while keeping genuine-but-processed documents below it. If not,
the posture is indicator-only (set doctamper_threshold > 1.0).

Reference layout (assemble under ~/Downloads/doc_ref/):
  edited/    -> at least one genuine document with a Photoshopped field (POSITIVE)
  genuine/   -> the same/other genuine documents, cleanly scanned            (NEGATIVE)
  processed/ -> genuine documents screenshotted and/or JPEG-recompressed     (NEGATIVE)
"""
import os, glob, asyncio
from types import SimpleNamespace
from app.signals import doctamper as dt

REF = os.path.expanduser("~/Downloads/doc_ref")


async def score(path, settings):
    dt.get_settings = lambda: settings
    dt._DOC_GATE = None
    sig = dt.DocTamperSignal()
    r = await sig.analyze(dt.ImageInput(data=open(path, "rb").read(), filename=path, content_type=None))
    return r.manipulation_score


async def main():
    settings = SimpleNamespace(doctamper_enabled=True, doctamper_threshold=0.0,
                               doctamper_doc_gate_threshold=0.55, doctamper_backbone="ViT-L-14")
    groups = {g: sorted(glob.glob(os.path.join(REF, g, "*"))) for g in ("edited", "genuine", "processed")}
    scores = {}
    for g, paths in groups.items():
        scores[g] = []
        for p in paths:
            s = await score(p, settings)
            scores[g].append((os.path.basename(p), s))
            print(f"  {g:10} {s if s is not None else 'n/a':>6}  {os.path.basename(p)}")
    pos = [s for _, s in scores["edited"] if s is not None]
    neg = [s for _, s in scores["genuine"] + scores["processed"] if s is not None]
    if pos and neg:
        margin = min(pos) - max(neg)
        print(f"min(edited)={min(pos):.3f}  max(genuine/processed)={max(neg):.3f}  margin={margin:.3f}")
        print("POSTURE:", "ELEVATE (set threshold between them)" if margin > 0 else "INDICATOR-ONLY (no separating threshold)")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Assemble the reference set + run.** Gather documents into `~/Downloads/doc_ref/{edited,genuine,processed}/` (a genuine document + a Photoshopped-field copy; plus screenshot/recompressed copies of genuine ones). Then:

```bash
cd backend && PYTHONPATH=$PWD .venv/bin/python scripts/doctamper_gate.py
```

- [ ] **Step 3: Decide the posture + set the threshold.**
  - If `margin > 0` (edited separates from all genuine/processed): set `doctamper_threshold` in `config.py` to a value between `max(neg)` and `min(pos)`, and set `DOC_TAMPER_THRESHOLD` in `app.jsx` to match. Verdict elevation is ON.
  - If `margin <= 0` (ELA can't separate — the expected outcome): set `doctamper_threshold = 1.01` in `config.py` and `DOC_TAMPER_THRESHOLD = 1.01` in `app.jsx`. This ships **indicator-only** — the heatmap still shows on documents (Task 6 renders it whenever present), but the verdict never elevates. Record the decision in the commit message.

- [ ] **Step 4: Commit the locked configuration**

```bash
git add backend/app/config.py app.jsx backend/scripts/doctamper_gate.py
git commit -m "feat(doctamper): lock threshold + posture via acceptance gate"
```

---

## Task 8: End-to-end demo verification

**Files:** none (verification only)

- [ ] **Step 1: Restart the backend**

```bash
cd backend && PYTHONPATH=$PWD .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- [ ] **Step 2: Confirm the signal is available**

Run: `curl -s http://127.0.0.1:8000/health | python3 -m json.tool`
Expected: `available_signals` includes `doctamper`.

- [ ] **Step 3: Verify behavior via `/scan`**

```bash
cd /Users/sarthakhans/ai-detector-repo
# a document (license) -> gets a doctamper score + heatmap in raw
curl -s -X POST http://127.0.0.1:8000/scan -F "media=@$HOME/Downloads/photos/WhatsApp Image 2026-06-14 at 12.05.14.jpeg" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);s=[x for x in d['signals'] if x['name']=='doctamper'][0];print('doc score=',s.get('manipulation_score'),'heatmap?', bool((s.get('raw') or {}).get('heatmap_png_b64')),'status=',s['status'])"
# a non-document (bicycle) -> null score (document-gate rejects)
curl -s -X POST http://127.0.0.1:8000/scan -F "media=@$HOME/Downloads/photos/real_ai_demo_1_bicycle.jpg" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);s=[x for x in d['signals'] if x['name']=='doctamper'][0];print('bicycle score=',s.get('manipulation_score'),'(want null)')"
```
Expected: the license gets a numeric score + `heatmap? True`; the bicycle gets `null` (document-gate rejects it).

- [ ] **Step 4: Verify in the UI** — with `npm run dev` running, upload a document and confirm the heatmap renders; confirm the verdict elevates only if the gate posture was ELEVATE (otherwise it stays as-is with the heatmap shown).

- [ ] **Step 5: Final commit (if thresholds were nudged during verification).**

---

## Self-Review

- **Spec coverage:** §4.1 signal → Tasks 2–5; document-gate (CLIP reuse) → Task 4; on-device localizer (ELA per option A) → Task 5; heatmap delivery → Task 5 (`raw`) + Task 6 (render); verdict integration + "verify with issuer" copy → Task 6; acceptance gate + indicator-only fallback (§7/§8) → Task 7; graceful degradation (§2.5) → Task 2/3; reuse loaded CLIP backbone (§5) → Task 4; `manipulation_score` axis (§5) → Task 2. All covered. Non-goals (OCR/checksum, fabrication, AI-gen, video) correctly excluded.
- **Placeholder scan:** none — every code step has concrete content.
- **Type consistency:** `_load_doc_gate`/`_localize_tamper`/`_should_flag`/`TamperResult`/`ElaResult`/`compute_ela`/`_hot_bbox` consistent across Tasks 1/2/4/5; `manipulation_score` used throughout; frontend reads `deepfake` (= manipulation_score via `toDetector`) and `raw.heatmap_png_b64`, matching the backend `raw` contract; `DOC_TAMPER_THRESHOLD` ↔ `doctamper_threshold` sync noted in Tasks 6/7.
