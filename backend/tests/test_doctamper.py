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
