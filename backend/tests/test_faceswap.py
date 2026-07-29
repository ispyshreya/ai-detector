import io
from types import SimpleNamespace
from PIL import Image
from app.schemas import SignalClass
from app.config import get_settings
from app.signals import faceswap as fs


def test_manipulation_signal_class_exists():
    assert SignalClass.manipulation.value == "manipulation"


def test_faceswap_settings_defaults():
    s = get_settings()
    assert isinstance(s.faceswap_enabled, bool)
    assert isinstance(s.faceswap_model_id, str) and s.faceswap_model_id
    assert 0.0 < s.faceswap_threshold < 1.0


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


async def test_classifier_failure_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(fs, "get_settings", lambda: _settings())
    monkeypatch.setattr(fs, "_MODEL", None)
    monkeypatch.setattr(fs, "_load_face_detector", lambda: (lambda pil: [Image.new("RGB", (32, 32))]))
    def classifier_boom(crop):
        raise RuntimeError("classify boom")
    monkeypatch.setattr(fs, "_load_classifier", lambda: classifier_boom)
    result = await fs.FaceSwapSignal().analyze(_img())
    assert result.status.value == "error"
    assert "classify boom" in (result.error or "")
