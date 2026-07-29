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
