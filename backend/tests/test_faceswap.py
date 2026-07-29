from app.schemas import SignalClass
from app.config import get_settings


def test_manipulation_signal_class_exists():
    assert SignalClass.manipulation.value == "manipulation"


def test_faceswap_settings_defaults():
    s = get_settings()
    assert isinstance(s.faceswap_enabled, bool)
    assert isinstance(s.faceswap_model_id, str) and s.faceswap_model_id
    assert 0.0 < s.faceswap_threshold < 1.0
