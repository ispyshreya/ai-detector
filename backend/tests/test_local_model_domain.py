import asyncio
from io import BytesIO

from PIL import Image

from app.schemas import SignalStatus
from app.signals.base import ImageInput
from app.signals.local_model import LocalModelSignal


def _jpeg(width: int, height: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="JPEG")
    return output.getvalue()


def test_phone_sized_image_is_downsampled_and_still_scored():
    """Real-world photos are no longer skipped: they're downsampled to the
    checkpoint's 32x32 CIFAKE training resolution and scored, but with
    confidence discounted for the unvalidated resize (see local_model.py)."""
    image = ImageInput(
        data=_jpeg(2160, 3840),
        filename="iphone.jpg",
        content_type="image/jpeg",
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert result.ai_score is not None
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["was_resized"] is True
    assert result.raw["original_size"] == [2160, 3840]
    assert result.raw["training_native_size"] == [32, 32]
    assert any("downsampled" in note for note in result.notes)
    # Confidence factor is 0.7x the raw distance from 0.5, whose max is 1.0.
    assert result.confidence <= 0.7


def test_native_resolution_image_is_scored_without_confidence_discount():
    image = ImageInput(
        data=_jpeg(32, 32),
        filename="native.jpg",
        content_type="image/jpeg",
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert result.raw["was_resized"] is False
    assert not any("downsampled" in note for note in result.notes)
