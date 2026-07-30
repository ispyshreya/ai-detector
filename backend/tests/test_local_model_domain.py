import asyncio
from io import BytesIO

from PIL import Image

from app.schemas import SignalStatus
from app.signals.base import ImageInput
from app.signals.local_model import MAX_PATCHES, LocalModelSignal


def _jpeg(width: int, height: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (width, height), "white").save(output, format="JPEG")
    return output.getvalue()


def test_phone_sized_image_is_patched_and_still_scored():
    """Real-world photos are no longer skipped or shrunk whole: they're diced
    into overlapping native-resolution 32x32 patches and scored, but with
    confidence discounted for the unvalidated heuristic (see local_model.py)."""
    image = ImageInput(
        data=_jpeg(2160, 3840),
        filename="iphone.jpg",
        content_type="image/jpeg",
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.ok
    assert result.ai_score is not None
    assert 0.0 <= result.ai_score <= 1.0
    assert result.raw["was_tiled"] is True
    assert result.raw["original_size"] == [2160, 3840]
    assert result.raw["training_native_size"] == [32, 32]
    # Whole image patched, not shrunk -- the sampled patch count is capped.
    assert 0 < result.raw["tile_count"] <= MAX_PATCHES
    assert any("patch" in note for note in result.notes)
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
    assert result.raw["was_tiled"] is False
    assert result.raw["tile_count"] == 1
    assert not any("unvalidated domain-matching heuristic" in note for note in result.notes)
