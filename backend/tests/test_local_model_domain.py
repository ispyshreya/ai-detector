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


def test_phone_sized_image_is_skipped_before_model_load():
    image = ImageInput(
        data=_jpeg(2160, 3840),
        filename="iphone.jpg",
        content_type="image/jpeg",
    )
    result = asyncio.run(LocalModelSignal().analyze(image))

    assert result.status == SignalStatus.skipped
    assert result.ai_score is None
    assert result.raw["reason"] == "out_of_training_domain"
    assert result.raw["training_native_size"] == [32, 32]
    assert result.raw["input_size"] == [2160, 3840]
