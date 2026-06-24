from __future__ import annotations

from io import BytesIO
from time import perf_counter

from PIL import ExifTags, Image

from app.schemas import ExifResult, ExifSignal
from app.services.base import AnalysisContext


IMPORTANT_EXIF_FIELDS = (
    "Make",
    "Model",
    "DateTime",
    "DateTimeOriginal",
    "Software",
    "GPSInfo",
    "Orientation",
)

SUSPICIOUS_SOFTWARE_MARKERS = (
    "photoshop",
    "gimp",
    "comfyui",
    "stable diffusion",
    "midjourney",
    "openai",
    "canva",
    "image generator",
)


class ExifModule:
    name = "exif"

    async def analyze(self, context: AnalysisContext) -> ExifResult:
        started = perf_counter()
        try:
            image = Image.open(BytesIO(context.media_bytes))
            raw_exif = image.getexif()
        except Exception as exc:
            return ExifResult(
                status="error",
                summary="Failed to parse image metadata.",
                warnings=[],
                errors=[str(exc)],
                latency_ms=int((perf_counter() - started) * 1000),
                metadata_present=False,
                gps_present=False,
                anomaly_flags=["metadata_parse_error"],
                extracted_fields=[],
                raw_field_count=0,
            )

        named_fields: dict[str, str] = {}
        for tag_id, value in raw_exif.items():
            label = ExifTags.TAGS.get(tag_id, str(tag_id))
            named_fields[label] = str(value)

        extracted_fields = [
            ExifSignal(field=field, value=named_fields[field])
            for field in IMPORTANT_EXIF_FIELDS
            if field in named_fields
        ]
        anomaly_flags: list[str] = []
        warnings: list[str] = []

        if "Make" not in named_fields:
            anomaly_flags.append("missing_camera_make")
        if "Model" not in named_fields:
            anomaly_flags.append("missing_camera_model")
        if "DateTimeOriginal" not in named_fields:
            anomaly_flags.append("missing_capture_timestamp")

        software_tag = named_fields.get("Software")
        if software_tag and any(
            marker in software_tag.lower() for marker in SUSPICIOUS_SOFTWARE_MARKERS
        ):
            anomaly_flags.append("suspicious_software_signature")
            warnings.append(f"Software tag suggests editing or generation: {software_tag}")

        metadata_present = bool(named_fields)
        gps_present = "GPSInfo" in named_fields

        if not metadata_present:
            status = "warning"
            summary = "No EXIF metadata was present in the uploaded file."
        elif anomaly_flags:
            status = "warning"
            summary = "Metadata extracted with one or more anomaly flags."
        else:
            status = "ok"
            summary = "Metadata extracted successfully with no immediate anomalies."

        return ExifResult(
            status=status,
            summary=summary,
            warnings=warnings,
            errors=[],
            latency_ms=int((perf_counter() - started) * 1000),
            metadata_present=metadata_present,
            gps_present=gps_present,
            software_tag=software_tag,
            anomaly_flags=anomaly_flags,
            extracted_fields=extracted_fields,
            raw_field_count=len(named_fields),
        )
