from __future__ import annotations

from io import BytesIO
from time import perf_counter

from PIL import Image, ImageChops, ImageEnhance, ImageStat

from app.schemas import ElaRegion, ElaResult
from app.services.base import AnalysisContext


class ElaModule:
    name = "ela"

    async def analyze(self, context: AnalysisContext) -> ElaResult:
        started = perf_counter()
        try:
            original = Image.open(BytesIO(context.media_bytes)).convert("RGB")
        except Exception as exc:
            return ElaResult(
                status="error",
                summary="Failed to decode image for ELA analysis.",
                warnings=[],
                errors=[str(exc)],
                latency_ms=int((perf_counter() - started) * 1000),
                heatmap_url=None,
                max_intensity=0.0,
                mean_intensity=0.0,
                suspicious_regions=[],
            )

        recompressed_buffer = BytesIO()
        original.save(recompressed_buffer, "JPEG", quality=90)
        recompressed_buffer.seek(0)
        recompressed = Image.open(recompressed_buffer).convert("RGB")

        difference = ImageChops.difference(original, recompressed)
        max_diff = max(channel[1] for channel in difference.getextrema())
        scale = 1 if max_diff == 0 else 255.0 / max_diff
        enhanced = ImageEnhance.Brightness(difference).enhance(scale)
        grayscale = enhanced.convert("L")
        stats = ImageStat.Stat(grayscale)
        mean_intensity = float(stats.mean[0]) / 255.0
        max_intensity = float(max_diff) / 255.0

        context.artifact_dir.mkdir(parents=True, exist_ok=True)
        heatmap_filename = "ela_heatmap.png"
        heatmap_path = context.artifact_dir / heatmap_filename
        enhanced.save(heatmap_path, "PNG")

        region = self._find_hot_region(grayscale)
        suspicious_regions = [region] if region else []

        if max_intensity > 0.7 or mean_intensity > 0.2:
            status = "warning"
            summary = "ELA found elevated compression differences worth reviewing."
        else:
            status = "ok"
            summary = "ELA completed with relatively low compression-difference intensity."

        return ElaResult(
            status=status,
            summary=summary,
            warnings=[],
            errors=[],
            latency_ms=int((perf_counter() - started) * 1000),
            heatmap_url=f"{context.artifact_url_prefix}/{context.analysis_id}/{heatmap_filename}",
            max_intensity=round(max_intensity, 4),
            mean_intensity=round(mean_intensity, 4),
            suspicious_regions=suspicious_regions,
        )

    def _find_hot_region(self, grayscale: Image.Image) -> ElaRegion | None:
        width, height = grayscale.size
        threshold = 200
        hot_pixels = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if grayscale.getpixel((x, y)) >= threshold
        ]
        if not hot_pixels:
            return None

        xs = [x for x, _ in hot_pixels]
        ys = [y for _, y in hot_pixels]
        score = len(hot_pixels) / float(width * height)
        return ElaRegion(
            label="largest_hot_region",
            left=min(xs),
            top=min(ys),
            right=max(xs),
            bottom=max(ys),
            score=round(score, 4),
        )
