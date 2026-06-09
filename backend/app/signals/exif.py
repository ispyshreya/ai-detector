"""EXIF / metadata forensic signal.

This is a FORENSIC HEURISTIC, not a learned detector. We read the image's
embedded metadata (EXIF/XMP tags via `exifread`) and reason about its
*anomalies*. The intuition: a photo straight off a camera carries a rich,
self-consistent metadata trail (Make, Model, capture timestamp, exposure
settings); AI-generated or scrubbed images typically arrive with that trail
missing — or, occasionally, with a tell-tale Software signature naming the
generator that produced them.

None of this is dispositive (legitimate images are stripped of EXIF all the
time by messaging apps, screenshots, and social platforms), so this signal is
deliberately LOW confidence and only nudges Axis-1 (ai_score). It never touches
the manipulation axis — absence of metadata says nothing about pixel edits.
"""

from __future__ import annotations

import io
import time

import exifread

from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

# Software-field substrings (lowercased) that strongly imply synthetic origin.
# These name image *generators* — their presence is a fairly direct admission.
_GENERATOR_SIGNATURES = (
    "stable diffusion",
    "midjourney",
    "dall",            # DALL-E / DALL·E
    "adobe firefly",
    "firefly",
    "imagen",
    "comfyui",
    "automatic1111",
    "invokeai",
    "flux",
)

# Software-field substrings for generic editors. Weaker: lots of authentic
# photos pass through these. Presence just means "was opened in an editor",
# which mildly raises suspicion but is far from proof.
_EDITOR_SIGNATURES = (
    "photoshop",
    "lightroom",
    "gimp",
    "affinity",
    "pixelmator",
    "snapseed",
)

# EXIF tags (as exifread keys) we treat as "camera authenticity" evidence.
_CAMERA_TAGS = {
    "make": "Image Make",
    "model": "Image Model",
    "datetime_original": "EXIF DateTimeOriginal",
    "exposure_time": "EXIF ExposureTime",
    "iso": "EXIF ISOSpeedRatings",
    "fnumber": "EXIF FNumber",
}


class ExifSignal(Signal):
    name = "exif"
    signal_class = SignalClass.forensic

    def available(self) -> bool:
        # Purely local, no API key or network. Always runnable.
        return True

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()

        # exifread is synchronous but very fast (it reads a small header region,
        # not the whole pixel payload), so calling it inline in async is fine.
        try:
            tags = exifread.process_file(
                io.BytesIO(image.data),
                details=False,   # skip MakerNote/thumbnail blobs — we don't use them
            )
        except Exception as exc:  # noqa: BLE001 - malformed/unsupported input
            return SignalResult(
                name=self.name,
                signal_class=self.signal_class,
                status=SignalStatus.skipped,
                notes=[f"Could not parse image metadata: {exc}"],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # exifread returns {} for valid images that simply carry no EXIF (e.g.
        # PNGs, scrubbed JPEGs). That's a meaningful finding, not a parse error,
        # so we proceed rather than skip. We only skip if the bytes clearly
        # aren't a decodable image — heuristically, an empty/tiny payload.
        if not image.data or len(image.data) < 16:
            return SignalResult(
                name=self.name,
                signal_class=self.signal_class,
                status=SignalStatus.skipped,
                notes=["Empty or truncated payload; nothing to analyze."],
                latency_ms=(time.perf_counter() - started) * 1000,
            )

        # --- Extract the fields we reason about -----------------------------
        present: dict[str, str] = {}
        for key, exif_key in _CAMERA_TAGS.items():
            if exif_key in tags:
                present[key] = str(tags[exif_key]).strip()

        software = str(tags["Image Software"]).strip() if "Image Software" in tags else None
        has_gps = any(k.startswith("GPS") for k in tags.keys())

        has_make_model = "make" in present and "model" in present

        # --- Heuristic scoring ----------------------------------------------
        # Start from a neutral-ish prior and move it based on anomalies. All
        # constants below are hand-tuned heuristics, NOT calibrated probabilities.
        notes: list[str] = [
            "Heuristic signal: scores derive from metadata anomalies, not a "
            "learned model. Stripped EXIF is common in legitimate images "
            "(messaging apps, screenshots, social media), so treat as weak."
        ]

        software_is_generator = False
        software_is_editor = False
        if software:
            low = software.lower()
            software_is_generator = any(sig in low for sig in _GENERATOR_SIGNATURES)
            software_is_editor = any(sig in low for sig in _EDITOR_SIGNATURES)

        if not tags:
            # No metadata of any kind: strongest available anomaly here.
            ai_score = 0.7
            notes.append("No EXIF/metadata present at all — consistent with a "
                         "stripped or synthetically generated image.")
        elif not has_make_model:
            # Some tags, but no camera identity: still suspicious.
            ai_score = 0.65
            missing = [k for k in ("make", "model") if k not in present]
            notes.append(f"No camera identity present (missing: {', '.join(missing)}).")
        else:
            # Intact camera identity: this looks like a real capture. Reward it.
            ai_score = 0.2
            notes.append(
                f"Camera identity present: Make={present.get('make')!r}, "
                f"Model={present.get('model')!r}."
            )
            captured_extras = [k for k in
                               ("datetime_original", "exposure_time", "iso", "fnumber")
                               if k in present]
            if captured_extras:
                notes.append("Capture settings present: " + ", ".join(captured_extras) + ".")
            else:
                # Make/Model but none of the capture telemetry — mildly odd.
                ai_score += 0.1
                notes.append("Camera make/model present but no exposure/ISO/timestamp "
                             "telemetry — partially stripped.")

        # Software signature adjustments (applied on top of the structural score).
        if software_is_generator:
            ai_score = max(ai_score, 0.85)
            notes.append(f"Software field names an image generator: {software!r}.")
        elif software_is_editor:
            ai_score = min(1.0, ai_score + 0.1)
            notes.append(f"Software field names a generic editor: {software!r} "
                         "(weak indicator; routine for authentic photos).")
        elif software:
            notes.append(f"Software field present: {software!r} (unrecognized).")

        if has_gps:
            # GPS presence leans authentic (cameras/phones geotag captures), but
            # is easily spoofed/stripped, so it's only a soft hint. We do NOT
            # record the coordinates themselves — privacy.
            notes.append("GPS metadata present (coordinates withheld for privacy).")

        ai_score = max(0.0, min(1.0, ai_score))

        # --- Compact, stringified raw payload for audit ---------------------
        raw = {
            "fields_present": present,
            "software": software,
            "gps_present": has_gps,
            "exif_tag_count": len(tags),
            "software_is_generator": software_is_generator,
            "software_is_editor": software_is_editor,
        }

        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=SignalStatus.ok,
            ai_score=ai_score,
            manipulation_score=None,   # this signal does not assess pixel edits
            # Low by design: metadata is trivially stripped/forged, so even a
            # confident-looking anomaly is only weakly informative.
            confidence=0.4,
            notes=notes,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
