"""Reverse image search context signal (Layer: context).

This signal does NOT score AI-generation. Its job is CONTEXT: finding where and
when an image first appeared on the web. The earliest known appearance, matching
pages, and source attributions help a human reason about an image's origin — but
they are not, by themselves, a verdict on synthesis. So ai_score and
manipulation_score are always left None; the value lives entirely in the notes.

IMPORTANT LIMITATION (honest by design): SerpAPI's Google Lens / Google Reverse
Image engines take an image URL, not raw uploaded bytes. We only have bytes here,
so we cannot run the search yet. Rather than fake a result, we return
status=skipped with a clear explanation and leave request-building code ready for
when images are hosted (see the TODO below).
"""

from __future__ import annotations

import time

from app.config import get_settings
from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

# SerpAPI search endpoint. The "google_lens" engine gives the richest visual-match
# context; "google_reverse_image" is the classic alternative. Both need image_url.
_ENDPOINT = "https://serpapi.com/search"
_ENGINE = "google_lens"
# Per-signal timeout; the runner caps signals at ~6s (see config), so stay under it.
_TIMEOUT_SECONDS = 6.0


class ReverseSearchSignal(Signal):
    name = "reverse_search"
    signal_class = SignalClass.context

    def available(self) -> bool:
        """Configured only when a SerpAPI key is present."""
        return bool(get_settings().serpapi_key)

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()
        settings = get_settings()

        # We cannot proceed without a hosted URL for the image. SerpAPI's reverse
        # image / Lens engines accept `image_url=`, never raw multipart bytes. We
        # only receive bytes in ImageInput, so we honestly skip rather than guess.
        #
        # TODO: To finish this signal, host the bytes somewhere SerpAPI can fetch:
        #   1. Upload `image.data` to a short-lived public store (e.g. S3 presigned
        #      PUT, a temp bucket, or an image host) and get back a public URL.
        #   2. Pass that URL as `image_url` in the params below and send the request.
        #   3. Parse `visual_matches` / `image_results` from the JSON: extract the
        #      title, source/link, and any dates to report the earliest known
        #      appearance and notable hosting sites in `notes`.
        #   4. Clean up / let the temp URL expire after the request.
        #
        # The request is fully structured below and ready to fire once `image_url`
        # is real. It is intentionally not sent because `image_url` is a placeholder.
        if settings.serpapi_key:
            image_url = None  # <- set this to the hosted URL once uploading is wired.
            params = {
                "engine": _ENGINE,            # or "google_reverse_image"
                "api_key": settings.serpapi_key,
                "url": image_url,             # google_lens uses `url`; reverse_image uses `image_url`
                # "image_url": image_url,     # use this key instead for google_reverse_image
                "hl": "en",
            }

            # Ready-to-enable call (kept commented until image_url is a real URL):
            #
            # import httpx
            # try:
            #     async with httpx.AsyncClient(timeout=_TIMEOUT_SECONDS) as client:
            #         response = await client.get(_ENDPOINT, params=params)
            #     response.raise_for_status()
            #     payload = response.json()
            # except httpx.HTTPError as exc:
            #     return self._result(
            #         status=SignalStatus.error, started=started,
            #         error=f"reverse search request failed: {exc}",
            #     )
            # matches = payload.get("visual_matches") or payload.get("image_results") or []
            # notes = [
            #     f"{m.get('title', 'match')} — {m.get('source') or m.get('link')}"
            #     for m in matches[:5]
            # ] or ["No web matches found."]
            # return self._result(
            #     status=SignalStatus.ok, started=started, notes=notes, raw=payload,
            # )
            del params  # silence "assigned but unused" until the call is enabled.

        return self._result(
            status=SignalStatus.skipped,
            started=started,
            notes=[
                "Reverse image search not run: this is a CONTEXT signal (earliest "
                "known web appearance), and SerpAPI's Google Lens / reverse-image "
                "engines require a hosted image URL, not the raw uploaded bytes we "
                "have here. Host the image and pass its URL to enable this lookup. "
                "Note that web matches inform origin/context only — they do not "
                "directly indicate AI generation."
            ],
        )

    def _result(
        self,
        *,
        status: SignalStatus,
        started: float,
        notes: list[str] | None = None,
        error: str | None = None,
        raw: dict | None = None,
    ) -> SignalResult:
        """Single builder. ai_score/manipulation_score/confidence stay None: this
        signal contributes context, not a synthesis verdict."""
        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=status,
            ai_score=None,
            manipulation_score=None,
            confidence=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=notes or [],
            error=error,
            raw=raw,
        )
