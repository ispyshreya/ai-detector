"""C2PA Content Credentials provenance signal (Layer: provenance).

C2PA (Coalition for Content Provenance and Authenticity) embeds a cryptographically
signed manifest in an asset describing how it was created and edited — including
whether a generative-AI tool produced or modified it. When present and valid, this
is the STRONGEST kind of provenance evidence: it is signed, not inferred.

Two important asymmetries shape how we score:
  * A valid manifest that ASSERTS AI generation -> high ai_score (cryptographic claim).
  * A valid manifest from a normal camera/editor with NO AI assertion -> low ai_score.
  * NO manifest at all -> we SKIP. Most images on the web have no C2PA data, so the
    absence of provenance is NOT evidence of fakery — it tells us nothing either way.

The `c2pa` Python package is OPTIONAL. If it isn't installed we degrade to
status=unavailable (not an error) so the rest of the pipeline keeps working.
"""

from __future__ import annotations

import json
import time

from app.schemas import SignalClass, SignalResult, SignalStatus
from app.signals.base import ImageInput, Signal

# The c2pa SDK is optional — guard the import so a missing dependency never breaks
# module import (the registry constructs this signal eagerly).
try:
    import c2pa  # type: ignore
except ImportError:  # pragma: no cover - depends on optional install
    c2pa = None  # type: ignore[assignment]

# Scores we emit for the two decisive manifest outcomes. Provenance is cryptographic,
# so these sit near the extremes rather than hedging around 0.5.
_AI_ASSERTED_SCORE = 0.9
_NO_AI_SCORE = 0.1
# Confidence when a signed manifest is found — high, because the claim is signed.
_MANIFEST_CONFIDENCE = 0.9

# Substrings that, when found in an action/assertion label, indicate a generative-AI
# step. C2PA's vocabulary is still evolving, so we match defensively on keywords.
_AI_LABEL_HINTS = (
    "ai",
    "genai",
    "generative",
    "trainedalgorithmicmedia",  # c2pa.actions digitalSourceType for GenAI
    "compositewithtrainedalgorithmicmedia",
    "syntheticmedia",
)


class C2paSignal(Signal):
    name = "c2pa"
    signal_class = SignalClass.provenance

    def available(self) -> bool:
        """Always available as a signal slot. Whether the c2pa SDK is actually
        installed is reported per-scan via status=unavailable, so the engine can
        tell "not installed" apart from "no manifest in this image"."""
        return True

    async def analyze(self, image: ImageInput) -> SignalResult:
        started = time.perf_counter()

        # Optional dependency not installed -> unavailable, NOT an error.
        if c2pa is None:
            return self._result(
                status=SignalStatus.unavailable,
                started=started,
                notes=[
                    "C2PA provenance check skipped: the optional `c2pa` package "
                    "is not installed. Install it to read Content Credentials."
                ],
            )

        # Read the manifest store from the raw bytes. The c2pa API surface has shifted
        # across releases, so probe a few known entrypoints and degrade gracefully.
        try:
            manifest_json = _read_manifest_json(image.data, image.content_type)
        except Exception as exc:  # noqa: BLE001 - never raise for expected failures
            # A read failure usually just means "no/invalid manifest" rather than a
            # real fault, but surface it in .error for auditing.
            return self._result(
                status=SignalStatus.error,
                started=started,
                error=f"c2pa read failed: {exc}",
            )

        if not manifest_json:
            # No manifest present. This is the common case and is NOT evidence either
            # way — say so explicitly so downstream layers don't treat it as a signal.
            return self._result(
                status=SignalStatus.skipped,
                started=started,
                notes=[
                    "No C2PA manifest present. Absence of provenance is not "
                    "evidence of AI generation or tampering — most images carry "
                    "no Content Credentials."
                ],
            )

        # We have a manifest. Parse it defensively into a summary.
        try:
            summary = _summarize_manifest(manifest_json)
        except Exception as exc:  # noqa: BLE001
            return self._result(
                status=SignalStatus.error,
                started=started,
                error=f"c2pa parse failed: {exc}",
                raw={"manifest_json": _maybe_json(manifest_json)},
            )

        asserts_ai = summary["asserts_ai"]
        issuer = summary["issuer"] or "unknown issuer"
        notes: list[str] = []

        if asserts_ai:
            ai_score = _AI_ASSERTED_SCORE
            notes.append(
                f"Signed C2PA manifest asserts AI generation/editing "
                f"(issuer: {issuer})."
            )
        else:
            ai_score = _NO_AI_SCORE
            notes.append(
                f"Signed C2PA manifest present with no AI-generation assertion "
                f"(issuer: {issuer})."
            )
        if summary["actions"]:
            notes.append("Recorded actions: " + ", ".join(summary["actions"][:6]))

        return self._result(
            status=SignalStatus.ok,
            started=started,
            ai_score=ai_score,
            confidence=_MANIFEST_CONFIDENCE,
            notes=notes,
            raw={"manifest": summary, "manifest_json": _maybe_json(manifest_json)},
        )

    def _result(
        self,
        *,
        status: SignalStatus,
        started: float,
        ai_score: float | None = None,
        confidence: float | None = None,
        notes: list[str] | None = None,
        error: str | None = None,
        raw: dict | None = None,
    ) -> SignalResult:
        """Single builder so every exit path reports latency consistently."""
        return SignalResult(
            name=self.name,
            signal_class=self.signal_class,
            status=status,
            ai_score=ai_score,
            confidence=confidence,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            notes=notes or [],
            error=error,
            raw=raw,
        )


def _read_manifest_json(data: bytes, content_type: str | None) -> str | None:
    """Return the manifest store as a JSON string, or None if no manifest is found.

    The c2pa SDK has used several shapes over its releases. We try the modern
    `Reader` first, then fall back to older module-level helpers. Any genuine read
    error propagates to the caller; a "no manifest" condition returns None.
    """
    fmt = _format_from_content_type(content_type)

    # Modern API (c2pa-python >= 0.5): Reader(format, stream) -> .json().
    reader_cls = getattr(c2pa, "Reader", None)
    if reader_cls is not None:
        import io

        try:
            reader = reader_cls(fmt, io.BytesIO(data))  # type: ignore[call-arg]
        except TypeError:
            # Some versions take a single stream/bytes argument.
            reader = reader_cls(io.BytesIO(data))  # type: ignore[call-arg]
        try:
            return reader.json()  # type: ignore[no-any-return]
        finally:
            close = getattr(reader, "close", None)
            if callable(close):
                close()

    # Older API: read_ingredient_file / read_file or a bytes helper.
    for fn_name in ("read_manifest_from_bytes", "manifest_store_from_bytes"):
        fn = getattr(c2pa, fn_name, None)
        if callable(fn):
            return fn(data)  # type: ignore[no-any-return]

    # No recognizable entrypoint — treat as "cannot read", which the caller maps to
    # status=error so the limitation is visible rather than silently swallowed.
    raise RuntimeError("installed c2pa package exposes no known reader API")


def _format_from_content_type(content_type: str | None) -> str:
    """Map a MIME type to the short format string the c2pa SDK expects."""
    if not content_type:
        return "image/jpeg"
    return content_type


def _summarize_manifest(manifest_json: str) -> dict:
    """Distill the manifest store JSON into the few fields we score on.

    Defensive throughout: the structure varies by signer/version, so we walk it
    with .get() and tolerate missing keys.
    """
    store = json.loads(manifest_json)

    active_label = store.get("active_manifest")
    manifests = store.get("manifests", {}) or {}
    active = manifests.get(active_label) if active_label else None
    if active is None and manifests:
        # Fall back to the first manifest if no active pointer is set.
        active = next(iter(manifests.values()))
    active = active or {}

    # Issuer: prefer the signature_info issuer, fall back to claim_generator.
    sig_info = active.get("signature_info", {}) or {}
    issuer = sig_info.get("issuer") or active.get("claim_generator")

    # Collect action labels and any digitalSourceType hints from assertions.
    actions: list[str] = []
    asserts_ai = False
    for assertion in active.get("assertions", []) or []:
        label = str(assertion.get("label", ""))
        data = assertion.get("data", {}) or {}
        if "action" in label.lower():
            for action in data.get("actions", []) or []:
                act_label = str(action.get("action", ""))
                if act_label:
                    actions.append(act_label)
                source_type = str(action.get("digitalSourceType", ""))
                if _looks_like_ai(act_label) or _looks_like_ai(source_type):
                    asserts_ai = True
        # Some manifests carry a top-level AI/training assertion label directly.
        if _looks_like_ai(label):
            asserts_ai = True

    return {
        "active_label": active_label,
        "issuer": issuer,
        "actions": actions,
        "asserts_ai": asserts_ai,
    }


def _looks_like_ai(text: str) -> bool:
    """True if a label/source-type string hints at generative-AI involvement."""
    lowered = text.lower()
    return any(hint in lowered for hint in _AI_LABEL_HINTS)


def _maybe_json(manifest_json: str) -> object:
    """Parse the manifest JSON for the raw audit payload, or keep the raw string."""
    try:
        return json.loads(manifest_json)
    except ValueError:
        return manifest_json
