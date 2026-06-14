"""THE CONTRACT. Every signal module produces a SignalResult; /scan returns a
ScanResponse. Do not change field names without updating all signal modules and
the frontend, which consumes this envelope.

Two-axis design (see methodology): a signal answers EITHER "is this
AI-generated?" (ai_score) OR "was this real image manipulated/edited?"
(manipulation_score). Most signals fill one axis and leave the other None.
The triangulation engine fuses each axis separately so an edited-but-real photo
is not misread as fake.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SignalClass(str, Enum):
    detector = "detector"        # learned commercial model
    forensic = "forensic"        # pixel/metadata forensics
    provenance = "provenance"    # cryptographic / watermark provenance
    context = "context"          # external context (e.g. reverse search)


class SignalStatus(str, Enum):
    ok = "ok"                    # ran, produced a usable result
    error = "error"              # ran, failed (see .error)
    skipped = "skipped"          # not applicable to this image
    unavailable = "unavailable"  # not configured (missing key) — not an error


class SignalResult(BaseModel):
    """One signal's contribution to the envelope. The engine reads these."""

    name: str                                  # stable id, e.g. "sightengine"
    signal_class: SignalClass
    status: SignalStatus
    # Axis 1 — likelihood the image is AI-generated, in [0, 1]. None if N/A.
    ai_score: float | None = Field(default=None, ge=0.0, le=1.0)
    # Axis 2 — likelihood a real image was manipulated/edited, in [0, 1].
    manipulation_score: float | None = Field(default=None, ge=0.0, le=1.0)
    # This signal's self-reported confidence in its own output, in [0, 1].
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = None
    notes: list[str] = Field(default_factory=list)   # human-readable findings
    error: str | None = None
    raw: dict | None = None                    # raw upstream payload, for audit


class Aggregate(BaseModel):
    """Filled by the triangulation engine (Layer 2). Placeholder until built."""

    verdict: str | None = None                 # "Likely AI" | "Likely Authentic" | "Inconclusive"
    ai_score: float | None = None              # fused Axis-1 score
    manipulation_score: float | None = None    # fused Axis-2 score
    confidence: float | None = None
    disagreement: float | None = None          # spread across signals
    contributions: dict[str, float] = Field(default_factory=dict)


class ScanResponse(BaseModel):
    image_sha256: str
    filename: str | None = None
    signals: list[SignalResult]
    aggregate: Aggregate = Field(default_factory=Aggregate)
    explanation: str | None = None             # filled by Layer 3
