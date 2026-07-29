"""Layer 2 — the triangulation engine.

Fuses independent SignalResults into a single Aggregate: one verdict, one
calibrated confidence, disagreement across checks, and a plain-language trail
of why. This is the piece VEIL_CLIENT_PROMPT.md means by "evaluate the
reliability and relevance of every result before combining them" — it is
deliberately NOT a plain average of ai_score across signals.

Design rules, each tied to a client requirement:
  * Signals are weighted by evidence class (provenance > detector > forensic;
    context never scores) and by their own self-reported confidence, so a
    signed C2PA manifest can outweigh a noisy ELA heuristic. ("whether a
    signal is direct evidence, weak supporting evidence, or merely
    contextual")
  * unavailable/error/skipped signals contribute zero weight, never a
    penalty or a neutral 0.5. ("An unavailable check must contribute no
    positive or negative evidence.")
  * Real disagreement between direct-evidence signals forces Inconclusive
    instead of being averaged into a misleading midpoint. ("The UI must not
    represent the midpoint between opposing scores as '50% fake.'")
  * Too little total evidence mass also forces Inconclusive, even if the one
    available signal is very confident. ("A local detector returning 100% AI
    ... should trigger disagreement handling, not an automatic verdict.")
  * manipulation_score (Axis 2) can escalate the headline verdict without
    being confused for AI-generation evidence (Axis 1) itself.
"""

from __future__ import annotations

from app.schemas import Aggregate, SignalClass, SignalResult, SignalStatus

# How much an entire signal_class is trusted as evidence, before its own
# self-reported confidence is applied. Provenance (a signed C2PA manifest) is
# the strongest evidence available: it is cryptographically asserted, not
# inferred. Forensic heuristics (ELA/EXIF) are explicitly weak corroboration
# by their own module docstrings. Context signals (reverse search) never
# populate a score at all, so their weight is moot.
_CLASS_WEIGHT: dict[SignalClass, float] = {
    SignalClass.provenance: 1.0,
    SignalClass.detector: 0.8,
    SignalClass.forensic: 0.35,
    SignalClass.context: 0.0,
}

# Per-signal-name discount on top of the class weight. The in-house `local`
# checkpoint is only validated on 32x32 CIFAKE crops (it skips itself outside
# that domain — see local_model.py) and has not been evaluated on the
# representative datasets the client prompt's Quality Standard requires. A
# single confident `local` result alone must not be able to carry a verdict,
# so it is discounted relative to commercial detectors with broader (if still
# imperfect) validation.
_RELIABILITY_OVERRIDE: dict[str, float] = {
    "local": 0.6,
}

_DEFAULT_SIGNAL_CONFIDENCE = 0.5  # used only if a signal omits self-reported confidence

# Minimum total weighted evidence ("mass") required before committing to a
# risk band. Below this we do not have enough reliable evidence to say
# anything directional -- Inconclusive, not a guess dressed up as a score.
_MIN_RELIABLE_MASS = 0.5

# Spread (max - min) among ok, direct-evidence (provenance/detector) ai_scores
# at/above which we treat it as real disagreement, not noise.
_DISAGREEMENT_THRESHOLD = 0.45

# ai_score band edges -> verdict label, applied only once there is enough
# agreeing evidence to commit to one.
_BANDS: tuple[tuple[float, str], ...] = (
    (0.15, "Likely Authentic"),
    (0.40, "Low Risk"),
    (0.65, "Medium Risk"),
    (1.01, "High Risk"),
)

# A high, reasonably-attested manipulation_score (Axis 2) escalates the
# headline verdict even when Axis 1 (AI-generation) alone looked clean --
# principle #5 says don't confuse editing with full AI generation, but an
# edited real photo is still a warning worth surfacing.
_MANIPULATION_ESCALATE = 0.65
_MANIPULATION_MIN_MASS = 0.4


def _is_unvalidated_resize(signal: SignalResult) -> bool:
    """True for the `local` signal when its input had to be downsampled to
    the checkpoint's 32x32 CIFAKE training resolution (see
    signals/local_model.py). That resize is an unvalidated heuristic that has
    been observed in production to disagree sharply with validated detectors
    on ordinary real photos, dragging otherwise-confident real-world scans
    into Inconclusive. It still appears in the per-signal breakdown for
    transparency, but must not move the fused verdict or count toward
    disagreement until it's actually been evaluated (Quality Standard,
    VEIL_CLIENT_PROMPT.md)."""
    return (
        signal.name == "local"
        and isinstance(signal.raw, dict)
        and signal.raw.get("was_resized") is True
    )


def _weight(signal: SignalResult) -> float:
    if _is_unvalidated_resize(signal):
        return 0.0
    class_weight = _CLASS_WEIGHT.get(signal.signal_class, 0.0)
    override = _RELIABILITY_OVERRIDE.get(signal.name, 1.0)
    confidence = signal.confidence if signal.confidence is not None else _DEFAULT_SIGNAL_CONFIDENCE
    return class_weight * override * confidence


def _fuse_axis(
    signals: list[SignalResult], axis: str
) -> tuple[float | None, float, dict[str, float]]:
    """Weighted-average one score axis ("ai_score" or "manipulation_score").

    Only status=ok signals with a non-None value on this axis contribute.
    Returns (fused_score_or_None, total_mass, {signal_name: weight}).
    """
    total_weight = 0.0
    weighted_sum = 0.0
    contributions: dict[str, float] = {}
    for signal in signals:
        if signal.status != SignalStatus.ok:
            continue
        value = getattr(signal, axis)
        if value is None:
            continue
        weight = _weight(signal)
        if weight <= 0:
            continue
        weighted_sum += weight * value
        total_weight += weight
        contributions[signal.name] = round(weight, 4)
    fused = weighted_sum / total_weight if total_weight > 0 else None
    return fused, total_weight, contributions


def _band(score: float) -> str:
    for edge, label in _BANDS:
        if score < edge:
            return label
    return _BANDS[-1][1]


def _primary_ai_signals(signals: list[SignalResult]) -> list[SignalResult]:
    """Direct-evidence signals whose mutual disagreement actually matters —
    excludes weak forensic corroboration and non-scoring context signals."""
    return [
        s
        for s in signals
        if s.status == SignalStatus.ok
        and s.ai_score is not None
        and s.signal_class in (SignalClass.provenance, SignalClass.detector)
        and not _is_unvalidated_resize(s)
    ]


def triangulate(signals: list[SignalResult]) -> Aggregate:
    """Fuse this scan's SignalResults into one Aggregate verdict."""
    ai_score, ai_mass, ai_contrib = _fuse_axis(signals, "ai_score")
    manip_score, manip_mass, manip_contrib = _fuse_axis(signals, "manipulation_score")

    contributions: dict[str, float] = dict(ai_contrib)
    for name, weight in manip_contrib.items():
        contributions[name] = max(contributions.get(name, 0.0), weight)

    reasons: list[str] = []
    for signal in signals:
        if signal.status == SignalStatus.error:
            reasons.append(f"{signal.name} failed and contributed no evidence ({signal.error}).")
        elif signal.status == SignalStatus.unavailable:
            reasons.append(f"{signal.name} is not configured and contributed no evidence.")
        elif signal.status == SignalStatus.skipped and signal.notes:
            reasons.append(f"{signal.name} did not apply to this image: {signal.notes[0]}")
        elif _is_unvalidated_resize(signal):
            reasons.append(
                f"{signal.name} scored this image after downsampling it to 32x32, an "
                "unvalidated resize heuristic — it's shown for reference but was excluded "
                "from the verdict."
            )

    primary = _primary_ai_signals(signals)
    disagreement = 0.0
    if len(primary) >= 2:
        primary_scores = [s.ai_score for s in primary]
        disagreement = max(primary_scores) - min(primary_scores)  # type: ignore[type-var]

    if ai_score is None:
        verdict = "Inconclusive"
        confidence = 0.0
        reasons.insert(
            0,
            "No usable evidence: every check failed, was unavailable, or did not apply to this image.",
        )
    elif len(primary) >= 2 and disagreement >= _DISAGREEMENT_THRESHOLD:
        verdict = "Inconclusive"
        confidence = max(0.0, 0.35 * (1 - disagreement))
        named = ", ".join(f"{s.name} ({s.ai_score:.0%})" for s in primary)
        reasons.insert(
            0,
            f"Independent checks disagreed strongly ({named}); the combined score "
            "is not a meaningful midpoint.",
        )
    elif ai_mass < _MIN_RELIABLE_MASS:
        verdict = "Inconclusive"
        confidence = min(0.35, ai_mass)
        reasons.insert(
            0, "Only weak or limited evidence was available; not enough to reach a confident verdict."
        )
    else:
        verdict = _band(ai_score)
        agreement_factor = (
            1.0 if len(primary) < 2 else max(0.0, 1 - disagreement / _DISAGREEMENT_THRESHOLD)
        )
        certainty = abs(ai_score - 0.5) * 2
        mass_factor = min(1.0, ai_mass / 1.2)
        confidence = max(
            0.0, min(1.0, mass_factor * (0.55 + 0.30 * agreement_factor + 0.15 * certainty))
        )
        if len(ai_contrib) == 1:
            only = next(iter(ai_contrib))
            reasons.insert(
                0,
                f"Only one reliable check ({only}) contributed a score; treat this verdict as provisional.",
            )

    if (
        manip_score is not None
        and manip_mass >= _MANIPULATION_MIN_MASS
        and manip_score >= _MANIPULATION_ESCALATE
    ):
        if verdict in ("Likely Authentic", "Low Risk"):
            verdict = "Medium Risk"
        reasons.append(
            "Possible manipulation of a real image was detected independently of "
            f"AI-generation risk (score {manip_score:.0%})."
        )

    return Aggregate(
        verdict=verdict,
        ai_score=ai_score,
        manipulation_score=manip_score,
        confidence=round(confidence, 4),
        disagreement=round(disagreement, 4),
        contributions=contributions,
        reasons=reasons,
    )
