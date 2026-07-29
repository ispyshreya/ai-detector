from app.engine.triangulate import triangulate
from app.schemas import SignalClass, SignalResult, SignalStatus


def _signal(
    name: str,
    signal_class: SignalClass,
    status: SignalStatus = SignalStatus.ok,
    ai_score: float | None = None,
    manipulation_score: float | None = None,
    confidence: float | None = None,
    error: str | None = None,
    notes: list[str] | None = None,
    raw: dict | None = None,
) -> SignalResult:
    return SignalResult(
        name=name,
        signal_class=signal_class,
        status=status,
        ai_score=ai_score,
        manipulation_score=manipulation_score,
        confidence=confidence,
        error=error,
        notes=notes or [],
        raw=raw,
    )


def test_all_signals_missing_is_inconclusive_with_zero_confidence():
    signals = [
        _signal("local", SignalClass.detector, status=SignalStatus.skipped, notes=["out of domain"]),
        _signal("sightengine", SignalClass.detector, status=SignalStatus.unavailable),
        _signal("hive", SignalClass.detector, status=SignalStatus.unavailable),
        _signal("c2pa", SignalClass.provenance, status=SignalStatus.skipped, notes=["no manifest"]),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "Inconclusive"
    assert aggregate.ai_score is None
    assert aggregate.confidence == 0.0
    assert "No usable evidence" in aggregate.reasons[0]


def test_single_weak_forensic_signal_does_not_confidently_clear_or_convict():
    """A phone photo with intact EXIF but no detector/provenance signal must not
    be confidently labeled either way -- not enough reliable evidence."""
    signals = [
        _signal("local", SignalClass.detector, status=SignalStatus.skipped),
        _signal("sightengine", SignalClass.detector, status=SignalStatus.unavailable),
        _signal("hive", SignalClass.detector, status=SignalStatus.unavailable),
        _signal("exif", SignalClass.forensic, ai_score=0.2, confidence=0.4),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "Inconclusive"
    assert aggregate.confidence < 0.35
    assert any("weak or limited evidence" in r for r in aggregate.reasons)


def test_confident_local_only_result_is_inconclusive_not_high_risk():
    """The exact acceptance scenario from the client prompt: one internal model
    at ~100% AI must not, by itself, produce a confident High Risk verdict."""
    signals = [
        _signal("local", SignalClass.detector, ai_score=0.99, confidence=0.95),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "Inconclusive"
    assert aggregate.confidence <= 0.35


def test_strong_disagreement_between_detectors_forces_inconclusive():
    signals = [
        _signal("sightengine", SignalClass.detector, ai_score=0.92, confidence=0.85),
        _signal("hive", SignalClass.detector, ai_score=0.08, confidence=0.85),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "Inconclusive"
    assert aggregate.disagreement > 0.45
    assert "disagreed strongly" in aggregate.reasons[0]
    # Must not silently present the ~50% midpoint as a meaningful score.
    assert aggregate.confidence < 0.2


def test_agreeing_detectors_produce_confident_verdict():
    signals = [
        _signal("sightengine", SignalClass.detector, ai_score=0.9, confidence=0.85),
        _signal("hive", SignalClass.detector, ai_score=0.85, confidence=0.8),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "High Risk"
    assert aggregate.confidence > 0.6


def test_signed_provenance_dominates_a_weak_forensic_signal():
    signals = [
        _signal("c2pa", SignalClass.provenance, ai_score=0.9, confidence=0.9),
        _signal("exif", SignalClass.forensic, ai_score=0.25, confidence=0.4),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "High Risk"
    assert aggregate.ai_score > 0.75  # c2pa should dominate the fused score
    assert aggregate.confidence > 0.6


def test_single_external_detector_is_flagged_provisional():
    signals = [
        _signal("sightengine", SignalClass.detector, ai_score=0.95, confidence=0.9),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "High Risk"
    assert list(aggregate.contributions.keys()) == ["sightengine"]
    assert any("provisional" in r for r in aggregate.reasons)


def test_failed_and_unavailable_signals_contribute_no_evidence_either_way():
    agreeing = [
        _signal("sightengine", SignalClass.detector, ai_score=0.1, confidence=0.8),
        _signal("hive", SignalClass.detector, ai_score=0.12, confidence=0.8),
    ]
    with_noise = agreeing + [
        _signal("local", SignalClass.detector, status=SignalStatus.error, error="OOM"),
        _signal("reverse_search", SignalClass.context, status=SignalStatus.unavailable),
    ]

    baseline = triangulate(agreeing)
    noisy = triangulate(with_noise)

    assert noisy.ai_score == baseline.ai_score
    assert noisy.verdict == baseline.verdict == "Likely Authentic"
    assert any("OOM" in r for r in noisy.reasons)


def test_manipulation_evidence_escalates_verdict_even_when_ai_axis_is_clean():
    signals = [
        _signal("sightengine", SignalClass.detector, ai_score=0.1, manipulation_score=0.9, confidence=0.85),
    ]

    aggregate = triangulate(signals)

    assert aggregate.verdict == "Medium Risk"
    assert any("manipulation" in r.lower() for r in aggregate.reasons)


def test_resized_local_score_is_excluded_from_verdict_and_disagreement():
    """Regression: a real photo where sightengine/hive confidently agree it's
    authentic must not be dragged to Inconclusive by a noisy resized-to-32x32
    `local` read, even though local disagrees sharply on its own axis."""
    agreeing_only = [
        _signal("sightengine", SignalClass.detector, ai_score=0.0, confidence=0.99),
        _signal("hive", SignalClass.detector, ai_score=0.03, confidence=0.94),
    ]
    with_noisy_local = agreeing_only + [
        _signal(
            "local",
            SignalClass.detector,
            ai_score=0.87,
            confidence=0.55,
            raw={"was_resized": True, "original_size": [2160, 3840], "training_native_size": [32, 32]},
        ),
    ]

    baseline = triangulate(agreeing_only)
    with_local = triangulate(with_noisy_local)

    assert with_local.verdict == baseline.verdict == "Likely Authentic"
    assert with_local.ai_score == baseline.ai_score
    assert with_local.confidence == baseline.confidence
    assert "local" not in with_local.contributions
    assert any("excluded from the verdict" in r for r in with_local.reasons)


def test_native_resolution_local_score_still_counts():
    """Only the resized path is excluded -- a native 32x32 local result
    should still participate normally."""
    signals = [
        _signal(
            "local",
            SignalClass.detector,
            ai_score=0.9,
            confidence=0.8,
            raw={"was_resized": False},
        ),
    ]

    aggregate = triangulate(signals)

    assert "local" in aggregate.contributions
