from PIL import Image

from app.explain.vlm import (
    _parse_category_answer,
    fallback_explanation,
    format_explanation,
    make_views,
    parse_category_findings,
)


def test_make_views_returns_full_image_and_quadrants():
    image = Image.new("RGB", (100, 80))
    assert [view.size for view in make_views(image)] == [
        (100, 80), (50, 40), (50, 40), (50, 40), (50, 40)
    ]


def test_yes_answer_is_not_a_finding():
    assert _parse_category_answer("YES") is None
    assert _parse_category_answer("Yes, everything looks normal.") is None


def test_no_answer_yields_the_stated_problem():
    observation = _parse_category_answer(
        'NO\nOPEQ 24 HUORS $5.99\nSzALE TDOAYONLY!!'
    )
    assert observation is not None
    assert "OPEQ 24 HUORS" in observation


def test_ambiguous_answer_is_not_a_finding():
    assert _parse_category_answer("It's hard to tell from this angle.") is None


def test_technique_leakage_is_discarded():
    """The model sometimes describes our own 5-view grid instead of the
    image itself -- that must never surface as a finding."""
    leak = _parse_category_answer(
        "NO, this image is a collage of multiple images with a red banner."
    )
    assert leak is None


def test_absence_answered_as_no_is_not_a_finding():
    """Regression: model answered NO to 'is there text?' instead of NO to
    'is the text correct?', producing 'there is no text in the image' --
    an absence, not evidence of anything. Must not surface as a finding."""
    assert _parse_category_answer("NO, there is no text in the image") is None
    assert _parse_category_answer("No. There are no reflections present.") is None


def test_parse_category_findings_combines_only_no_answers():
    responses = {
        "Text/writing": 'NO\nOPEQ 24 HUORS $5.99 is misspelled in the top banner',
        "Hands/limbs/faces": "YES",
        "Reflections/lighting": "YES",
        "Patterns/edges": "YES",
    }
    assessment, findings = parse_category_findings(responses)
    assert assessment == "specific_artifacts_found"
    assert len(findings) == 1
    assert findings[0].region == "Text/writing"
    assert "OPEQ" in findings[0].observation


def test_parse_category_findings_all_normal_abstains():
    responses = {name: "YES" for name in ["Text/writing", "Hands/limbs/faces"]}
    assessment, findings = parse_category_findings(responses)
    assert assessment == "no_clear_artifacts"
    assert findings == ()


def test_parse_category_findings_caps_at_three():
    responses = {
        "Text/writing": "NO, the sign says GARBBLED TEXT in the corner",
        "Hands/limbs/faces": "NO, the hand has six visible fingers on it",
        "Reflections/lighting": "NO, the mirror reflection does not match the room",
        "Patterns/edges": "NO, the fence pattern repeats unnaturally at the edge",
    }
    _, findings = parse_category_findings(responses)
    assert len(findings) == 3


def test_fallback_does_not_repeat_high_score_as_evidence():
    text = fallback_explanation(0.99)
    assert "99" not in text
    assert "weigh the detector score above" in text


def test_fallback_never_claims_authenticity_against_a_risky_score():
    """Regression: fallback text must not open with "looks visually
    authentic" when the fused score already says High/Medium Risk -- that's
    a flat contradiction next to the verdict shown above it."""
    text = fallback_explanation(0.955)
    assert "looks visually authentic" not in text


def test_format_valid_findings_without_fallback():
    assessment, findings = parse_category_findings(
        {"Reflections/lighting": "NO, the window reflection bends unnaturally at its center edge"}
    )
    text, used_fallback = format_explanation(assessment, findings, 0.8)
    assert "window reflection" in text
    assert used_fallback is False


def test_findings_render_as_a_grounded_verdict_not_bare_facts():
    """Product requirement: the VLM must state a lean ("likely AI-generated")
    tied to the specific visible reason, not just list neutral observations."""
    assessment, findings = parse_category_findings(
        {"Hands/limbs/faces": "NO, the right hand has six visible fingers"}
    )
    text, used_fallback = format_explanation(assessment, findings, 0.8)
    assert used_fallback is False
    assert "likely AI-generated" in text
    assert "six visible fingers" in text


def test_authentic_case_also_states_a_verdict():
    assessment, findings = parse_category_findings({"Text/writing": "YES"})
    text, used_fallback = format_explanation(assessment, findings, 0.1)
    assert used_fallback is True
    assert "looks visually authentic" in text


def test_no_findings_uses_fallback():
    assessment, findings = parse_category_findings({"Text/writing": "YES"})
    text, used_fallback = format_explanation(assessment, findings, 0.8)
    assert used_fallback is True
    assert text == fallback_explanation(0.8)
