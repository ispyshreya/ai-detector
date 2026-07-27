from app.explain.vlm import build_prompt, fallback_explanation, normalize_output


def test_prompt_treats_score_as_context_not_proof():
    prompt = build_prompt(0.91)
    assert "91.0%" in prompt
    assert "context, not proof" in prompt
    assert "must disagree" in prompt


def test_normalize_output_limits_and_formats_bullets():
    text = "1. The sign has warped letters.\n* The left hand has fused fingers.\n- Reflections disagree.\n- Extra."
    assert normalize_output(text) == (
        "- The sign has warped letters.\n"
        "- The left hand has fused fingers.\n"
        "- Reflections disagree."
    )


def test_normalize_output_filters_known_hallucination_patterns():
    assert normalize_output("- The word Veil appears distorted.\n- The clock text is warped.") == (
        "- The clock text is warped."
    )


def test_fallback_is_cautious():
    text = fallback_explanation(0.9)
    assert "No clear visual artifacts" in text
    assert "independent proof" in text
