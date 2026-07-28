from PIL import Image

from app.explain.vlm import (
    build_prompt,
    fallback_explanation,
    format_explanation,
    make_views,
    parse_inspection,
)


def test_prompt_is_score_blind_and_structured():
    prompt = build_prompt()
    assert "do not know" in prompt
    assert "detector score" in prompt
    assert '"assessment"' in prompt
    assert "specific_artifacts_found" in prompt


def test_make_views_returns_full_image_and_quadrants():
    image = Image.new("RGB", (100, 80))
    assert [view.size for view in make_views(image)] == [
        (100, 80), (50, 40), (50, 40), (50, 40), (50, 40)
    ]


def test_parse_valid_structured_findings():
    raw = """```json
    {"assessment":"specific_artifacts_found","findings":[
      {"region":"store sign in upper-left","observation":"three letters have broken and inconsistent strokes","confidence":"medium"},
      {"region":"left hand near the table","observation":"two adjacent fingers merge without a visible boundary","confidence":"high"}
    ]}```"""
    assessment, findings = parse_inspection(raw)
    assert assessment == "specific_artifacts_found"
    assert len(findings) == 2
    assert findings[0].region == "store sign in upper-left"


def test_parse_rejects_generic_and_hallucinated_findings():
    raw = """{"assessment":"specific_artifacts_found","findings":[
      {"region":"image","observation":"this looks AI-generated","confidence":"high"},
      {"region":"lower-right clock","observation":"the printed numerals have duplicated strokes","confidence":"medium"}
    ]}"""
    assessment, findings = parse_inspection(raw)
    assert assessment == "specific_artifacts_found"
    assert len(findings) == 1
    assert findings[0].region == "lower-right clock"


def test_invalid_json_abstains():
    assert parse_inspection("The image might be fake.") == (
        "insufficient_visual_detail", ()
    )


def test_empty_specific_assessment_becomes_abstention():
    assert parse_inspection(
        '{"assessment":"specific_artifacts_found","findings":[]}'
    ) == ("no_clear_artifacts", ())


def test_fallback_does_not_repeat_high_score_as_evidence():
    text = fallback_explanation(0.99)
    assert "99" not in text
    assert "not visually corroborated" in text


def test_format_valid_findings_without_fallback():
    assessment, findings = parse_inspection(
        '{"assessment":"specific_artifacts_found","findings":['
        '{"region":"window reflection on right","observation":"the reflected frame bends at the center","confidence":"low"}]}'
    )
    text, used_fallback = format_explanation(assessment, findings, 0.8)
    assert "window reflection on right" in text
    assert used_fallback is False
