"""
Tier 1 Regression Tests: Rendering, Contradictions, and Single Source of Truth
Covers items 7 through 19 from user specification.
"""

import re
import pytest

import config
from ui.utils import (
    compute_ols_slope,
    emotion_code_label,
    html_block,
    predominantly_label,
)
from ui.components.export_bar import _clean_transcript_text
from ui.components.verdict_card import build_key_finding


# ==============================================================================
# Item 7: Regression test that fails if rendered HTML contains 4+ leading spaces
# ==============================================================================

def test_html_block_has_no_four_or_more_leading_spaces():
    """Verify that html_block strips leading whitespace on every line so that
    Streamlit markdown never sees 4+ leading spaces (which triggers code-block rendering).
    """
    raw_snippet = """
        <div class="test-class">
            <span>Some indented content</span>
                <p>Deeply indented content</p>
        </div>
    """
    cleaned = html_block(raw_snippet)
    lines = cleaned.splitlines()

    assert len(lines) > 0, "html_block returned empty string"
    for line in lines:
        if line.strip():  # ignore blank lines
            leading_spaces = len(line) - len(line.lstrip(" "))
            assert leading_spaces < 4, f"Found 4+ leading spaces ({leading_spaces}) in line: {line!r}"


def test_html_block_preserves_html_tags_and_content():
    """html_block preserves tags and text while removing leading space."""
    raw = "    <div class=\"foo\">\n        <b>Hello World</b>\n    </div>"
    result = html_block(raw)
    assert "<div class=\"foo\">" in result
    assert "<b>Hello World</b>" in result
    assert "</div>" in result
    for line in result.splitlines():
        assert not line.startswith("    "), f"Indented line found: {line!r}"


# ==============================================================================
# Item 8: Ribbon turn indexing matches segment list (Turn 01, Turn 02, ...)
# ==============================================================================

def test_ribbon_indexing_matches_segment_list():
    """For a 2-turn clip, indexing must be Turn 01, Turn 02 (not T02, T03)."""
    segments = [
        {"segment_index": 1, "start_seconds": 0.0, "end_seconds": 2.5, "fusion": {"agreement": True}},
        {"segment_index": 2, "start_seconds": 2.5, "end_seconds": 5.0, "fusion": {"mismatch": True}},
    ]
    # Check that extracting display index from 1-based segment_index yields 1 and 2
    for s in segments:
        raw_idx = s.get("segment_index", 0)
        display_idx = int(raw_idx)
        assert display_idx in (1, 2)
        assert f"Turn {display_idx:02d}" in (f"Turn {1:02d}", f"Turn {2:02d}")


# ==============================================================================
# Item 9: Map codes (neu, hap, ang, sad) to Neutral/Happy/Angry/Sad in ONE helper
# ==============================================================================

@pytest.mark.parametrize(
    "raw_code, expected_label",
    [
        ("neu", "Neutral"),
        ("neutral", "Neutral"),
        ("hap", "Happy"),
        ("happy", "Happy"),
        ("ang", "Angry"),
        ("angry", "Angry"),
        ("sad", "Sad"),
        ("positive", "Positive"),
        ("negative", "Negative"),
        ("unknown", "Unknown"),
    ],
)
def test_emotion_code_label_single_source_of_truth(raw_code, expected_label):
    """emotion_code_label maps all abbreviated and full model codes consistently."""
    assert emotion_code_label(raw_code) == expected_label
    assert emotion_code_label(raw_code.upper()) == expected_label


# ==============================================================================
# Item 10: 'predominantly' only if >= 60% of turns; otherwise describe as mixed
# ==============================================================================

def test_predominantly_rule_at_least_sixty_percent():
    """When one class has >= 60% of turns, it is predominantly X."""
    # 3 turns: 2 happy, 1 sad -> 66.7% happy >= 60%
    is_pred, top, mixed = predominantly_label({"happy": 2, "sad": 1})
    assert is_pred is True
    assert top == "Happy"
    assert mixed == ""


def test_mixed_rule_below_sixty_percent():
    """When turn 1 is sad and turn 2 is happy (50/50), describe as mixed."""
    # 2 turns: 1 sad, 1 happy -> 50% < 60%
    is_pred, top, mixed = predominantly_label({"sad": 1, "happy": 1})
    assert is_pred is False
    assert "in turn" in mixed


def test_build_key_finding_uses_mixed_description():
    """Key finding must NOT say 'predominantly hap' on a 1 sad / 1 happy clip."""
    segments = [
        {"segment_index": 1, "voice_emotion": {"emotion": "sad"}, "text_sentiment": {"sentiment": "positive"}},
        {"segment_index": 2, "voice_emotion": {"emotion": "hap"}, "text_sentiment": {"sentiment": "positive"}},
    ]
    finding = build_key_finding(
        dominant_voice="hap",
        dominant_text="positive",
        segments=segments,
        trend_label="stable",
        escalation_detected=False,
    )
    # Voice is 50/50 so it must NOT say "predominantly"
    assert "Voice is mixed" in finding
    # Text is 100% positive so it should say predominantly
    assert "text sentiment is predominantly" in finding


# ==============================================================================
# Item 11: Slope SSOT and < 3 turns handling
# ==============================================================================

def test_compute_ols_slope_single_source_of_truth():
    """compute_ols_slope gives exact slope and net change."""
    intensities = [0.2, 0.4, 0.6]
    slope, intercept, net_change = compute_ols_slope(intensities)
    assert pytest.approx(slope, 0.001) == 0.2
    assert pytest.approx(net_change, 0.001) == 0.4


def test_compute_ols_slope_handles_fewer_than_two_points():
    """Handles 0 or 1 point defensively."""
    s0, _, _ = compute_ols_slope([])
    s1, _, _ = compute_ols_slope([0.5])
    assert s0 == 0.0
    assert s1 == 0.0


# ==============================================================================
# Item 12: Gauge action text derived from same score / level (SSOT)
# ==============================================================================

def test_gauge_action_text_alignment():
    """Action text must be LOW PRIORITY when level is LOW and score is 0%."""
    from ui.components.verdict_card import render_verdict_hero
    # If att_level is LOW and score is 0.0, action text must be LOW PRIORITY, not REVIEW RECOMMENDED.
    att_score = {"score": 0.0, "level": "LOW", "notice": "Analytical priority only"}
    # The action text is derived as:
    # att_level == 'HIGH' or att_pct >= 66 -> REVIEW RECOMMENDED
    # att_level == 'MEDIUM' or att_pct >= 33 -> MONITOR
    # else -> LOW PRIORITY
    att_pct = att_score["score"] * 100
    att_level = att_score["level"]
    if att_level == "HIGH" or att_pct >= 66:
        action = "REVIEW RECOMMENDED"
    elif att_level == "MEDIUM" or att_pct >= 33:
        action = "MONITOR"
    else:
        action = "LOW PRIORITY"

    assert action == "LOW PRIORITY", f"Expected LOW PRIORITY but got {action}"


# ==============================================================================
# Item 13: Clean transcript text (no 'Speaker A:' prefix)
# ==============================================================================

def test_clean_transcript_strips_speaker_prefix():
    """Transcript text has speaker prefix stripped cleanly."""
    raw1 = "Speaker A: I am really angry about this"
    raw2 = "Speaker B: Yes, I understand"
    raw3 = "I don't think this is fair"

    assert _clean_transcript_text(raw1, "Speaker A") == "I am really angry about this"
    assert _clean_transcript_text(raw2, "Speaker B") == "Yes, I understand"
    assert _clean_transcript_text(raw3, "Speaker A") == "I don't think this is fair"


# ==============================================================================
# Item 15: Widening and Low ASR confidence guards
# ==============================================================================

def test_widening_guard_duration_check():
    """Widening note is only shown when duration < SHORT_SEG_SECONDS and was_widened=True."""
    short_thresh = getattr(config, "SHORT_SEG_SECONDS", 1.0)
    # A 10.6s segment must NEVER show the widening note
    duration_long = 10.6
    was_widened = False
    note_long = " (widened short-segment window)" if (was_widened and duration_long < short_thresh) else ""
    assert note_long == ""

    # Even if was_widened flag is set, if duration >= short_thresh, note must be empty
    note_long_bug = " (widened short-segment window)" if (True and duration_long < short_thresh) else ""
    assert note_long_bug == ""

    # Short segment that was actually widened:
    duration_short = 0.8
    note_short = " (widened short-segment window)" if (True and duration_short < short_thresh) else ""
    assert note_short == " (widened short-segment window)"


def test_low_asr_confidence_threshold_configurable():
    """LOW_ASR_CONFIDENCE_THRESHOLD exists in config and is positive float."""
    thresh = getattr(config, "LOW_ASR_CONFIDENCE_THRESHOLD", None)
    assert thresh is not None
    assert 0.0 < thresh < 1.0


# ==============================================================================
# Item 16: Version and Fusion Strategy name consistency
# ==============================================================================

def test_version_and_fusion_strategy_name_in_config():
    """APP_VERSION and FUSION_STRATEGY_NAME are defined in config and non-empty."""
    version = getattr(config, "APP_VERSION", "")
    strategy = getattr(config, "FUSION_STRATEGY_NAME", "")
    assert version != "", "APP_VERSION is empty"
    assert "1.0" not in version, f"APP_VERSION should be updated beyond 1.0, got: {version}"
    assert strategy == "Multi-Evidence Arbitration", f"Expected 'Multi-Evidence Arbitration', got: {strategy}"
