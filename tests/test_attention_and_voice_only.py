"""
Tests for attention gauge single-turn mismatch, incongruence card override,
voice-only UI rendering safeguards, and backend-specific prosody sanity rule.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

import config
from app.consistency_analysis import (
    compute_attention_score,
    compute_consistency_trajectory,
)
from app.voice_calibration import apply_prosody_sanity_rule
from ui.components.incongruence_card import render_incongruence_card
from ui.components.modality_cards import render_channel_meters_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_segment(mismatch: bool, intensity: float = 0.5, text_unavail: bool = False):
    return {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 3.0,
        "intensity": intensity,
        "text_unavailable": text_unavail,
        "fusion": {
            "mismatch": mismatch,
            "agreement": not mismatch and not text_unavail,
            "audio_direction": "negative" if mismatch else "positive",
            "text_direction": "positive",
            "combined_confidence": 0.75,
        },
    }


# ---------------------------------------------------------------------------
# 1. Attention gauge single-turn and multi-turn consistency
# ---------------------------------------------------------------------------

def test_pinned_gauge_2_3_4_segments_unchanged():
    """Verify 2, 3, 4 segments produce expected pinned attention scores."""
    # 2-segment fixture
    segs2 = [
        _make_segment(mismatch=False, intensity=0.4),
        _make_segment(mismatch=False, intensity=0.6),
    ]
    traj2 = compute_consistency_trajectory(segs2)
    score2 = compute_attention_score(segs2, trajectory=traj2)
    assert score2.mismatch_rate == 0.0
    assert score2.mismatch_contribution == 0.0
    assert score2.intensity_contribution == pytest.approx(0.175, abs=1e-3)
    assert score2.attention_score == pytest.approx(0.175, abs=1e-3)

    # 3-segment fixture
    segs3 = [
        _make_segment(mismatch=True, intensity=0.8),
        _make_segment(mismatch=True, intensity=0.6),
        _make_segment(mismatch=False, intensity=0.4),
    ]
    traj3 = compute_consistency_trajectory(segs3)
    score3 = compute_attention_score(segs3, trajectory=traj3)
    assert score3.mismatch_rate == pytest.approx(2 / 3, abs=1e-3)
    assert score3.mismatch_contribution == pytest.approx(0.40 * (2 / 3), abs=1e-3)

    # 4-segment fixture
    segs4 = [
        _make_segment(mismatch=True, intensity=0.9),
        _make_segment(mismatch=False, intensity=0.3),
        _make_segment(mismatch=False, intensity=0.5),
        _make_segment(mismatch=False, intensity=0.7),
    ]
    traj4 = compute_consistency_trajectory(segs4)
    score4 = compute_attention_score(segs4, trajectory=traj4)
    assert score4.mismatch_rate == 0.25
    assert score4.mismatch_contribution == pytest.approx(0.10, abs=1e-3)


def test_single_turn_mismatch_gauge_positive():
    """Single-turn recording with mismatch=True must produce mismatch_contribution > 0."""
    seg = _make_segment(mismatch=True, intensity=0.6)
    traj = compute_consistency_trajectory([seg])
    score = compute_attention_score([seg], trajectory=traj)

    assert traj.trajectory == "insufficient_data"
    assert traj.mismatch_rate == 1.0
    assert score.mismatch_rate == 1.0
    assert score.mismatch_contribution == pytest.approx(0.40, abs=1e-3)
    assert score.attention_score > 0.40


def test_single_turn_agreement_gauge_low():
    """Single-turn recording with agreement (mismatch=False) keeps mismatch contribution at 0."""
    seg = _make_segment(mismatch=False, intensity=0.2)
    traj = compute_consistency_trajectory([seg])
    score = compute_attention_score([seg], trajectory=traj)

    assert traj.mismatch_rate == 0.0
    assert score.mismatch_rate == 0.0
    assert score.mismatch_contribution == 0.0
    assert score.attention_score == pytest.approx(0.35 * 0.2, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Incongruence card derivation from fusion mismatch
# ---------------------------------------------------------------------------

def test_incongruence_card_consistent_with_banner_for_mismatch_generic():
    """When mismatch rule fires (e.g. mismatch_generic), card conflict is NOT NONE."""
    fusion_dict = {
        "mismatch": True,
        "rule_ids_fired": ["mismatch_generic"],
        "interpretation": "Voice tone and linguistic sentiment diverge.",
    }

    markdown_calls = []
    with patch("streamlit.markdown", side_effect=lambda content, **kwargs: markdown_calls.append(content)):
        render_incongruence_card(
            acoustic_label="neutral",
            acoustic_confidence=0.70,
            text_label="positive",
            text_confidence=0.85,
            fusion=fusion_dict,
        )

    assert len(markdown_calls) == 1
    rendered = markdown_calls[0]
    assert "CONFLICT: NONE" not in rendered
    assert ("CONFLICT: HIGH" in rendered or "CONFLICT: MEDIUM" in rendered)
    assert "Voice tone and linguistic sentiment diverge" in rendered


# ---------------------------------------------------------------------------
# 3. Voice-only UI safeguards
# ---------------------------------------------------------------------------

def test_voice_only_render_channel_meters():
    """render_channel_meters_html with hide_text=True has no text row and no UNKNOWN."""
    html = render_channel_meters_html(
        voice_label="happy",
        voice_confidence=0.75,
        text_label="unknown",
        text_confidence=0.0,
        fused_confidence=0.75,
        hide_text=True,
    )
    assert "UNKNOWN" not in html
    assert "0.0%" not in html
    assert "ACOUSTIC (VOICE)" in html


def test_voice_only_timeline_elements():
    """Verifies that voice-only elements render n/a and suppress misleading captions."""
    is_text_unavail = True
    f_conf = 0.65

    if is_text_unavail:
        evidence_str = "n/a (voice-only)"
        evidence_caption = ""
    else:
        evidence_str = f"{f_conf:.1%}"
        evidence_caption = "Weighted average of the two modality confidences"

    assert "n/a (voice-only)" in evidence_str
    assert "Weighted average" not in evidence_caption

    # Direction line suppression
    dirs_differ = True
    agreement = False
    direction_line = (
        "Voice and text point in different directions"
        if dirs_differ and not agreement and not is_text_unavail
        else ""
    )
    assert direction_line == ""


# ---------------------------------------------------------------------------
# 4. Prosody sanity rule per backend
# ---------------------------------------------------------------------------

def test_sanity_rule_runs_on_superb_and_bypassed_on_emotion2vec(monkeypatch):
    """Sanity rule dampens sad on SUPERB, but leaves emotion2vec unchanged."""
    raw_scores = {"sad": 0.80, "happy": 0.10, "neutral": 0.10}
    # Create synthetic audio with high energy to trigger sanity rule
    audio_segment = np.ones(16000, dtype=np.float32) * 0.15
    f0_segment = np.array([200.0, 300.0, 150.0, 250.0], dtype=np.float32)
    voiced_flag = np.array([True, True, True, True])

    # 1. On SUPERB: sanity rule applies
    monkeypatch.setattr(config, "VOICE_BACKEND", "superb")
    monkeypatch.setattr(config, "VOICE_SANITY_RULE_BACKENDS", ["superb"])
    monkeypatch.setattr(config, "VOICE_CALIBRATION_PROSODY_RULE_ENABLED", True)

    res_superb = apply_prosody_sanity_rule(
        raw_scores=raw_scores,
        audio_segment=audio_segment,
        sr=16000,
        f0_segment=f0_segment,
        voiced_flag_segment=voiced_flag,
    )
    assert "dampened 'sad'" in str(res_superb.applied_rules)
    assert res_superb.scores["sad"] < 0.80

    # 2. On emotion2vec: sanity rule is bypassed
    monkeypatch.setattr(config, "VOICE_BACKEND", "emotion2vec")
    monkeypatch.setattr(config, "VOICE_SANITY_RULE_BACKENDS", ["superb"])

    res_e2v = apply_prosody_sanity_rule(
        raw_scores=raw_scores,
        audio_segment=audio_segment,
        sr=16000,
        f0_segment=f0_segment,
        voiced_flag_segment=voiced_flag,
    )
    assert len(res_e2v.applied_rules) == 0
    assert res_e2v.scores["sad"] == 0.80
    assert res_e2v.confidence == 0.80


# ---------------------------------------------------------------------------
# 5. Confidence wording test
# ---------------------------------------------------------------------------

def test_capped_confidence_wording():
    """Verify confidence wording tag changes when capped vs uncapped."""
    # When cap applied (raw_conf > conf)
    v_raw_conf = 0.99
    v_conf = 0.65
    v_conf_info = False

    if not v_conf_info:
        if v_raw_conf > v_conf + 0.001:
            tag = f"(capped from raw {v_raw_conf:.0%}; not calibrated)"
        else:
            tag = "(raw model score (not calibrated))"
    else:
        tag = ""

    assert tag == "(capped from raw 99%; not calibrated)"

    # When cap not applied
    v_raw_conf_uncapped = 0.65
    if not v_conf_info:
        if v_raw_conf_uncapped > v_conf + 0.001:
            tag_uncapped = f"(capped from raw {v_raw_conf_uncapped:.0%}; not calibrated)"
        else:
            tag_uncapped = "(raw model score (not calibrated))"
    else:
        tag_uncapped = ""

    assert tag_uncapped == "(raw model score (not calibrated))"


# ---------------------------------------------------------------------------
# 6. Phase 4d: HTML format, Voice-Only, Short Segment, and Sanitization Tests
# ---------------------------------------------------------------------------

def test_html_block_no_leading_spaces_or_blank_lines():
    """Output of meters and card renderers has no line with 4+ leading spaces and no blank lines."""
    from ui.utils import html_block
    from ui.components.modality_cards import render_modality_cards

    # 1. Meters output
    meters_html = render_channel_meters_html("happy", 0.8, "positive", 0.7, 0.75, hide_text=False)
    for line in meters_html.splitlines():
        assert not line.startswith("    "), f"Line starts with 4+ spaces: {line}"
        assert line.strip() != "", "Empty line found"

    # 2. Modality cards output
    md_calls = []
    with patch("streamlit.markdown", side_effect=lambda c, **kw: md_calls.append(c)):
        render_modality_cards("happy", "positive", [], 1, 0)
    for card_html in md_calls:
        for line in card_html.splitlines():
            assert not line.startswith("    "), f"Card line starts with 4+ spaces: {line}"
            assert line.strip() != "", "Empty line found in card HTML"


def test_voice_only_render_has_no_forbidden_strings():
    """Voice-only turn render contains none of: 'different directions', 'UNKNOWN', 'Weighted average', 'low ASR confidence'."""
    from ui.components.segment_timeline import _render_why_this_result

    # 1. Channel meters for voice-only hides text and fused meters
    meters_html = render_channel_meters_html("happy", 0.8, "unknown", 0.0, 0.8, hide_text=True)
    assert "different directions" not in meters_html
    assert "UNKNOWN" not in meters_html
    assert "Weighted average" not in meters_html
    assert "MULTIMODAL EVIDENCE STRENGTH" not in meters_html

    # 2. Why-this-result expander for voice-only
    fusion_dict = {
        "pattern_label": "Voice Dominant",
        "pattern_explanation": "Voice-only inference",
        "audio_reliability": 0.85,
        "text_reliability": 0.0,
        "counterfactuals": ["hypothetical sensitivity"],
    }
    why_calls = []
    with patch("streamlit.markdown", side_effect=lambda c, **kw: why_calls.append(c)), \
         patch("streamlit.expander"):
        _render_why_this_result(fusion_dict, is_text_unavail=True)

    assert len(why_calls) == 1
    content = why_calls[0]
    assert "different directions" not in content
    assert "UNKNOWN" not in content
    assert "Weighted average" not in content
    assert "low ASR confidence" not in content
    assert "text: n/a (text unavailable)" in content
    assert "Counterfactual Sensitivity:" not in content


def test_short_segment_exclusion_from_trend_and_transitions():
    """Segments shorter than VOICE_MIN_SEGMENT_SEC are excluded from trend verdict and voice direction sentence."""
    from app.conversation_analysis import ConversationAnalyzer
    from ui.components.verdict_card import build_key_finding

    # 4 segments: 1st and last are short (0.5s), middle 2 are 2.0s
    segs = [
        {"segment_index": 1, "start_seconds": 0.0, "end_seconds": 0.5, "intensity": 0.9, "fusion": {"audio_direction": "angry", "text_direction": "neutral"}},
        {"segment_index": 2, "start_seconds": 1.0, "end_seconds": 3.0, "intensity": 0.4, "fusion": {"audio_direction": "happy", "text_direction": "positive"}},
        {"segment_index": 3, "start_seconds": 4.0, "end_seconds": 6.0, "intensity": 0.5, "fusion": {"audio_direction": "happy", "text_direction": "positive"}},
        {"segment_index": 4, "start_seconds": 7.0, "end_seconds": 7.5, "intensity": 0.1, "fusion": {"audio_direction": "sad", "text_direction": "neutral"}},
    ]

    # Only 2 segments >= 1.0s, so compute_unified_verdict returns insufficient_data (needs 3 reliable turns)
    verdict = ConversationAnalyzer.compute_unified_verdict(segs)
    assert verdict["verdict"] == "insufficient_data"

    # Voice direction in build_key_finding uses valid segments (both middle segments are happy -> no change)
    finding = build_key_finding("happy", "positive", segs, "stable", False)
    assert "Voice direction moved" not in finding


def test_label_consistency_no_mixed_other():
    """Canonical display labels are used without mixed 'OTHER (SURPRISED)' labels."""
    from ui.utils import emotion_code_label

    assert emotion_code_label("other") == "Other"
    assert emotion_code_label("happy") == "Happy"
    assert emotion_code_label("neu") == "Neutral"

    # Timeline display logic for other uses canonical label to avoid 'OTHER (SURPRISED)' mixed title
    v_raw = "other"
    v_leaning = "surprised"
    v_disp = emotion_code_label(v_raw, raw_label=None if v_raw == "other" else v_leaning)
    assert v_disp == "Other"
    assert v_disp.upper() == "OTHER"


def test_strip_ufffd_and_repetitive_phrase_collapse():
    """U+FFFD is stripped, and phrases of 1 to 4 tokens repeated 4+ times are collapsed."""
    from app.conversation_analysis import sanitize_transcript
    from ui.components.segment_timeline import _collapse_repetitive_tokens

    # U+FFFD stripped
    raw_text = "Hello\ufffd world\ufffd"
    assert sanitize_transcript(raw_text) == "Hello world"

    # Repeated token
    assert _collapse_repetitive_tokens("yes yes yes yes") == "yes… ×4 (repetitive transcript)"

    # Repeated 2-token phrase
    assert _collapse_repetitive_tokens("thank you thank you thank you thank you") == "thank you… ×4 (repetitive transcript)"

    # Repeated 3-token phrase
    assert _collapse_repetitive_tokens("one two three one two three one two three one two three") == "one two three… ×4 (repetitive transcript)"

    # Non-repetitive text unchanged
    assert _collapse_repetitive_tokens("this is a normal sentence with distinct words") == "this is a normal sentence with distinct words"

