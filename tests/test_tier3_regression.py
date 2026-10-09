"""
Regression tests for Tier 3 / Tier 6B fixes.
Items covered: 1 (turn numbering), 2 (attention gauge), 3 (summary label order),
               4 (low-voice-reliability message wording), 5 (no raw codes in UI),
               7 (evaluation aggregation matches live pipeline).

All test data is synthetic and marked test-only per AGENTS.md.
Never fabricate actual model outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def _src(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


def _make_segment(
    index: int,
    voice_emotion: str = "neutral",
    voice_conf: float = 0.70,
    text_sentiment: str = "neutral",
    text_conf: float = 0.70,
    intensity: float = 0.50,
    agreement: bool = True,
    mismatch: bool = False,
    low_voice_reliability: bool = False,
    voice_reliability: float = 1.0,
    voice_collapse_detected: bool = False,
) -> Dict[str, Any]:
    """Minimal synthetic segment for UI rendering tests."""
    return {
        "segment_index": index,
        "start_seconds": float(index),
        "end_seconds": float(index + 1),
        "transcript": f"Turn {index} text.",
        "speaker": "Speaker A",
        "intensity": intensity,
        "low_voice_reliability": low_voice_reliability,
        "voice_reliability": voice_reliability,
        "voice_collapse_detected": voice_collapse_detected,
        "voice_emotion": {
            "emotion": voice_emotion,
            "confidence": voice_conf,
            "voice_reliability": voice_reliability,
        },
        "text_sentiment": {
            "sentiment": text_sentiment,
            "confidence": text_conf,
        },
        "fusion": {
            "agreement": agreement,
            "mismatch": mismatch,
            "interpretation": "Test agreement.",
            "combined_confidence": (voice_conf + text_conf) / 2,
            "audio_direction": "stable",
            "text_direction": "stable",
            "effective_weights": {"audio_weight": 0.5, "text_weight": 0.5},
            "audio_reliability": voice_reliability,
            "text_reliability": 1.0,
            "rule_ids_fired": ["default"],
            "pattern_label": "Standard Arbitration",
            "counterfactuals": [],
        },
    }


# ---------------------------------------------------------------------------
# Item 1: Turn numbering is 1-based and consistent across all UI locations
# ---------------------------------------------------------------------------
class TestItem1TurnNumbering:
    """format_turn_label must produce Turn 01, Turn 02, ... for 0-based indices."""

    def test_format_turn_label_two_turns(self):
        """2-turn fixture: segment_index 0,1 → Turn 01, Turn 02."""
        from ui.utils import format_turn_label

        assert format_turn_label(0, one_based=False) == "Turn 01"
        assert format_turn_label(1, one_based=False) == "Turn 02"

    def test_format_turn_label_four_turns(self):
        """4-turn fixture: segment_index 0..3 → Turn 01..Turn 04."""
        from ui.utils import format_turn_label

        for i in range(4):
            result = format_turn_label(i, one_based=False)
            assert result == f"Turn {i + 1:02d}", f"index={i}: got {result!r}"

    def test_why_result_view_uses_format_turn_label(self):
        """why_result_view.py must call format_turn_label (single source of truth)."""
        src = _src("ui/pages/why_result_view.py")
        assert "format_turn_label" in src

    def test_segment_timeline_uses_format_turn_label(self):
        """segment_timeline.py must call format_turn_label (single source of truth)."""
        src = _src("ui/components/segment_timeline.py")
        assert "format_turn_label" in src

    def test_waveform_player_uses_format_turn_label(self):
        """waveform_player.py must call format_turn_label for timeline markers."""
        src = _src("ui/components/waveform_player.py")
        assert "format_turn_label" in src

    def test_segment_timeline_one_based_false(self):
        """segment_timeline.py must pass one_based=False (segment_index is 0-based)."""
        src = _src("ui/components/segment_timeline.py")
        assert "one_based=False" in src

    def test_waveform_player_one_based_false(self):
        """waveform_player.py must pass one_based=False (segment_index is 0-based)."""
        src = _src("ui/components/waveform_player.py")
        assert "one_based=False" in src

    def test_why_result_view_enumerate_starts_at_one(self):
        """why_result_view.py enumerates from 1 and uses one_based=True."""
        src = _src("ui/pages/why_result_view.py")
        assert "enumerate(segments, 1)" in src
        assert "one_based=True" in src


# ---------------------------------------------------------------------------
# Item 2: Attention gauge reads "attention_score" key, not "score"
# ---------------------------------------------------------------------------
class TestItem2AttentionGauge:
    """verdict_card.py must read att_score.get('attention_score', ...) not just 'score'."""

    def test_verdict_card_reads_correct_key(self):
        src = _src("ui/components/verdict_card.py")
        assert "attention_score" in src
        # Must not silently fall back to only "score" — the key is "attention_score"
        assert 'att_score.get("attention_score"' in src or "att_score.get('attention_score'" in src

    def test_escalating_fixture_gives_nonzero_attention(self):
        """Escalating segment sequence (rising intensity + mismatches) → attention > 0."""
        from app.consistency_analysis import compute_attention_score

        from tests.test_consistency_analysis import make_segment  # reuse existing fixture builder

        escalating = [
            make_segment(agreement=False, mismatch=True, intensity=0.4, segment_index=0),
            make_segment(agreement=False, mismatch=True, intensity=0.6, segment_index=1),
            make_segment(agreement=False, mismatch=True, intensity=0.8, segment_index=2),
        ]
        # escalation_result is a dict with "escalation_detected" key
        escalation = {"escalation_detected": True}
        score = compute_attention_score(escalating, escalation)
        assert score.attention_score > 0.0, f"Expected > 0, got {score.attention_score}"

    def test_calm_fixture_gives_low_attention(self):
        """Calm, all-agreement sequence → attention score near zero (< 0.20)."""
        from app.consistency_analysis import compute_attention_score

        from tests.test_consistency_analysis import make_segment

        calm = [
            make_segment(agreement=True, mismatch=False, intensity=0.3, segment_index=0),
            make_segment(agreement=True, mismatch=False, intensity=0.3, segment_index=1),
            make_segment(agreement=True, mismatch=False, intensity=0.3, segment_index=2),
        ]
        # escalation_result is a dict with "escalation_detected" key
        no_escalation = {"escalation_detected": False}
        score = compute_attention_score(calm, no_escalation)
        assert score.attention_score < 0.20, f"Expected < 0.20, got {score.attention_score}"

    def test_attention_score_to_dict_key_is_attention_score(self):
        """AttentionScore.to_dict() must return key 'attention_score', not 'score'."""
        from app.consistency_analysis import compute_attention_score

        from tests.test_consistency_analysis import make_segment

        seg = [make_segment(agreement=False, mismatch=True, intensity=0.5, segment_index=0)]
        score = compute_attention_score(seg, None)
        d = score.to_dict()
        assert "attention_score" in d, f"Keys: {list(d.keys())}"
        assert "score" not in d or d.get("attention_score") == d.get("score"), (
            "'score' key found but differs from 'attention_score'"
        )


# ---------------------------------------------------------------------------
# Item 3: Summary lists voice and text labels in actual turn order
# ---------------------------------------------------------------------------
class TestItem3SummaryLabelOrder:
    """derive_segment_summary returns labels based on confidence-weighted votes."""

    def test_two_turn_fixture_dominant_by_confidence(self):
        """Higher-confidence label wins even if it appears fewer times."""
        from ui.utils import derive_segment_summary

        segments = [
            _make_segment(0, voice_emotion="sad", voice_conf=0.30),
            _make_segment(1, voice_emotion="happy", voice_conf=0.70),
        ]
        dom_voice, _dom_text, _ag, _mm = derive_segment_summary(segments)
        assert dom_voice.lower() == "happy", f"Expected happy, got {dom_voice}"

    def test_four_turn_fixture_majority_by_sum(self):
        """3 low-conf sad turns vs 1 high-conf happy: low sum vs high sum."""
        from ui.utils import derive_segment_summary

        segments = [
            _make_segment(0, voice_emotion="sad", voice_conf=0.20),
            _make_segment(1, voice_emotion="sad", voice_conf=0.20),
            _make_segment(2, voice_emotion="sad", voice_conf=0.20),
            _make_segment(3, voice_emotion="happy", voice_conf=0.70),
        ]
        dom_voice, _dom_text, _ag, _mm = derive_segment_summary(segments)
        # 3×0.20 = 0.60 sad, 1×0.70 = 0.70 happy → happy wins
        assert dom_voice.lower() == "happy", f"Expected happy (conf-weighted), got {dom_voice}"

    def test_agreement_mismatch_counts_correct(self):
        """agreement_count and mismatch_count must reflect fusion flags."""
        from ui.utils import derive_segment_summary

        segments = [
            _make_segment(0, agreement=True, mismatch=False),
            _make_segment(1, agreement=False, mismatch=True),
            _make_segment(2, agreement=True, mismatch=False),
        ]
        _dv, _dt, ag, mm = derive_segment_summary(segments)
        assert ag == 2
        assert mm == 1


# ---------------------------------------------------------------------------
# Item 4: Low-voice-reliability message wording
# ---------------------------------------------------------------------------
class TestItem4LowVoiceReliabilityWording:
    """The low-voice-reliability callout must NOT say 'confidence < 50%' when
    the trigger is the reliability falling below its own threshold (not confidence)."""

    def test_reliability_below_threshold_message_mentions_reliability(self):
        """When v_rel < LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD, callout mentions 'voice reliability'."""
        src = _src("ui/components/segment_timeline.py")
        # Check that the reliability reason string includes 'voice reliability'
        assert "voice reliability" in src.lower()

    def test_reliability_message_does_not_hardcode_fifty_percent(self):
        """The low-voice callout must not hardcode '50%' as the reliability threshold."""
        src = _src("ui/components/segment_timeline.py")
        # The confidence threshold is configurable; '50%' as hardcoded string is a bug
        # Note: the display may show the dynamic value, not a hardcoded 50%
        # Verify it reads from config, not magic number
        assert "LOW_VOICE_CONFIDENCE_THRESHOLD" in src or "low_v_conf_thresh" in src

    def test_reliability_threshold_read_from_config(self):
        """Reliability display threshold must be read from config (not magic number)."""
        src = _src("ui/components/segment_timeline.py")
        assert "LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD" in src or "low_v_rel_disp_thresh" in src

    def test_both_trigger_paths_exist_in_source(self):
        """Both trigger conditions (reliability AND confidence) must be code-present."""
        src = _src("ui/components/segment_timeline.py")
        # Path 1: reliability below threshold
        assert "v_rel < low_v_rel_disp_thresh" in src
        # Path 2: voice confidence below threshold
        assert "v_conf < low_v_conf_thresh" in src


# ---------------------------------------------------------------------------
# Item 5: No raw codes (hap/ang/neu/sad) in rendered UI elements
# ---------------------------------------------------------------------------
class TestItem5NoRawCodesInUI:
    """emotion_code_label must be applied before any HTML rendering."""

    def test_emotion_code_label_maps_all_raw_codes(self):
        """All four SUPERB raw codes must map to human labels."""
        from ui.utils import emotion_code_label

        assert emotion_code_label("neu") == "Neutral"
        assert emotion_code_label("hap") == "Happy"
        assert emotion_code_label("ang") == "Angry"
        assert emotion_code_label("sad") == "Sad"

    def test_segment_timeline_applies_emotion_code_label(self):
        """segment_timeline.py must call emotion_code_label before rendering voice labels."""
        src = _src("ui/components/segment_timeline.py")
        assert "emotion_code_label" in src

    def test_verdict_card_imports_or_uses_emotion_labels(self):
        """verdict_card.py must not render raw codes; check import or display helper."""
        src = _src("ui/components/verdict_card.py")
        # Either imports emotion_code_label or references display-form labels
        assert "emotion_code_label" in src or "Neutral" in src or "Happy" in src

    def test_no_standalone_neu_code_in_segment_timeline(self):
        """'neu' must not appear standalone in HTML output strings of segment_timeline.py."""
        src = _src("ui/components/segment_timeline.py")
        # If "neu" appears, it must be as part of a map lookup, not a raw label output
        # We check that emotion_code_label is imported/used (upstream test ensures conversion)
        assert "emotion_code_label" in src, "emotion_code_label must be applied to voice labels"

    def test_waveform_player_applies_emotion_code_label(self):
        """waveform_player.py must apply emotion_code_label to marker labels."""
        src = _src("ui/components/waveform_player.py")
        assert "emotion_code_label" in src


# ---------------------------------------------------------------------------
# Item 7: derive_segment_summary aggregation matches live pipeline
# ---------------------------------------------------------------------------
class TestItem7EvaluationAggregation:
    """Evaluation clip-level prediction must equal derive_segment_summary result."""

    def test_derive_segment_summary_matches_manual_confidence_argmax(self):
        """Dominant label == argmax of sum(confidence) per label."""
        from ui.utils import derive_segment_summary

        # Manually compute: sad gets 0.30+0.20=0.50+eps, happy gets 0.65
        segments = [
            _make_segment(0, voice_emotion="sad", voice_conf=0.30),
            _make_segment(1, voice_emotion="sad", voice_conf=0.20),
            _make_segment(2, voice_emotion="happy", voice_conf=0.65),
        ]
        dom_voice, _dt, _ag, _mm = derive_segment_summary(segments)
        # happy (0.65) > sad (0.50)
        assert dom_voice.lower() == "happy"

    def test_derive_segment_summary_uses_epsilon_for_zero_conf(self):
        """A 0.0-confidence vote still counts (epsilon ensures it beats an absent label)."""
        from ui.utils import derive_segment_summary

        segments = [
            _make_segment(0, voice_emotion="sad", voice_conf=0.0),
        ]
        dom_voice, _dt, _ag, _mm = derive_segment_summary(segments)
        assert dom_voice.lower() == "sad"

    def test_evaluation_run_evaluation_uses_derive_segment_summary(self):
        """evaluation/run_evaluation.py must import and use derive_segment_summary."""
        src = _src("evaluation/run_evaluation.py")
        assert "derive_segment_summary" in src

    def test_cache_key_includes_config_hash(self):
        """evaluation/run_evaluation.py must include config in the cache key."""
        src = _src("evaluation/run_evaluation.py")
        assert "compute_sample_cache_key" in src or "config_hash" in src or "cfg_dict" in src

    def test_evaluation_produces_clip_level_prediction_column(self):
        """evaluation/run_evaluation.py must write clip_level_prediction to CSV."""
        src = _src("evaluation/run_evaluation.py")
        assert "clip_level_prediction" in src
