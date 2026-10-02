"""
Comprehensive tests for Prompt G Counterfactual Sensitivity:
  - Segment-level rule sensitivity (text confidence crossing, tone alignment, abstention buffer)
  - Conversation trajectory verdict sensitivity (dip tolerance, net rise, slope perturbations)
  - Invariance and side-effect-free evaluation
"""

import pytest
import config
from app.fusion import MultimodalFusion
from app.conversation_analysis import ConversationAnalyzer


class TestSegmentCounterfactuals:
    """Verify segment-level counterfactual sensitivity generation."""

    @pytest.fixture
    def fusion(self):
        return MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=0.60,
            abstention_threshold=0.35,
        )

    def test_counterfactual_when_vocal_tone_mismatches_positive_text(self, fusion):
        """When voice is angry and text is positive, reports what happens if tone aligned."""
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert any("If vocal tone aligned with text sentiment (positive)" in c for c in cfs)
        assert any("Abstention buffer" in c for c in cfs)

    def test_counterfactual_when_vocal_tone_matches_negative_text(self, fusion):
        """When both are negative, reports what happens if text shifted to opposite direction."""
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert any("If text sentiment shifted from negative to positive" in c for c in cfs)

    def test_counterfactual_softens_when_text_confidence_falls(self, fusion):
        """When text confidence is currently high, checks sensitivity to dipping below threshold."""
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.75, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert any("If text confidence fell below 0.60 (currently 0.75)" in c for c in cfs)

    def test_counterfactual_sub_abstention_buffer_message(self, fusion):
        """When below abstention threshold, message indicates the required rise to produce classification."""
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 0.10}
        text = {"sentiment": "positive", "confidence": 0.80, "text_reliability": 0.10}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert res["abstention"] is True
        assert any("If combined reliability rose above 0.35" in c for c in cfs)

    def test_counterfactual_above_abstention_buffer_message(self, fusion):
        """When above abstention threshold, reports the exact margin above cut-off."""
        audio = {"emotion": "happy", "confidence": 0.85, "voice_reliability": 0.85}
        text = {"sentiment": "positive", "confidence": 0.80, "text_reliability": 0.85}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert res["abstention"] is False
        assert any("Abstention buffer: combined reliability (0.85) is 0.50 above" in c for c in cfs)

    def test_counterfactual_high_arousal_angry_tone(self, fusion):
        """Arousal correctly reflects in counterfactual pattern naming."""
        audio = {"emotion": "angry", "confidence": 0.80, "voice_reliability": 0.9}
        text = {"sentiment": "positive", "confidence": 0.50, "text_reliability": 0.9}
        res = fusion.fuse(audio, text)

        cfs = res["counterfactuals"]
        assert any("pattern would flip from" in c for c in cfs)

    def test_counterfactual_evaluations_are_deterministic_and_side_effect_free(self, fusion):
        """Calling compute_counterfactuals multiple times does not mutate inputs."""
        audio = {"emotion": "sad", "confidence": 0.70, "voice_reliability": 0.8}
        text = {"sentiment": "positive", "confidence": 0.65, "text_reliability": 0.8}

        res1 = fusion.fuse(audio, text)
        res2 = fusion.fuse(audio, text)

        assert res1["counterfactuals"] == res2["counterfactuals"]
        assert audio["confidence"] == 0.70
        assert text["confidence"] == 0.65


class TestTrajectoryVerdictCounterfactuals:
    """Verify conversation-level verdict counterfactual sensitivity."""

    def test_trajectory_counterfactual_escalating_dip_tolerance(self):
        """Escalating verdict notes dip tolerance boundary when dips occur."""
        # 4 segments with a minor dip within tolerance
        # Net rise: 0.70 - 0.20 = 0.50; dip: 0.40 - 0.35 = 0.05 <= 0.08
        intensities = [0.20, 0.40, 0.35, 0.70]
        res = ConversationAnalyzer.compute_unified_verdict([{"intensity": i} for i in intensities])

        assert res["verdict"] == "escalating"
        cfs = res["counterfactuals"]
        assert any("Verdict flips from escalating if dip tolerance falls below" in c for c in cfs)

    def test_trajectory_counterfactual_escalating_near_net_rise_threshold(self):
        """Escalating verdict notes sensitivity if net rise is close to minimum."""
        # Net rise: 0.22 (just above 0.20 threshold by 0.02 < 0.10)
        intensities = [0.20, 0.28, 0.36, 0.42]
        res = ConversationAnalyzer.compute_unified_verdict([{"intensity": i} for i in intensities])


        assert res["verdict"] == "escalating"
        cfs = res["counterfactuals"]
        assert any("If net intensity rise fell below" in c for c in cfs)

    def test_trajectory_counterfactual_deescalating_spike_tolerance(self):
        """De-escalating verdict notes sensitivity if small upward blips exist."""
        # Net drop: 0.20 - 0.70 = -0.50; spike: 0.55 - 0.50 = 0.05 <= 0.08
        intensities = [0.70, 0.50, 0.55, 0.20]
        res = ConversationAnalyzer.compute_unified_verdict([{"intensity": i} for i in intensities])

        assert res["verdict"] == "de-escalating"
        cfs = res["counterfactuals"]
        assert any("Verdict flips from de-escalating if upward spike tolerance falls below" in c for c in cfs)

    def test_trajectory_counterfactual_fluctuating_swing_tolerance(self):
        """Fluctuating verdict notes swing tolerance requirement for stability."""
        intensities = [0.20, 0.80, 0.20, 0.80]
        res = ConversationAnalyzer.compute_unified_verdict([{"intensity": i} for i in intensities])

        assert res["verdict"] == "fluctuating"
        cfs = res["counterfactuals"]
        assert any("Verdict would become stable if largest swing" in c for c in cfs)

    def test_trajectory_counterfactual_stable_slope_increase_requirement(self):
        """Stable verdict reports slope increase required to become escalating."""
        intensities = [0.40, 0.41, 0.40, 0.42]
        res = ConversationAnalyzer.compute_unified_verdict([{"intensity": i} for i in intensities])

        assert res["verdict"] == "stable"
        cfs = res["counterfactuals"]
        assert any("Verdict would become escalating if per-turn slope increased by" in c for c in cfs)
