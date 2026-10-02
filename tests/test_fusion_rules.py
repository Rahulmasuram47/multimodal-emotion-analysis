"""
Comprehensive tests for all declarative multimodal fusion rules, edge cases,
and dynamic weight renormalizations (Phase 5 & Prompt G).
"""

import pytest
import config
from app.fusion import MultimodalFusion


class TestDeclarativeFusionRules:
    """Test every specific rule ID in _evaluate_pattern_rules."""

    @pytest.fixture
    def fusion(self):
        return MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=0.6,
            abstention_threshold=0.35,
        )

    def test_rule_agreement_negative_high_arousal(self, fusion):
        # Angry audio (arousal >= 0.6) + negative text
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "agreement_negative_high_arousal" in res["rule_ids_fired"]
        assert "high-arousal negative pattern" in res["pattern_label"]
        assert res["agreement"] is True
        assert res["mismatch"] is False

    def test_rule_agreement_negative_low_arousal(self, fusion):
        # Sad audio (arousal < 0.6) + negative text
        audio = {"emotion": "sad", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "agreement_negative_low_arousal" in res["rule_ids_fired"]
        assert "low-arousal negative pattern" in res["pattern_label"]
        assert res["agreement"] is True
        assert res["mismatch"] is False

    def test_rule_agreement_positive(self, fusion):
        # Happy audio + positive text
        audio = {"emotion": "happy", "confidence": 0.90, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.85, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "agreement_positive" in res["rule_ids_fired"]
        assert "positive pattern" in res["pattern_label"]
        assert res["agreement"] is True

    def test_rule_agreement_neutral(self, fusion):
        # Neutral audio + neutral text
        audio = {"emotion": "neutral", "confidence": 0.75, "voice_reliability": 1.0}
        text = {"sentiment": "neutral", "confidence": 0.70, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "agreement_neutral" in res["rule_ids_fired"]
        assert "neutral pattern" in res["pattern_label"]
        assert res["agreement"] is True

    def test_rule_mismatch_positive_text_negative_voice(self, fusion):
        # Angry audio + positive text (both high confidence)
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "mismatch_positive_text_negative_voice" in res["rule_ids_fired"]
        assert "incongruence: positive text, negative-toned voice" in res["pattern_label"]
        assert res["mismatch"] is True
        assert res["agreement"] is False

    def test_rule_mismatch_negative_text_positive_voice(self, fusion):
        # Happy audio + negative text (both high confidence)
        audio = {"emotion": "happy", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "mismatch_negative_text_positive_voice" in res["rule_ids_fired"]
        assert "incongruence: negative text, positive-toned voice" in res["pattern_label"]
        assert res["mismatch"] is True
        assert res["agreement"] is False

    def test_rule_mismatch_generic(self, fusion):
        # Neutral audio + positive text (both high confidence >= 0.6)
        audio = {"emotion": "neutral", "confidence": 0.75, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.70, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "mismatch_generic" in res["rule_ids_fired"]
        assert "incongruence: directional mismatch" in res["pattern_label"]
        assert res["mismatch"] is True

    def test_rule_weak_disagreement(self, fusion):
        # Audio angry (0.85) + text positive (0.45 < 0.60 threshold)
        audio = {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0}
        text = {"sentiment": "positive", "confidence": 0.45, "text_reliability": 1.0}
        res = fusion.fuse(audio, text)

        assert "weak_disagreement" in res["rule_ids_fired"]
        assert "weak or low-confidence signal" in res["pattern_label"]
        assert res["mismatch"] is False
        assert res["agreement"] is False

    def test_rule_abstention_on_degraded_inputs(self, fusion):
        # Both voice and text severely degraded
        audio = {"emotion": "neutral", "confidence": 0.50, "voice_reliability": 0.15}
        text = {"sentiment": "neutral", "confidence": 0.50, "text_reliability": 0.20}
        res = fusion.fuse(audio, text)

        assert res["abstention"] is True
        assert "abstention_low_reliability" in res["rule_ids_fired"]
        assert res["pattern_label"] == "Insufficient reliability. Review audio"


class TestDynamicWeightRenormalization:
    """Verify reliability-aware proportional weight adjustment."""

    def test_weight_shift_when_voice_reliability_drops(self):
        fusion = MultimodalFusion(audio_weight=0.5, text_weight=0.5)
        audio = {"emotion": "happy", "confidence": 0.80, "voice_reliability": 0.50}
        text = {"sentiment": "happy", "confidence": 0.80, "text_reliability": 1.00}
        res = fusion.fuse(audio, text)

        # Raw weights: 0.5 * 0.50 = 0.25; 0.5 * 1.0 = 0.50. Total = 0.75.
        # Normalized: audio = 0.25 / 0.75 = 0.3333, text = 0.50 / 0.75 = 0.6667
        eff = res["effective_weights"]
        assert pytest.approx(eff["audio_weight"], abs=0.01) == 0.33
        assert pytest.approx(eff["text_weight"], abs=0.01) == 0.67

    def test_weight_fallback_when_both_reliabilities_are_zero(self):
        fusion = MultimodalFusion(audio_weight=0.4, text_weight=0.6)
        eff_audio, eff_text = fusion._effective_weights(audio_reliability=0.0, text_reliability=0.0)
        # Should fallback safely to base weights without division by zero

        assert pytest.approx(eff_audio, abs=0.001) == 0.40
        assert pytest.approx(eff_text, abs=0.001) == 0.60
