"""
Unit and regression tests for Tier 2: Correctness of Numbers (Items 20–27).

Follows AGENTS.md: Never fabricate data. All test data synthetic and marked test-only.
"""

import math
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pytest

import config
from app.conversation_analysis import ConversationAnalyzer, set_seed
from app.fusion import MultimodalFusion
from ui.charts.dynamics import render_valence_arousal_map
from ui.components.transparency_panels import (
    render_fusion_explainer,
    render_model_transparency_section,
    render_professional_limitations_section,
    _has_ground_truth_eval_results,
)


ROOT = Path(__file__).resolve().parents[1]


def _src(rel_path: str) -> str:
    return (ROOT / rel_path).read_text(encoding="utf-8")


# -----------------------------------------------------------------------------
# Item 20: Valence-Arousal Map Fusion, Tooltip, and Non-overlapping Labels
# -----------------------------------------------------------------------------
class TestItem20ValenceArousal:
    def test_valence_fuses_voice_and_text(self):
        """Valence must fuse both voice and text modalities via effective weights."""
        src = _src("ui/charts/dynamics.py")
        assert "sentiment_valence_map" in src
        assert "voice_valence_map" in src
        assert "w_audio * v_voice + w_text * v_text" in src or "fused_v" in src

    def test_non_overlapping_label_positions(self):
        """Label positions must alternate across turns to prevent text collision."""
        src = _src("ui/charts/dynamics.py")
        assert "label_positions" in src
        assert "text_positions = [label_positions" in src

    def test_valence_arousal_formula_tooltip(self):
        """A formula explanation caption/tooltip must be displayed above the chart."""
        src = _src("ui/charts/dynamics.py")
        assert "Affective Coordinates Formula" in src
        assert "Valence ($X$)" in src
        assert "Arousal ($Y$)" in src



# -----------------------------------------------------------------------------
# Item 21: Combined Confidence & No Weak Disagreement Boost
# -----------------------------------------------------------------------------
class TestItem21CombinedConfidence:
    def test_combined_confidence_bounded_by_modalities(self):
        """Combined confidence is a convex combination and must never exceed max(audio, text)."""
        fusion = MultimodalFusion(audio_weight=0.6, text_weight=0.4)

        # Agreement case
        res_agree = fusion.fuse(
            {"emotion": "happy", "confidence": 0.85},
            {"sentiment": "positive", "confidence": 0.65},
        )
        assert 0.65 <= res_agree["combined_confidence"] <= 0.85
        assert abs(res_agree["combined_confidence"] - (0.6 * 0.85 + 0.4 * 0.65)) < 1e-4

        # Weak disagreement case: audio negative (0.80), text positive (0.30)
        res_weak = fusion.fuse(
            {"emotion": "angry", "confidence": 0.80},
            {"sentiment": "positive", "confidence": 0.30},
        )
        assert res_weak["agreement"] is False
        # Must strictly be within [0.30, 0.80] and no agreement bonus
        assert 0.30 <= res_weak["combined_confidence"] <= 0.80
        assert res_weak["combined_confidence"] == pytest.approx(0.6 * 0.80 + 0.4 * 0.30, abs=1e-4)

    def test_no_false_agreement_bonus_in_transparency(self):
        """Transparency panel must not claim an artificial 'agreement bonus'."""
        src = _src("ui/components/transparency_panels.py")
        assert "agreement bonus" not in src.lower()


# -----------------------------------------------------------------------------
# Item 22: Intensity v2 Component Exposure
# -----------------------------------------------------------------------------
class TestItem22IntensityV2:
    def test_intensity_v2_weights_sum_to_one(self):
        """Intensity v2 weights must sum to 1.0 (0.35 + 0.35 + 0.30)."""
        w_prosody = config.INTENSITY_V2_WEIGHT_PROSODY
        w_voice = config.INTENSITY_V2_WEIGHT_VOICE_AROUSAL
        w_text = config.INTENSITY_V2_WEIGHT_TEXT_NEGATIVITY
        assert abs((w_prosody + w_voice + w_text) - 1.0) < 1e-6
        assert w_prosody == 0.35
        assert w_voice == 0.35
        assert w_text == 0.30

    def test_intensity_components_rendered_in_segment_timeline(self):
        """Segment timeline must render the Intensity v2 components breakdown."""
        src = _src("ui/components/segment_timeline.py")
        assert "Intensity v2:" in src
        assert "Prosody (w=" in src
        assert "Expected Arousal (Intensity v2, w=" in src
        assert "Text Negativity (w=" in src


# -----------------------------------------------------------------------------
# Item 23: Determinism & Seeding
# -----------------------------------------------------------------------------
class TestItem23Determinism:
    def test_set_seed_exists_and_seeds_all_frameworks(self):
        """set_seed must seed random, numpy, and torch."""
        src = _src("app/conversation_analysis.py")
        assert "def set_seed(seed: int = 42)" in src
        assert "random.seed(seed)" in src
        assert "np.random.seed(seed)" in src
        assert "torch.manual_seed(seed)" in src

    def test_speech_to_text_has_deterministic_params(self):
        """SpeechToText transcribe must set temperature=0.0 and condition_on_previous_text=False."""
        src = _src("app/speech_to_text.py")
        assert "temperature=0.0" in src
        assert "condition_on_previous_text=False" in src

    def test_repeatability_on_synthetic_features(self):
        """Repeated calls with set_seed produce identical outputs."""
        set_seed(42)
        arr1 = np.random.randn(10, 10)
        set_seed(42)
        arr2 = np.random.randn(10, 10)
        np.testing.assert_array_equal(arr1, arr2)


# -----------------------------------------------------------------------------
# Item 24: Explainable Multimodal Fusion Architecture
# -----------------------------------------------------------------------------
class TestItem24FusionTransparency:
    def test_transparency_uses_actual_rule_ids(self):
        """Transparency explainer must list actual rule IDs from app/fusion.py."""
        src = _src("ui/components/transparency_panels.py")
        actual_rule_ids = [
            "agreement_negative_high_arousal",
            "agreement_negative_low_arousal",
            "agreement_positive",
            "agreement_neutral",
            "mismatch_positive_text_negative_voice",
            "mismatch_negative_text_positive_voice",
            "mismatch_generic",
            "weak_disagreement",
            "abstention_low_reliability",
        ]
        for rule_id in actual_rule_ids:
            assert rule_id in src, f"Expected rule ID {rule_id} in transparency panels"

    def test_unsupported_claims_removed(self):
        """Unsupported claims like Asymmetric Modality Dominance or Neutral Mediation must be absent."""
        src = _src("ui/components/transparency_panels.py")
        assert "Asymmetric Modality Dominance" not in src
        assert "Neutral Mediation" not in src


# -----------------------------------------------------------------------------
# Item 25: Model Table Derived from Configuration
# -----------------------------------------------------------------------------
class TestItem25ModelTable:
    def test_config_has_diarization_model_name(self):
        """config.py must declare DIARIZATION_MODEL_NAME."""
        assert hasattr(config, "DIARIZATION_MODEL_NAME")
        assert config.DIARIZATION_MODEL_NAME == "pyannote/speaker-diarization-community-1"

    def test_model_table_derives_from_config(self):
        """Model table must read models dynamically from config."""
        src = _src("ui/components/transparency_panels.py")
        assert "AUDIO_EMOTION_MODEL_NAME" in src
        assert "TEXT_SENTIMENT_MODEL_NAME" in src
        assert "DIARIZATION_MODEL_NAME" in src
        assert "Pyannote Community-1" in src


# -----------------------------------------------------------------------------
# Item 26: Conditional Limitations Statement
# -----------------------------------------------------------------------------
class TestItem26ConditionalLimitations:
    def test_limitations_conditional_on_evaluation(self):
        """Limitations section must conditionally display evaluation status."""
        src = _src("ui/components/transparency_panels.py")
        assert "_has_ground_truth_eval_results()" in src
        assert "_build_eval_summary()" in src
        # Phase 15: stale sentence replaced with dynamic eval summary
        assert "No local evaluation results yet" not in src


# -----------------------------------------------------------------------------
# Item 27: MFCC Chart (MFCC 2–13, mean ± std, educational caption)
# -----------------------------------------------------------------------------
class TestItem27MFCCChart:
    def test_mfcc_plots_coefficients_2_to_13(self):
        """MFCC chart must plot MFCC 2–13, skipping index 0 (MFCC 1)."""
        src = _src("ui/charts/inspection.py")
        assert "mfcc_mean[1:]" in src
        assert "mfcc_std[1:]" in src
        assert "MFCC 2–13" in src

    def test_mfcc_label_mean_std(self):
        """MFCC series must be labeled 'mean ± std across frames'."""
        src = _src("ui/charts/inspection.py")
        assert "mean ± std across frames" in src

    def test_mfcc_plain_english_explanation(self):
        """MFCC chart must include educational explanation without emotion claims."""
        src = _src("ui/charts/inspection.py")
        assert "spectral envelope" in src
        assert "not discrete emotional categories directly" in src

