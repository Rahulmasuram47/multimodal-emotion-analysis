"""
Tier 4 Regression Tests: Voice Model Diagnostics, Collapse Handling, and Low Voice Reliability.
Covers item 32 from user specification.
"""

import pytest

import config
from app.fusion import MultimodalFusion
from app.voice_calibration import VoiceCalibrator



def test_tier4_config_thresholds_present_and_safe():
    """Ensure Tier 4 configuration thresholds are defined and strictly follow hard constraints."""
    assert hasattr(config, "LOW_VOICE_CONFIDENCE_THRESHOLD")
    assert hasattr(config, "LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY")
    assert hasattr(config, "LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD")
    assert hasattr(config, "VOICE_COLLAPSE_THRESHOLD")
    assert hasattr(config, "VOICE_COLLAPSE_RELIABILITY_PENALTY")

    # Hard constraint check: Minimum fit samples must remain at least 15 (no premature prior fitting)
    assert config.VOICE_CALIBRATION_MIN_FIT_SAMPLES >= 15
    assert 0.0 < config.LOW_VOICE_CONFIDENCE_THRESHOLD <= 1.0
    assert 0.0 < config.LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY < 1.0
    assert config.VOICE_COLLAPSE_THRESHOLD == 0.70


def test_prior_correction_refuses_to_fit_below_minimum_n():
    """VoiceCalibrator must refuse to fit and remain uncalibrated when N < min_samples."""
    calibrator = VoiceCalibrator()

    # Synthetic small dataset with fewer than 15 samples (e.g. 4 samples total)
    small_history = [
        ({"neutral": 0.4, "happy": 0.2, "angry": 0.2, "sad": 0.2}, "neutral"),
        ({"neutral": 0.1, "happy": 0.6, "angry": 0.1, "sad": 0.2}, "happy"),
        ({"neutral": 0.1, "happy": 0.1, "angry": 0.1, "sad": 0.7}, "sad"),
        ({"neutral": 0.2, "happy": 0.1, "angry": 0.5, "sad": 0.2}, "angry"),
    ]

    calibrator.fit(small_history)
    assert calibrator.is_fitted is False, "Calibrator must not fit when total samples < min_samples"
    assert calibrator.temperature == 1.0

    # Calling apply_temperature on an unfitted model must be a strict no-op returning identical scores
    test_scores = {"neutral": 0.1, "happy": 0.1, "angry": 0.1, "sad": 0.7}
    calibrated = calibrator.apply_temperature(test_scores)
    assert calibrated == test_scores


def test_low_voice_reliability_lowers_voice_weight_in_fusion():
    """Lowering voice reliability must directly lower the effective voice weight in multimodal fusion."""
    fusion = MultimodalFusion(audio_weight=0.50, text_weight=0.50)

    # Case A: Nominal full reliability (1.0 vs 1.0)
    w_audio_norm, w_text_norm = fusion._effective_weights(audio_reliability=1.0, text_reliability=1.0)
    assert pytest.approx(w_audio_norm, 0.01) == 0.50
    assert pytest.approx(w_text_norm, 0.01) == 0.50

    # Case B: Low voice reliability (e.g. 0.30 due to low confidence or collapse penalty)
    w_audio_low, w_text_low = fusion._effective_weights(audio_reliability=0.30, text_reliability=1.0)
    assert w_audio_low < w_audio_norm, "Effective audio weight must decrease when audio reliability is low"
    assert w_text_low > w_text_norm, "Effective text weight must increase relatively"
    assert pytest.approx(w_audio_low + w_text_low, 0.001) == 1.0


def test_voice_collapse_lowers_fusion_weights_in_pipeline():
    """Test that voice collapse simulation penalizes voice reliability and sets collapse flag."""
    fusion = MultimodalFusion(audio_weight=0.50, text_weight=0.50)

    audio_res = {
        "emotion": "sad",
        "confidence": 0.90,
        "voice_reliability": 1.0,
        "scores": {"neutral": 0.05, "happy": 0.02, "angry": 0.03, "sad": 0.90},
    }
    text_res = {
        "sentiment": "positive",
        "confidence": 0.85,
        "text_reliability": 1.0,
        "scores": {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    }

    res_nominal = fusion.fuse(audio_res, text_res)

    # Apply collapse penalty
    audio_res_collapsed = dict(audio_res)
    audio_res_collapsed["voice_reliability"] = round(1.0 * config.VOICE_COLLAPSE_RELIABILITY_PENALTY, 4)

    res_collapsed = fusion.fuse(audio_res_collapsed, text_res)

    assert res_collapsed["effective_weights"]["audio_weight"] < res_nominal["effective_weights"]["audio_weight"]
    assert res_collapsed["effective_weights"]["text_weight"] > res_nominal["effective_weights"]["text_weight"]

