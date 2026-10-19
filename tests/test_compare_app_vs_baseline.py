"""Tests for evaluation/compare_app_vs_baseline.py."""
import pytest
from unittest.mock import MagicMock, patch

from evaluation.compare_app_vs_baseline import (
    parse_true_label,
    run_app_voice_whole_clip,
    run_comparison,
)
import config


def test_filename_parser_toy_and_crema_names():
    """Verify filename parser handles both CREMA-D naming conventions and toy names."""
    # CREMA-D patterns
    assert parse_true_label("1001_DFA_ANG_XX.wav") == "angry"
    assert parse_true_label("1001_IEO_HAP_HI.wav") == "happy"
    assert parse_true_label("1002_TSI_SAD_MD.wav") == "sad"
    assert parse_true_label("1003_ITH_NEU_XX.wav") == "neutral"
    assert parse_true_label("1004_IEO_DIS_LO.wav") == "disgust"
    assert parse_true_label("1005_IEO_FEA_HI.wav") == "fear"

    # Toy names
    assert parse_true_label("Sad1.wav") == "sad"
    assert parse_true_label("Happy2.wav") == "happy"
    assert parse_true_label("Neutral1.wav") == "neutral"
    assert parse_true_label("angry_sample.wav") == "angry"
    assert parse_true_label("rising_tone.wav") == "rising"


def test_script_and_app_use_same_backend_and_confidence():
    """Script and app use the same backend and confidence fields."""
    mock_analyzer = MagicMock()
    mock_analyzer.emotion_analyzer.backend_used = "emotion2vec"
    mock_analyzer.emotion_analyzer.backend_info.return_value = {"backend_used": "emotion2vec"}
    mock_analyzer.emotion_analyzer.confidence_informative = False
    mock_analyzer.emotion_analyzer.analyze.return_value = {
        "emotion": "happy",
        "confidence": 0.98,
        "raw_confidence": 0.98,
    }

    with patch("evaluation.compare_app_vs_baseline.load_and_preprocess_audio", return_value=(None, 16000, 2.0)):
        res = run_app_voice_whole_clip(mock_analyzer, "dummy.wav")

    assert res["backend_used"] == "emotion2vec"
    assert res["top_label"] == "happy"
    # Saturated confidence on uninformative backend clamped to VOICE_CONF_CAP
    assert res["final_confidence"] == config.VOICE_CONF_CAP
    assert res["raw_confidence"] == 0.98
