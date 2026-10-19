"""Phase 12b: Tests for post-calibration confidence capping on uncalibrated voice backends."""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.conversation_analysis import ConversationAnalyzer
import config


def test_saturated_scores_emotion2vec_capped_at_voice_conf_cap():
    """Saturated scores (0.999) through _analyze_segment with mocked emotion2vec end with confidence <= cap."""
    analyzer = ConversationAnalyzer(whisper_model_size="base")
    mock_voice = MagicMock()
    mock_voice.confidence_informative = False

    # Return saturated scores
    mock_voice.analyze_segment.return_value = {
        "emotion": "happy",
        "confidence": 0.999,
        "raw_confidence": 0.999,
        "scores": {"happy": 0.999, "neutral": 0.001, "angry": 0.0, "sad": 0.0},
    }
    analyzer.emotion_analyzer = mock_voice

    dummy_audio = np.zeros(32000, dtype=np.float32)
    bounds = [(0.0, 2.0)]
    raw_segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "Everything is wonderful and great today.",
        "speaker": "Speaker A",
    }
    seg = analyzer._analyze_segment(
        audio=dummy_audio,
        sr=16000,
        raw_segment=raw_segment,
        index=0,
        all_segment_bounds=bounds,
        audio_duration_seconds=2.0,
    )

    assert seg is not None
    voice = seg["voice_emotion"]
    assert voice["confidence"] <= config.VOICE_CONF_CAP
    assert voice["confidence"] == config.VOICE_CONF_CAP
    assert voice["confidence_capped"] is True
    assert voice["raw_confidence"] == 0.999


def test_superb_path_confidence_uncapped():
    """SUPERB path (confidence_informative == True) leaves confidence uncapped."""
    analyzer = ConversationAnalyzer(whisper_model_size="base")
    mock_voice = MagicMock()
    mock_voice.confidence_informative = True

    mock_voice.analyze_segment.return_value = {
        "emotion": "happy",
        "confidence": 0.95,
        "raw_confidence": 0.95,
        "scores": {"happy": 0.95, "neutral": 0.05, "angry": 0.0, "sad": 0.0},
    }
    analyzer.emotion_analyzer = mock_voice

    dummy_audio = np.zeros(32000, dtype=np.float32)
    bounds = [(0.0, 2.0)]
    raw_segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "Everything is wonderful and great today.",
        "speaker": "Speaker A",
    }
    seg = analyzer._analyze_segment(
        audio=dummy_audio,
        sr=16000,
        raw_segment=raw_segment,
        index=0,
        all_segment_bounds=bounds,
        audio_duration_seconds=2.0,
    )

    assert seg is not None
    voice = seg["voice_emotion"]
    assert voice["confidence"] == 0.95
    assert voice["confidence_capped"] is False


def test_uncertain_class_does_not_crash():
    """Uncertain class from other-group does not crash _analyze_segment with capping."""
    analyzer = ConversationAnalyzer(whisper_model_size="base")
    mock_voice = MagicMock()
    mock_voice.confidence_informative = False

    mock_voice.analyze_segment.return_value = {
        "emotion": "uncertain",
        "leaning": "fearful",
        "confidence": 0.30,
        "raw_confidence": 0.85,
        "scores": {"neutral": 0.25, "happy": 0.25, "angry": 0.25, "sad": 0.25},
    }
    analyzer.emotion_analyzer = mock_voice

    dummy_audio = np.zeros(32000, dtype=np.float32)
    bounds = [(0.0, 2.0)]
    raw_segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "I am not sure what is going on.",
        "speaker": "Speaker A",
    }
    seg = analyzer._analyze_segment(
        audio=dummy_audio,
        sr=16000,
        raw_segment=raw_segment,
        index=0,
        all_segment_bounds=bounds,
        audio_duration_seconds=2.0,
    )

    assert seg is not None
    voice = seg["voice_emotion"]
    assert voice["emotion"] == "uncertain"
    assert voice["confidence"] == 0.30
    assert voice["confidence_capped"] is False
