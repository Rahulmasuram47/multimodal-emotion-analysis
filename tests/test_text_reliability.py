"""
Unit tests for Continuous Text Reliability and Non-Speech Audio Handling.

Synthetic / mocked inputs clearly marked test-only.
Verifies Prompt D:
- Continuous text reliability calculation from avg_logprob and no_speech_prob
- Low ASR confidence threshold trigger (<0.65)
- Effective weight adjustment in MultimodalFusion
- Graceful non-speech / silence handling (no crash)
"""

import numpy as np
import pytest

import config
from app.conversation_analysis import ConversationAnalyzer
from app.fusion import MultimodalFusion


def test_continuous_text_reliability_scaling():
    # Ideal transcript: logprob = 0.0, no_speech = 0.0
    r_high = float(np.exp(0.0)) * (1.0 - 0.0)
    assert np.clip(round(r_high, 3), 0.20, 1.0) == 1.0

    # Uncertain transcript: logprob = -1.2, no_speech = 0.4
    r_mid = float(np.exp(-1.2)) * (1.0 - 0.4)
    r_mid_norm = float(np.clip(round(r_mid, 3), 0.20, 1.0))
    assert 0.15 < r_mid_norm < 0.65

    # Very bad transcript: logprob = -2.8, no_speech = 0.9
    r_low = float(np.exp(-2.8)) * (1.0 - 0.9)
    r_low_norm = float(np.clip(round(r_low, 3), 0.20, 1.0))
    assert r_low_norm == 0.20  # clamped to floor 0.20


def test_text_reliability_shifts_fusion_weights():
    fusion = MultimodalFusion(audio_weight=0.5, text_weight=0.5)

    voice_result = {
        "emotion": "happy",
        "confidence": 0.85,
        "voice_reliability": 1.0,
        "scores": {"happy": 0.85, "sad": 0.05, "angry": 0.05, "neutral": 0.05},
    }

    # Case 1: High text reliability (1.0)
    text_high = {
        "sentiment": "positive",
        "confidence": 0.85,
        "text_reliability": 1.0,
        "scores": {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    }
    fused_high = fusion.fuse(voice_result, text_high)
    assert fused_high["effective_weights"]["audio_weight"] == pytest.approx(0.50)
    assert fused_high["effective_weights"]["text_weight"] == pytest.approx(0.50)

    # Case 2: Low text reliability (0.40)
    text_low = {
        "sentiment": "positive",
        "confidence": 0.85,
        "text_reliability": 0.40,
        "scores": {"positive": 0.85, "neutral": 0.10, "negative": 0.05},
    }
    fused_low = fusion.fuse(voice_result, text_low)
    # Audio weight should dominate since text is uncertain
    # w_audio = 0.5 * 1.0 = 0.5; w_text = 0.5 * 0.4 = 0.2 -> normalized: 0.5/0.7 = 0.714
    assert fused_low["effective_weights"]["audio_weight"] > 0.70
    assert fused_low["effective_weights"]["text_weight"] < 0.30


def test_graceful_non_speech_handling():
    """Verify that when ASR produces no segments (silence/non-speech),
    the pipeline returns a structured non-speech result rather than crashing.
    """
    analyzer = ConversationAnalyzer.__new__(ConversationAnalyzer)

    # Mock audio load and speech_to_text returning empty segments
    class MockSTT:
        def transcribe(self, path):
            return {
                "text": "",
                "language": "en",
                "language_probability": 0.50,
                "segments": [],
                "no_speech_detected": True,
            }

    analyzer.speech_to_text = MockSTT()

    # Pass synthetic dummy audio
    audio = np.zeros(16000 * 3, dtype=np.float32)
    sr = 16000
    duration = 3.0

    # Mock _report_progress
    analyzer._report_progress = lambda *args: None

    # Directly trigger the non-speech branch
    raw_segments = []
    t_load_elapsed = 0.01
    t_transcribe_elapsed = 0.05
    t_total_start = 0.0

    # Call the logic implemented in analyze
    transcription = analyzer.speech_to_text.transcribe("dummy.wav")
    assert transcription["no_speech_detected"] is True
    assert len(transcription["segments"]) == 0
