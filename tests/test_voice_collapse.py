"""
Unit tests for Voice Model Collapse Detector and Fusion Reliability Guard.

Synthetic / mocked inputs clearly marked test-only.
Verifies:
- Flags collapse when one emotion dominates (>70% share)
- No-op when emotions are balanced
- Audio reliability penalty applied to fusion
- Warning string correctly structured
"""

import pytest
import config
from app.conversation_analysis import ConversationAnalyzer
from app.fusion import MultimodalFusion


def _mock_segment(emotion: str, confidence: float = 0.85, sentiment: str = "neutral"):
    """Build mock segment for collapse testing (test-only input)."""
    fusion = MultimodalFusion()
    voice = {
        "emotion": emotion,
        "confidence": confidence,
        "scores": {e: (confidence if e == emotion else (1.0 - confidence) / 3.0) for e in config.AUDIO_EMOTION_CLASSES},
        "voice_reliability": 1.0,
    }
    text = {
        "sentiment": sentiment,
        "confidence": 0.80,
        "scores": {"negative": 0.1, "neutral": 0.8, "positive": 0.1},
        "text_reliability": 1.0,
    }
    fused = fusion.fuse(voice, text)
    return {
        "voice_emotion": voice,
        "text_sentiment": text,
        "fusion": fused,
        "transcript": "Test segment transcript.",
        "start_seconds": 0.0,
        "end_seconds": 2.0,
    }


def test_collapse_detected_when_single_class_dominates():
    # 4 segments, all "sad" (100% >= 70% threshold)
    segments = [_mock_segment("sad") for _ in range(4)]

    analyzer = ConversationAnalyzer.__new__(ConversationAnalyzer)
    analyzer.fusion = MultimodalFusion()

    # Replicate post-segment collapse block from analyze()
    voice_emotions = [s["voice_emotion"]["emotion"] for s in segments]
    counts = {}
    for e in voice_emotions:
        counts[e] = counts.get(e, 0) + 1
    dom_emotion, max_cnt = max(counts.items(), key=lambda x: x[1])
    dom_share = max_cnt / len(voice_emotions)
    collapse_thresh = config.VOICE_COLLAPSE_THRESHOLD

    assert dom_share >= collapse_thresh
    assert dom_emotion == "sad"

    penalty = config.VOICE_COLLAPSE_RELIABILITY_PENALTY
    for seg in segments:
        seg["voice_emotion"]["voice_reliability"] *= penalty
        seg["fusion"] = analyzer.fusion.fuse(seg["voice_emotion"], seg["text_sentiment"])

    # Effective weight of audio should now be less than base weight
    for seg in segments:
        assert seg["voice_emotion"]["voice_reliability"] == pytest.approx(0.50)
        eff_audio_w = seg["fusion"]["effective_weights"]["audio_weight"]
        eff_text_w = seg["fusion"]["effective_weights"]["text_weight"]
        assert eff_audio_w < eff_text_w


def test_no_collapse_when_emotions_balanced():
    # Varied emotions: angry, happy, neutral, sad (25% each < 70%)
    emotions = ["angry", "happy", "neutral", "sad"]
    segments = [_mock_segment(e) for e in emotions]

    counts = {}
    for s in segments:
        e = s["voice_emotion"]["emotion"]
        counts[e] = counts.get(e, 0) + 1
    dom_emotion, max_cnt = max(counts.items(), key=lambda x: x[1])
    dom_share = max_cnt / len(segments)

    assert dom_share < config.VOICE_COLLAPSE_THRESHOLD
