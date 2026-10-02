"""
Automated tests for conversation-level analysis logic.

Tests:
- Emotional trend detection
- Escalation detection
- Insufficient segment handling
- Emotional intensity calculation
"""


from app.conversation_analysis import ConversationAnalyzer


def make_segment(intensity):
    """Create a minimal segment for trend/escalation testing."""
    return {
        "intensity": intensity,
    }


def make_fusion_segment(
    combined_confidence,
    agreement,
    mismatch,
):
    """Create a minimal segment for intensity testing."""
    return {
        "fusion": {
            "combined_confidence": combined_confidence,
            "agreement": agreement,
            "mismatch": mismatch,
        }
    }


# ---------------------------------------------------------
# Emotional trend tests
# ---------------------------------------------------------


def test_emotional_trend_increasing():
    """Trend should be increasing when intensity rises by >= 0.10."""

    segments = [
        make_segment(0.20),
        make_segment(0.35),
        make_segment(0.50),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "increasing"
    assert result["change"] == 0.30
    assert result["intensity_scores"] == [0.20, 0.35, 0.50]


def test_emotional_trend_decreasing():
    """Trend should be decreasing when intensity falls by >= 0.10."""

    segments = [
        make_segment(0.80),
        make_segment(0.60),
        make_segment(0.40),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "decreasing"
    assert result["change"] == -0.40


def test_emotional_trend_stable():
    """Small changes should be classified as stable."""

    segments = [
        make_segment(0.40),
        make_segment(0.45),
        make_segment(0.43),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "stable"


def test_emotional_trend_with_insufficient_segments():
    """Trend requires at least two segments."""

    segments = [
        make_segment(0.40),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "insufficient_data"
    assert result["intensity_scores"] == []
    assert result["change"] == 0.0


# ---------------------------------------------------------
# Escalation tests
# ---------------------------------------------------------


def test_escalation_fires_on_continuously_rising_intensity():
    """Continuous increase of >= 0.20 should trigger escalation."""

    segments = [
        make_segment(0.20),
        make_segment(0.40),
        make_segment(0.60),
        make_segment(0.80),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is True
    assert result["intensity_change"] == 0.60


def test_no_escalation_when_intensity_is_flat():
    """Flat or fluctuating intensity should not trigger escalation."""

    segments = [
        make_segment(0.30),
        make_segment(0.31),
        make_segment(0.29),
        make_segment(0.30),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is False


def test_no_escalation_when_intensity_is_not_continuously_increasing():
    """A temporary decrease should prevent escalation."""

    segments = [
        make_segment(0.20),
        make_segment(0.50),
        make_segment(0.40),
        make_segment(0.80),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is False


def test_no_escalation_when_less_than_three_segments():
    """Escalation requires at least three segments."""

    segments = [
        make_segment(0.20),
        make_segment(0.50),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is False
    assert result["reason"] == "At least 3 segments are required."


# ---------------------------------------------------------
# Intensity tests
# ---------------------------------------------------------


def test_intensity_increases_when_modalities_agree():
    """Agreement should increase the calculated intensity."""

    segment = make_fusion_segment(
        combined_confidence=0.70,
        agreement=True,
        mismatch=False,
    )

    intensity = ConversationAnalyzer._calculate_intensity(segment)

    assert intensity == 0.80


def test_intensity_is_reduced_for_strong_mismatch():
    """Strong disagreement should reduce calculated intensity."""

    segment = make_fusion_segment(
        combined_confidence=0.80,
        agreement=False,
        mismatch=True,
    )

    intensity = ConversationAnalyzer._calculate_intensity(segment)

    assert intensity == 0.68


def test_intensity_is_reduced_for_low_confidence_disagreement():
    """Weak disagreement should receive a smaller adjustment."""

    segment = make_fusion_segment(
        combined_confidence=0.80,
        agreement=False,
        mismatch=False,
    )

    intensity = ConversationAnalyzer._calculate_intensity(segment)

    assert intensity == 0.72


def test_intensity_is_capped_at_one():
    """Intensity should never exceed 1.0."""

    segment = make_fusion_segment(
        combined_confidence=0.98,
        agreement=True,
        mismatch=False,
    )

    intensity = ConversationAnalyzer._calculate_intensity(segment)

    assert intensity == 1.0


def test_intensity_is_never_negative():
    """Intensity should never be below 0.0."""

    segment = make_fusion_segment(
        combined_confidence=0.0,
        agreement=False,
        mismatch=True,
    )

    intensity = ConversationAnalyzer._calculate_intensity(segment)

    assert intensity == 0.0

# ---------------------------------------------------------------------
# Regression test: sentiment analysis must always run after voice
# calibration — whether calibration succeeds or fails — and the
# segment must not be silently dropped.
#
# Bug being guarded against: sentiment analysis was previously nested
# INSIDE voice calibration's `except` block, with a stray unconditional
# `return None` at the end of the whole calibration `if` block. That
# meant sentiment analysis only ran when calibration FAILED, and
# _analyze_segment returned None unconditionally regardless of outcome
# — so every segment silently vanished whenever calibration succeeded
# (the normal, expected case), producing a generic "No valid
# conversation segments were produced" error with no exception ever
# logged.
# ---------------------------------------------------------------------

import numpy as np
import config
from app.conversation_analysis import ConversationAnalyzer


class _DummyCalibrationResult:
    def __init__(self, emotion="sad", confidence=0.7, scores=None, applied_rules=None):
        self.emotion = emotion
        self.confidence = confidence
        self.scores = scores or {"sad": 0.7, "neutral": 0.2, "angry": 0.05, "happy": 0.05}
        self.applied_rules = applied_rules or []


class _DummyEmotionAnalyzer:
    def analyze_segment(self, audio, sr, start, end):
        return {
            "emotion": "sad",
            "confidence": 0.9,
            "scores": {"sad": 0.9, "neutral": 0.05, "angry": 0.03, "happy": 0.02},
        }


class _DummySentimentAnalyzer:
    def __init__(self):
        self.call_count = 0

    def analyze(self, text):
        self.call_count += 1
        return {
            "sentiment": "negative",
            "confidence": 0.8,
            "scores": {"negative": 0.8, "neutral": 0.15, "positive": 0.05},
        }


class _DummyFusion:
    def fuse(self, voice_result, sentiment_result):
        return {
            "combined_confidence": 0.85,
            "agreement": True,
            "mismatch": False,
            "pattern_label": "test-pattern",
        }


class _DummyVoiceCalibratorSuccess:
    """Simulates calibration succeeding — the normal, everyday case."""

    def calibrate_segment(self, raw_scores, audio_segment, sr):
        return _DummyCalibrationResult()


class _DummyVoiceCalibratorFailure:
    """Simulates calibration raising an exception."""

    def calibrate_segment(self, raw_scores, audio_segment, sr):
        raise RuntimeError("simulated calibration failure")


def _make_analyzer_with_calibrator(calibrator):
    analyzer = ConversationAnalyzer.__new__(ConversationAnalyzer)
    analyzer.emotion_analyzer = _DummyEmotionAnalyzer()
    analyzer.sentiment_analyzer = _DummySentimentAnalyzer()
    analyzer.fusion = _DummyFusion()
    analyzer.voice_calibrator = calibrator
    return analyzer


def test_segment_not_dropped_when_calibration_succeeds(monkeypatch):
    """
    This is the exact regression case: calibration succeeds, so the
    buggy code's `except` block (which was the only place sentiment
    analysis used to run) never fires. Before the fix, this always
    returned None. After the fix, sentiment analysis must still run
    and the segment must be produced.
    """
    monkeypatch.setattr(config, "VOICE_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "INTENSITY_FORMULA_VERSION", "v1")

    analyzer = _make_analyzer_with_calibrator(_DummyVoiceCalibratorSuccess())

    audio = np.zeros(16000 * 5, dtype=np.float32)
    sr = 16000
    raw_segment = {
        "start": 0.5,
        "end": 3.0,
        "text": "This needs to be fixed immediately.",
    }
    all_segment_bounds = [(0.5, 3.0)]

    result = analyzer._analyze_segment(
        audio, sr, raw_segment, index=0,
        all_segment_bounds=all_segment_bounds,
        audio_duration_seconds=5.0,
    )

    assert result is not None, (
        "_analyze_segment returned None even though calibration, "
        "sentiment, and fusion should all have succeeded. This is "
        "the exact regression this test guards against."
    )
    assert analyzer.sentiment_analyzer.call_count == 1, (
        "Sentiment analysis was not called when calibration succeeded."
    )
    assert result["text_sentiment"]["sentiment"] == "negative"
    assert result["voice_emotion"]["emotion"] == "sad"
    assert "intensity" in result


def test_segment_not_dropped_when_calibration_fails(monkeypatch):
    """
    Companion case: sentiment analysis and segment production must
    ALSO still succeed when calibration itself fails, since that was
    the one path the original buggy code happened to exercise.
    """
    monkeypatch.setattr(config, "VOICE_CALIBRATION_ENABLED", True)
    monkeypatch.setattr(config, "INTENSITY_FORMULA_VERSION", "v1")

    analyzer = _make_analyzer_with_calibrator(_DummyVoiceCalibratorFailure())

    audio = np.zeros(16000 * 5, dtype=np.float32)
    sr = 16000
    raw_segment = {
        "start": 0.5,
        "end": 3.0,
        "text": "This needs to be fixed immediately.",
    }
    all_segment_bounds = [(0.5, 3.0)]

    result = analyzer._analyze_segment(
        audio, sr, raw_segment, index=0,
        all_segment_bounds=all_segment_bounds,
        audio_duration_seconds=5.0,
    )

    assert result is not None
    assert analyzer.sentiment_analyzer.call_count == 1
    assert result["text_sentiment"]["sentiment"] == "negative"