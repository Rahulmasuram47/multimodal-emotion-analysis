"""
Automated tests for conversation-level analysis logic.

Tests:
- Emotional trend detection
- Escalation detection
- Insufficient segment handling
"""


from app.conversation_analysis import ConversationAnalyzer


def make_segment(intensity):
    """Create a minimal segment for trend/escalation testing."""
    return {
        "intensity": intensity,
    }


def test_emotional_trend_increasing():
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
    segments = [
        make_segment(0.80),
        make_segment(0.60),
        make_segment(0.40),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "decreasing"
    assert result["change"] == -0.40


def test_emotional_trend_stable():
    segments = [
        make_segment(0.40),
        make_segment(0.45),
        make_segment(0.43),
    ]

    result = ConversationAnalyzer.detect_emotional_trend(segments)

    assert result["trend"] == "stable"


def test_escalation_fires_on_continuously_rising_intensity():
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
    segments = [
        make_segment(0.30),
        make_segment(0.31),
        make_segment(0.29),
        make_segment(0.30),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is False


def test_no_escalation_when_less_than_three_segments():
    segments = [
        make_segment(0.20),
        make_segment(0.50),
    ]

    result = ConversationAnalyzer.detect_escalation(segments)

    assert result["escalation_detected"] is False
    assert result["reason"] == "At least 3 segments are required."