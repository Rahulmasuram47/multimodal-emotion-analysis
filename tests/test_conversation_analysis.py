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