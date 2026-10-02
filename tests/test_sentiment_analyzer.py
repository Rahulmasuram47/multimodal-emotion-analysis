"""
Automated tests for the Text Sentiment Analysis Module.

Covers:
- Clearly positive text
- Clearly negative text
- Neutral / factual text
- Empty text rejection
- Score dictionary integrity
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from app.sentiment_analyzer import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    """Load the sentiment model once for all tests in this file."""
    return SentimentAnalyzer()


def test_positive_sentence(analyzer):
    result = analyzer.analyze(
        "I absolutely loved this, it made my day and everything worked perfectly!"
    )
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.5


def test_negative_sentence(analyzer):
    result = analyzer.analyze(
        "This is terrible, I am really upset and nothing about this worked."
    )
    assert result["sentiment"] == "negative"
    assert result["confidence"] > 0.5


def test_neutral_sentence(analyzer):
    result = analyzer.analyze(
        "The meeting is scheduled for 3 PM in the second-floor conference room."
    )
    assert result["sentiment"] == "neutral"


def test_empty_text_raises(analyzer):
    with pytest.raises(ValueError):
        analyzer.analyze("")

    with pytest.raises(ValueError):
        analyzer.analyze("   ")


def test_scores_dictionary_integrity(analyzer):
    result = analyzer.analyze("This product is okay, nothing special.")

    assert set(result["scores"].keys()) == {"positive", "negative", "neutral"}

    total = sum(result["scores"].values())
    assert abs(total - 1.0) < 1e-3

    # The reported sentiment must be the highest-scoring class.
    top_label = max(result["scores"], key=result["scores"].get)
    assert top_label == result["sentiment"]