"""
Automated tests for the Multimodal Fusion Module.

Covers:
- Negative emotion + negative sentiment agreement
- High-confidence voice/text mismatch
- Neutral agreement
- Low-confidence disagreement
- Wav2Vec2 emotion-label normalization
- Weighted combined confidence
"""


from app.fusion import MultimodalFusion


def test_negative_emotion_and_negative_sentiment_agree():
    """Negative voice emotion and negative text sentiment should agree."""

    fusion = MultimodalFusion()

    audio_result = {
        "emotion": "angry",
        "confidence": 0.90,
    }

    text_result = {
        "sentiment": "negative",
        "confidence": 0.85,
    }

    result = fusion.fuse(audio_result, text_result)

    assert result["agreement"] is True
    assert result["mismatch"] is False
    assert result["audio_direction"] == "negative"
    assert result["text_direction"] == "negative"


def test_high_confidence_voice_text_mismatch():
    """Strongly negative voice and strongly positive text should mismatch."""

    fusion = MultimodalFusion()

    audio_result = {
        "emotion": "angry",
        "confidence": 0.90,
    }

    text_result = {
        "sentiment": "positive",
        "confidence": 0.85,
    }

    result = fusion.fuse(audio_result, text_result)

    assert result["agreement"] is False
    assert result["mismatch"] is True
    assert result["audio_direction"] == "negative"
    assert result["text_direction"] == "positive"


def test_neutral_emotion_and_neutral_sentiment_agree():
    """Neutral voice emotion and neutral text sentiment should agree."""

    fusion = MultimodalFusion()

    audio_result = {
        "emotion": "neutral",
        "confidence": 0.80,
    }

    text_result = {
        "sentiment": "neutral",
        "confidence": 0.75,
    }

    result = fusion.fuse(audio_result, text_result)

    assert result["agreement"] is True
    assert result["mismatch"] is False
    assert result["audio_direction"] == "neutral"
    assert result["text_direction"] == "neutral"


def test_low_confidence_disagreement_is_not_strong_mismatch():
    """
    Disagreement between low-confidence modalities should not
    be classified as a strong mismatch.
    """

    fusion = MultimodalFusion()

    audio_result = {
        "emotion": "angry",
        "confidence": 0.30,
    }

    text_result = {
        "sentiment": "positive",
        "confidence": 0.35,
    }

    result = fusion.fuse(audio_result, text_result)

    assert result["agreement"] is False
    assert result["mismatch"] is False
    assert result["audio_direction"] == "negative"
    assert result["text_direction"] == "positive"


def test_short_emotion_labels_are_normalized():
    """
    Wav2Vec2 SUPERB labels such as 'hap' should be normalized
    to standard emotion names.
    """

    fusion = MultimodalFusion()

    audio_result = {
        "emotion": "hap",
        "confidence": 0.90,
    }

    text_result = {
        "sentiment": "positive",
        "confidence": 0.90,
    }

    result = fusion.fuse(audio_result, text_result)

    assert result["audio_emotion"] == "happy"
    assert result["audio_direction"] == "positive"
    assert result["text_direction"] == "positive"
    assert result["agreement"] is True
    assert result["mismatch"] is False


def test_combined_confidence_is_weighted():
    """Combined confidence should respect the configured modality weights."""

    fusion = MultimodalFusion(
        audio_weight=0.6,
        text_weight=0.4,
    )

    audio_result = {
        "emotion": "sad",
        "confidence": 0.80,
    }

    text_result = {
        "sentiment": "negative",
        "confidence": 0.60,
    }

    result = fusion.fuse(
        audio_result,
        text_result,
    )

    # 0.6 * 0.80 + 0.4 * 0.60 = 0.72
    assert abs(result["combined_confidence"] - 0.72) < 1e-6