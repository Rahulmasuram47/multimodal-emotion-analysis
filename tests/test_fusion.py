"""
Automated tests for the Multimodal Fusion Module.

Covers:
- Negative emotion + negative sentiment agreement
- High-confidence voice/text mismatch
- Neutral agreement
"""


from app.fusion import MultimodalFusion


def test_negative_emotion_and_negative_sentiment_agree():
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