"""
Automated unit tests for Learned Fusion Baseline Module.

Covers:
- Feature extraction and dimensionality
- Insufficient data handling (status INSUFFICIENT_DATA, no fabrication)
- Deterministic feature extraction
- Model training with synthetic TEST FIXTURES ONLY
- Prediction format and probability ranges
- Protection against mutation of production fusion configuration
"""

import pytest

from config import DEFAULT_AUDIO_WEIGHT, DEFAULT_TEXT_WEIGHT, CONFIDENCE_THRESHOLD
from app.fusion import MultimodalFusion
from evaluation.learned_fusion import (
    extract_fusion_features,
    LearnedFusionBaseline,
    LearnedFusionResult,
    FEATURE_NAMES,
    MIN_TRAINING_SAMPLES,
)


def test_feature_extraction_dimension_and_range():
    """Features extracted from audio and text results must be exactly 12 normalized numbers."""
    audio_res = {
        "emotion": "happy",
        "confidence": 0.85,
        "scores": {"neutral": 0.05, "happy": 0.85, "angry": 0.05, "sad": 0.05},
    }
    text_res = {
        "sentiment": "positive",
        "confidence": 0.90,
        "probabilities": {"positive": 0.90, "negative": 0.02, "neutral": 0.08},
    }

    features = extract_fusion_features(audio_res, text_res)

    assert len(features) == len(FEATURE_NAMES)
    assert len(features) == 12
    # All values should be within [0.0, 1.0]
    for val in features:
        assert 0.0 <= val <= 1.0

    # In this case: happy + positive -> agreement=1.0, mismatch=0.0
    assert features[10] == 1.0  # agreement
    assert features[11] == 0.0  # mismatch


def test_deterministic_feature_extraction():
    """Extracting features from identical inputs must produce identical vectors."""
    audio_res = {
        "emotion": "angry",
        "confidence": 0.77,
        "scores": {"neutral": 0.1, "happy": 0.0, "angry": 0.77, "sad": 0.13},
    }
    text_res = {
        "sentiment": "negative",
        "confidence": 0.88,
        "probabilities": {"positive": 0.02, "negative": 0.88, "neutral": 0.10},
    }

    f1 = extract_fusion_features(audio_res, text_res)
    f2 = extract_fusion_features(audio_res, text_res)

    assert f1 == f2


def test_insufficient_data_handling():
    """When fewer than MIN_TRAINING_SAMPLES are provided, reports INSUFFICIENT_DATA without fabricating metrics."""
    baseline = LearnedFusionBaseline()

    # Provide only 3 samples (fewer than MIN_TRAINING_SAMPLES=10)
    samples_data = [
        {"audio_result": {"confidence": 0.8}, "text_result": {"confidence": 0.7}},
        {"audio_result": {"confidence": 0.6}, "text_result": {"confidence": 0.5}},
        {"audio_result": {"confidence": 0.9}, "text_result": {"confidence": 0.9}},
    ]
    ground_truth = ["positive", "negative", "positive"]

    result = baseline.train_and_evaluate(samples_data, ground_truth)

    assert isinstance(result, LearnedFusionResult)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.accuracy is None
    assert result.f1_macro is None
    assert result.confusion_matrix is None
    assert "Insufficient labelled data" in result.message


def test_training_on_synthetic_test_fixtures_only():
    """Test model training and evaluation using purely in-memory synthetic fixtures."""
    baseline = LearnedFusionBaseline(random_state=42)

    # Generate 20 synthetic samples (10 positive, 10 negative)
    samples_data = []
    ground_truth = []

    for i in range(10):
        # Positive samples
        samples_data.append({
            "audio_result": {
                "emotion": "happy",
                "confidence": 0.8 + (i * 0.01),
                "scores": {"neutral": 0.1, "happy": 0.8, "angry": 0.05, "sad": 0.05},
            },
            "text_result": {
                "sentiment": "positive",
                "confidence": 0.85,
                "probabilities": {"positive": 0.85, "negative": 0.05, "neutral": 0.10},
            },
        })
        ground_truth.append("positive")

        # Negative samples
        samples_data.append({
            "audio_result": {
                "emotion": "angry",
                "confidence": 0.75 + (i * 0.01),
                "scores": {"neutral": 0.1, "happy": 0.05, "angry": 0.75, "sad": 0.10},
            },
            "text_result": {
                "sentiment": "negative",
                "confidence": 0.80,
                "probabilities": {"positive": 0.05, "negative": 0.80, "neutral": 0.15},
            },
        })
        ground_truth.append("negative")

    assert len(samples_data) >= MIN_TRAINING_SAMPLES

    result = baseline.train_and_evaluate(samples_data, ground_truth, test_size=0.30)

    assert result.status == "OK"
    assert result.accuracy is not None
    assert result.f1_macro is not None
    assert 0.0 <= result.accuracy <= 1.0
    assert 0.0 <= result.f1_macro <= 1.0
    assert result.train_samples > 0
    assert result.test_samples > 0
    assert result.confusion_matrix is not None


def test_prediction_output_format():
    """Predict and predict_proba return expected structure and valid probabilities."""
    baseline = LearnedFusionBaseline(random_state=42)

    # 12-dim synthetic features
    X_train = [
        [0.1, 0.8, 0.05, 0.05, 0.8, 0.1, 0.1, 0.8, 0.8, 0.8, 1.0, 0.0],
        [0.1, 0.85, 0.02, 0.03, 0.85, 0.05, 0.1, 0.85, 0.85, 0.85, 1.0, 0.0],
        [0.1, 0.05, 0.8, 0.05, 0.05, 0.85, 0.1, 0.8, 0.85, 0.8, 0.0, 0.0],
        [0.1, 0.02, 0.85, 0.03, 0.03, 0.88, 0.09, 0.85, 0.88, 0.85, 0.0, 0.0],
    ]
    y_train = ["positive", "positive", "negative", "negative"]

    baseline.fit(X_train, y_train)

    X_test = [
        [0.1, 0.78, 0.06, 0.06, 0.82, 0.08, 0.1, 0.8, 0.8, 0.8, 1.0, 0.0],
        [0.1, 0.04, 0.82, 0.04, 0.06, 0.84, 0.1, 0.82, 0.84, 0.82, 0.0, 0.0],
    ]

    preds = baseline.predict(X_test)
    assert len(preds) == 2
    assert all(p in ["positive", "negative"] for p in preds)

    probs = baseline.predict_proba(X_test)
    assert "positive" in probs and "negative" in probs
    for i in range(len(X_test)):
        total_p = probs["positive"][i] + probs["negative"][i]
        assert pytest.approx(total_p, rel=1e-2) == 1.0


def test_production_fusion_remains_unmutated():
    """Verify production fusion defaults remain exactly 0.50, 0.50, and 0.40."""
    assert DEFAULT_AUDIO_WEIGHT == 0.50
    assert DEFAULT_TEXT_WEIGHT == 0.50
    assert CONFIDENCE_THRESHOLD == 0.40

    prod_fusion = MultimodalFusion()
    assert prod_fusion.audio_weight == 0.50
    assert prod_fusion.text_weight == 0.50
    assert prod_fusion.confidence_threshold == 0.40
