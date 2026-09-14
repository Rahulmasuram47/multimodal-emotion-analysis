"""
Unit tests for the Actionable Recommendation Engine (app/recommendation_engine.py).
"""

import pytest
from app.recommendation_engine import (
    ActionableRecommendation,
    generate_recommendation,
    STATUS_LOW_STABLE,
    STATUS_MODERATE,
    STATUS_HIGH_ESCALATION,
    STATUS_HIGH_INCONGRUENCE,
    STATUS_LOW_CONFIDENCE,
    MANDATORY_RECOMMENDATION_DISCLAIMER,
)


def test_empty_or_none_result():
    rec1 = generate_recommendation({})
    assert rec1.status == STATUS_LOW_CONFIDENCE
    assert rec1.disclaimer == MANDATORY_RECOMMENDATION_DISCLAIMER

    rec2 = generate_recommendation({"segments": []})
    assert rec2.status == STATUS_LOW_CONFIDENCE


def test_low_confidence_trigger():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "neutral", "confidence": 0.25},
                "text_sentiment": {"sentiment": "neutral", "confidence": 0.30},
                "fusion": {"combined_confidence": 0.28},
            }
        ],
        "escalation": {"escalation_detected": False},
        "emotional_trend": {"trend": "stable"},
        "attention_score": {"score": 0.1},
    }
    rec = generate_recommendation(analysis)
    assert rec.status == STATUS_LOW_CONFIDENCE
    assert "low" in rec.reason.lower() or "reliability" in rec.reason.lower()


def test_high_escalation_trigger():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "angry", "confidence": 0.85},
                "text_sentiment": {"sentiment": "negative", "confidence": 0.80},
                "fusion": {"combined_confidence": 0.82},
            }
        ],
        "escalation": {
            "escalation_detected": True,
            "intensity_increase": 0.35,
        },
        "emotional_trend": {"trend": "increasing"},
        "attention_score": {"score": 0.75},
    }
    rec = generate_recommendation(analysis)
    assert rec.status == STATUS_HIGH_ESCALATION
    assert rec.status_level == "critical"
    assert "de-escalation" in rec.recommended_action.lower()


def test_high_incongruence_trigger():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "angry", "confidence": 0.85},
                "text_sentiment": {"sentiment": "positive", "confidence": 0.80},
                "fusion": {"combined_confidence": 0.82},
            }
        ],
        "escalation": {"escalation_detected": False},
        "emotional_trend": {"trend": "stable"},
        "attention_score": {"score": 0.50},
        "modality_incongruence": {
            "high_conflict_count": 1,
            "conflict_segments_count": 1,
        },
    }
    rec = generate_recommendation(analysis)
    assert rec.status == STATUS_HIGH_INCONGRUENCE
    assert rec.status_level == "warning"
    assert "sarcasm" in rec.recommended_action.lower() or "context" in rec.recommended_action.lower()


def test_moderate_trigger():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "sad", "confidence": 0.70},
                "text_sentiment": {"sentiment": "negative", "confidence": 0.75},
                "fusion": {"combined_confidence": 0.72},
            }
        ],
        "escalation": {"escalation_detected": False},
        "emotional_trend": {"trend": "stable"},
        "attention_score": {"score": 0.48},
    }
    rec = generate_recommendation(analysis)
    assert rec.status == STATUS_MODERATE
    assert rec.status_level == "warning"


def test_low_stable_trigger():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "happy", "confidence": 0.85},
                "text_sentiment": {"sentiment": "positive", "confidence": 0.90},
                "fusion": {"combined_confidence": 0.87},
            }
        ],
        "escalation": {"escalation_detected": False},
        "emotional_trend": {"trend": "stable"},
        "attention_score": {"score": 0.15},
    }
    rec = generate_recommendation(analysis)
    assert rec.status == STATUS_LOW_STABLE
    assert rec.status_level == "success"
    assert "no immediate intervention" in rec.recommended_action.lower()


def test_deterministic_and_disclaimer():
    analysis = {
        "segments": [
            {
                "voice_emotion": {"emotion": "neutral", "confidence": 0.65},
                "text_sentiment": {"sentiment": "neutral", "confidence": 0.60},
                "fusion": {"combined_confidence": 0.62},
            }
        ],
        "escalation": {"escalation_detected": False},
        "emotional_trend": {"trend": "stable"},
        "attention_score": {"score": 0.2},
    }
    rec1 = generate_recommendation(analysis)
    rec2 = generate_recommendation(analysis)
    assert rec1.to_dict() == rec2.to_dict()
    assert rec1.disclaimer == MANDATORY_RECOMMENDATION_DISCLAIMER
