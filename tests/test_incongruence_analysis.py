"""
Automated unit tests for Modality Incongruence and Conflict Detection Module.

Covers:
- Positive text + angry voice -> conflict detected
- Positive text + sad voice -> conflict detected
- Negative text + happy voice -> conflict detected
- Same-direction modalities -> no conflict detected
- Low confidence in either modality -> no high-confidence conflict
- Deterministic output behavior
- Conflict strength validity ("none", "low", "medium", "high")
- Scientific explanation and disclaimer existence
- Conversation segment aggregation
"""

import pytest

from app.incongruence_analysis import (
    detect_modality_incongruence,
    analyze_conversation_incongruence,
    IncongruenceResult,
    MANDATORY_HEURISTIC_NOTICE,
)


def test_positive_text_angry_acoustic_high_conflict():
    """Positive text with angry voice at high confidence triggers HIGH conflict."""
    res = detect_modality_incongruence(
        text_label="positive",
        text_confidence=0.91,
        acoustic_label="angry",
        acoustic_confidence=0.82,
    )
    assert res.conflict_detected is True
    assert res.conflict_strength == "high"
    assert res.conflict_type == "positive_text_angry_voice"
    assert "Strong disagreement" in res.explanation
    assert MANDATORY_HEURISTIC_NOTICE in res.heuristic_notice


def test_positive_text_sad_acoustic_conflict():
    """Positive text with sad voice triggers conflict."""
    res = detect_modality_incongruence(
        text_label="positive",
        text_confidence=0.75,
        acoustic_label="sad",
        acoustic_confidence=0.70,
    )
    assert res.conflict_detected is True
    assert res.conflict_strength == "high"
    assert res.conflict_type == "positive_text_sad_voice"


def test_negative_text_happy_acoustic_conflict():
    """Negative text with happy voice triggers conflict."""
    res = detect_modality_incongruence(
        text_label="negative",
        text_confidence=0.88,
        acoustic_label="happy",
        acoustic_confidence=0.84,
    )
    assert res.conflict_detected is True
    assert res.conflict_strength == "high"
    assert res.conflict_type == "negative_text_happy_voice"


def test_same_direction_modalities_no_conflict():
    """Concordant modalities (positive text + happy acoustic) do not trigger conflict."""
    res = detect_modality_incongruence(
        text_label="positive",
        text_confidence=0.85,
        acoustic_label="happy",
        acoustic_confidence=0.80,
    )
    assert res.conflict_detected is False
    assert res.conflict_strength == "none"
    assert res.conflict_type == "none"
    assert "No strong modality conflict" in res.explanation


def test_concordant_negative_no_conflict():
    """Negative text with angry or sad voice does not trigger conflict."""
    res_angry = detect_modality_incongruence(
        text_label="negative",
        text_confidence=0.85,
        acoustic_label="angry",
        acoustic_confidence=0.80,
    )
    assert res_angry.conflict_detected is False
    assert res_angry.conflict_strength == "none"

    res_sad = detect_modality_incongruence(
        text_label="negative",
        text_confidence=0.85,
        acoustic_label="sad",
        acoustic_confidence=0.80,
    )
    assert res_sad.conflict_detected is False
    assert res_sad.conflict_strength == "none"


def test_low_confidence_prevents_high_conflict():
    """Opposing modalities with low confidence must NOT produce high conflict."""
    res = detect_modality_incongruence(
        text_label="positive",
        text_confidence=0.35,  # Below default min 0.40
        acoustic_label="angry",
        acoustic_confidence=0.85,
    )
    assert res.conflict_detected is False
    assert res.conflict_strength == "low"
    assert "low confidence" in res.explanation.lower()


def test_moderate_confidence_produces_medium_conflict():
    """Opposing modalities with moderate confidence produce medium conflict."""
    res = detect_modality_incongruence(
        text_label="positive",
        text_confidence=0.55,  # between min (0.40) and high (0.65)
        acoustic_label="angry",
        acoustic_confidence=0.75,
    )
    assert res.conflict_detected is True
    assert res.conflict_strength == "medium"


def test_deterministic_output():
    """Repeated calls with identical inputs produce identical outputs."""
    r1 = detect_modality_incongruence("angry", 0.77, "positive", 0.88)
    r2 = detect_modality_incongruence("angry", 0.77, "positive", 0.88)

    assert r1.conflict_detected == r2.conflict_detected
    assert r1.conflict_strength == r2.conflict_strength
    assert r1.explanation == r2.explanation
    assert r1.heuristic_notice == r2.heuristic_notice


def test_valid_conflict_strength_values():
    """Conflict strength values must be one of allowed strings."""
    valid_strengths = {"none", "low", "medium", "high"}

    test_cases = [
        ("happy", 0.9, "positive", 0.9),
        ("angry", 0.9, "positive", 0.9),
        ("angry", 0.5, "positive", 0.5),
        ("angry", 0.2, "positive", 0.9),
    ]

    for ac_lbl, ac_conf, tx_lbl, tx_conf in test_cases:
        res = detect_modality_incongruence(ac_lbl, ac_conf, tx_lbl, tx_conf)
        assert res.conflict_strength in valid_strengths


def test_mandatory_disclaimer_exists():
    """Disclaimer must be present and contain safety wording."""
    res = detect_modality_incongruence("angry", 0.8, "positive", 0.8)
    d = res.to_dict()
    assert "heuristic_notice" in d
    assert "computational speech and text signals" in d["heuristic_notice"]
    assert "does not reliably determine sarcasm, deception" in d["heuristic_notice"]


def test_analyze_conversation_incongruence_empty():
    """Empty segment list handled safely."""
    res = analyze_conversation_incongruence([])
    assert res["status"] == "NO_SEGMENTS"
    assert res["conflict_segments_count"] == 0
    assert res["has_incongruence"] is False


def test_analyze_conversation_incongruence_with_segments():
    """Multi-segment conversation correctly counts incongruent turns."""
    segments = [
        {
            "segment_index": 1,
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "transcript": "Everything is great!",
            "voice_emotion": {"emotion": "angry", "confidence": 0.85},
            "text_sentiment": {"sentiment": "positive", "confidence": 0.90},
        },
        {
            "segment_index": 2,
            "start_seconds": 2.5,
            "end_seconds": 5.0,
            "transcript": "Yes I am happy.",
            "voice_emotion": {"emotion": "happy", "confidence": 0.80},
            "text_sentiment": {"sentiment": "positive", "confidence": 0.85},
        },
    ]

    res = analyze_conversation_incongruence(segments)
    assert res["status"] == "OK"
    assert res["total_segments"] == 2
    assert res["conflict_segments_count"] == 1
    assert res["high_conflict_count"] == 1
    assert res["has_incongruence"] is True
    assert len(res["segment_results"]) == 2
    assert res["segment_results"][0]["conflict_strength"] == "high"
    assert res["segment_results"][1]["conflict_strength"] == "none"
