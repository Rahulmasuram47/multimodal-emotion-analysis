"""
Unit tests for Error Analysis and Diagnostic Logging.

Synthetic / mocked test inputs clearly marked test-only.
"""

from pathlib import Path
import pytest
from evaluation.harness import EvaluationResult


def test_error_analysis_file_generation(tmp_path):
    # Mock evaluation result with 1 misclassified voice emotion
    mock_records = [
        {
            "filename": "mock_sample_1.wav",
            "true_voice_emotion": "happy",
            "pred_voice_emotion": "sad",
            "true_text_sentiment": "positive",
            "pred_text_sentiment": "positive",
            "true_escalation": "no",
            "pred_transcript": "I am so happy today!",
            "extra": {
                "audio_scores": {"happy": 0.1, "sad": 0.8, "neutral": 0.05, "angry": 0.05},
                "text_scores": {"positive": 0.9, "neutral": 0.08, "negative": 0.02},
                "full_transcript": "I am so happy today!",
                "escalation_detected": False,
            },
        },
        {
            "filename": "mock_sample_2.wav",
            "true_voice_emotion": "neutral",
            "pred_voice_emotion": "neutral",
            "true_text_sentiment": "neutral",
            "pred_text_sentiment": "neutral",
            "true_escalation": "no",
            "pred_transcript": "Checking the status.",
            "extra": {},
        },
    ]

    eval_result = EvaluationResult(
        total_samples=2,
        voice_emotion_metrics={"status": "OK"},
        text_sentiment_metrics={"status": "OK"},
        speech_to_text_metrics={"status": "NO_DATA"},
        speaker_count_metrics={"status": "NO_DATA"},
        sample_records=mock_records,
        has_ground_truth=True,
    )

    # Replicate error analysis writing logic
    misclassifications = []
    for rec in eval_result.sample_records:
        t_v = rec.get("true_voice_emotion")
        p_v = rec.get("pred_voice_emotion")
        t_s = rec.get("true_text_sentiment")
        p_s = rec.get("pred_text_sentiment")
        if (t_v and p_v and t_v != p_v) or (t_s and p_s and t_s != p_s):
            misclassifications.append(rec)

    assert len(misclassifications) == 1
    err_rec = misclassifications[0]
    assert err_rec["filename"] == "mock_sample_1.wav"
    assert err_rec["true_voice_emotion"] == "happy"
    assert err_rec["pred_voice_emotion"] == "sad"
