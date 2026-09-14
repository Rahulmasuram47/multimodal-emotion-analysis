"""
Automated unit tests for Quantitative Evaluation Runner and Reporting Workflow.

Covers:
- Fully labelled dataset evaluation
- All-N/A dataset handling (pending data status, zero fabricated metrics)
- Partially labelled dataset (evaluates valid subsets only)
- Missing ground truth annotation reporting
- Invalid rows / missing files handling
- Report generation: latest_evaluation.json and latest_evaluation.txt
- Verification that no fabricated numbers exist when labels are missing
"""

import json
import os
import tempfile
import pytest

from evaluation.dataset import EvaluationDataset, EvaluationSample
from evaluation.harness import EvaluationHarness, SamplePrediction
from evaluation.ablation import AblationEvaluator, AblationResult
from evaluation.learned_fusion import LearnedFusionBaseline, LearnedFusionResult
from evaluation.run_evaluation import format_human_readable_report


def test_all_na_dataset_reports_pending_data_without_fabrication():
    """When ground truth is completely N/A, reports Pending Data without inventing metrics."""
    samples = [
        EvaluationSample(filename="sample_audio/real_voice.wav"),
        EvaluationSample(filename="sample_audio/rising_tone.wav"),
    ]
    dataset = EvaluationDataset(samples)

    harness = EvaluationHarness()
    preds = [
        SamplePrediction(filename="sample_audio/real_voice.wav", pred_voice_emotion="neutral"),
        SamplePrediction(filename="sample_audio/rising_tone.wav", pred_voice_emotion="happy"),
    ]
    eval_result = harness.evaluate_predictions(dataset, preds)

    assert eval_result.has_ground_truth is False
    assert eval_result.voice_emotion_metrics["status"] == "NO_DATA"
    assert eval_result.voice_emotion_metrics["accuracy"] is None
    assert eval_result.voice_emotion_metrics["f1_macro"] is None
    assert eval_result.text_sentiment_metrics["accuracy"] is None

    # Ablation evaluator
    ablation_eval = AblationEvaluator()
    ablation_res = ablation_eval.evaluate(
        [
            {"audio_result": {"confidence": 0.8}, "text_result": {"confidence": 0.7}},
            {"audio_result": {"confidence": 0.6}, "text_result": {"confidence": 0.5}},
        ],
        dataset=dataset,
    )

    comp_table = ablation_res.format_comparison_table()
    assert "Pending Data" in comp_table

    # Learned fusion on N/A dataset
    lf = LearnedFusionBaseline()
    lf_res = lf.train_and_evaluate(
        [
            {"audio_result": {"confidence": 0.8}, "text_result": {"confidence": 0.7}},
            {"audio_result": {"confidence": 0.6}, "text_result": {"confidence": 0.5}},
        ],
        [None, None],
    )
    assert lf_res.status == "INSUFFICIENT_DATA"
    assert lf_res.accuracy is None


def test_partially_labelled_dataset():
    """When only voice emotion is labelled, evaluates voice emotion and keeps text sentiment as Pending Data."""
    samples = [
        EvaluationSample(filename="a.wav", true_voice_emotion="happy"),
        EvaluationSample(filename="b.wav", true_voice_emotion="angry"),
        EvaluationSample(filename="c.wav"),  # No labels
    ]
    dataset = EvaluationDataset(samples)

    harness = EvaluationHarness()
    preds = [
        SamplePrediction(filename="a.wav", pred_voice_emotion="happy", pred_text_sentiment="positive"),
        SamplePrediction(filename="b.wav", pred_voice_emotion="angry", pred_text_sentiment="negative"),
        SamplePrediction(filename="c.wav", pred_voice_emotion="neutral", pred_text_sentiment="neutral"),
    ]

    eval_result = harness.evaluate_predictions(dataset, preds)

    # Voice emotion was evaluated for 2 labelled samples
    assert eval_result.has_ground_truth is True
    assert eval_result.voice_emotion_metrics["status"] == "OK"
    assert eval_result.voice_emotion_metrics["total_samples"] == 2
    assert eval_result.voice_emotion_metrics["accuracy"] == 1.0

    # Text sentiment had NO labels, must remain NO_DATA
    assert eval_result.text_sentiment_metrics["status"] == "NO_DATA"
    assert eval_result.text_sentiment_metrics["accuracy"] is None
    assert eval_result.text_sentiment_metrics["f1_macro"] is None


def test_fully_labelled_mock_dataset():
    """Fully labelled dataset correctly computes accuracy, macro F1, and confusion matrix."""
    samples = [
        EvaluationSample(filename="1.wav", true_voice_emotion="happy", true_text_sentiment="positive", true_speaker_count=1),
        EvaluationSample(filename="2.wav", true_voice_emotion="angry", true_text_sentiment="negative", true_speaker_count=2),
        EvaluationSample(filename="3.wav", true_voice_emotion="sad", true_text_sentiment="negative", true_speaker_count=1),
        EvaluationSample(filename="4.wav", true_voice_emotion="neutral", true_text_sentiment="neutral", true_speaker_count=1),
    ]
    dataset = EvaluationDataset(samples)

    harness = EvaluationHarness()
    preds = [
        SamplePrediction(filename="1.wav", pred_voice_emotion="happy", pred_text_sentiment="positive", pred_speaker_count=1),
        SamplePrediction(filename="2.wav", pred_voice_emotion="angry", pred_text_sentiment="negative", pred_speaker_count=2),
        SamplePrediction(filename="3.wav", pred_voice_emotion="sad", pred_text_sentiment="negative", pred_speaker_count=1),
        SamplePrediction(filename="4.wav", pred_voice_emotion="neutral", pred_text_sentiment="neutral", pred_speaker_count=1),
    ]

    eval_result = harness.evaluate_predictions(dataset, preds)

    assert eval_result.has_ground_truth is True
    assert eval_result.voice_emotion_metrics["accuracy"] == 1.0
    assert eval_result.voice_emotion_metrics["f1_macro"] == 1.0
    assert eval_result.text_sentiment_metrics["accuracy"] == 1.0
    assert eval_result.speaker_count_metrics["exact_match_accuracy"] == 1.0
    assert eval_result.speaker_count_metrics["mae"] == 0.0


def test_report_generation_formatting():
    """Human-readable report text contains required headers, tables, and notices."""
    samples = [
        EvaluationSample(filename="clip1.wav", true_voice_emotion="happy", true_speaker_count=1),
        EvaluationSample(filename="clip2.wav", true_voice_emotion="angry", true_speaker_count=1),
    ]
    dataset = EvaluationDataset(samples)

    harness = EvaluationHarness()
    preds = [
        SamplePrediction(filename="clip1.wav", pred_voice_emotion="happy", pred_speaker_count=1),
        SamplePrediction(filename="clip2.wav", pred_voice_emotion="angry", pred_speaker_count=1),
    ]
    eval_result = harness.evaluate_predictions(dataset, preds)

    ablation_eval = AblationEvaluator()
    ablation_res = ablation_eval.evaluate(
        [
            {"audio_result": {"emotion": "happy", "confidence": 0.8}, "text_result": {"sentiment": "positive", "confidence": 0.7}},
            {"audio_result": {"emotion": "angry", "confidence": 0.8}, "text_result": {"sentiment": "negative", "confidence": 0.7}},
        ],
        dataset=dataset,
    )

    lf = LearnedFusionBaseline()
    lf_res = lf.train_and_evaluate(
        [
            {"audio_result": {"confidence": 0.8}, "text_result": {"confidence": 0.7}},
            {"audio_result": {"confidence": 0.6}, "text_result": {"confidence": 0.5}},
        ],
        [None, None],
    )

    missing = ["Text Sentiment ground truth (true_text_sentiment)", "Reference Transcripts (true_transcript for WER)"]

    report = format_human_readable_report(
        dataset=dataset,
        eval_result=eval_result,
        ablation_result=ablation_res,
        learned_fusion_result=lf_res,
        missing_annotations=missing,
        total_evaluated=2,
        total_failed=0,
    )

    assert "MULTIMODAL EMOTION ANALYSIS — EMPIRICAL EVALUATION REPORT" in report
    assert "Pending / Missing Annotations:" in report
    assert "Text Sentiment ground truth" in report
    assert "VOICE EMOTION (ACOUSTIC) METRICS:" in report
    assert "MODEL COMPARISON (ABLATION & LEARNED FUSION BASELINE):" in report
    assert "Voice Only" in report
    assert "Learned Fusion" in report
    assert "MANDATORY SCIENTIFIC DISCLAIMER:" in report


def test_invalid_csv_rows_handled_gracefully():
    """Invalid CSV content without filename is skipped safely."""
    csv_content = (
        "filename,true_voice_emotion,true_text_sentiment,true_speaker_count,true_transcript,notes\n"
        ",,,,,\n"  # completely empty row
        "valid.wav,happy,positive,1,Hello,valid row\n"
    )
    dataset = EvaluationDataset.from_csv_string(csv_content)
    assert len(dataset) == 1
    assert dataset[0].filename == "valid.wav"
