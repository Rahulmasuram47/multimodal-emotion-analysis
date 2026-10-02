"""
Unit tests for Phase 2 Evaluation Tooling, Metrics, and Dataset extensions.

Covers:
- Extended CSV schema loading and round-trip
- Blank cells excluded from metrics and never treated as a class
- Character Error Rate (CER) computation
- Binary detection metrics (escalation and incongruence) with precision/recall/F1 and bootstrap CI
- Bootstrap confidence intervals calculation
- Leave-One-Out Cross-Validation (LOO-CV) for Learned Fusion Baseline
- Insufficient data handling in learned fusion (N < 10)
- 4-way ablation structure (Voice-Only, Text-Only, Rule-Based, Learned)
- Annotation Helper save and overwrite safety logic
- Minimum N threshold enforcement (EVAL_MIN_N)
"""

import io
import os
import tempfile
import pytest

from config import EVAL_MIN_N, EVAL_BOOTSTRAP_ITERATIONS, AUDIO_EMOTION_CLASSES, TEXT_SENTIMENT_CLASSES
from evaluation.dataset import (
    EvaluationDataset,
    EvaluationSample,
    EXTENDED_COLUMNS,
    normalize_voice_emotion_label,
    normalize_text_sentiment_label,
)
from evaluation.metrics import (
    bootstrap_ci,
    compute_character_error_rate,
    compute_word_error_rate,
    compute_classification_metrics,
    compute_speaker_metrics,
    compute_binary_detection_metrics,
)
from evaluation.learned_fusion import (
    LearnedFusionBaseline,
    LearnedFusionResult,
    MIN_TRAINING_SAMPLES,
    MIN_LOO_CV_THRESHOLD,
)
from evaluation.ablation import AblationEvaluator, AblationResult, ModalityAblationRecord


# ── 1. Extended Schema & Blank Handling ──────────────────────────────────────

def test_extended_dataset_parsing_and_blank_exclusion():
    csv_content = (
        "file,source,true_voice_emotion,true_text_sentiment,true_transcript,"
        "true_speaker_count,true_escalation,true_incongruent,notes,annotator\n"
        "recordings/s1.wav,recordings,happy,positive,I feel great,1,no,no,Clear speech,SR\n"
        "recordings/s2.wav,recordings,,,,,yes,yes,Ambiguous audio,AK\n"
        "recordings/s3.wav,recordings,neutral,N/A,Hello world,2,na,none,,RV\n"
    )

    ds = EvaluationDataset.from_csv_string(csv_content)
    assert len(ds) == 3

    # Sample 1: completely annotated
    s1 = ds[0]
    assert s1.filename == "recordings/s1.wav"
    assert s1.source == "recordings"
    assert s1.true_voice_emotion == "happy"
    assert s1.true_text_sentiment == "positive"
    assert s1.true_transcript == "I feel great"
    assert s1.true_speaker_count == 1
    assert s1.true_escalation == "no"
    assert s1.true_incongruent == "no"
    assert s1.annotator == "SR"

    # Sample 2: blank fields must be None (never a class or string "None")
    s2 = ds[1]
    assert s2.true_voice_emotion is None
    assert s2.true_text_sentiment is None
    assert s2.true_transcript is None
    assert s2.true_speaker_count is None
    assert s2.true_escalation == "yes"
    assert s2.true_incongruent == "yes"

    # Sample 3: N/A, na, none should normalize to None
    s3 = ds[2]
    assert s3.true_voice_emotion == "neutral"
    assert s3.true_text_sentiment is None
    assert s3.true_speaker_count == 2
    assert s3.true_escalation is None
    assert s3.true_incongruent is None


def test_extended_dataset_csv_roundtrip():
    samples = [
        EvaluationSample(
            filename="recordings/s1.wav",
            source="recordings",
            true_voice_emotion="angry",
            true_text_sentiment="negative",
            true_transcript="Stop this immediately.",
            true_speaker_count=1,
            true_escalation="yes",
            true_incongruent="no",
            notes="Escalating turn",
            annotator="SR",
        )
    ]
    ds = EvaluationDataset(samples)
    buf = io.StringIO()
    ds.to_csv(buf)
    content = buf.getvalue()

    reloaded = EvaluationDataset.from_csv_string(content)
    assert len(reloaded) == 1
    r = reloaded[0]
    assert r.filename == "recordings/s1.wav"
    assert r.source == "recordings"
    assert r.true_voice_emotion == "angry"
    assert r.true_text_sentiment == "negative"
    assert r.true_transcript == "Stop this immediately."
    assert r.true_speaker_count == 1
    assert r.true_escalation == "yes"
    assert r.true_incongruent == "no"
    assert r.annotator == "SR"


# ── 2. Character Error Rate (CER) ───────────────────────────────────────────

def test_character_error_rate_identical():
    res = compute_character_error_rate("hello world", "hello world")
    assert res is not None
    assert res["cer"] == 0.0
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert res["insertions"] == 0


def test_character_error_rate_different():
    res = compute_character_error_rate("cat", "bat")
    assert res is not None
    assert res["cer"] == pytest.approx(1 / 3, abs=0.01)
    assert res["substitutions"] == 1


def test_character_error_rate_none_or_empty():
    assert compute_character_error_rate(None, "hello") is None
    res = compute_character_error_rate("", "")
    assert res is not None
    assert res["cer"] == 0.0


# ── 3. Binary Detection Metrics (Escalation & Incongruence) ─────────────────

def test_binary_detection_metrics_perfect():
    y_true = ["yes", "no", "yes", "no"]
    y_pred = ["yes", "no", "yes", "no"]
    res = compute_binary_detection_metrics(y_true, y_pred, positive_class="yes")

    assert res["status"] == "OK"
    assert res["n"] == 4
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["precision_ci"]["ci_lower"] is not None


def test_binary_detection_metrics_with_unannotated_blanks():
    # Blank (None) values must be excluded, not treated as "no"
    y_true = ["yes", None, "no", None, "yes"]
    y_pred = ["yes", "no", "no", "yes", "no"]
    res = compute_binary_detection_metrics(y_true, y_pred, positive_class="yes")

    assert res["status"] == "OK"
    assert res["n"] == 3  # Only 3 valid ground truth rows
    assert res["tp"] == 1
    assert res["fn"] == 1
    assert res["fp"] == 0
    assert res["precision"] == 1.0
    assert res["recall"] == 0.5
    assert res["f1"] == pytest.approx(0.6667, abs=0.01)


def test_binary_detection_metrics_no_data():
    res = compute_binary_detection_metrics([None, None], ["yes", "no"])
    assert res["status"] == "NO_DATA"
    assert res["n"] == 0
    assert res["f1"] is None


# ── 4. Bootstrap Confidence Intervals ───────────────────────────────────────

def test_bootstrap_ci_valid():
    values = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    ci = bootstrap_ci(values, n_iterations=500, seed=42)
    assert ci["n"] == 8
    assert ci["mean"] == pytest.approx(0.75, abs=0.01)
    assert 0.0 <= ci["ci_lower"] <= ci["ci_upper"] <= 1.0


def test_bootstrap_ci_empty():
    ci = bootstrap_ci([], n_iterations=100)
    assert ci["n"] == 0
    assert ci["mean"] is None
    assert ci["ci_lower"] is None
    assert ci["ci_upper"] is None


def test_bootstrap_ci_single():
    ci = bootstrap_ci([0.8], n_iterations=100)
    assert ci["n"] == 1
    assert ci["mean"] == 0.8
    assert ci["ci_lower"] == 0.8
    assert ci["ci_upper"] == 0.8


# ── 5. Leave-One-Out CV for Learned Fusion ─────────────────────────────────

def test_learned_fusion_loo_cv_selected_for_small_n():
    baseline = LearnedFusionBaseline(random_state=42)

    # Create 12 synthetic samples (MIN_TRAINING_SAMPLES <= 12 < MIN_LOO_CV_THRESHOLD)
    samples_data = []
    ground_truth = []
    for i in range(12):
        cls = "positive" if i % 2 == 0 else "negative"
        ground_truth.append(cls)
        conf = 0.9 if cls == "positive" else 0.2
        samples_data.append({
            "audio_result": {"emotion": "happy" if cls == "positive" else "angry", "confidence": conf},
            "text_result": {"sentiment": cls, "confidence": conf, "probabilities": {cls: conf}},
            "extra": {"intensity": 0.8},
        })

    res = baseline.train_and_evaluate(samples_data, ground_truth)
    assert res.status == "OK"
    assert "Leave-One-Out CV" in res.message
    assert res.test_samples == 12
    assert res.accuracy is not None
    assert res.f1_macro is not None


def test_learned_fusion_insufficient_data():
    baseline = LearnedFusionBaseline(random_state=42)
    # Only 4 samples, below MIN_TRAINING_SAMPLES (10)
    samples_data = [
        {"audio_result": {"emotion": "happy", "confidence": 0.8}, "text_result": {"sentiment": "positive", "confidence": 0.8}}
        for _ in range(4)
    ]
    gt = ["positive", "negative", "positive", "negative"]
    res = baseline.train_and_evaluate(samples_data, gt)

    assert res.status == "INSUFFICIENT_DATA"
    assert res.train_samples == 0
    assert res.test_samples == 0
    assert "minimum required is 10" in res.message


# ── 6. 4-Way Modality Ablation ──────────────────────────────────────────────

def test_four_way_model_comparison_in_ablation():
    evaluator = AblationEvaluator()
    samples = [
        {
            "filename": "f1.wav",
            "audio_result": {"emotion": "happy", "confidence": 0.8},
            "text_result": {"sentiment": "positive", "confidence": 0.7},
            "ground_truth_direction": "positive",
        },
        {
            "filename": "f2.wav",
            "audio_result": {"emotion": "angry", "confidence": 0.9},
            "text_result": {"sentiment": "negative", "confidence": 0.8},
            "ground_truth_direction": "negative",
        },
    ]

    res = evaluator.evaluate(samples)
    four_way = res.get_four_way_model_comparison()

    assert len(four_way) == 4
    models = [m["model"] for m in four_way]
    assert "Voice-Only" in models
    assert "Text-Only" in models
    assert "Current Fusion (Rule-Based)" in models
    assert "Learned Fusion (Logistic Regression)" in models

    # Check that to_dict includes four_way_models
    d = res.to_dict()
    assert "four_way_models" in d
    assert len(d["four_way_models"]) == 4


# ── 7. Annotation Overwrite Safety ──────────────────────────────────────────

def test_annotation_save_row_logic():
    from ui.pages.annotation_view import _save_row

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
        tmp_csv = tf.name

    try:
        ds = EvaluationDataset()
        sample1 = EvaluationSample(
            filename="recordings/test1.wav",
            true_voice_emotion="happy",
            annotator="SR",
        )
        _save_row(ds, sample1, tmp_csv)

        # Reload from disk
        ds_reloaded = EvaluationDataset.from_csv(tmp_csv)
        assert len(ds_reloaded) == 1
        assert ds_reloaded[0].filename == "recordings/test1.wav"
        assert ds_reloaded[0].true_voice_emotion == "happy"

        # Update the same file
        sample1_updated = EvaluationSample(
            filename="recordings/test1.wav",
            true_voice_emotion="neutral",
            annotator="AK",
        )
        _save_row(ds_reloaded, sample1_updated, tmp_csv)

        ds_reloaded2 = EvaluationDataset.from_csv(tmp_csv)
        assert len(ds_reloaded2) == 1  # Must not duplicate
        assert ds_reloaded2[0].true_voice_emotion == "neutral"
        assert ds_reloaded2[0].annotator == "AK"
    finally:
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)


# ── 8. Minimum N Threshold ──────────────────────────────────────────────────

def test_minimum_n_threshold_defined():
    assert EVAL_MIN_N >= 5
    assert isinstance(EVAL_MIN_N, int)
