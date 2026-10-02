"""
Tests for Quantitative Evaluation Metrics.

Covers:
- Classification metrics: accuracy, precision, recall, F1, confusion matrix
- Speech-to-text WER calculation and alignments
- Speaker count comparisons (accuracy and MAE)
- Handling of edge cases and empty data
"""

import pytest

from evaluation.metrics import (
    compute_classification_metrics,
    compute_word_error_rate,
    compute_speaker_metrics,
)


def test_classification_metrics_perfect_match():
    y_true = ["neutral", "happy", "angry", "sad"]
    y_pred = ["neutral", "happy", "angry", "sad"]

    res = compute_classification_metrics(y_true, y_pred, labels=["neutral", "happy", "angry", "sad"])

    assert res["status"] == "OK"
    assert res["accuracy"] == 1.0
    assert res["precision_macro"] == 1.0
    assert res["recall_macro"] == 1.0
    assert res["f1_macro"] == 1.0
    assert res["confusion_matrix"]["happy"]["happy"] == 1
    assert res["confusion_matrix"]["happy"]["angry"] == 0


def test_classification_metrics_partial_match():
    y_true = ["positive", "positive", "negative", "neutral"]
    y_pred = ["positive", "negative", "negative", "neutral"]

    res = compute_classification_metrics(y_true, y_pred, labels=["positive", "negative", "neutral"])

    assert res["status"] == "OK"
    assert res["accuracy"] == 0.75  # 3 of 4
    assert res["per_class"]["positive"]["support"] == 2
    assert res["per_class"]["positive"]["precision"] == 1.0
    assert res["per_class"]["positive"]["recall"] == 0.5
    assert res["confusion_matrix"]["positive"]["negative"] == 1


def test_classification_metrics_empty_data():
    res = compute_classification_metrics([], [])
    assert res["status"] == "NO_DATA"
    assert res["total_samples"] == 0
    assert res["accuracy"] is None


def test_classification_metrics_length_mismatch_raises():
    with pytest.raises(ValueError, match="Length mismatch"):
        compute_classification_metrics(["a"], ["a", "b"])


# ---------------------------------------------------------
# WER Tests
# ---------------------------------------------------------

def test_wer_exact_match():
    ref = "the quick brown fox"
    hyp = "the quick brown fox"
    res = compute_word_error_rate(ref, hyp)

    assert res is not None
    assert res["wer"] == 0.0
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert res["insertions"] == 0
    assert res["reference_words"] == 4


def test_wer_with_punctuation_and_case():
    ref = "Hello, World! How are you?"
    hyp = "hello world how are you"
    res = compute_word_error_rate(ref, hyp)

    assert res is not None
    assert res["wer"] == 0.0


def test_wer_substitution():
    ref = "turn left at the signal"
    hyp = "turn right at the signal"
    res = compute_word_error_rate(ref, hyp)

    assert res is not None
    assert res["substitutions"] == 1
    assert res["reference_words"] == 5
    assert res["wer"] == 0.20


def test_wer_deletion():
    ref = "the weather is very nice"
    hyp = "the weather is nice"
    res = compute_word_error_rate(ref, hyp)

    assert res is not None
    assert res["deletions"] == 1
    assert res["reference_words"] == 5
    assert res["wer"] == 0.20


def test_wer_insertion():
    ref = "start analysis"
    hyp = "please start analysis now"
    res = compute_word_error_rate(ref, hyp)

    assert res is not None
    assert res["insertions"] == 2
    assert res["reference_words"] == 2
    assert res["wer"] == 1.0


def test_wer_missing_reference_returns_none():
    assert compute_word_error_rate(None, "some text") is None


def test_wer_empty_both():
    res = compute_word_error_rate("", "")
    assert res is not None
    assert res["wer"] == 0.0


# ---------------------------------------------------------
# Speaker Count Tests
# ---------------------------------------------------------

def test_speaker_metrics_calculation():
    y_true = [2, 1, 3, None]
    y_pred = [2, 2, 3, 1]

    res = compute_speaker_metrics(y_true, y_pred)
    assert res["status"] == "OK"
    assert res["count"] == 3  # None was skipped
    assert res["exact_match_accuracy"] == round(2 / 3, 4)
    # Errors: |2-2|=0, |1-2|=1, |3-3|=0 -> mean = 1/3
    assert res["mae"] == round(1 / 3, 4)


def test_speaker_metrics_no_data():
    y_true = [None, None]
    y_pred = [1, 2]
    res = compute_speaker_metrics(y_true, y_pred)
    assert res["status"] == "NO_DATA"
    assert res["count"] == 0
    assert res["exact_match_accuracy"] is None
    assert res["mae"] is None
