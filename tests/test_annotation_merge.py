"""
Unit tests for Inter-Annotator Agreement (Cohen's Kappa), ECE, and Annotation Merge Tooling.

All inputs clearly marked test-only / synthetic.
"""

import pytest
from evaluation.metrics import compute_cohens_kappa, compute_expected_calibration_error
from scripts.merge_annotations import analyze_agreement, write_consensus_dataset


def test_cohens_kappa_perfect_agreement():
    r1 = ["happy", "sad", "angry", "neutral"]
    r2 = ["happy", "sad", "angry", "neutral"]
    res = compute_cohens_kappa(r1, r2)
    assert res["kappa"] == 1.0
    assert res["observed_agreement"] == 1.0
    assert res["n"] == 4


def test_cohens_kappa_partial_agreement():
    r1 = ["happy", "happy", "sad", "neutral"]
    r2 = ["happy", "sad", "sad", "neutral"]
    res = compute_cohens_kappa(r1, r2)
    assert res["n"] == 4
    assert res["observed_agreement"] == 0.75
    assert 0.0 < res["kappa"] < 1.0


def test_cohens_kappa_handles_blanks():
    r1 = ["happy", "", None, "neutral"]
    r2 = ["happy", "sad", "angry", "neutral"]
    res = compute_cohens_kappa(r1, r2)
    assert res["n"] == 2
    assert res["kappa"] == 1.0


def test_expected_calibration_error():
    # 4 predictions with varying confidence and accuracy
    confidences = [0.90, 0.85, 0.60, 0.40]
    accuracies = [1, 1, 0, 0]
    ece_res = compute_expected_calibration_error(confidences, accuracies, n_bins=5)
    assert ece_res["n"] == 4
    assert 0.0 <= ece_res["ece"] <= 1.0
    assert len(ece_res["bins"]) == 5


def test_multi_annotator_merge_flags_conflict(tmp_path):
    # Synthetic annotator records (test-only)
    annotator_data = {
        "SR": {
            "clip1.wav": {"file": "clip1.wav", "true_voice_emotion": "happy", "true_text_sentiment": "positive", "source": "test"},
            "clip2.wav": {"file": "clip2.wav", "true_voice_emotion": "angry", "true_text_sentiment": "negative", "source": "test"},
        },
        "AK": {
            "clip1.wav": {"file": "clip1.wav", "true_voice_emotion": "happy", "true_text_sentiment": "positive", "source": "test"},
            "clip2.wav": {"file": "clip2.wav", "true_voice_emotion": "sad", "true_text_sentiment": "negative", "source": "test"},
        },
    }

    analysis = analyze_agreement(annotator_data)
    assert len(analysis["annotators"]) == 2
    assert analysis["total_files"] == 2
    # clip1 agreed
    assert "clip1.wav" in analysis["consensus_records"]
    assert analysis["consensus_records"]["clip1.wav"]["true_voice_emotion"] == "happy"
    # clip2 disagreed on voice emotion (angry vs sad)
    assert len(analysis["disagreements"]) == 1
    assert analysis["disagreements"][0]["file"] == "clip2.wav"
    assert "true_voice_emotion" in analysis["disagreements"][0]["conflicts"]

    # Write consensus
    out_file = tmp_path / "consensus.csv"
    saved = write_consensus_dataset(analysis["consensus_records"], out_file)
    assert saved == 1
    assert out_file.exists()
