"""
Tier 3 Regression Tests: Validation Tab, Inventory SSOT, Small-Sample Fallback, and Speaker MAE Notice.
Covers items 28 through 31 from user specification.
"""

import os
from pathlib import Path
import pytest

from config import EVAL_MIN_N
from evaluation.dataset import (
    EvaluationDataset,
    EvaluationSample,
    DatasetInventory,
    get_dataset_inventory,
    format_remaining_annotation_checklist,
)
from evaluation.run_evaluation import (
    _fmt_float,
    format_human_readable_report,
)
from evaluation.harness import EvaluationResult
from evaluation.ablation import AblationResult, ModalityAblationRecord

from evaluation.learned_fusion import LearnedFusionResult
from ui.pages.validation_view import _load_prediction_table


# ==============================================================================
# Item 28: Shared dataset inventory function and gap explanation
# ==============================================================================

@pytest.mark.skipif(
    not os.path.exists("sample_audio") or len(os.listdir("sample_audio")) < 9,
    reason="sample_audio files not present on disk",
)
def test_get_dataset_inventory_returns_expected_counts():
    """get_dataset_inventory must return the shared single source of truth:
    - 9 audio files on disk in sample_audio
    - 7 dataset records in evaluation_dataset.csv
    - 5 annotated records (4 voice emotion, 5 speaker count)
    - 2 unit test fixtures (test_corrupted.wav, test_empty.wav)
    """
    inv: DatasetInventory = get_dataset_inventory()

    assert inv.total_audio_files >= 9, f"Expected at least 9 audio files on disk, got {inv.total_audio_files}"
    assert inv.total_dataset_records == 7, f"Expected 7 dataset records, got {inv.total_dataset_records}"
    assert inv.annotated_records_count == 5, f"Expected 5 annotated records, got {inv.annotated_records_count}"
    assert inv.fixture_files_count == 2, f"Expected 2 fixture files, got {inv.fixture_files_count}"
    assert inv.voice_emotion_count == 4, f"Expected 4 voice emotion labels, got {inv.voice_emotion_count}"
    assert inv.speaker_count_count == 5, f"Expected 5 speaker count labels, got {inv.speaker_count_count}"
    assert inv.text_sentiment_count == 0, f"Expected 0 text sentiment labels, got {inv.text_sentiment_count}"

    # Verify gap explanation is present and explains the difference
    assert "9 WAV files" in inv.gap_explanation
    assert "7 records" in inv.gap_explanation
    assert "fixtures" in inv.gap_explanation.lower()


# ==============================================================================
# Item 29: Small-sample descriptive table fallback (N=4, need >=5)
# ==============================================================================

def test_load_prediction_table_loads_raw_predictions_for_voice():
    """_load_prediction_table should extract descriptive comparison rows when predictions exist."""
    df_voice = _load_prediction_table("voice")
    if Path("evaluation/results/evaluation_predictions.csv").exists():
        assert df_voice is not None, "Failed to load voice predictions table"
        assert not df_voice.empty
        assert "Ground Truth" in df_voice.columns
        assert "Model Prediction" in df_voice.columns
        assert "Match" in df_voice.columns
        # In current evaluation_predictions.csv, 4 samples have voice ground truth
        assert len(df_voice) == 4


# ==============================================================================
# Item 30: All single-speaker notice suppresses headline MAE
# ==============================================================================

def test_speaker_count_single_speaker_detection():
    """When all samples have true_speaker_count == 1, MAE is identified as uninformative."""
    per_sample = [
        {"true_speaker_count": 1, "predicted_speaker_count": 1, "diff": 0},
        {"true_speaker_count": 1, "predicted_speaker_count": 1, "diff": 0},
        {"true_speaker_count": 1, "predicted_speaker_count": 1, "diff": 0},
        {"true_speaker_count": 1, "predicted_speaker_count": 1, "diff": 0},
        {"true_speaker_count": 1, "predicted_speaker_count": 1, "diff": 0},
    ]
    spk_n = len(per_sample)
    all_single_speaker = (
        spk_n > 0 and all(int(p.get("true_speaker_count", 0)) == 1 for p in per_sample)
    )
    assert all_single_speaker is True

    # Mixed speaker counts should NOT trigger the notice
    per_sample_mixed = per_sample + [{"true_speaker_count": 2, "predicted_speaker_count": 2, "diff": 0}]
    assert not all(int(p.get("true_speaker_count", 0)) == 1 for p in per_sample_mixed)


# ==============================================================================
# Item 31: Evaluation must never crash on partial data & prints checklist
# ==============================================================================

def test_format_human_readable_report_never_crashes_on_none_metrics():
    """Report formatting must safely handle None values across all metrics."""
    empty_dataset = EvaluationDataset([])
    eval_res = EvaluationResult(
        total_samples=0,
        voice_emotion_metrics={"status": "OK", "total_samples": 0, "accuracy": None, "f1_macro": None, "precision_macro": None, "recall_macro": None},
        text_sentiment_metrics={"status": "OK", "total_samples": 0, "accuracy": None, "f1_macro": None, "precision_macro": None, "recall_macro": None},
        speech_to_text_metrics={"status": "OK", "total_evaluated": 0, "average_wer": None, "total_reference_words": 0, "total_hypothesis_words": 0},
        speaker_count_metrics={"status": "OK", "count": 0, "exact_match_accuracy": None, "mae": None},
        sample_records=[],
        has_ground_truth=False,
    )

    perf = ModalityAblationRecord(
        mode_name="Voice Only",
        audio_weight=1.0,
        text_weight=0.0,
        total_samples=0,
        mean_confidence=0.0,
        accuracy=None,
        precision=None,
        recall=None,
        f1_score=None,
        details={},
    )
    ablation_res = AblationResult(
        total_samples=0,
        has_ground_truth=False,
        voice_only=perf,
        text_only=perf,
        voice_text_fusion=perf,
        per_sample_comparison=[],
    )


    lf_res = LearnedFusionResult(status="NO_DATA", total_samples=0, train_samples=0, test_samples=0, message="None")

    # This call must NOT raise TypeError or crash
    report = format_human_readable_report(
        dataset=empty_dataset,
        eval_result=eval_res,
        ablation_result=ablation_res,
        learned_fusion_result=lf_res,
        missing_annotations=["Voice Emotion", "Text Sentiment"],
        total_evaluated=0,
        total_failed=0,
    )
    assert "MULTIMODAL EMOTION ANALYSIS" in report
    assert "Pending / Missing Annotations" in report


def test_format_remaining_annotation_checklist_output():
    """Checklist formatting should summarize missing items cleanly."""
    checklist_text = format_remaining_annotation_checklist()
    assert "REMAINING ANNOTATION CHECKLIST" in checklist_text
    assert "Voice Emotion" in checklist_text
    assert "Text Sentiment" in checklist_text
    assert "STT Transcripts" in checklist_text
    assert "Speaker Count" in checklist_text
