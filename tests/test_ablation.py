"""
Tests for the Ablation Framework.

Covers:
- Voice-only vs Text-only vs Multimodal Fusion confidence arbitration
- Correct weight assignment (voice-only: audio_weight=1.0, text-only: text_weight=1.0)
- Production config is NOT modified by the ablation framework
- Ground truth unavailable path (pending data reporting)
- Ground truth available path (accuracy/F1 computed correctly)
- Per-sample comparison records
"""

import pytest

from app.fusion import MultimodalFusion
from config import DEFAULT_AUDIO_WEIGHT, DEFAULT_TEXT_WEIGHT, CONFIDENCE_THRESHOLD
from evaluation.ablation import AblationEvaluator, AblationResult


def make_sample_data(audio_emotion, audio_conf, text_sentiment, text_conf, gt_dir=None, filename="test.wav"):
    """Build a minimal ablation sample dict."""
    return {
        "filename": filename,
        "audio_result": {"emotion": audio_emotion, "confidence": audio_conf},
        "text_result": {"sentiment": text_sentiment, "confidence": text_conf},
        "ground_truth_direction": gt_dir,
    }


# ---------------------------------------------------------
# Production config remains untouched
# ---------------------------------------------------------

def test_ablation_does_not_modify_production_config():
    """Running the ablation evaluator must not change config.py constants."""
    import config
    original_audio_weight = config.DEFAULT_AUDIO_WEIGHT
    original_text_weight = config.DEFAULT_TEXT_WEIGHT
    original_threshold = config.CONFIDENCE_THRESHOLD

    evaluator = AblationEvaluator()
    samples = [
        make_sample_data("angry", 0.80, "positive", 0.70),
        make_sample_data("neutral", 0.60, "neutral", 0.65),
    ]
    evaluator.evaluate(samples)

    # Production config constants must be completely unchanged
    assert config.DEFAULT_AUDIO_WEIGHT == original_audio_weight
    assert config.DEFAULT_TEXT_WEIGHT == original_text_weight
    assert config.CONFIDENCE_THRESHOLD == original_threshold


# ---------------------------------------------------------
# Ablation weight assignment
# ---------------------------------------------------------

def test_voice_only_uses_audio_weight_one():
    evaluator = AblationEvaluator()
    assert evaluator.voice_only_fusion.audio_weight == 1.0
    assert evaluator.voice_only_fusion.text_weight == 0.0


def test_text_only_uses_text_weight_one():
    evaluator = AblationEvaluator()
    assert evaluator.text_only_fusion.audio_weight == 0.0
    assert evaluator.text_only_fusion.text_weight == 1.0


def test_multimodal_uses_equal_weights():
    evaluator = AblationEvaluator()
    assert evaluator.multimodal_fusion.audio_weight == 0.5
    assert evaluator.multimodal_fusion.text_weight == 0.5


# ---------------------------------------------------------
# Confidence arbitration is modality-dependent
# ---------------------------------------------------------

def test_voice_only_confidence_reflects_audio_only():
    """Voice-only should produce confidence purely from the audio modality."""
    evaluator = AblationEvaluator()

    samples = [make_sample_data("angry", 0.90, "positive", 0.10)]
    result = evaluator.evaluate(samples)

    # Voice-only mean confidence = audio confidence (text weight=0)
    assert result.voice_only.mean_confidence == pytest.approx(0.90, abs=0.01)


def test_text_only_confidence_reflects_text_only():
    """Text-only should produce confidence purely from the text modality."""
    evaluator = AblationEvaluator()

    samples = [make_sample_data("angry", 0.10, "positive", 0.85)]
    result = evaluator.evaluate(samples)

    # Text-only mean confidence = text confidence (audio weight=0)
    assert result.text_only.mean_confidence == pytest.approx(0.85, abs=0.01)


def test_multimodal_confidence_is_average():
    """Multimodal mean confidence should be the average of audio and text confidences."""
    evaluator = AblationEvaluator()

    samples = [make_sample_data("angry", 0.80, "negative", 0.60)]
    result = evaluator.evaluate(samples)

    expected = (0.80 + 0.60) / 2
    assert result.voice_text_fusion.mean_confidence == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------
# No ground truth path
# ---------------------------------------------------------

def test_ablation_without_ground_truth_reports_pending():
    evaluator = AblationEvaluator()

    samples = [
        make_sample_data("neutral", 0.70, "neutral", 0.65),
        make_sample_data("happy", 0.80, "positive", 0.75),
    ]
    result = evaluator.evaluate(samples)

    assert result.has_ground_truth is False
    assert result.total_samples == 2
    assert result.voice_only.accuracy is None
    assert result.text_only.accuracy is None
    assert result.voice_text_fusion.accuracy is None

    table = result.format_table()
    assert "Pending Data" in table
    assert "Voice-Only" in table
    assert "Text-Only" in table
    assert "Voice + Text Fusion" in table


# ---------------------------------------------------------
# Ground truth available path
# ---------------------------------------------------------

def test_ablation_with_ground_truth_computes_accuracy():
    evaluator = AblationEvaluator()

    # 4 samples where text modality should be more correct
    samples = [
        make_sample_data("angry", 0.90, "positive", 0.80, gt_dir="positive"),
        make_sample_data("sad", 0.85, "negative", 0.75, gt_dir="negative"),
        make_sample_data("angry", 0.90, "positive", 0.80, gt_dir="positive"),
        make_sample_data("neutral", 0.70, "neutral", 0.75, gt_dir="neutral"),
    ]

    result = evaluator.evaluate(samples)

    assert result.has_ground_truth is True
    assert result.total_samples == 4
    assert result.text_only.accuracy is not None
    assert result.voice_only.accuracy is not None
    assert result.voice_text_fusion.accuracy is not None

    # Text-only should be 100% here (text matches gt perfectly)
    assert result.text_only.accuracy == 1.0


# ---------------------------------------------------------
# Per-sample comparison records
# ---------------------------------------------------------

def test_per_sample_comparison_contains_required_fields():
    evaluator = AblationEvaluator()

    samples = [
        make_sample_data("happy", 0.80, "positive", 0.70, filename="clip1.wav"),
    ]
    result = evaluator.evaluate(samples)

    assert len(result.per_sample_comparison) == 1
    rec = result.per_sample_comparison[0]
    assert "filename" in rec
    assert "voice_only_confidence" in rec
    assert "text_only_confidence" in rec
    assert "fusion_confidence" in rec
    assert "voice_direction" in rec
    assert "text_direction" in rec
    assert "fusion_direction" in rec
    assert "agreement" in rec
    assert "mismatch" in rec


def test_ablation_empty_samples():
    evaluator = AblationEvaluator()
    result = evaluator.evaluate([])

    assert result.total_samples == 0
    assert result.has_ground_truth is False
