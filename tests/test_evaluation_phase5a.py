import csv
import io
import os
import tempfile
import pytest

import config
from evaluation.harness import EvaluationHarness, SamplePrediction, EvaluationResult
from evaluation.dataset import EvaluationDataset, EvaluationSample
from evaluation.run_evaluation import compute_sample_cache_key


def test_new_columns_in_csv_export():
    """Verify that all Phase 5a columns are present in evaluation_predictions CSV."""
    sample = EvaluationSample(
        filename="dummy.wav",
        true_voice_emotion="happy",
        true_text_sentiment="positive",
        true_speaker_count=1,
    )
    dataset = EvaluationDataset([sample])

    pred = SamplePrediction(
        filename="dummy.wav",
        pred_voice_emotion="happy",
        pred_text_sentiment="positive",
        pred_transcript="hello world",
        pred_speaker_count=1,
        extra={
            "voice_backend": "emotion2vec",
            "voice_conf_cap": 0.65,
            "whisper_model_size": "base",
            "sanity_rule_applied": False,
            "n_segments": 2,
            "uncertain_rate": 0.0,
            "asr_flags": "none",
            "mean_raw_confidence": 0.88,
            "detected_language": "en",
            "clip_level_prediction": "happy",
            "per_segment_predictions": "Turn 1 Happy, Turn 2 Happy",
            "aggregation_rule": "dominant_voice_emotion argmax",
        },
    )

    harness = EvaluationHarness()
    eval_res = harness.evaluate_predictions(dataset, [pred])

    # Check that record forwarded the new columns
    rec = eval_res.sample_records[0]
    assert rec["voice_backend"] == "emotion2vec"
    assert rec["voice_conf_cap"] == 0.65
    assert rec["whisper_model_size"] == "base"
    assert rec["sanity_rule_applied"] is False
    assert rec["n_segments"] == 2
    assert rec["uncertain_rate"] == 0.0
    assert rec["asr_flags"] == "none"
    assert rec["mean_raw_confidence"] == 0.88
    assert rec["detected_language"] == "en"


def test_uncertain_rate_and_incorrect_metric_treatment():
    """'uncertain' is treated as incorrect and uncertain_rate is calculated."""
    s1 = EvaluationSample(filename="1.wav", true_voice_emotion="happy")
    s2 = EvaluationSample(filename="2.wav", true_voice_emotion="sad")
    dataset = EvaluationDataset([s1, s2])

    # 1 prediction matches, 1 prediction is uncertain
    p1 = SamplePrediction(filename="1.wav", pred_voice_emotion="happy")
    p2 = SamplePrediction(filename="2.wav", pred_voice_emotion="uncertain")

    harness = EvaluationHarness()
    res = harness.evaluate_predictions(dataset, [p1, p2])

    ve = res.voice_emotion_metrics
    # Accuracy should be 1/2 = 0.5 because 'uncertain' != 'sad'
    assert ve["accuracy"] == 0.5
    assert ve["uncertain_rate"] == 0.5
    assert ve["uncertain_count"] == 1

    summary = res.format_summary()
    assert "Uncertain rate: 50.0%" in summary or "Uncertain: 50.0%" in summary


def test_cache_invalidation_for_each_key_component(tmp_path, monkeypatch):
    """Every specified config value, model ID, and file content change invalidates cache key."""
    test_file = tmp_path / "audio.wav"
    test_file.write_bytes(b"INITIAL_AUDIO_BYTES_12345")

    base_key = compute_sample_cache_key(str(test_file))

    # 1. File bytes change
    test_file.write_bytes(b"MODIFIED_AUDIO_BYTES_67890")
    key_file_change = compute_sample_cache_key(str(test_file))
    assert key_file_change != base_key
    test_file.write_bytes(b"INITIAL_AUDIO_BYTES_12345")  # restore

    # 2. VOICE_BACKEND change
    monkeypatch.setattr(config, "VOICE_BACKEND", "superb")
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "VOICE_BACKEND", "emotion2vec")

    # 3. VOICE_CONF_CAP change
    monkeypatch.setattr(config, "VOICE_CONF_CAP", 0.50)
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "VOICE_CONF_CAP", 0.65)

    # 4. VOICE_SANITY_RULE_BACKENDS change
    monkeypatch.setattr(config, "VOICE_SANITY_RULE_BACKENDS", ["emotion2vec", "superb"])
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "VOICE_SANITY_RULE_BACKENDS", ["superb"])

    # 5. VOICE_CALIBRATION_PROSODY_RULE_ENABLED change
    monkeypatch.setattr(config, "VOICE_CALIBRATION_PROSODY_RULE_ENABLED", False)
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "VOICE_CALIBRATION_PROSODY_RULE_ENABLED", True)

    # 6. STT_MODEL_SIZE change
    monkeypatch.setattr(config, "STT_MODEL_SIZE", "small")
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "STT_MODEL_SIZE", "base")

    # 7. STT_LANGUAGE change
    monkeypatch.setattr(config, "STT_LANGUAGE", "en")
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "STT_LANGUAGE", "auto")

    # 8. VOICE_MIN_SEGMENT_SEC change
    monkeypatch.setattr(config, "VOICE_MIN_SEGMENT_SEC", 1.5)
    assert compute_sample_cache_key(str(test_file)) != base_key
    monkeypatch.setattr(config, "VOICE_MIN_SEGMENT_SEC", 1.0)

    # 9. Model IDs change
    monkeypatch.setattr(config, "AUDIO_EMOTION_MODEL_NAME", "custom/model")
    assert compute_sample_cache_key(str(test_file)) != base_key


def test_old_results_loading_without_new_fields():
    """Old results without Phase 5a fields load safely without KeyError/None errors."""
    sample = EvaluationSample(filename="old.wav", true_voice_emotion="neutral")
    dataset = EvaluationDataset([sample])

    # Old prediction lacking all new Phase 5a fields in extra
    old_pred = SamplePrediction(
        filename="old.wav",
        pred_voice_emotion="neutral",
        extra={},
    )

    harness = EvaluationHarness()
    eval_res = harness.evaluate_predictions(dataset, [old_pred])

    rec = eval_res.sample_records[0]
    # Check fallback/None-safe behavior
    assert rec.get("filename") == "old.wav"
    assert rec.get("true_voice_emotion") == "neutral"
    assert rec.get("pred_voice_emotion") == "neutral"

    # None-safe summary generation
    summary = eval_res.format_summary()
    assert "MULTIMODAL EMOTION ANALYSIS" in summary
