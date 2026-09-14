"""
Tests for Evaluation Dataset Loader and Schema.

Covers:
- Standard CSV parsing
- Missing / invalid header detection
- Missing ground truth (N/A, empty, case-insensitive)
- Normalization of voice emotion and text sentiment labels
- Integer speaker count parsing
- Dataset filtering and export
"""

import io
import pytest

from evaluation.dataset import (
    EvaluationDataset,
    EvaluationSample,
    normalize_voice_emotion_label,
    normalize_text_sentiment_label,
)


def test_evaluation_sample_normalization():
    """Verify normalization of emotion labels and missing ground truth."""
    sample = EvaluationSample(
        filename="test.wav",
        true_voice_emotion="ANG",
        true_text_sentiment="POS",
        true_speaker_count="2",
        true_transcript="Hello world.",
        notes="Test note",
    )

    assert sample.filename == "test.wav"
    assert sample.true_voice_emotion == "angry"
    assert sample.true_text_sentiment == "positive"
    assert sample.true_speaker_count == 2
    assert sample.true_transcript == "Hello world."
    assert sample.notes == "Test note"


def test_missing_ground_truth_parsing():
    """N/A, na, empty strings, and null should all parse as None."""
    for na_val in ["N/A", "n/a", "NA", "none", "", "  ", None]:
        sample = EvaluationSample(
            filename="test.wav",
            true_voice_emotion=na_val,
            true_text_sentiment=na_val,
            true_speaker_count=na_val,
            true_transcript=na_val,
        )
        assert sample.true_voice_emotion is None
        assert sample.true_text_sentiment is None
        assert sample.true_speaker_count is None
        assert sample.true_transcript is None


def test_dataset_from_csv_string_valid():
    csv_text = (
        "filename,true_voice_emotion,true_text_sentiment,true_speaker_count,true_transcript,notes\n"
        "audio1.wav,happy,positive,1,Great morning!,Speaker 1\n"
        "audio2.wav,sad,negative,2,I feel down.,Speaker 2\n"
        "audio3.wav,N/A,neutral,N/A,N/A,Unlabeled voice\n"
    )

    ds = EvaluationDataset.from_csv_string(csv_text)
    assert len(ds) == 3
    assert ds[0].filename == "audio1.wav"
    assert ds[0].true_voice_emotion == "happy"
    assert ds[0].true_text_sentiment == "positive"
    assert ds[0].true_speaker_count == 1

    assert ds[2].filename == "audio3.wav"
    assert ds[2].true_voice_emotion is None
    assert ds[2].true_text_sentiment == "neutral"
    assert ds[2].true_speaker_count is None
    assert ds[2].true_transcript is None


def test_dataset_missing_column_raises():
    bad_csv = "filename,true_voice_emotion\naudio1.wav,happy\n"
    with pytest.raises(ValueError, match="CSV missing required columns"):
        EvaluationDataset.from_csv_string(bad_csv)


def test_dataset_filtering_utilities():
    csv_text = (
        "filename,true_voice_emotion,true_text_sentiment,true_speaker_count,true_transcript,notes\n"
        "f1.wav,happy,N/A,2,Hello,Sample 1\n"
        "f2.wav,N/A,positive,N/A,N/A,Sample 2\n"
        "f3.wav,N/A,N/A,N/A,N/A,Sample 3\n"
    )
    ds = EvaluationDataset.from_csv_string(csv_text)

    assert len(ds.filter_with_voice_emotion()) == 1
    assert ds.filter_with_voice_emotion()[0].filename == "f1.wav"

    assert len(ds.filter_with_text_sentiment()) == 1
    assert ds.filter_with_text_sentiment()[0].filename == "f2.wav"

    assert len(ds.filter_with_speaker_count()) == 1
    assert ds.filter_with_speaker_count()[0].filename == "f1.wav"

    assert len(ds.filter_with_transcript()) == 1
    assert ds.filter_with_transcript()[0].filename == "f1.wav"


def test_dataset_to_csv_roundtrip():
    ds = EvaluationDataset([
        EvaluationSample(filename="test1.wav", true_voice_emotion="angry", true_speaker_count=1),
        EvaluationSample(filename="test2.wav", true_text_sentiment="neutral"),
    ])

    buf = io.StringIO()
    ds.to_csv(buf)
    content = buf.getvalue()

    reloaded = EvaluationDataset.from_csv_string(content)
    assert len(reloaded) == 2
    assert reloaded[0].filename == "test1.wav"
    assert reloaded[0].true_voice_emotion == "angry"
    assert reloaded[0].true_speaker_count == 1
    assert reloaded[1].filename == "test2.wav"
    assert reloaded[1].true_text_sentiment == "neutral"
