"""
Tests for the speaker diarization module.

These tests focus on:
- pipeline loading behavior
- annotation selection
- speaker label mapping
- timestamp overlap assignment
- nearest-turn fallback
- unknown speaker handling
- same-speaker turn merging
- input validation

The actual pyannote Community-1 model is NOT downloaded or executed.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from app.diarization import SpeakerDiarizer


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def diarizer():
    """Return a SpeakerDiarizer with a fake token."""
    return SpeakerDiarizer(hf_token="test-token")


# ---------------------------------------------------------------------
# Initialization / pipeline loading
# ---------------------------------------------------------------------


def test_diarizer_initializes_with_token():
    """Diarizer should store the Hugging Face token and model name."""

    diarizer = SpeakerDiarizer(hf_token="test-token")

    assert diarizer.hf_token == "test-token"
    assert diarizer.model_name == "pyannote/speaker-diarization-community-1"
    assert diarizer._pipeline is None


def test_load_pipeline_requires_hf_token():
    """Pipeline loading should fail clearly when no HF token exists."""

    diarizer = SpeakerDiarizer(hf_token="")

    with patch.object(diarizer, "hf_token", None):
        with pytest.raises(
            RuntimeError,
            match="No Hugging Face token found",
        ):
            diarizer._load_pipeline()


def test_load_pipeline_import_error():
    """A missing pyannote installation should produce a clear error."""

    diarizer = SpeakerDiarizer(hf_token="test-token")

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise ImportError("pyannote.audio not installed")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(
            RuntimeError,
            match="pyannote.audio is not installed",
        ):
            diarizer._load_pipeline()


def test_load_pipeline_is_cached(diarizer):
    """Pipeline should not be recreated after the first successful load."""

    fake_pipeline = Mock()

    diarizer._pipeline = fake_pipeline

    result = diarizer._load_pipeline()

    assert result is fake_pipeline


# ---------------------------------------------------------------------
# Annotation selection
# ---------------------------------------------------------------------


def test_get_annotation_prefers_exclusive_diarization():
    """Community-1 exclusive diarization should be preferred."""

    exclusive = Mock()
    normal = Mock()

    output = SimpleNamespace(
        exclusive_speaker_diarization=exclusive,
        speaker_diarization=normal,
    )

    result = SpeakerDiarizer._get_annotation(output)

    assert result is exclusive


def test_get_annotation_falls_back_to_regular_diarization():
    """Regular speaker diarization should be used when exclusive is absent."""

    normal = Mock()

    output = SimpleNamespace(
        exclusive_speaker_diarization=None,
        speaker_diarization=normal,
    )

    result = SpeakerDiarizer._get_annotation(output)

    assert result is normal


def test_get_annotation_accepts_annotation_like_object():
    """An Annotation-like object should be accepted as a fallback."""

    # Use SimpleNamespace so missing attributes really are missing.
    annotation = SimpleNamespace()

    def itertracks():
        return []

    annotation.itertracks = itertracks

    result = SpeakerDiarizer._get_annotation(annotation)

    assert result is annotation


def test_get_annotation_rejects_unexpected_output():
    """Unexpected pipeline output should raise a clear RuntimeError."""

    output = SimpleNamespace(
        exclusive_speaker_diarization=None,
        speaker_diarization=None,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected output format",
    ):
        SpeakerDiarizer._get_annotation(output)


# ---------------------------------------------------------------------
# Interval calculations
# ---------------------------------------------------------------------


def test_overlap_returns_correct_duration():
    """Overlap should return the shared duration between intervals."""

    result = SpeakerDiarizer._overlap(
        1.0,
        4.0,
        2.0,
        5.0,
    )

    assert result == 2.0


def test_overlap_returns_zero_when_no_overlap():
    """Non-overlapping intervals should have zero overlap."""

    result = SpeakerDiarizer._overlap(
        0.0,
        2.0,
        3.0,
        5.0,
    )

    assert result == 0.0


def test_distance_returns_zero_for_overlapping_intervals():
    """Distance should be zero when intervals overlap."""

    result = SpeakerDiarizer._distance_to_turn(
        1.0,
        4.0,
        2.0,
        5.0,
    )

    assert result == 0.0


def test_distance_returns_gap_between_intervals():
    """Distance should return the gap between non-overlapping intervals."""

    result = SpeakerDiarizer._distance_to_turn(
        0.0,
        2.0,
        3.0,
        5.0,
    )

    assert result == 1.0


# ---------------------------------------------------------------------
# Speaker label mapping
# ---------------------------------------------------------------------


def test_build_label_map_assigns_speaker_a_and_b():
    """First two raw speaker IDs should become Speaker A and Speaker B."""

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "SPEAKER_01",
        },
    ]

    label_map = SpeakerDiarizer._build_label_map(turns)

    assert label_map["SPEAKER_00"] == "Speaker A"
    assert label_map["SPEAKER_01"] == "Speaker B"


def test_build_label_map_does_not_duplicate_existing_speaker():
    """Repeated speaker IDs should keep the same human-readable label."""

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 4.0,
            "end": 6.0,
            "speaker": "SPEAKER_01",
        },
    ]

    label_map = SpeakerDiarizer._build_label_map(turns)

    assert len(label_map) == 2
    assert label_map["SPEAKER_00"] == "Speaker A"
    assert label_map["SPEAKER_01"] == "Speaker B"


# ---------------------------------------------------------------------
# Speaker assignment
# ---------------------------------------------------------------------


def test_assign_speaker_using_maximum_overlap(diarizer):
    """A segment should be assigned to the turn with greatest overlap."""

    segments = [
        {
            "segment_index": 0,
            "start_seconds": 1.0,
            "end_seconds": 3.0,
            "transcript": "Hello",
        }
    ]

    turns = [
        {
            "start": 0.0,
            "end": 1.5,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 1.5,
            "end": 4.0,
            "speaker": "SPEAKER_01",
        },
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert len(result) == 1
    assert result[0]["speaker"] == "Speaker B"


def test_assign_multiple_segments_to_correct_speakers(diarizer):
    """Multiple transcription segments should receive correct speakers."""

    segments = [
        {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "transcript": "Hello",
        },
        {
            "segment_index": 1,
            "start_seconds": 2.0,
            "end_seconds": 4.0,
            "transcript": "How are you?",
        },
        {
            "segment_index": 2,
            "start_seconds": 4.0,
            "end_seconds": 6.0,
            "transcript": "I am fine.",
        },
    ]

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 2.0,
            "end": 4.0,
            "speaker": "SPEAKER_01",
        },
        {
            "start": 4.0,
            "end": 6.0,
            "speaker": "SPEAKER_00",
        },
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert [item["speaker"] for item in result] == [
        "Speaker A",
        "Speaker B",
        "Speaker A",
    ]


def test_assign_speaker_uses_custom_label_map(diarizer):
    """A supplied label map should be respected."""

    segments = [
        {
            "start_seconds": 0.0,
            "end_seconds": 2.0,
        }
    ]

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        }
    ]

    label_map = {
        "SPEAKER_00": "Customer",
    }

    result = diarizer.assign_speakers(
        segments,
        turns,
        label_map=label_map,
    )

    assert result[0]["speaker"] == "Customer"


def test_assign_speaker_uses_nearest_turn_fallback(diarizer):
    """A nearby turn without sufficient overlap should yield Uncertain Speaker, never forcing a label."""

    segments = [
        {
            "start_seconds": 2.0,
            "end_seconds": 2.5,
            "transcript": "Hello",
        }
    ]

    turns = [
        {
            "start": 0.0,
            "end": 1.5,
            "speaker": "SPEAKER_00",
        },
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert result[0]["speaker"] == "Uncertain Speaker"


def test_assign_speaker_returns_unknown_when_turn_is_too_far(diarizer):
    """A distant diarization turn should not be assigned."""

    segments = [
        {
            "start_seconds": 10.0,
            "end_seconds": 11.0,
            "transcript": "Hello",
        }
    ]

    turns = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
        },
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert result[0]["speaker"] == "Unknown"


def test_assign_speakers_returns_unknown_when_turns_are_empty(diarizer):
    """Segments should remain usable when no diarization turns exist."""

    segments = [
        {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "transcript": "Hello",
        },
        {
            "segment_index": 1,
            "start_seconds": 2.0,
            "end_seconds": 4.0,
            "transcript": "World",
        },
    ]

    result = diarizer.assign_speakers(segments, [])

    assert len(result) == 2
    assert result[0]["speaker"] == "Unknown"
    assert result[1]["speaker"] == "Unknown"


def test_assign_speakers_returns_empty_for_empty_segments(diarizer):
    """No transcription segments should produce an empty result."""

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        }
    ]

    result = diarizer.assign_speakers([], turns)

    assert result == []


def test_assign_speaker_preserves_existing_segment_data(diarizer):
    """Speaker assignment should not remove existing segment information."""

    segments = [
        {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "transcript": "Hello there",
            "voice_emotion": "happy",
            "text_sentiment": "positive",
        }
    ]

    turns = [
        {
            "start": 0.0,
            "end": 2.0,
            "speaker": "SPEAKER_00",
        }
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert result[0]["segment_index"] == 0
    assert result[0]["transcript"] == "Hello there"
    assert result[0]["voice_emotion"] == "happy"
    assert result[0]["text_sentiment"] == "positive"
    assert result[0]["speaker"] == "Speaker A"


# ---------------------------------------------------------------------
# Diarization execution
# ---------------------------------------------------------------------


def test_diarize_rejects_empty_audio_path(diarizer):
    """An empty audio path should raise ValueError."""

    with pytest.raises(
        ValueError,
        match="audio_path cannot be empty",
    ):
        diarizer.diarize("")


def test_diarize_rejects_missing_audio_file(diarizer):
    """A nonexistent audio file should raise FileNotFoundError."""

    with pytest.raises(
        FileNotFoundError,
        match="Audio file not found",
    ):
        diarizer.diarize("does_not_exist.wav")


def test_diarize_extracts_and_merges_speaker_turns(diarizer):
    """
    Diarization should extract turns and merge consecutive turns
    belonging to the same speaker when their gap is <= 0.25 seconds.
    """

    fake_annotation = Mock()

    fake_annotation.itertracks.return_value = [
        (
            SimpleNamespace(start=0.0, end=2.0),
            None,
            "SPEAKER_00",
        ),
        (
            SimpleNamespace(start=2.1, end=4.0),
            None,
            "SPEAKER_00",
        ),
        (
            SimpleNamespace(start=4.0, end=6.0),
            None,
            "SPEAKER_01",
        ),
    ]

    fake_output = SimpleNamespace(
        exclusive_speaker_diarization=fake_annotation,
        speaker_diarization=None,
    )

    fake_pipeline = Mock(return_value=fake_output)

    diarizer._pipeline = fake_pipeline

    with patch("os.path.exists", return_value=True):
        result = diarizer.diarize("sample.wav")

    assert len(result) == 2

    assert result[0]["speaker"] == "SPEAKER_00"
    assert result[0]["start"] == 0.0
    assert result[0]["end"] == 4.0

    assert result[1]["speaker"] == "SPEAKER_01"
    assert result[1]["start"] == 4.0
    assert result[1]["end"] == 6.0

    fake_pipeline.assert_called_once_with("sample.wav")


def test_diarize_skips_invalid_turns(diarizer):
    """Turns where end <= start should be ignored."""

    fake_annotation = Mock()

    fake_annotation.itertracks.return_value = [
        (
            SimpleNamespace(start=2.0, end=1.0),
            None,
            "SPEAKER_00",
        ),
        (
            SimpleNamespace(start=0.0, end=2.0),
            None,
            "SPEAKER_01",
        ),
    ]

    fake_output = SimpleNamespace(
        exclusive_speaker_diarization=fake_annotation,
        speaker_diarization=None,
    )

    diarizer._pipeline = Mock(return_value=fake_output)

    with patch("os.path.exists", return_value=True):
        result = diarizer.diarize("sample.wav")

    assert len(result) == 1
    assert result[0]["speaker"] == "SPEAKER_01"


def test_diarize_raises_when_no_valid_turns(diarizer):
    """Diarization should fail clearly if no valid speaker turns exist."""

    fake_annotation = Mock()

    fake_annotation.itertracks.return_value = [
        (
            SimpleNamespace(start=2.0, end=1.0),
            None,
            "SPEAKER_00",
        )
    ]

    fake_output = SimpleNamespace(
        exclusive_speaker_diarization=fake_annotation,
        speaker_diarization=None,
    )

    diarizer._pipeline = Mock(return_value=fake_output)

    with patch("os.path.exists", return_value=True):
        with pytest.raises(
            RuntimeError,
            match="no speaker turns were detected",
        ):
            diarizer.diarize("sample.wav")


# ---------------------------------------------------------------------
# End-to-end local logic test
# ---------------------------------------------------------------------


def test_diarization_to_segment_assignment_flow(diarizer):
    """
    Verify the complete local flow:

    diarization turns
        ->
    Speaker A / Speaker B labels
        ->
    Whisper-style segments
        ->
    speaker assignment
    """

    turns = [
        {
            "start": 0.0,
            "end": 3.0,
            "speaker": "SPEAKER_00",
        },
        {
            "start": 3.0,
            "end": 6.0,
            "speaker": "SPEAKER_01",
        },
    ]

    segments = [
        {
            "segment_index": 0,
            "start_seconds": 0.5,
            "end_seconds": 2.5,
            "transcript": "Hello, how are you?",
        },
        {
            "segment_index": 1,
            "start_seconds": 3.2,
            "end_seconds": 5.5,
            "transcript": "I am doing well.",
        },
    ]

    result = diarizer.assign_speakers(segments, turns)

    assert len(result) == 2

    assert result[0]["speaker"] == "Speaker A"
    assert result[1]["speaker"] == "Speaker B"

    assert result[0]["transcript"] == "Hello, how are you?"
    assert result[1]["transcript"] == "I am doing well."