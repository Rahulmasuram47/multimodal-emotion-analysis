"""
Unit tests for Diarization Badge States, Uncertain Speaker Overlap Gating, and Multi-Speaker Analytics.

Synthetic / mocked inputs clearly marked test-only.
"""

from unittest.mock import patch
import pytest
import config
from app.diarization import SpeakerDiarizer, get_diarization_status
from app.speaker_analytics import (
    compute_speaker_analytics,
    find_possible_escalation_initiator,
    get_identified_speakers,
)


def test_badge_status_token_present():
    with patch.dict("os.environ", {"HF_TOKEN": "hf_test_secret_token_12345"}):
        status = get_diarization_status()
        assert status["token_present"] is True
        assert "..." in status["token_preview"]
        # Never leak full secret
        assert "hf_test_secret_token_12345" not in status["token_preview"]


def test_badge_status_no_token():
    with patch.dict("os.environ", {}, clear=True):
        status = get_diarization_status()
        assert status["token_present"] is False
        assert status["token_preview"] == ""


def test_uncertain_speaker_when_overlap_below_threshold():
    diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
    ]
    label_map = {"SPEAKER_00": "Speaker A"}

    # Segment is 10.0s long (0.0 to 10.0s). Overlap with turn is 1.0s.
    # Overlap ratio = 1.0 / 10.0 = 0.10 < DIARIZATION_MIN_OVERLAP_RATIO (0.35)
    segments = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "transcript": "Testing low overlap."}
    ]

    labeled = diarizer.assign_speakers(segments, turns, label_map=label_map)
    assert len(labeled) == 1
    assert labeled[0]["speaker"] == "Uncertain Speaker"


def test_confident_speaker_when_overlap_above_threshold():
    diarizer = SpeakerDiarizer.__new__(SpeakerDiarizer)
    turns = [
        {"start": 0.0, "end": 8.0, "speaker": "SPEAKER_00"},
    ]
    label_map = {"SPEAKER_00": "Speaker A"}

    # Segment is 10.0s long (0.0 to 10.0s). Overlap is 8.0s -> ratio 0.80 >= 0.35
    segments = [
        {"start_seconds": 0.0, "end_seconds": 10.0, "transcript": "Testing high overlap."}
    ]

    labeled = diarizer.assign_speakers(segments, turns, label_map=label_map)
    assert len(labeled) == 1
    assert labeled[0]["speaker"] == "Speaker A"


def test_speaker_analytics_gated_when_single_speaker():
    # Only 1 speaker: analytics should return empty list
    segments = [
        {"speaker": "Speaker A", "start_seconds": 0.0, "end_seconds": 3.0, "intensity": 0.20},
        {"speaker": "Speaker A", "start_seconds": 3.5, "end_seconds": 6.0, "intensity": 0.35},
    ]
    analytics = compute_speaker_analytics(segments, total_duration_seconds=6.0)
    assert analytics == []
    initiator = find_possible_escalation_initiator(segments)
    assert initiator is None


def test_speaker_analytics_renders_when_two_speakers():
    segments = [
        {"speaker": "Speaker A", "start_seconds": 0.0, "end_seconds": 4.0, "intensity": 0.20, "segment_index": 1, "transcript": "Hi"},
        {"speaker": "Speaker B", "start_seconds": 4.5, "end_seconds": 8.0, "intensity": 0.65, "segment_index": 2, "transcript": "Why is this broken?"},
        {"speaker": "Speaker A", "start_seconds": 8.5, "end_seconds": 12.0, "intensity": 0.30, "segment_index": 3, "transcript": "Let me help"},
        {"speaker": "Speaker B", "start_seconds": 12.5, "end_seconds": 16.0, "intensity": 0.80, "segment_index": 4, "transcript": "I need it now!"},
    ]
    analytics = compute_speaker_analytics(segments, total_duration_seconds=16.0)
    assert len(analytics) == 2
    speakers = [a.speaker for a in analytics]
    assert "Speaker A" in speakers
    assert "Speaker B" in speakers

    # Talk time percentage sum should approximate total speech share
    total_talk = sum(a.talk_time_pct for a in analytics)
    assert total_talk > 0.0

    # Escalation initiator: Speaker B initiated a jump from 0.20 to 0.65 (+0.45 >= 0.15)
    initiator = find_possible_escalation_initiator(segments)
    assert initiator is not None
    assert initiator["speaker"] == "Speaker B"
    assert initiator["intensity_rise"] == pytest.approx(0.50)
