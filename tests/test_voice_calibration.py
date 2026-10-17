"""
Unit tests for app/voice_calibration.py.

Synthetic / mocked inputs clearly marked test-only.
Verifies:
- Window widening for short segments (<1.0s)
- Window bounds clipping against neighbors and audio duration
- Prosody sanity rule dampening of 'sad' when energy and pitch range are high
- Prosody sanity rule no-op for monotone/quiet 'sad' or non-sad emotions
- Temperature scaling fitting gate (N >= 15 threshold)
- Transparent applied_rules logging
"""

import numpy as np
import pytest

import config
from app.voice_calibration import (
    apply_prosody_sanity_rule,
    compute_widened_window,
    VoiceCalibrator,
    CalibrationResult,
)


@pytest.fixture(autouse=True)
def _use_superb_backend(monkeypatch):
    """Voice calibration unit tests were written for the SUPERB backend."""
    monkeypatch.setattr(config, "VOICE_BACKEND", "superb")


def _synthetic_signal(freq: float, duration: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generate synthetic sine signal (test-only input)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compute_widened_window_normal_segment():
    # Segment >= 1.0s should not be widened
    start, end, widened = compute_widened_window(
        start_seconds=1.0,
        end_seconds=2.5,
        all_segment_bounds=[(1.0, 2.5)],
        audio_duration_seconds=5.0,
    )
    assert widened is False
    assert start == 1.0
    assert end == 2.5


def test_compute_widened_window_short_segment():
    # Segment 0.4s should be widened by 0.5s padding on each side
    start, end, widened = compute_widened_window(
        start_seconds=1.0,
        end_seconds=1.4,
        all_segment_bounds=[(1.0, 1.4)],
        audio_duration_seconds=5.0,
    )
    assert widened is True
    assert start == 0.5  # 1.0 - 0.5
    assert end == 1.9    # 1.4 + 0.5


def test_compute_widened_window_clamps_to_neighbors():
    # Segment flanked closely by neighbors at 0.8s and 1.6s
    bounds = [(0.0, 0.8), (1.0, 1.4), (1.6, 2.5)]
    start, end, widened = compute_widened_window(
        start_seconds=1.0,
        end_seconds=1.4,
        all_segment_bounds=bounds,
        audio_duration_seconds=5.0,
    )
    assert widened is True
    assert start >= 0.8  # clamped to previous end
    assert end <= 1.6    # clamped to next start


def test_prosody_sanity_rule_fires_on_loud_varied_sad():
    raw_scores = {"sad": 0.80, "neutral": 0.10, "angry": 0.05, "happy": 0.05}
    # Synthetic loud signal with wide frequency range (100 Hz to 400 Hz sweep)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    freq_sweep = 100.0 + 300.0 * t
    audio = (0.6 * np.sin(2 * np.pi * freq_sweep * t)).astype(np.float32)

    # Pass precomputed F0 with large semitone range (e.g. 100 Hz to 400 Hz is 24 semitones)
    f0 = np.linspace(100.0, 400.0, 30)
    voiced = np.ones(30, dtype=bool)

    res = apply_prosody_sanity_rule(
        raw_scores=raw_scores,
        audio_segment=audio,
        sr=16000,
        recording_percentiles=(-40.0, -10.0),
        f0_segment=f0,
        voiced_flag_segment=voiced,
    )

    assert len(res.applied_rules) == 1
    assert "prosody_sanity_rule" in res.applied_rules[0]
    assert res.scores["sad"] < raw_scores["sad"]
    assert res.scores["angry"] > raw_scores["angry"]


def test_prosody_sanity_rule_no_op_on_quiet_sad():
    raw_scores = {"sad": 0.80, "neutral": 0.10, "angry": 0.05, "happy": 0.05}
    # Very quiet signal
    quiet = _synthetic_signal(200.0, 1.0, amp=0.005)
    f0 = np.full(30, 200.0)
    voiced = np.ones(30, dtype=bool)

    res = apply_prosody_sanity_rule(
        raw_scores=raw_scores,
        audio_segment=quiet,
        sr=16000,
        recording_percentiles=(-40.0, -10.0),
        f0_segment=f0,
        voiced_flag_segment=voiced,
    )

    # Rule should not fire for quiet audio
    assert len(res.applied_rules) == 0
    assert res.scores["sad"] == pytest.approx(0.80)


def test_prosody_sanity_rule_no_op_on_non_sad():
    raw_scores = {"happy": 0.70, "sad": 0.10, "neutral": 0.10, "angry": 0.10}
    loud = _synthetic_signal(300.0, 1.0, amp=0.8)
    res = apply_prosody_sanity_rule(raw_scores, loud, sr=16000)
    assert len(res.applied_rules) == 0
    assert res.emotion == "happy"


def test_voice_calibrator_fit_gate():
    calibrator = VoiceCalibrator()
    # Below N=15 threshold, fit should be a no-op
    tiny_dataset = [({"sad": 0.7, "neutral": 0.3}, "sad")] * 4
    calibrator.fit(tiny_dataset)
    assert calibrator.is_fitted is False
    assert calibrator.temperature == 1.0

    # With N=15 samples, fit succeeds
    adequate_dataset = [
        ({"sad": 0.8, "angry": 0.1, "happy": 0.05, "neutral": 0.05}, "sad")
        for _ in range(16)
    ]
    calibrator.fit(adequate_dataset)
    assert calibrator.is_fitted is True
    assert calibrator.temperature > 0.0
