"""
Unit tests for app/prosody.py (Prosody v2 Rebuild).

Synthetic / mocked inputs clearly marked test-only.
Verifies:
- Loudness percentile extraction
- F0 and voiced ratio tracking
- Speech rate calculation from timestamps
- Component breakdown in [0, 1]
- Voice arousal expected value
- Text negativity extraction
"""

import numpy as np
import pytest

import config
from app.prosody import (
    compute_recording_loudness_percentiles,
    compute_recording_f0,
    compute_prosodic_arousal,
    compute_voice_arousal,
    compute_text_negativity,
    ProsodyResult,
)


def _generate_synthetic_tone(freq: float, duration: float, sr: int = 16000, amp: float = 0.5) -> np.ndarray:
    """Generate a clean synthetic sine tone (test-only input)."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_compute_recording_loudness_percentiles():
    # Empty audio returns safe defaults
    p_low, p_high = compute_recording_loudness_percentiles(np.array([]))
    assert p_high > p_low

    # Quiet vs loud synthetic audio
    quiet = _generate_synthetic_tone(440.0, 1.0, amp=0.01)
    loud = _generate_synthetic_tone(440.0, 1.0, amp=0.8)
    combined = np.concatenate([quiet, loud])

    low_db, high_db = compute_recording_loudness_percentiles(combined)
    assert high_db > low_db
    assert low_db < -20.0
    assert high_db > -10.0


def test_compute_recording_f0():
    # Empty audio returns empty
    f0, voiced = compute_recording_f0(np.array([]))
    assert len(f0) == 0

    # 440 Hz tone has voiced frames around 440 Hz
    audio = _generate_synthetic_tone(440.0, 0.5, sr=16000, amp=0.5)
    f0, voiced = compute_recording_f0(audio, sr=16000)
    assert len(f0) > 0
    valid_f0 = f0[voiced & ~np.isnan(f0)]
    if len(valid_f0) > 0:
        assert np.median(valid_f0) == pytest.approx(440.0, abs=20.0)


def test_compute_prosodic_arousal_loud_vs_quiet():
    sr = 16000
    quiet = _generate_synthetic_tone(440.0, 1.0, sr=sr, amp=0.01)
    loud = _generate_synthetic_tone(440.0, 1.0, sr=sr, amp=0.7)

    res_quiet = compute_prosodic_arousal(
        audio_segment=quiet,
        sr=sr,
        word_count=2,
        duration_seconds=1.0,
        recording_percentiles=(-40.0, -10.0),
    )
    res_loud = compute_prosodic_arousal(
        audio_segment=loud,
        sr=sr,
        word_count=2,
        duration_seconds=1.0,
        recording_percentiles=(-40.0, -10.0),
    )

    assert 0.0 <= res_quiet.arousal <= 1.0
    assert 0.0 <= res_loud.arousal <= 1.0
    assert res_loud.arousal > res_quiet.arousal
    assert res_loud.components["loudness_normalized"] > res_quiet.components["loudness_normalized"]


def test_compute_prosodic_arousal_speech_rate():
    sr = 16000
    audio = _generate_synthetic_tone(300.0, 2.0, sr=sr, amp=0.3)

    # Slow rate (2 words in 2 seconds = 1 wps)
    slow_res = compute_prosodic_arousal(audio, sr=sr, word_count=2, duration_seconds=2.0)
    # Fast rate (8 words in 2 seconds = 4 wps)
    fast_res = compute_prosodic_arousal(audio, sr=sr, word_count=8, duration_seconds=2.0)

    assert fast_res.components["speech_rate_words_per_sec"] > slow_res.components["speech_rate_words_per_sec"]
    assert fast_res.components["speech_rate_normalized"] >= slow_res.components["speech_rate_normalized"]


def test_compute_prosodic_arousal_with_timestamps():
    sr = 16000
    audio = _generate_synthetic_tone(220.0, 1.5, sr=sr, amp=0.4)
    words = [
        {"word": "hello", "start": 0.2, "end": 0.6},
        {"word": "there", "start": 0.7, "end": 1.1},
    ]
    res = compute_prosodic_arousal(
        audio_segment=audio,
        sr=sr,
        word_count=2,
        duration_seconds=1.5,
        word_timestamps=words,
    )
    assert res.components["speech_rate_words_per_sec"] > 0
    assert "loudness" in res.components["weights"]


def test_compute_voice_arousal():
    # Empty
    assert compute_voice_arousal({}) == 0.0

    # 100% angry (high arousal = 1.0)
    angry_scores = {"angry": 1.0, "happy": 0.0, "neutral": 0.0, "sad": 0.0}
    assert compute_voice_arousal(angry_scores) == 1.0

    # 100% sad (low arousal = 0.15)
    sad_scores = {"angry": 0.0, "happy": 0.0, "neutral": 0.0, "sad": 1.0}
    assert compute_voice_arousal(sad_scores) == 0.15

    # Mixed angry + happy vs angry + sad
    high_arousal_mix = {"angry": 0.5, "happy": 0.5, "neutral": 0.0, "sad": 0.0}
    low_arousal_mix = {"angry": 0.5, "sad": 0.5, "neutral": 0.0, "happy": 0.0}
    assert compute_voice_arousal(high_arousal_mix) > compute_voice_arousal(low_arousal_mix)


def test_compute_text_negativity():
    assert compute_text_negativity({}) == 0.0
    assert compute_text_negativity({"negative": 0.85, "neutral": 0.10, "positive": 0.05}) == 0.85
