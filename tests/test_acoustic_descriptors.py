import numpy as np
import pytest

from app.prosody import (
    compute_segment_acoustic_descriptors,
    compute_recording_frame_rms,
)
from ui.components.export_bar import build_json_export_payload


def test_pitch_trend_synthetic_curves():
    # 30 frames: rising pitch
    # first 10 frames: 150 Hz, last 10 frames: 250 Hz (delta st = 12 * log2(250/150) = ~8.8 st > 2.0)
    f0_rising = np.concatenate([
        np.full(10, 150.0),
        np.full(10, 200.0),
        np.full(10, 250.0),
    ])
    voiced = np.ones(30, dtype=bool)
    rms = np.full(30, 0.05)

    res_rising = compute_segment_acoustic_descriptors(
        f0_slice=f0_rising,
        voiced_slice=voiced,
        rms_slice=rms,
        n_words=5,
        duration=2.0,
        rec_rms_percentiles=(-40.0, -10.0),
        n_recording_segments=3,
    )
    assert res_rising["pitch_trend"] == "rising"
    assert res_rising["pitch_trend_semitones"] > 2.0
    assert res_rising["pitch_range_st"] > 0

    # Falling pitch
    f0_falling = np.concatenate([
        np.full(10, 250.0),
        np.full(10, 200.0),
        np.full(10, 150.0),
    ])
    res_falling = compute_segment_acoustic_descriptors(
        f0_slice=f0_falling,
        voiced_slice=voiced,
        rms_slice=rms,
        n_words=5,
        duration=2.0,
        rec_rms_percentiles=(-40.0, -10.0),
        n_recording_segments=3,
    )
    assert res_falling["pitch_trend"] == "falling"
    assert res_falling["pitch_trend_semitones"] < -2.0

    # Flat pitch
    f0_flat = np.full(30, 200.0)
    res_flat = compute_segment_acoustic_descriptors(
        f0_slice=f0_flat,
        voiced_slice=voiced,
        rms_slice=rms,
        n_words=5,
        duration=2.0,
        rec_rms_percentiles=(-40.0, -10.0),
        n_recording_segments=3,
    )
    assert res_flat["pitch_trend"] == "flat"
    assert abs(res_flat["pitch_trend_semitones"]) <= 2.0


def test_pitch_trend_unvoiced_or_insufficient():
    # < 10 voiced frames in total
    f0 = np.full(30, 200.0)
    voiced = np.zeros(30, dtype=bool)
    voiced[:5] = True  # only 5 voiced frames

    res = compute_segment_acoustic_descriptors(
        f0_slice=f0,
        voiced_slice=voiced,
        rms_slice=np.full(30, 0.05),
        n_words=4,
        duration=2.0,
    )
    assert res["pitch_trend"] == "n/a"

    # < 4 voiced frames in one third
    voiced_sparse = np.zeros(30, dtype=bool)
    voiced_sparse[0:2] = True   # only 2 in first third
    voiced_sparse[10:18] = True # 8 in middle
    voiced_sparse[20:25] = True # 5 in last third (total 15 voiced, but first third has 2)
    res_sparse = compute_segment_acoustic_descriptors(
        f0_slice=f0,
        voiced_slice=voiced_sparse,
        rms_slice=np.full(30, 0.05),
        n_words=4,
        duration=2.0,
    )
    assert res_sparse["pitch_trend"] == "n/a"


def test_energy_level_and_few_segments():
    rms_slice = np.full(20, 0.05)
    # When n_recording_segments < 3, energy_level must be None
    res_few = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=rms_slice,
        n_words=4,
        duration=2.0,
        rec_rms_percentiles=(-30.0, -15.0),
        n_recording_segments=2,
    )
    assert res_few["energy_level"] is None
    assert res_few["energy_dbfs"] is not None

    # When >= 3 segments, relative level should trigger
    # 0.05 amplitude -> 20*log10(0.05) ~ -26 dB
    res_low = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=rms_slice,
        n_words=4,
        duration=2.0,
        rec_rms_percentiles=(-20.0, -10.0),  # -26 is below -20
        n_recording_segments=3,
    )
    assert res_low["energy_level"] == "low"

    res_high = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=rms_slice,
        n_words=4,
        duration=2.0,
        rec_rms_percentiles=(-40.0, -30.0),  # -26 is above -30
        n_recording_segments=3,
    )
    assert res_high["energy_level"] == "high"


def test_speaking_rate_and_invalidation():
    # Normal segment >= 1.5s
    res = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=np.full(20, 0.05),
        n_words=6,
        duration=2.0,
    )
    assert res["speaking_rate_wps"] == 3.0
    assert res["rate_band"] == "moderate"

    # Short segment < 1.5s -> n/a
    res_short = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=np.full(10, 0.05),
        n_words=3,
        duration=1.2,
    )
    assert res_short["speaking_rate_wps"] is None
    assert res_short["rate_band"] == "n/a"

    # Text unavailable or repetition -> n/a
    res_unavail = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=np.full(20, 0.05),
        n_words=6,
        duration=2.0,
        is_text_unavailable=True,
    )
    assert res_unavail["speaking_rate_wps"] is None
    assert res_unavail["rate_band"] == "n/a"


def test_export_payload_includes_acoustic_descriptors():
    res = {
        "segments": [
            {
                "segment_index": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "transcript": "Hello world",
                "acoustic_descriptors": {
                    "pitch_trend": "rising",
                    "pitch_trend_semitones": 3.2,
                    "pitch_range_st": 5.4,
                    "energy_level": "medium",
                    "energy_dbfs": -22.1,
                    "pause_ratio": 0.15,
                    "speaking_rate_wps": 2.5,
                    "rate_band": "moderate",
                },
            }
        ]
    }
    payload = build_json_export_payload(res)
    assert payload["segments"][0]["acoustic_descriptors"]["pitch_trend"] == "rising"


def test_energy_dbfs_tone_plus_digital_silence_gaps():
    # 17 seconds segment: 3 seconds speech/tone (0.1 amplitude ~ -20 dBFS), 14 seconds digital silence (0.0 amplitude)
    # Hop length 512 at 16kHz -> ~31.25 fps
    speech_frames = np.full(90, 0.1)  # ~2.9s of speech
    silence_frames = np.full(440, 0.0)  # ~14.1s of silence (dominant)
    rms_slice = np.concatenate([speech_frames, silence_frames])

    # Case A: voiced_slice provided
    voiced_slice = np.concatenate([np.ones(90, dtype=bool), np.zeros(440, dtype=bool)])
    res_voiced = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=voiced_slice,
        rms_slice=rms_slice,
        n_words=10,
        duration=17.0,
        n_recording_segments=1,
    )
    # Energy must be computed over voiced/speech frames (~ -20 dBFS), not collapsing to <= -100 dBFS
    assert res_voiced["energy_dbfs"] is not None
    assert -25.0 <= res_voiced["energy_dbfs"] <= -15.0

    # Case B: voiced_slice is None (falls back to non-silent frames above pause_thresh)
    res_unvoiced = compute_segment_acoustic_descriptors(
        f0_slice=None,
        voiced_slice=None,
        rms_slice=rms_slice,
        n_words=10,
        duration=17.0,
        n_recording_segments=1,
    )
    assert res_unvoiced["energy_dbfs"] is not None
    assert -25.0 <= res_unvoiced["energy_dbfs"] <= -15.0

