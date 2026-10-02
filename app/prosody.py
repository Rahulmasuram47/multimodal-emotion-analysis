"""
Prosodic Arousal Module (v2 Rebuild).

Computes a segment's prosodic arousal — a proxy for vocal energy and agitation
derived purely from the segment's own acoustics and transcription timing,
independent of model classification certainty.

Four sub-features, normalized to [0, 1] and combined via configurable weights:
  1. Relative loudness: Segment RMS in dB vs recording's P10-P90 percentiles.
  2. Pitch variability: Semitone range (or std) over voiced frames via librosa.pyin F0.
  3. Voiced ratio: Proportion of frames with active voicing.
  4. Speech rate: Words per second from Whisper word timestamps (or duration).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class ProsodyResult:
    """Prosodic arousal score plus its component breakdown, for
    transparency ("why is this segment's intensity 0.62?").
    """
    arousal: float
    components: Dict[str, float] = field(default_factory=dict)


def compute_recording_loudness_percentiles(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512,
    p_low: float = config.PROSODY_LOUDNESS_PERCENTILE_LOW,
    p_high: float = config.PROSODY_LOUDNESS_PERCENTILE_HIGH,
) -> Tuple[float, float]:
    """Compute low and high loudness percentiles in dB across an entire recording.

    Returns (p_low_db, p_high_db). If audio is empty or flat, returns safe defaults.
    """
    if len(audio) == 0:
        return -60.0, -10.0

    n_frames = max(1, len(audio) // hop_length)
    frame_len = hop_length * 2
    rms_vals = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(len(audio), start + frame_len)
        frame = audio[start:end]
        if len(frame) > 0:
            rms_vals.append(np.sqrt(np.mean(np.square(frame))))

    if not rms_vals:
        return -60.0, -10.0

    rms_arr = np.array(rms_vals)
    rms_db = 20.0 * np.log10(rms_arr + 1e-9)
    low_db = float(np.percentile(rms_db, p_low))
    high_db = float(np.percentile(rms_db, p_high))

    if high_db <= low_db:
        high_db = low_db + 12.0  # safe minimum 12 dB dynamic range spread

    return low_db, high_db


def compute_recording_f0(
    audio: np.ndarray,
    sr: int = 16000,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract fundamental frequency (F0) and voiced flags over the entire audio
    using librosa.pyin. Fast single pass over the recording.
    """
    if len(audio) == 0:
        return np.array([]), np.array([], dtype=bool)

    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio,
            fmin=config.PROSODY_F0_MIN_HZ,
            fmax=config.PROSODY_F0_MAX_HZ,
            sr=sr,
            hop_length=hop_length,
        )
        return f0, voiced_flag
    except Exception:
        logger.exception("Whole-recording F0 extraction failed; returning empty arrays.")
        return np.array([]), np.array([], dtype=bool)


def _compute_segment_f0(
    audio_segment: np.ndarray,
    sr: int,
    hop_length: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract F0 for a single segment when precomputed recording F0 is not provided."""
    if len(audio_segment) < hop_length * 2:
        return np.array([]), np.array([], dtype=bool)

    try:
        import librosa
        f0, voiced_flag, _ = librosa.pyin(
            audio_segment,
            fmin=config.PROSODY_F0_MIN_HZ,
            fmax=config.PROSODY_F0_MAX_HZ,
            sr=sr,
            hop_length=hop_length,
        )
        return f0, voiced_flag
    except Exception:
        return np.array([]), np.array([], dtype=bool)


def compute_prosodic_arousal(
    audio_segment: np.ndarray,
    sr: int,
    word_count: int,
    duration_seconds: float,
    recording_percentiles: Optional[Tuple[float, float]] = None,
    f0_segment: Optional[np.ndarray] = None,
    voiced_flag_segment: Optional[np.ndarray] = None,
    word_timestamps: Optional[List[Dict[str, Any]]] = None,
) -> ProsodyResult:
    """Compute this segment's prosodic arousal from its own audio and timings.

    Args:
        audio_segment: Raw audio samples of the segment.
        sr: Sample rate (Hz).
        word_count: Words count in segment transcript.
        duration_seconds: Duration in seconds.
        recording_percentiles: (p10_db, p90_db) from whole recording, or None.
        f0_segment: Slice of precomputed F0 for this segment, or None.
        voiced_flag_segment: Slice of precomputed voiced flags, or None.
        word_timestamps: List of Whisper word dicts [{'start': ..., 'end': ..., 'word': ...}], or None.

    Returns:
        ProsodyResult with normalized arousal in [0, 1] and component breakdown.
    """
    try:
        # 1. Relative Loudness
        rms = float(np.sqrt(np.mean(np.square(audio_segment)))) if len(audio_segment) > 0 else 0.0
        rms_db = float(20.0 * np.log10(rms + 1e-9))

        if recording_percentiles is not None and recording_percentiles[1] > recording_percentiles[0]:
            p_low, p_high = recording_percentiles
            loudness_norm = float(np.clip((rms_db - p_low) / (p_high - p_low), 0.0, 1.0))
        else:
            # Fallback to absolute reference maximum
            ref_max = getattr(config, "PROSODY_RMS_REFERENCE_MAX", 0.15)
            loudness_norm = float(np.clip(rms / ref_max, 0.0, 1.0)) if ref_max > 0 else 0.0

        # 2. F0 variability and Voiced Ratio
        if f0_segment is None:
            f0_segment, voiced_flag_segment = _compute_segment_f0(audio_segment, sr)

        if f0_segment is not None and len(f0_segment) > 0:
            if voiced_flag_segment is not None and len(voiced_flag_segment) == len(f0_segment):
                voiced_mask = voiced_flag_segment & (~np.isnan(f0_segment)) & (f0_segment > 0)
            else:
                voiced_mask = (~np.isnan(f0_segment)) & (f0_segment > 0)

            voiced_f0 = f0_segment[voiced_mask]
            total_frames = max(1, len(f0_segment))
            voiced_ratio = float(np.clip(len(voiced_f0) / total_frames, 0.0, 1.0))

            if len(voiced_f0) >= 2:
                # Convert F0 in Hz to semitones relative to A4 (440 Hz): ST = 12 * log2(F0 / 440) + 69
                semitones = 12.0 * np.log2(voiced_f0 / 440.0) + 69.0
                semitone_range = float(np.max(semitones) - np.min(semitones))
                semitone_std = float(np.std(semitones))
                max_range = getattr(config, "PROSODY_F0_SEMITONE_RANGE_MAX", 24.0)
                pitch_var_norm = float(np.clip(semitone_range / max_range, 0.0, 1.0)) if max_range > 0 else 0.0
            else:
                semitone_range = 0.0
                semitone_std = 0.0
                pitch_var_norm = 0.0
        else:
            voiced_ratio = 0.0
            semitone_range = 0.0
            semitone_std = 0.0
            pitch_var_norm = 0.0

        # 3. Speech Rate
        if word_timestamps and len(word_timestamps) >= 2:
            t_start = word_timestamps[0].get("start", 0.0)
            t_end = word_timestamps[-1].get("end", duration_seconds)
            active_duration = max(0.1, t_end - t_start)
            speech_rate = len(word_timestamps) / active_duration
        else:
            speech_rate = (word_count / duration_seconds) if duration_seconds > 0 else 0.0

        max_speech_rate = getattr(config, "PROSODY_SPEECH_RATE_REFERENCE_MAX", 4.0)
        speech_rate_norm = float(np.clip(speech_rate / max_speech_rate, 0.0, 1.0)) if max_speech_rate > 0 else 0.0

        # 4. Weighted Prosodic Arousal
        w_loudness = getattr(config, "PROSODY_WEIGHT_LOUDNESS", 0.35)
        w_pitch = getattr(config, "PROSODY_WEIGHT_PITCH_VARIATION", 0.30)
        w_voiced = getattr(config, "PROSODY_WEIGHT_VOICED_RATIO", 0.15)
        w_rate = getattr(config, "PROSODY_WEIGHT_SPEECH_RATE", 0.20)

        arousal = (
            w_loudness * loudness_norm
            + w_pitch * pitch_var_norm
            + w_voiced * voiced_ratio
            + w_rate * speech_rate_norm
        )
        arousal = round(float(np.clip(arousal, 0.0, 1.0)), 4)

        components = {
            "rms": round(rms, 4),
            "rms_db": round(rms_db, 2),
            "loudness_normalized": round(loudness_norm, 4),
            "semitone_range": round(semitone_range, 2),
            "semitone_std": round(semitone_std, 2),
            "pitch_variation_normalized": round(pitch_var_norm, 4),
            "voiced_ratio": round(voiced_ratio, 4),
            "speech_rate_words_per_sec": round(speech_rate, 2),
            "speech_rate_normalized": round(speech_rate_norm, 4),
            "weights": {
                "loudness": w_loudness,
                "pitch_variation": w_pitch,
                "voiced_ratio": w_voiced,
                "speech_rate": w_rate,
            },
        }

        return ProsodyResult(arousal=arousal, components=components)

    except Exception:
        logger.exception("Prosodic arousal computation failed; defaulting to 0.0.")
        return ProsodyResult(arousal=0.0, components={})


def compute_voice_arousal(voice_scores: Dict[str, float]) -> float:
    """Expected-value arousal from the voice-emotion class probabilities,
    using config.VOICE_EMOTION_AROUSAL_MAP. This is deliberately NOT just
    top-class confidence — it weights every class by both its arousal
    level and its probability.
    """
    if not voice_scores:
        return 0.0
    total = sum(voice_scores.values())
    if total <= 0:
        return 0.0
    arousal_map = getattr(config, "VOICE_EMOTION_AROUSAL_MAP", {
        "angry": 1.0,
        "happy": 0.75,
        "neutral": 0.25,
        "sad": 0.15,
    })
    weighted = sum(
        prob * arousal_map.get(label, 0.5)
        for label, prob in voice_scores.items()
    )
    return round(float(np.clip(weighted / total, 0.0, 1.0)), 4)


def compute_text_negativity(text_scores: Dict[str, float]) -> float:
    """Probability mass on the 'negative' text-sentiment class."""
    return round(float(text_scores.get("negative", 0.0)), 4)