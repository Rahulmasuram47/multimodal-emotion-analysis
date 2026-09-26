"""
Prosodic Arousal Module.

Computes a segment's prosodic arousal — a proxy for vocal energy/agitation
derived purely from the segment's own audio, independent of any ML model's
classification. Used as one input to the Phase 4 intensity formula
(config.INTENSITY_FORMULA_VERSION == 'v2').

Three sub-features, each normalized to [0, 1] against a configurable
reference maximum (config.py), then combined with configurable weights:

  1. Relative loudness   — RMS amplitude of this segment's audio.
  2. Pitch variation      — zero-crossing-rate standard deviation, a cheap
                             proxy for pitch movement (full F0 tracking via
                             librosa.pyin is available in the Acoustic
                             Inspection tab but is too expensive to run
                             per-segment on every analysis).
  3. Speech rate           — words per second, from the segment's word
                             count and duration.

This module deliberately reuses the same lightweight RMS/ZCR approach as
app/voice_calibration.py's prosody sanity rule, for consistency and to
avoid a second, more expensive pitch-tracking pass per segment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class ProsodyResult:
    """Prosodic arousal score plus its component breakdown, for
    transparency ("why is this segment's intensity 0.82?").
    """
    arousal: float
    components: Dict[str, float] = field(default_factory=dict)


def _compute_rms(audio_segment: np.ndarray) -> float:
    if len(audio_segment) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_segment))))


def _compute_zcr_std(audio_segment: np.ndarray, sr: int) -> float:
    """Zero-crossing-rate standard deviation across ~20ms frames, used as
    a fast proxy for pitch movement (see module docstring)."""
    if len(audio_segment) == 0:
        return 0.0
    frame_len = max(1, sr // 50)
    n_frames = len(audio_segment) // frame_len
    if n_frames == 0:
        return 0.0
    zcr_values = []
    for i in range(n_frames):
        frame = audio_segment[i * frame_len:(i + 1) * frame_len]
        signs = np.sign(frame)
        signs[signs == 0] = 1
        zcr = np.mean(signs[:-1] != signs[1:]) if len(signs) > 1 else 0.0
        zcr_values.append(zcr)
    return float(np.std(zcr_values)) if zcr_values else 0.0


def compute_prosodic_arousal(
    audio_segment: np.ndarray,
    sr: int,
    word_count: int,
    duration_seconds: float,
) -> ProsodyResult:
    """Compute this segment's prosodic arousal from its own audio.

    Args:
        audio_segment: The segment's raw audio samples (NOT the widened
            acoustic window from short-segment handling — this should be
            the segment's actual transcript-aligned audio, matching what
            speech rate is computed against).
        sr: Sample rate.
        word_count: Number of words in the segment's transcript, used for
            speech-rate calculation.
        duration_seconds: Segment duration (end - start), used for
            speech-rate calculation.

    Returns:
        ProsodyResult with the combined arousal score in [0, 1] and a
        breakdown of each normalized sub-component for transparency.
    """
    try:
        rms = _compute_rms(audio_segment)
        rms_norm = min(1.0, rms / config.PROSODY_RMS_REFERENCE_MAX) if config.PROSODY_RMS_REFERENCE_MAX > 0 else 0.0

        zcr_std = _compute_zcr_std(audio_segment, sr)
        pitch_var_norm = min(1.0, zcr_std / config.PROSODY_ZCR_STD_REFERENCE_MAX) if config.PROSODY_ZCR_STD_REFERENCE_MAX > 0 else 0.0

        speech_rate = (word_count / duration_seconds) if duration_seconds > 0 else 0.0
        speech_rate_norm = min(1.0, speech_rate / config.PROSODY_SPEECH_RATE_REFERENCE_MAX) if config.PROSODY_SPEECH_RATE_REFERENCE_MAX > 0 else 0.0

        arousal = (
            config.PROSODY_WEIGHT_RMS * rms_norm
            + config.PROSODY_WEIGHT_PITCH_VARIATION * pitch_var_norm
            + config.PROSODY_WEIGHT_SPEECH_RATE * speech_rate_norm
        )
        arousal = round(min(1.0, max(0.0, arousal)), 4)

        return ProsodyResult(
            arousal=arousal,
            components={
                "rms": round(rms, 4),
                "rms_normalized": round(rms_norm, 4),
                "zcr_std": round(zcr_std, 4),
                "pitch_variation_normalized": round(pitch_var_norm, 4),
                "speech_rate_words_per_sec": round(speech_rate, 4),
                "speech_rate_normalized": round(speech_rate_norm, 4),
            },
        )
    except Exception:
        logger.exception("Prosodic arousal computation failed; defaulting to 0.0.")
        return ProsodyResult(arousal=0.0, components={})


def compute_voice_arousal(voice_scores: Dict[str, float]) -> float:
    """Expected-value arousal from the voice-emotion class probabilities,
    using config.VOICE_EMOTION_AROUSAL_MAP. This is deliberately NOT just
    top-class confidence — it weights every class by both its arousal
    level and its probability, so a 55%-confident 'angry' with 30% 'happy'
    (both high-arousal) scores higher than a 55%-confident 'angry' with
    30% 'neutral'.
    """
    if not voice_scores:
        return 0.0
    total = sum(voice_scores.values())
    if total <= 0:
        return 0.0
    weighted = sum(
        prob * config.VOICE_EMOTION_AROUSAL_MAP.get(label, 0.5)
        for label, prob in voice_scores.items()
    )
    return round(min(1.0, max(0.0, weighted / total)), 4)


def compute_text_negativity(text_scores: Dict[str, float]) -> float:
    """Probability mass on the 'negative' text-sentiment class."""
    return round(float(text_scores.get("negative", 0.0)), 4)