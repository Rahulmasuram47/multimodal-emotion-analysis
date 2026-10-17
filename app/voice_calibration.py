"""
Voice Emotion Calibration Module.

Two independent correction mechanisms, both OFF by default until proven
useful, and both fully transparent (every adjustment is logged in the
result so it can be explained in the UI/viva):

1. Prosody sanity rule (RULE-BASED, always usable regardless of data size):
   High relative loudness + wide pitch range is acoustically inconsistent
   with "sad" (typically low-energy, monotone). When both conditions hold
   and "sad" is the top prediction, its probability mass is redistributed
   toward "angry"/"happy" before renormalizing. This is a fixed heuristic,
   not fit from data, so it can run even with very few labeled samples.

2. Temperature scaling + per-class prior correction (DATA-FITTED):
   Requires config.VOICE_CALIBRATION_MIN_FIT_SAMPLES labeled samples per
   class before it will do anything. Below that threshold, calibrate() is
   a documented no-op — it does NOT guess a correction from too little
   data, since that would just overfit to those exact few clips.

Also provides compute_widened_window(), used by ConversationAnalyzer to
widen the acoustic-only audio window for very short segments (see
config.SHORT_SEGMENT_THRESHOLD_SECONDS), giving the acoustic model more
signal without changing the segment's transcript/text boundaries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class CalibrationResult:
    """Wraps a (possibly adjusted) score dict with an explanation trail."""
    scores: Dict[str, float]
    emotion: str
    confidence: float
    applied_rules: List[str] = field(default_factory=list)


def _renormalize(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return scores
    return {k: v / total for k, v in scores.items()}


def _top_class(scores: Dict[str, float]) -> Tuple[str, float]:
    label = max(scores, key=scores.get)
    return label, scores[label]


def compute_widened_window(
    start_seconds: float,
    end_seconds: float,
    all_segment_bounds: List[Tuple[float, float]],
    audio_duration_seconds: float,
) -> Tuple[float, float, bool]:
    """Widen a short segment's time window for acoustic-only inference.

    Returns (widened_start, widened_end, was_widened). The transcript/text
    window is NEVER changed by this — only the audio slice used for the
    acoustic emotion model. Widening never crosses into a neighboring
    segment's own time range, so it only borrows silence/breath/trailing-
    tone context, never another segment's speech content.

    all_segment_bounds: list of (start, end) for every segment in the
        conversation (including this one), used to avoid overlapping a
        neighbor's speech.
    """
    duration = end_seconds - start_seconds
    if duration >= config.SHORT_SEGMENT_THRESHOLD_SECONDS:
        return start_seconds, end_seconds, False

    padding = config.SHORT_SEGMENT_CONTEXT_PADDING_SECONDS

    # Find the nearest neighbor bounds so we don't eat into their speech.
    prev_end = 0.0
    next_start = audio_duration_seconds
    for (other_start, other_end) in all_segment_bounds:
        if other_end <= start_seconds and other_end > prev_end:
            prev_end = other_end
        if other_start >= end_seconds and other_start < next_start:
            next_start = other_start

    widened_start = max(prev_end, start_seconds - padding, 0.0)
    widened_end = min(next_start, end_seconds + padding, audio_duration_seconds)

    # Guard against a degenerate widened window (shouldn't happen given the
    # clamps above, but never allow it to invert or return empty).
    if widened_end <= widened_start:
        return start_seconds, end_seconds, False

    return widened_start, widened_end, True


def apply_prosody_sanity_rule(
    raw_scores: Dict[str, float],
    audio_segment: np.ndarray,
    sr: int,
    recording_percentiles: Optional[Tuple[float, float]] = None,
    f0_segment: Optional[np.ndarray] = None,
    voiced_flag_segment: Optional[np.ndarray] = None,
    backend: Optional[str] = None,
) -> CalibrationResult:
    """Dampen 'sad' when the segment's own acoustics look high-arousal.

    Uses relative RMS (vs recording percentiles) and F0 semitone range
    over voiced frames. Both conditions must indicate active vocal energy
    to override a 'sad' classification. Logs every rule firing for transparency.
    """
    applied: List[str] = []
    scores = dict(raw_scores)

    active_backend = backend if backend is not None else getattr(config, "VOICE_BACKEND", "superb")
    allowed_backends = getattr(config, "VOICE_SANITY_RULE_BACKENDS", ["superb"])
    if not config.VOICE_CALIBRATION_PROSODY_RULE_ENABLED or (active_backend not in allowed_backends):
        top_label, top_conf = _top_class(scores)
        return CalibrationResult(scores=scores, emotion=top_label, confidence=top_conf, applied_rules=applied)

    top_label, _ = _top_class(scores)

    if top_label == "sad" and len(audio_segment) > 0:
        try:
            rms = float(np.sqrt(np.mean(np.square(audio_segment))))
            rms_db = float(20.0 * np.log10(rms + 1e-9))

            # 1. Check relative loudness
            if recording_percentiles is not None and recording_percentiles[1] > recording_percentiles[0]:
                p_low, p_high = recording_percentiles
                rel_loudness = float(np.clip((rms_db - p_low) / (p_high - p_low), 0.0, 1.0))
                thresh_loudness = getattr(config, "VOICE_CALIBRATION_RELATIVE_LOUDNESS_THRESHOLD", 0.40)
                loud_enough = rel_loudness >= thresh_loudness
            else:
                rel_loudness = rms / getattr(config, "PROSODY_RMS_REFERENCE_MAX", 0.15)
                loud_enough = rms >= getattr(config, "VOICE_CALIBRATION_RMS_THRESHOLD", 0.05)

            # 2. Check F0 variability (semitone range over voiced frames)
            if f0_segment is None:
                # Local F0 computation
                from app.prosody import _compute_segment_f0
                f0_segment, voiced_flag_segment = _compute_segment_f0(audio_segment, sr)

            if f0_segment is not None and len(f0_segment) > 0:
                if voiced_flag_segment is not None and len(voiced_flag_segment) == len(f0_segment):
                    voiced_mask = voiced_flag_segment & (~np.isnan(f0_segment)) & (f0_segment > 0)
                else:
                    voiced_mask = (~np.isnan(f0_segment)) & (f0_segment > 0)

                voiced_f0 = f0_segment[voiced_mask]
                if len(voiced_f0) >= 2:
                    semitones = 12.0 * np.log2(voiced_f0 / 440.0) + 69.0
                    f0_range = float(np.max(semitones) - np.min(semitones))
                else:
                    f0_range = 0.0
            else:
                f0_range = 0.0

            thresh_f0 = getattr(config, "VOICE_CALIBRATION_F0_RANGE_SEMITONES_THRESHOLD", 6.0)
            varied_enough = f0_range >= thresh_f0

            # Fallback to ZCR if F0 unvoiced or pyin failed
            if not varied_enough and f0_range == 0.0:
                frame_len = max(1, sr // 50)
                n_frames = len(audio_segment) // frame_len
                zcr_values = []
                for i in range(n_frames):
                    frame = audio_segment[i * frame_len:(i + 1) * frame_len]
                    signs = np.sign(frame)
                    signs[signs == 0] = 1
                    zcr = np.mean(signs[:-1] != signs[1:]) if len(signs) > 1 else 0.0
                    zcr_values.append(zcr)
                zcr_std = float(np.std(zcr_values)) if zcr_values else 0.0
                varied_enough = zcr_std >= getattr(config, "VOICE_CALIBRATION_ZCR_STD_THRESHOLD", 0.02)

            if loud_enough and varied_enough:
                dampening = getattr(config, "VOICE_CALIBRATION_SAD_DAMPENING_FACTOR", 0.5)
                removed_mass = scores.get("sad", 0.0) * (1.0 - dampening)
                scores["sad"] = scores.get("sad", 0.0) * dampening

                # Redistribute removed mass toward angry/happy proportionally
                redistribute_targets = {
                    k: v for k, v in scores.items() if k in ("angry", "happy") and v > 0
                }
                target_total = sum(redistribute_targets.values())
                if target_total > 0:
                    for k in redistribute_targets:
                        scores[k] += removed_mass * (redistribute_targets[k] / target_total)
                else:
                    # If neither had mass, split evenly between angry and happy
                    scores["angry"] = scores.get("angry", 0.0) + removed_mass * 0.5
                    scores["happy"] = scores.get("happy", 0.0) + removed_mass * 0.5

                scores = _renormalize(scores)
                rule_desc = (
                    f"prosody_sanity_rule: relative_loudness={rel_loudness:.2f}, "
                    f"f0_range={f0_range:.1f}st, dampened 'sad' by factor {dampening}"
                )
                applied.append(rule_desc)
                logger.info("Voice calibration sanity rule fired: %s", rule_desc)
        except Exception:
            logger.exception("Prosody sanity rule failed for this segment; leaving scores unchanged.")

    top_label, top_conf = _top_class(scores)
    return CalibrationResult(scores=scores, emotion=top_label, confidence=top_conf, applied_rules=applied)


class VoiceCalibrator:
    """Stateful calibrator: fits temperature scaling from labeled data,
    and always offers the (data-free) prosody sanity rule.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0  # 1.0 = no-op
        self.is_fitted: bool = False
        self.fit_n_samples: int = 0

    def fit(self, labeled_scores: List[Tuple[Dict[str, float], str]]) -> None:
        """Fit temperature scaling from (raw_scores, true_label) pairs.

        Refuses to fit (stays a no-op) below config.VOICE_CALIBRATION_MIN_FIT_SAMPLES,
        to avoid overfitting a correction to a handful of clips.
        """
        self.fit_n_samples = len(labeled_scores)

        if self.fit_n_samples < config.VOICE_CALIBRATION_MIN_FIT_SAMPLES:
            logger.info(
                "Skipping temperature-scaling fit: only %d labeled samples "
                "(need >= %d). Calibration will use identity temperature.",
                self.fit_n_samples,
                config.VOICE_CALIBRATION_MIN_FIT_SAMPLES,
            )
            self.temperature = 1.0
            self.is_fitted = False
            return

        # Simple grid search over temperature to minimize negative log-
        # likelihood on the labeled set. Deliberately simple (no external
        # optimizer dependency) since N will typically be small (tens, not
        # thousands) even once "enough" data exists for a mini-project.
        best_temp, best_nll = 1.0, float("inf")
        for temp in np.arange(0.5, 3.01, 0.1):
            nll = 0.0
            for raw_scores, true_label in labeled_scores:
                logits = np.array([np.log(max(v, 1e-8)) for v in raw_scores.values()])
                labels = list(raw_scores.keys())
                scaled = logits / temp
                exp_scaled = np.exp(scaled - np.max(scaled))
                probs = exp_scaled / np.sum(exp_scaled)
                true_idx = labels.index(true_label) if true_label in labels else None
                if true_idx is not None:
                    nll -= np.log(max(probs[true_idx], 1e-8))
            if nll < best_nll:
                best_nll, best_temp = nll, temp

        self.temperature = float(best_temp)
        self.is_fitted = True
        logger.info("Temperature scaling fitted: T=%.2f on N=%d samples.", self.temperature, self.fit_n_samples)

    def apply_temperature(self, raw_scores: Dict[str, float]) -> Dict[str, float]:
        if not self.is_fitted or self.temperature == 1.0:
            return dict(raw_scores)
        labels = list(raw_scores.keys())
        logits = np.array([np.log(max(v, 1e-8)) for v in raw_scores.values()])
        scaled = logits / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))
        probs = exp_scaled / np.sum(exp_scaled)
        return {label: float(p) for label, p in zip(labels, probs)}

    def calibrate_segment(
        self,
        raw_scores: Dict[str, float],
        audio_segment: np.ndarray,
        sr: int,
        recording_percentiles: Optional[Tuple[float, float]] = None,
        f0_segment: Optional[np.ndarray] = None,
        voiced_flag_segment: Optional[np.ndarray] = None,
        backend: Optional[str] = None,
    ) -> CalibrationResult:
        """Full calibration pipeline for one segment: temperature (if fitted)
        then prosody sanity rule, both logged in applied_rules.
        """
        applied: List[str] = []
        scores = dict(raw_scores)

        if self.is_fitted:
            scores = self.apply_temperature(scores)
            applied.append(f"temperature_scaling: T={self.temperature:.2f}")

        active_backend = backend or getattr(config, "VOICE_BACKEND", "superb")
        prosody_result = apply_prosody_sanity_rule(
            raw_scores=scores,
            audio_segment=audio_segment,
            sr=sr,
            recording_percentiles=recording_percentiles,
            f0_segment=f0_segment,
            voiced_flag_segment=voiced_flag_segment,
            backend=active_backend,
        )
        applied.extend(prosody_result.applied_rules)

        return CalibrationResult(
            scores=prosody_result.scores,
            emotion=prosody_result.emotion,
            confidence=prosody_result.confidence,
            applied_rules=applied,
        )