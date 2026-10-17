"""
Conversation Analysis Module.

Builds segment-level multimodal analysis from:
- Audio
- Voice emotion
- Speech-to-text
- Text sentiment
- Multimodal fusion

Also provides:
- Emotional trend detection (first-vs-last delta)
- Emotional escalation detection (strict and tolerant modes)
- A single unified conversation verdict (escalating / de-escalating /
  stable / fluctuating / insufficient_data)
- Segment-level emotional intensity (v1 confidence-based, v2 prosody-based)
- Speaker diarization (Speaker A / Speaker B labels)
- Modality consistency trajectory (voice vs. text direction agreement)
- Explainable review-priority attention score (heuristic only)
"""

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def sanitize_transcript(text: str) -> str:
    """Strip U+FFFD and trailing partial / replacement characters."""
    if not text:
        return ""
    s = re.sub(r'[\ufffd\uFFFE\uFFFF]+', '', str(text))
    return s.strip()

import config
from app.audio_processor import load_and_preprocess_audio
from app.emotion_analyzer import VoiceEmotionAnalyzer
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer
from app.fusion import MultimodalFusion
from app.diarization import SpeakerDiarizer
from app.voice_calibration import VoiceCalibrator, compute_widened_window
from app.prosody import (
    compute_prosodic_arousal,
    compute_voice_arousal,
    compute_text_negativity,
    compute_recording_loudness_percentiles,
    compute_recording_f0,
    compute_recording_frame_rms,
    compute_segment_acoustic_descriptors,
)
from app.consistency_analysis import (
    compute_consistency_trajectory,
    compute_attention_score,
)
from app.incongruence_analysis import analyze_conversation_incongruence
from app.recommendation_engine import generate_recommendation

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Seed Python random, NumPy, and PyTorch for deterministic repeatability."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# Signature: callback(stage: str, current: int, total: int) -> None
ProgressCallback = Callable[[str, int, int], None]


class ConversationAnalyzer:
    """Analyze a conversation segment by segment."""

    def __init__(
        self,
        emotion_analyzer: VoiceEmotionAnalyzer | None = None,
        speech_to_text: SpeechToText | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        fusion: MultimodalFusion | None = None,
        diarizer: SpeakerDiarizer | None = None,
        voice_calibrator: VoiceCalibrator | None = None,
        whisper_model_size: Optional[str] = None,
        whisper_language: Optional[str] = None,
    ):
        logger.info("Initializing conversation analysis pipeline...")

        self.emotion_analyzer = emotion_analyzer or VoiceEmotionAnalyzer()

        self.speech_to_text = speech_to_text or SpeechToText(
            model_size=whisper_model_size, language=whisper_language
        )

        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()

        self.fusion = fusion or MultimodalFusion()

        # Lazy-loads the pyannote pipeline on first .diarize() call.
        self.diarizer = diarizer or SpeakerDiarizer()

        # Voice emotion calibration (prosody sanity rule + temperature scaling).
        self.voice_calibrator = voice_calibrator or VoiceCalibrator()

        logger.info("Conversation analysis pipeline initialized.")

    # ---------------------------------------------------------
    # Progress helper
    # ---------------------------------------------------------

    @staticmethod
    def _report_progress(
        callback: Optional[ProgressCallback],
        stage: str,
        current: int,
        total: int,
    ) -> None:
        """Call the progress callback if provided; never let it break the pipeline."""
        if callback is None:
            return

        try:
            callback(stage, current, total)
        except Exception:
            logger.exception("Progress callback raised an exception; ignoring.")

    # ---------------------------------------------------------
    # Voice reliability from class-probability margin
    # ---------------------------------------------------------

    @staticmethod
    def _margin_reliability(scores: Dict[str, float]) -> float:
        """Reliability factor in [floor, 1.0] from the top-1 vs top-2
        probability margin. A near-tie between classes (e.g. 37% vs 33%)
        lowers reliability, so a coin-flip voice label no longer counts as
        fully reliable in fusion."""
        if not getattr(config, "VOICE_MARGIN_RELIABILITY_ENABLED", True):
            return 1.0

        try:
            probs = sorted((float(v) for v in (scores or {}).values()), reverse=True)
        except (TypeError, ValueError):
            return 1.0

        if len(probs) < 2:
            return 1.0

        floor = getattr(config, "VOICE_MARGIN_RELIABILITY_FLOOR", 0.50)
        full_credit = getattr(config, "VOICE_MARGIN_FULL_CREDIT", 0.30)
        if full_credit <= 0:
            return 1.0

        margin = probs[0] - probs[1]
        return floor + (1.0 - floor) * min(1.0, margin / full_credit)

    # ---------------------------------------------------------
    # Intensity — v1 (legacy, confidence-based)
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_intensity(segment: Dict[str, Any]) -> float:
        """
        LEGACY (v1) intensity formula: pure fusion-confidence based.

        Agreement between voice and text increases confidence. Strong
        disagreement reduces the influence of the combined confidence so
        that one unreliable modality does not dominate.

        Returns:
            Float between 0.0 and 1.0.
        """
        fusion = segment["fusion"]

        confidence = float(fusion.get("combined_confidence", 0.0))
        agreement = bool(fusion.get("agreement", False))
        mismatch = bool(fusion.get("mismatch", False))

        if agreement:
            intensity = confidence + 0.10
        elif mismatch:
            intensity = confidence * 0.85
        else:
            intensity = confidence * 0.90

        return round(min(1.0, max(0.0, intensity)), 4)

    # ---------------------------------------------------------
    # Intensity — v2 (prosody-based)
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_intensity_v2(
        voice_scores: Dict[str, float],
        text_scores: Dict[str, float],
        audio_segment: np.ndarray,
        sr: int,
        word_count: int,
        duration_seconds: float,
        recording_percentiles: Optional[Tuple[float, float]] = None,
        f0_segment: Optional[np.ndarray] = None,
        voiced_flag_segment: Optional[np.ndarray] = None,
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        v2 intensity: weighted combination of prosodic arousal, expected
        voice arousal from class probabilities, and text negativity.

        Returns (intensity, components) where components breaks down every
        sub-score for transparency.
        """
        prosody_result = compute_prosodic_arousal(
            audio_segment=audio_segment,
            sr=sr,
            word_count=word_count,
            duration_seconds=duration_seconds,
            recording_percentiles=recording_percentiles,
            f0_segment=f0_segment,
            voiced_flag_segment=voiced_flag_segment,
            word_timestamps=word_timestamps,
        )
        voice_arousal = compute_voice_arousal(voice_scores)

        text_available = bool(text_scores)
        if text_available:
            text_negativity = compute_text_negativity(text_scores)
            w_p = config.INTENSITY_V2_WEIGHT_PROSODY
            w_v = config.INTENSITY_V2_WEIGHT_VOICE_AROUSAL
            w_t = config.INTENSITY_V2_WEIGHT_TEXT_NEGATIVITY
            tot_w = w_p + w_v + w_t
            if tot_w > 0:
                w_p_eff, w_v_eff, w_t_eff = w_p / tot_w, w_v / tot_w, w_t / tot_w
            else:
                w_p_eff, w_v_eff, w_t_eff = 0.35, 0.35, 0.30
            intensity = (
                w_p_eff * prosody_result.arousal
                + w_v_eff * voice_arousal
                + w_t_eff * text_negativity
            )
            intensity = round(min(1.0, max(0.0, intensity)), 4)
            components = {
                "formula_version": "v2",
                "prosodic_arousal": prosody_result.arousal,
                "prosody_components": prosody_result.components,
                "voice_arousal": voice_arousal,
                "text_negativity": text_negativity,
                "weights": {
                    "prosody": round(w_p_eff, 4),
                    "voice_arousal": round(w_v_eff, 4),
                    "text_negativity": round(w_t_eff, 4),
                },
            }
        else:
            text_negativity = 0.0
            w_p = config.INTENSITY_V2_WEIGHT_PROSODY
            w_v = config.INTENSITY_V2_WEIGHT_VOICE_AROUSAL
            tot_w = w_p + w_v
            if tot_w > 0:
                w_p_eff = round(w_p / tot_w, 4)
                w_v_eff = round(w_v / tot_w, 4)
            else:
                w_p_eff, w_v_eff = 0.5, 0.5
            w_t_eff = 0.0
            intensity = (
                w_p_eff * prosody_result.arousal
                + w_v_eff * voice_arousal
            )
            intensity = round(min(1.0, max(0.0, intensity)), 4)
            components = {
                "formula_version": "v2",
                "prosodic_arousal": prosody_result.arousal,
                "prosody_components": prosody_result.components,
                "voice_arousal": voice_arousal,
                "text_negativity": None,
                "weights": {
                    "prosody": w_p_eff,
                    "voice_arousal": w_v_eff,
                    "text_negativity": 0.0,
                },
            }

        return intensity, components

    # ---------------------------------------------------------
    # Emotional trend (first-vs-last delta)
    # ---------------------------------------------------------

    @staticmethod
    def detect_emotional_trend(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect the overall emotional intensity trend (first-vs-last delta).

        Increasing: last intensity at least 0.10 higher than the first.
        Decreasing: last intensity at least 0.10 lower than the first.
        Stable:     change smaller than 0.10.

        Returns both "change" (original key) and "delta" (alias read by
        streamlit_app.py — previously the UI read "delta" while this
        function only returned "change", so the delta always showed 0.0).
        """
        if len(segments) < 2:
            return {
                "trend": "insufficient_data",
                "intensity_scores": [],
                "change": 0.0,
                "delta": 0.0,
            }

        intensities = [float(segment.get("intensity", 0.0)) for segment in segments]

        change = round(intensities[-1] - intensities[0], 4)

        threshold = 0.10

        if change >= threshold:
            trend = "increasing"
        elif change <= -threshold:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "intensity_scores": intensities,
            "change": change,
            "delta": change,
        }

    # ---------------------------------------------------------
    # Escalation — strict (legacy)
    # ---------------------------------------------------------

    @staticmethod
    def detect_escalation(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        LEGACY strict escalation rule.

        Requires at least three segments, every consecutive step
        increasing, and a total increase of at least 0.20.
        """
        if len(segments) < 3:
            return {
                "escalation_detected": False,
                "reason": "At least 3 segments are required.",
                "intensity_change": 0.0,
            }

        intensities = [float(segment.get("intensity", 0.0)) for segment in segments]

        consecutive_increase = all(
            intensities[i] > intensities[i - 1] for i in range(1, len(intensities))
        )

        total_change = round(intensities[-1] - intensities[0], 4)

        escalation_detected = consecutive_increase and total_change >= 0.20

        if escalation_detected:
            reason = (
                "Emotional intensity increased continuously "
                "and exceeded the escalation threshold."
            )
        else:
            reason = (
                "Emotional intensity did not show a sufficient "
                "continuous increase."
            )

        return {
            "escalation_detected": escalation_detected,
            "reason": reason,
            "intensity_change": total_change,
        }

    # ---------------------------------------------------------
    # Escalation — tolerant (robust-slope based)
    # ---------------------------------------------------------

    @staticmethod
    def _ols_slope(values: List[float]) -> float:
        """OLS slope of values against their index. 0.0 for fewer than 2 points."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        slope, _intercept = np.polyfit(x, y, 1)
        return float(slope)

    @staticmethod
    def detect_escalation_advanced(
        segments: List[Dict[str, Any]],
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Escalation detection with two selectable modes:

          'strict'   — delegates to the legacy rule.
          'tolerant' — (default, config.ESCALATION_MODE) fits a linear
                       slope and allows single dips up to
                       config.ESCALATION_DIP_TOLERANCE. Requires ALL of:
                       enough segments, slope >= ESCALATION_MIN_SLOPE,
                       net rise >= ESCALATION_MIN_NET_RISE, and no
                       single-step drop above the tolerance.
        """
        mode = mode or config.ESCALATION_MODE

        if mode == "strict":
            legacy = ConversationAnalyzer.detect_escalation(segments)
            legacy["mode"] = "strict"
            legacy["slope"] = None
            legacy["dip_violations"] = []
            return legacy

        if len(segments) < config.ESCALATION_MIN_SEGMENTS:
            return {
                "escalation_detected": False,
                "reason": f"At least {config.ESCALATION_MIN_SEGMENTS} segments are required.",
                "intensity_change": 0.0,
                "mode": "tolerant",
                "slope": 0.0,
                "dip_violations": [],
            }

        intensities = [float(segment.get("intensity", 0.0)) for segment in segments]

        slope = ConversationAnalyzer._ols_slope(intensities)
        net_rise = round(intensities[-1] - intensities[0], 4)

        dip_violations = []
        for i in range(1, len(intensities)):
            drop = intensities[i - 1] - intensities[i]
            if drop > config.ESCALATION_DIP_TOLERANCE:
                dip_violations.append({"index": i, "drop": round(drop, 4)})

        escalation_detected = (
            slope >= config.ESCALATION_MIN_SLOPE
            and net_rise >= config.ESCALATION_MIN_NET_RISE
            and len(dip_violations) == 0
        )

        if escalation_detected:
            reason = (
                f"Intensity rose steadily (slope={slope:.3f}/turn, "
                f"net rise={net_rise:+.2f}) with no dip exceeding the "
                f"{config.ESCALATION_DIP_TOLERANCE:.2f} tolerance."
            )
        elif dip_violations:
            reason = (
                f"Intensity trend was upward overall (slope={slope:.3f}/turn, "
                f"net rise={net_rise:+.2f}) but a dip of "
                f"{dip_violations[0]['drop']:.2f} at turn {dip_violations[0]['index'] + 1} "
                f"exceeded the {config.ESCALATION_DIP_TOLERANCE:.2f} dip tolerance."
            )
        elif slope < config.ESCALATION_MIN_SLOPE:
            reason = (
                f"Slope ({slope:.3f}/turn) did not reach the minimum "
                f"required rate of {config.ESCALATION_MIN_SLOPE:.2f}/turn."
            )
        else:
            reason = (
                f"Net rise ({net_rise:+.2f}) did not reach the minimum "
                f"threshold of {config.ESCALATION_MIN_NET_RISE:.2f}."
            )

        return {
            "escalation_detected": escalation_detected,
            "reason": reason,
            "intensity_change": net_rise,
            "mode": "tolerant",
            "slope": round(slope, 4),
            "dip_violations": dip_violations,
        }

    # ---------------------------------------------------------
    # Unified verdict
    # ---------------------------------------------------------

    @staticmethod
    def compute_unified_verdict(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        A single conversation-level verdict: escalating / de-escalating /
        fluctuating / stable / insufficient_data.

        Fewer than config.VERDICT_MIN_SEGMENTS (default 3) turns yields
        'insufficient_data' — a slope or "Stable" claim from only two
        turns is not meaningful.
        """
        min_turns = getattr(config, "VERDICT_MIN_SEGMENTS", 3)
        min_sec = getattr(config, "VOICE_MIN_SEGMENT_SEC", 1.0)
        # Exclude segments shorter than VOICE_MIN_SEGMENT_SEC from trend verdict
        trend_segments = [
            s for s in segments
            if ("start_seconds" not in s and "end_seconds" not in s)  # no time info → include always
            or (float(s.get("end_seconds", 0.0) or 0.0) - float(s.get("start_seconds", 0.0) or 0.0)) >= min_sec
        ]
        if len(trend_segments) < min_turns:
            return {
                "verdict": "insufficient_data",
                "reason": (
                    f"At least {min_turns} reliable turns (>= {min_sec:.1f}s) are needed to assess a trend "
                    f"(found {len(trend_segments)})."
                ),
                "slope": 0.0,
                "net_change": 0.0,
                "counterfactuals": [],
            }

        intensities = [float(segment.get("intensity", 0.0)) for segment in trend_segments]

        slope = ConversationAnalyzer._ols_slope(intensities)
        net_change = round(intensities[-1] - intensities[0], 4)

        max_drop = 0.0
        max_rise = 0.0
        for i in range(1, len(intensities)):
            delta = intensities[i] - intensities[i - 1]
            if delta < 0:
                max_drop = max(max_drop, -delta)
            else:
                max_rise = max(max_rise, delta)

        is_escalating = (
            slope >= config.ESCALATION_MIN_SLOPE
            and net_change >= config.ESCALATION_MIN_NET_RISE
            and max_drop <= config.ESCALATION_DIP_TOLERANCE
        )
        is_deescalating = (
            slope <= -config.ESCALATION_MIN_SLOPE
            and net_change <= -config.ESCALATION_MIN_NET_RISE
            and max_rise <= config.ESCALATION_DIP_TOLERANCE
        )
        has_notable_swing = (
            max_drop > config.ESCALATION_DIP_TOLERANCE * 1.5
            or max_rise > config.ESCALATION_DIP_TOLERANCE * 1.5
        )

        if is_escalating:
            verdict = "escalating"
            reason = (
                f"Intensity rose steadily across the conversation "
                f"(slope={slope:+.3f}/turn, net change={net_change:+.2f})."
            )
        elif is_deescalating:
            verdict = "de-escalating"
            reason = (
                f"Intensity fell steadily across the conversation "
                f"(slope={slope:+.3f}/turn, net change={net_change:+.2f})."
            )
        elif has_notable_swing:
            verdict = "fluctuating"
            reason = (
                f"Intensity moved back and forth without a clear net "
                f"direction (largest rise={max_rise:.2f}, "
                f"largest drop={max_drop:.2f}, net change={net_change:+.2f})."
            )
        else:
            verdict = "stable"
            reason = (
                f"Intensity stayed roughly constant across the conversation "
                f"(net change={net_change:+.2f}, slope={slope:+.3f}/turn)."
            )

        # Counterfactual sensitivity for the verdict
        verdict_cfs: List[str] = []
        if is_escalating:
            if max_drop > 0:
                verdict_cfs.append(
                    f"Verdict flips from escalating if dip tolerance falls below {max_drop:.2f} "
                    f"(current tolerance: {config.ESCALATION_DIP_TOLERANCE:.2f})."
                )
            if (net_change - config.ESCALATION_MIN_NET_RISE) < 0.10:
                verdict_cfs.append(
                    f"If net intensity rise fell below {config.ESCALATION_MIN_NET_RISE:.2f} "
                    f"(currently {net_change:+.2f}), verdict would become stable."
                )
        elif is_deescalating:
            if max_rise > 0:
                verdict_cfs.append(
                    f"Verdict flips from de-escalating if upward spike tolerance falls below {max_rise:.2f} "
                    f"(current tolerance: {config.ESCALATION_DIP_TOLERANCE:.2f})."
                )
        elif verdict == "fluctuating":
            verdict_cfs.append(
                f"Verdict would become stable if largest swing ({max(max_rise, max_drop):.2f}) "
                f"fell within tolerance ({config.ESCALATION_DIP_TOLERANCE * 1.5:.2f})."
            )
        else:
            diff_slope = config.ESCALATION_MIN_SLOPE - slope
            if diff_slope > 0:
                verdict_cfs.append(
                    f"Verdict would become escalating if per-turn slope increased by +{diff_slope:.3f} "
                    f"and net rise reached +{config.ESCALATION_MIN_NET_RISE:.2f}."
                )

        return {
            "verdict": verdict,
            "reason": reason,
            "slope": round(slope, 4),
            "net_change": net_change,
            "counterfactuals": verdict_cfs,
        }

    # ---------------------------------------------------------
    # Speaker diarization
    # ---------------------------------------------------------

    def _apply_diarization(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Attach speaker labels to transcript segments using pyannote.

        Diarization is an enhancement, so a failure does not abort the
        analysis. The error is returned separately so the dashboard can
        explain why labels are Unknown.
        """
        try:
            turns = self.diarizer.diarize(audio_path)

            logger.info("Speaker diarization returned %d turns.", len(turns))
            logger.info("Diarization turns: %s", turns)

            labeled_segments = self.diarizer.assign_speakers(segments, turns)

            logger.info(
                "Speaker assignments: %s",
                [
                    {
                        "start": segment.get("start_seconds"),
                        "end": segment.get("end_seconds"),
                        "speaker": segment.get("speaker"),
                    }
                    for segment in labeled_segments
                ],
            )

            return labeled_segments, None

        except Exception as exc:
            logger.exception(
                "Speaker diarization failed; continuing without speaker labels."
            )
            for segment in segments:
                segment["speaker"] = "Unknown"
            return segments, str(exc)

    # ---------------------------------------------------------
    # Single segment analysis
    # ---------------------------------------------------------

    def _analyze_segment(
        self,
        audio,
        sr,
        raw_segment: Dict[str, Any],
        index: int,
        all_segment_bounds: List[Tuple[float, float]],
        audio_duration_seconds: float,
        recording_percentiles: Optional[Tuple[float, float]] = None,
        rec_f0: Optional[np.ndarray] = None,
        rec_voiced_flags: Optional[np.ndarray] = None,
        rec_rms: Optional[np.ndarray] = None,
        n_recording_segments: int = 1,
        asr_recording_flags: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single transcript segment.

        Returns None (and logs) if the segment is malformed or any stage
        fails for this segment. A failure here does not abort analysis of
        the remaining segments.
        """
        start = float(raw_segment.get("start", 0.0) or 0.0)
        end = float(raw_segment.get("end", 0.0) or 0.0)
        text = str(raw_segment.get("text", "") or "").strip()

        if not text:
            logger.info("Skipping segment %d: empty transcript.", index + 1)
            return None

        if end < start:
            logger.warning(
                "Skipping segment %d: end (%.2f) is before start (%.2f).",
                index + 1,
                end,
                start,
            )
            return None

        logger.info("Analyzing conversation segment %d: %.2f-%.2fs", index + 1, start, end)

        hop_len = getattr(config, "HOP_LENGTH", 512)
        f0_seg = None
        voiced_seg = None
        rms_seg = None
        if rec_f0 is not None and len(rec_f0) > 0:
            idx_start = max(0, int((start * sr) // hop_len))
            idx_end = min(len(rec_f0), max(idx_start + 1, int((end * sr) // hop_len)))
            f0_seg = rec_f0[idx_start:idx_end]
            if rec_voiced_flags is not None and len(rec_voiced_flags) == len(rec_f0):
                voiced_seg = rec_voiced_flags[idx_start:idx_end]

        if rec_rms is not None and len(rec_rms) > 0:
            rms_idx_start = max(0, int((start * sr) // hop_len))
            rms_idx_end = min(len(rec_rms), max(rms_idx_start + 1, int((end * sr) // hop_len)))
            rms_seg = rec_rms[rms_idx_start:rms_idx_end]

        # Raw (unwidened) audio slice for this segment's transcript window,
        # used by voice calibration and the v2 intensity formula.
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        segment_audio = audio[start_sample:end_sample]

        # Widen the acoustic window for short segments so the emotion model
        # has enough signal. Falls back to the original window on failure.
        acoustic_start, acoustic_end, was_widened = start, end, False
        try:
            acoustic_start, acoustic_end, was_widened = compute_widened_window(
                start, end, all_segment_bounds, audio_duration_seconds
            )
        except Exception:
            logger.exception(
                "Short-segment window widening failed for segment %d; "
                "using original (unwidened) window.",
                index + 1,
            )
            acoustic_start, acoustic_end, was_widened = start, end, False

        try:
            voice_result = self.emotion_analyzer.analyze_segment(
                audio,
                sr,
                acoustic_start,
                acoustic_end,
            )
        except Exception:
            logger.exception(
                "Voice emotion analysis failed for segment %d; skipping.",
                index + 1,
            )
            return None

        voice_result["voice_reliability"] = (
            config.SHORT_SEGMENT_RELIABILITY_SCORE if was_widened else 1.0
        )
        voice_result["acoustic_window_widened"] = was_widened

        # Voice calibration (prosody sanity rule + temperature scaling).
        # Raw values are preserved alongside calibrated ones.
        if config.VOICE_CALIBRATION_ENABLED:
            try:
                calibration = self.voice_calibrator.calibrate_segment(
                    raw_scores=voice_result["scores"],
                    audio_segment=segment_audio,
                    sr=sr,
                    recording_percentiles=recording_percentiles,
                    f0_segment=f0_seg,
                    voiced_flag_segment=voiced_seg,
                )

                voice_result["raw_emotion"] = voice_result["emotion"]
                if "raw_confidence" not in voice_result:
                    voice_result["raw_confidence"] = voice_result["confidence"]
                voice_result["raw_scores"] = dict(voice_result["scores"])
                voice_result["emotion"] = calibration.emotion
                voice_result["confidence"] = calibration.confidence
                voice_result["scores"] = calibration.scores
                voice_result["calibration_applied_rules"] = calibration.applied_rules
            except Exception:
                logger.exception(
                    "Voice calibration failed for segment %d; using raw model output.",
                    index + 1,
                )

        # Check backend confidence_informative
        conf_informative = getattr(self.emotion_analyzer, "confidence_informative", True)
        voice_result["confidence_informative"] = conf_informative

        if conf_informative:
            # Margin-based reliability: a near-tie between the top two voice
            # classes lowers voice reliability (applied to final scores).
            margin_factor = self._margin_reliability(voice_result.get("scores", {}))
            voice_result["voice_margin_reliability"] = round(margin_factor, 4)
            voice_result["voice_reliability"] = round(
                voice_result.get("voice_reliability", 1.0) * margin_factor, 4
            )

            # Tier 4 (Item 32): Low voice confidence check
            low_v_conf_thresh = getattr(config, "LOW_VOICE_CONFIDENCE_THRESHOLD", 0.50)
            v_conf = float(voice_result.get("confidence", 0.0))
            if v_conf < low_v_conf_thresh:
                conf_penalty = getattr(config, "LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY", 0.60)
                voice_result["voice_reliability"] = round(
                    voice_result.get("voice_reliability", 1.0) * conf_penalty, 4
                )
                voice_result["is_low_voice_confidence"] = True
        else:
            voice_result["voice_margin_reliability"] = 1.0
            voice_result["is_low_voice_confidence"] = False

        # Item 4 & 5: If emotion is 'other'/'uncertain' or ensemble disagreed, clamp voice reliability
        floor_other = getattr(config, "VOICE_RELIABILITY_FLOOR_OTHER", 0.35)
        if voice_result.get("emotion") in ("other", "uncertain") or voice_result.get("uncertain", False) or voice_result.get("ensemble_disagreement", False):
            # "uncertain" from emotion2vec uses a stricter floor of 0.2
            if voice_result.get("emotion") == "uncertain":
                voice_result["voice_reliability"] = min(voice_result.get("voice_reliability", 1.0), 0.2)
            else:
                voice_result["voice_reliability"] = min(voice_result.get("voice_reliability", 1.0), floor_other)

        low_v_rel_display_thresh = getattr(config, "LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD", 0.65)
        is_low_v_rel = bool(
            voice_result.get("voice_reliability", 1.0) < low_v_rel_display_thresh
            or (conf_informative and voice_result.get("is_low_voice_confidence", False))
        )
        voice_result["low_voice_reliability"] = is_low_v_rel

        # ASR validation checks
        asr_flags: List[str] = list(asr_recording_flags or [])
        tokens = [t.lower() for t in text.split() if t.strip()]
        if len(tokens) >= 8:
            rep_ratio_thresh = getattr(config, "REPETITION_RATIO_MAX", 0.8)
            unique_tokens = len(set(tokens))
            rep_ratio = 1.0 - (unique_tokens / float(len(tokens)))
            if rep_ratio >= rep_ratio_thresh:
                if "repetition" not in asr_flags:
                    asr_flags.append("repetition")

        is_text_unavailable = ("non_english" in asr_flags or "repetition" in asr_flags)

        if is_text_unavailable:
            sentiment_result = None
            sentiment_scores = {}
            text_rel = 0.0
        else:
            try:
                sentiment_result = self.sentiment_analyzer.analyze(sanitize_transcript(text))
            except Exception:
                logger.exception(
                    "Text sentiment analysis failed for segment %d; skipping.",
                    index + 1,
                )
                return None

            # Text reliability from the ASR confidence flag, so fusion can
            # down-weight a segment whose transcript itself was uncertain.
            if "text_reliability" in raw_segment:
                sentiment_result["text_reliability"] = float(raw_segment["text_reliability"])
            else:
                is_low_confidence_asr = bool(raw_segment.get("is_low_confidence", False))
                sentiment_result["text_reliability"] = (
                    config.FUSION_TEXT_LOW_ASR_RELIABILITY if is_low_confidence_asr else 1.0
                )
            sentiment_scores = sentiment_result.get("scores", {})
            text_rel = float(sentiment_result.get("text_reliability", 1.0))

        try:
            fusion_result = self.fusion.fuse(voice_result, sentiment_result)
            fusion_result["acoustic_window_widened"] = was_widened
            fusion_result["segment_duration"] = max(0.0, end - start)
        except Exception:
            logger.exception(
                "Multimodal fusion failed for segment %d; skipping.",
                index + 1,
            )
            return None

        segment_result = {
            "segment_index": index + 1,
            "start_seconds": start,
            "end_seconds": end,
            "transcript": text,
            "voice_emotion": voice_result,
            "text_sentiment": sentiment_result,
            "text_unavailable": is_text_unavailable,
            "asr_flags": asr_flags,
            "fusion": fusion_result,
            "text_reliability": text_rel,
            "voice_reliability": float(voice_result.get("voice_reliability", 1.0)),
            "low_voice_reliability": is_low_v_rel,
            "is_low_confidence": bool(raw_segment.get("is_low_confidence", False)),
            "acoustic_window_widened": was_widened,
            "segment_duration": max(0.0, end - start),
        }

        # Intensity — v2 by default, with a defensive fallback to v1.
        try:
            if config.INTENSITY_FORMULA_VERSION == "v2":
                word_count = len(text.split())
                seg_duration = max(0.0, end - start)
                intensity_v2, intensity_components = self._calculate_intensity_v2(
                    voice_scores=voice_result.get("scores", {}),
                    text_scores=sentiment_scores,
                    audio_segment=segment_audio,
                    sr=sr,
                    word_count=word_count,
                    duration_seconds=seg_duration,
                    recording_percentiles=recording_percentiles,
                    f0_segment=f0_seg,
                    voiced_flag_segment=voiced_seg,
                    word_timestamps=raw_segment.get("words"),
                )
                segment_result["intensity"] = intensity_v2
                segment_result["intensity_components"] = intensity_components
            else:
                segment_result["intensity"] = self._calculate_intensity(segment_result)
                segment_result["intensity_components"] = {"formula_version": "v1"}
        except Exception:
            logger.exception(
                "Intensity calculation (v2) failed for segment %d; "
                "falling back to legacy v1 formula.",
                index + 1,
            )
            try:
                segment_result["intensity"] = self._calculate_intensity(segment_result)
                segment_result["intensity_components"] = {"formula_version": "v1_fallback"}
            except Exception:
                logger.exception(
                    "Legacy intensity fallback also failed for segment %d; defaulting to 0.0.",
                    index + 1,
                )
                segment_result["intensity"] = 0.0
                segment_result["intensity_components"] = {}

        # Acoustic Descriptors (independent of Intensity v2)
        try:
            seg_duration = max(0.0, end - start)
            word_count = len(text.split())
            segment_result["acoustic_descriptors"] = compute_segment_acoustic_descriptors(
                f0_slice=f0_seg,
                voiced_slice=voiced_seg,
                rms_slice=rms_seg,
                n_words=word_count,
                duration=seg_duration,
                rec_rms_percentiles=recording_percentiles,
                n_recording_segments=n_recording_segments,
                is_text_unavailable=is_text_unavailable,
                asr_flags=asr_flags,
            )
        except Exception:
            logger.exception("Acoustic descriptors computation failed for segment %d.", index + 1)
            segment_result["acoustic_descriptors"] = {
                "pitch_trend": "n/a",
                "pitch_trend_semitones": None,
                "pitch_range_st": None,
                "energy_level": None,
                "energy_dbfs": None,
                "pause_ratio": 0.0,
                "speaking_rate_wps": None,
                "rate_band": "n/a",
            }

        return segment_result

    # ---------------------------------------------------------
    # Main analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        audio_path: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Analyze an audio conversation segment by segment.

        Args:
            audio_path: Path to the audio file to analyze.
            progress_callback: Optional callable invoked as
                callback(stage, current, total) at each pipeline stage.
                Exceptions raised inside it are logged and ignored.

        Raises:
            ValueError: If the audio is empty/degenerate, or if no usable
                segments could be produced.
        """
        logger.info("Starting conversation analysis: %s", audio_path)
        set_seed(42)

        t_total_start = time.perf_counter()

        # ---- Load audio once ----
        self._report_progress(progress_callback, "Loading audio", 0, 1)

        t_load_start = time.perf_counter()
        audio, sr, duration = load_and_preprocess_audio(audio_path)
        t_load_elapsed = time.perf_counter() - t_load_start

        if duration is None or duration <= 0 or audio is None or len(audio) == 0:
            raise ValueError("The audio file appears to be empty or could not be decoded.")

        # ---- Speech-to-text ----
        self._report_progress(progress_callback, "Transcribing speech", 0, 1)

        t_transcribe_start = time.perf_counter()
        transcription = self.speech_to_text.transcribe(audio_path)
        t_transcribe_elapsed = time.perf_counter() - t_transcribe_start

        raw_segments = transcription.get("segments", [])

        if not raw_segments:
            logger.info("No speech detected by STT (VAD removed all frames).")
            t_total_elapsed = time.perf_counter() - t_total_start
            return {
                "audio_path": audio_path,
                "duration_seconds": duration,
                "sample_rate": sr,
                "language": transcription.get("language", "unknown"),
                "language_probability": float(transcription.get("language_probability", 0.0)),
                "full_transcript": "",
                "segments": [],
                "no_speech_detected": True,
                "no_speech_message": "No speech detected in this audio file. The recording appears to be silence or non-speech audio.",
                "emotional_trend": {"trend": "stable", "delta": 0.0, "change": 0.0},
                "escalation": {"escalation_detected": False, "reason": "No speech detected in recording."},
                "escalation_legacy_strict": {"escalation_detected": False, "reason": "No speech detected."},
                "verdict": {"verdict": "insufficient_data", "reason": "No speech detected."},
                "consistency_trajectory": {"trajectory": [], "mean_consistency": 0.0},
                "attention_score": {"attention_score": 0.0, "priority_band": "LOW"},
                "modality_incongruence": {"conflict_detected": False, "conflict_strength": "none", "explanation": "No speech detected."},
                "processing_time": {
                    "audio_preprocessing_seconds": round(t_load_elapsed, 3),
                    "transcription_whisper_seconds": round(t_transcribe_elapsed, 3),
                    "total_processing_seconds": round(t_total_elapsed, 3),
                },
                "skipped_segments": 0,
                "diarization_error": "No speech detected",
                "voice_collapse": {"detected": False},
                "recommendation": {
                    "status": "NORMAL",
                    "status_level": "info",
                    "recommended_action": "No speech detected in this audio file. The recording appears to be silence or non-speech audio.",
                    "reason": "Voice activity detection removed all audio frames as silence or non-speech.",
                    "contributing_factors": ["Silence or non-speech audio"],
                },
            }

        analyzed_segments: List[Dict[str, Any]] = []
        total = len(raw_segments)

        # Pre-compute all segment bounds so short-segment widening can see neighbors.
        all_segment_bounds: List[Tuple[float, float]] = [
            (float(s.get("start", 0.0) or 0.0), float(s.get("end", 0.0) or 0.0))
            for s in raw_segments
        ]

        try:
            recording_percentiles = compute_recording_loudness_percentiles(audio, sr)
            rec_f0, rec_voiced_flags = compute_recording_f0(audio, sr)
            rec_rms = compute_recording_frame_rms(audio, sr)
        except Exception:
            logger.exception(
                "Failed to compute recording prosody baseline; falling back to segment defaults."
            )
            recording_percentiles = None
            rec_f0, rec_voiced_flags = None, None
            rec_rms = None

        # Check recording-level ASR language flags
        detected_lang = str(transcription.get("language", "unknown") or "unknown").strip().lower()
        lang_prob = float(transcription.get("language_probability", 0.0) or 0.0)
        user_forced_language = bool(
            getattr(self.speech_to_text, "language", None)
            and self.speech_to_text.language != "auto"
        )
        non_english_min_prob = getattr(config, "NON_ENGLISH_MIN_PROB", 0.6)
        low_lang_conf_thresh = getattr(config, "LOW_LANG_CONF", 0.5)

        recording_asr_flags: List[str] = []
        if (
            not user_forced_language
            and detected_lang not in ("en", "unknown")
            and lang_prob >= non_english_min_prob
        ):
            recording_asr_flags.append("non_english")

        if lang_prob < low_lang_conf_thresh:
            recording_asr_flags.append("low_language_confidence")

        # ---- Analyze each transcript segment ----
        t_segments_start = time.perf_counter()
        for index, raw_segment in enumerate(raw_segments):

            self._report_progress(
                progress_callback,
                f"Analyzing segment {index + 1} of {total}",
                index + 1,
                total,
            )

            segment_result = self._analyze_segment(
                audio,
                sr,
                raw_segment,
                index,
                all_segment_bounds,
                duration,
                recording_percentiles,
                rec_f0,
                rec_voiced_flags,
                rec_rms=rec_rms,
                n_recording_segments=total,
                asr_recording_flags=recording_asr_flags,
            )

            if segment_result is not None:
                # Stamp recording-level language detection onto each segment so
                # the UI badge shows the actual detected language, not config.
                segment_result["language"] = detected_lang
                segment_result["language_probability"] = lang_prob
                analyzed_segments.append(segment_result)
        t_segments_elapsed = time.perf_counter() - t_segments_start

        if not analyzed_segments:
            raise ValueError("No valid conversation segments were produced.")

        # ---- Collapse detector: low voice-model discrimination ----
        voice_collapse: Dict[str, Any] = {"detected": False}
        voice_emotions = [
            s.get("voice_emotion", {}).get("emotion")
            for s in analyzed_segments
            if s.get("voice_emotion", {}).get("emotion")
        ]
        if len(voice_emotions) >= 3:
            counts: Dict[str, int] = {}
            for e in voice_emotions:
                counts[e] = counts.get(e, 0) + 1
            dom_emotion, max_cnt = max(counts.items(), key=lambda x: x[1])
            dom_share = max_cnt / len(voice_emotions)
            collapse_thresh = getattr(config, "VOICE_COLLAPSE_THRESHOLD", 0.70)
            if dom_share >= collapse_thresh:
                penalty = getattr(config, "VOICE_COLLAPSE_RELIABILITY_PENALTY", 0.50)
                voice_collapse = {
                    "detected": True,
                    "dominant_emotion": dom_emotion,
                    "share": round(dom_share, 4),
                    "threshold": collapse_thresh,
                    "penalty": penalty,
                    "warning": "Voice model shows low discrimination on this input; treat voice labels as low reliability.",
                }
                logger.warning(
                    "Voice collapse detected: dominant '%s' occupies %.1f%% of segments (>= %.0f%%). "
                    "Down-weighting voice reliability by factor %.2f.",
                    dom_emotion, dom_share * 100, collapse_thresh * 100, penalty,
                )
                for seg in analyzed_segments:
                    if "voice_emotion" in seg:
                        seg["voice_emotion"]["voice_reliability"] = round(
                            seg["voice_emotion"].get("voice_reliability", 1.0) * penalty, 4
                        )
                        seg["voice_emotion"]["collapse_warning"] = voice_collapse["warning"]
                        seg["voice_emotion"]["voice_collapse_detected"] = True
                        seg["voice_emotion"]["low_voice_reliability"] = True
                        seg["voice_collapse_detected"] = True
                        seg["low_voice_reliability"] = True
                        seg["voice_reliability"] = seg["voice_emotion"]["voice_reliability"]
                        if "fusion" in seg:
                            try:
                                seg["fusion"] = self.fusion.fuse(
                                    seg["voice_emotion"],
                                    seg.get("text_sentiment"),
                                )

                            except Exception:
                                logger.exception(
                                    "Re-fusion after collapse penalty failed for segment %s.",
                                    seg.get("segment_index"),
                                )

        # ---- Speaker diarization ----
        self._report_progress(progress_callback, "Identifying speakers", 0, 1)

        t_diarize_start = time.perf_counter()
        analyzed_segments, diarization_error = self._apply_diarization(
            audio_path, analyzed_segments
        )
        t_diarize_elapsed = time.perf_counter() - t_diarize_start

        # ---- Emotional trend ----
        self._report_progress(progress_callback, "Computing emotional trend", 0, 1)

        t_analytics_start = time.perf_counter()
        trend_result = self.detect_emotional_trend(analyzed_segments)

        # ---- Escalation (legacy strict + advanced) ----
        escalation_legacy_strict = self.detect_escalation(analyzed_segments)
        escalation_result = self.detect_escalation_advanced(analyzed_segments)

        # ---- Unified verdict ----
        verdict_result = self.compute_unified_verdict(analyzed_segments)

        # ---- Modality consistency trajectory ----
        self._report_progress(progress_callback, "Computing consistency trajectory", 0, 1)

        consistency_trajectory = compute_consistency_trajectory(analyzed_segments)

        # ---- Review-priority attention score (heuristic only) ----
        attention_score = compute_attention_score(
            segments=analyzed_segments,
            escalation_result=escalation_result,
            trajectory=consistency_trajectory,
        )

        # ---- Modality incongruence & conflict analysis ----
        incongruence_result = analyze_conversation_incongruence(analyzed_segments)
        t_analytics_elapsed = time.perf_counter() - t_analytics_start

        t_total_elapsed = time.perf_counter() - t_total_start

        processing_time = {
            "audio_preprocessing_seconds": round(t_load_elapsed, 3),
            "transcription_whisper_seconds": round(t_transcribe_elapsed, 3),
            "segment_emotion_sentiment_fusion_seconds": round(t_segments_elapsed, 3),
            "speaker_diarization_seconds": round(t_diarize_elapsed, 3),
            "analytics_and_trajectory_seconds": round(t_analytics_elapsed, 3),
            "total_processing_seconds": round(t_total_elapsed, 3),
        }

        # ---- Final result assembly & recommendation ----
        result = {
            "audio_path": audio_path,
            "duration_seconds": duration,
            "sample_rate": sr,
            "language": transcription.get("language", "unknown"),
            "language_probability": float(transcription.get("language_probability", 0.0)),
            "asr_flags": recording_asr_flags,
            "full_transcript": transcription.get("text", ""),
            "segments": analyzed_segments,
            "emotional_trend": trend_result,
            # "escalation" holds the ADVANCED result (strict or tolerant per
            # config.ESCALATION_MODE). The strict-only result is kept
            # separately for transparency/ablation.
            "escalation": escalation_result,
            "escalation_legacy_strict": escalation_legacy_strict,
            "verdict": verdict_result,
            "consistency_trajectory": consistency_trajectory.to_dict(),
            "attention_score": attention_score.to_dict(),
            "modality_incongruence": incongruence_result,
            "processing_time": processing_time,
            "skipped_segments": total - len(analyzed_segments),
            "diarization_error": diarization_error,
            "voice_collapse": voice_collapse,
        }

        recommendation = generate_recommendation(result)
        result["recommendation"] = recommendation.to_dict()

        self._report_progress(progress_callback, "Done", total, total)

        logger.info(
            "Conversation analysis completed: %d/%d segments analyzed in %.2fs",
            len(analyzed_segments),
            total,
            t_total_elapsed,
        )

        return result