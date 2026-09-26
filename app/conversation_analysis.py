"""
Conversation Analysis Module.

Builds segment-level multimodal analysis from:
- Audio
- Voice emotion
- Speech-to-text
- Text sentiment
- Multimodal fusion

Also provides:
- Emotional trend detection (legacy, first-vs-last delta)
- Emotional escalation detection (strict AND tolerant modes — Phase 4)
- A single unified conversation verdict (Phase 4: escalating /
  de-escalating / stable / fluctuating / insufficient_data)
- Segment-level emotional intensity (legacy v1 confidence-based formula,
  and Phase 4's v2 prosody-based formula, selectable via
  config.INTENSITY_FORMULA_VERSION)
- Speaker diarization (Speaker A / Speaker B labels)
- Modality consistency trajectory (voice vs. text direction agreement)
- Explainable review-priority attention score (heuristic only)
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

import config
from app.audio_processor import load_and_preprocess_audio
from app.emotion_analyzer import VoiceEmotionAnalyzer
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer
from app.fusion import MultimodalFusion
from app.diarization import SpeakerDiarizer
from app.voice_calibration import VoiceCalibrator, compute_widened_window
from app.prosody import compute_prosodic_arousal, compute_voice_arousal, compute_text_negativity
from app.consistency_analysis import (
    compute_consistency_trajectory,
    compute_attention_score,
)
from app.incongruence_analysis import analyze_conversation_incongruence
from app.recommendation_engine import generate_recommendation

logger = logging.getLogger(__name__)

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
    ):
        logger.info(
            "Initializing conversation analysis pipeline..."
        )

        self.emotion_analyzer = (
            emotion_analyzer
            or VoiceEmotionAnalyzer()
        )

        self.speech_to_text = (
            speech_to_text
            or SpeechToText()
        )

        self.sentiment_analyzer = (
            sentiment_analyzer
            or SentimentAnalyzer()
        )

        self.fusion = (
            fusion
            or MultimodalFusion()
        )

        # Lazy-loads the pyannote pipeline on first .diarize() call, so
        # construction here is cheap even if diarization never runs.
        self.diarizer = (
            diarizer
            or SpeakerDiarizer()
        )

        # Phase 3: voice emotion calibration (prosody sanity rule +
        # temperature scaling). Inert (identity behavior) until enough
        # labeled samples exist to fit temperature scaling; the prosody
        # rule can still apply from very small N since it is data-free.
        self.voice_calibrator = (
            voice_calibrator
            or VoiceCalibrator()
        )

        logger.info(
            "Conversation analysis pipeline initialized."
        )

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
        """Call the progress callback if provided, never let it break the pipeline."""

        if callback is None:
            return

        try:
            callback(stage, current, total)
        except Exception:
            logger.exception(
                "Progress callback raised an exception; ignoring."
            )

    # ---------------------------------------------------------
    # Intensity — v1 (legacy, confidence-based)
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_intensity(
        segment: Dict[str, Any]
    ) -> float:
        """
        LEGACY (v1) intensity formula: pure fusion-confidence based.

        Kept unchanged and still directly unit-tested — this is the
        pre-Phase-4 formula. Used when config.INTENSITY_FORMULA_VERSION
        == 'v1', and always available for the ablation comparison against
        the new v2 (prosody-based) formula in _calculate_intensity_v2.

        Agreement between voice and text increases confidence.
        Strong disagreement reduces the influence of the combined
        confidence so that one unreliable modality does not dominate
        the final intensity.

        Returns:
            Float between 0.0 and 1.0.
        """

        fusion = segment["fusion"]

        confidence = float(
            fusion.get(
                "combined_confidence",
                0.0
            )
        )

        agreement = bool(
            fusion.get(
                "agreement",
                False
            )
        )

        mismatch = bool(
            fusion.get(
                "mismatch",
                False
            )
        )

        if agreement:
            # Agreement provides stronger multimodal evidence.
            intensity = confidence + 0.10

        elif mismatch:
            # Reduce intensity when modalities disagree.
            intensity = confidence * 0.85

        else:
            # Low-confidence disagreement.
            intensity = confidence * 0.90

        return round(
            min(1.0, max(0.0, intensity)),
            4
        )

    # ---------------------------------------------------------
    # Intensity — v2 (Phase 4, prosody-based)
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_intensity_v2(
        voice_scores: Dict[str, float],
        text_scores: Dict[str, float],
        audio_segment: np.ndarray,
        sr: int,
        word_count: int,
        duration_seconds: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Phase 4 intensity formula: weighted combination of

          1. prosodic_arousal   — this segment's own loudness, pitch
                                   movement, and speech rate (app/prosody.py)
          2. voice_arousal      — expected arousal from the voice-emotion
                                   class probabilities (angry/happy = high,
                                   sad/neutral = low), not just top-class
                                   confidence
          3. text_negativity    — probability mass on the 'negative' text
                                   sentiment class

        Unlike v1, this is NOT purely a function of model confidence — a
        confidently-classified "sad" segment does not automatically read
        as more intense than an unsure one; actual vocal energy and
        linguistic negativity now drive the score.

        Returns (intensity, components) where components is a dict
        breaking down every sub-score, for the "why is this 0.82?"
        transparency requirement.
        """
        prosody_result = compute_prosodic_arousal(
            audio_segment, sr, word_count, duration_seconds
        )
        voice_arousal = compute_voice_arousal(voice_scores)
        text_negativity = compute_text_negativity(text_scores)

        intensity = (
            config.INTENSITY_V2_WEIGHT_PROSODY * prosody_result.arousal
            + config.INTENSITY_V2_WEIGHT_VOICE_AROUSAL * voice_arousal
            + config.INTENSITY_V2_WEIGHT_TEXT_NEGATIVITY * text_negativity
        )
        intensity = round(min(1.0, max(0.0, intensity)), 4)

        components = {
            "formula_version": "v2",
            "prosodic_arousal": prosody_result.arousal,
            "prosody_components": prosody_result.components,
            "voice_arousal": voice_arousal,
            "text_negativity": text_negativity,
            "weights": {
                "prosody": config.INTENSITY_V2_WEIGHT_PROSODY,
                "voice_arousal": config.INTENSITY_V2_WEIGHT_VOICE_AROUSAL,
                "text_negativity": config.INTENSITY_V2_WEIGHT_TEXT_NEGATIVITY,
            },
        }

        return intensity, components

    # ---------------------------------------------------------
    # Emotional trend (legacy — first-vs-last delta)
    # ---------------------------------------------------------

    @staticmethod
    def detect_emotional_trend(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect the overall emotional intensity trend (first-vs-last
        delta only — kept unchanged from pre-Phase-4 for backward
        compatibility and direct unit-test coverage). See
        compute_unified_verdict() for the new Phase 4 slope-based signal
        that is now the primary user-facing indicator.

        Rules:

        Increasing:
            Last intensity is at least 0.10 higher than the first.

        Decreasing:
            Last intensity is at least 0.10 lower than the first.

        Stable:
            Change is smaller than 0.10.
        """

        if len(segments) < 2:
            return {
                "trend": "insufficient_data",
                "intensity_scores": [],
                "change": 0.0,
            }

        intensities = [
            float(
                segment.get(
                    "intensity",
                    0.0
                )
            )
            for segment in segments
        ]

        first = intensities[0]
        last = intensities[-1]

        change = round(
            last - first,
            4
        )

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
        }

    # ---------------------------------------------------------
    # Escalation — strict (legacy, unchanged, still unit-tested)
    # ---------------------------------------------------------

    @staticmethod
    def detect_escalation(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        LEGACY strict escalation rule. Kept byte-for-byte unchanged from
        pre-Phase-4 so existing unit tests keep passing, and so it remains
        available as config.ESCALATION_MODE == 'strict' / the ablation
        baseline against the new tolerant rule below.

        Escalation requires:

        1. At least three segments.
        2. Intensity continuously increases (every consecutive step, no
           exceptions).
        3. Total increase is at least 0.20.

        Example:
            0.40 → 0.65 → 0.90
        triggers escalation.

        Example:
            0.75 → 0.67 → 0.97
        does NOT trigger escalation because the increase is not
        continuous — a single dip breaks the strict rule even though the
        overall trend and net rise are both upward. (This is exactly the
        "Trend: Increasing / Escalation: Not Detected" contradiction that
        Phase 4's tolerant mode below addresses.)
        """

        if len(segments) < 3:
            return {
                "escalation_detected": False,
                "reason": "At least 3 segments are required.",
                "intensity_change": 0.0,
            }

        intensities = [
            float(
                segment.get(
                    "intensity",
                    0.0
                )
            )
            for segment in segments
        ]

        consecutive_increase = all(
            intensities[i] > intensities[i - 1]
            for i in range(1, len(intensities))
        )

        total_change = round(
            intensities[-1] - intensities[0],
            4
        )

        escalation_detected = (
            consecutive_increase
            and total_change >= 0.20
        )

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
    # Escalation — tolerant (Phase 4, robust-slope based)
    # ---------------------------------------------------------

    @staticmethod
    def _ols_slope(values: List[float]) -> float:
        """Ordinary-least-squares slope of values against their index
        (0, 1, 2, ...). Returns 0.0 for fewer than 2 points."""
        n = len(values)
        if n < 2:
            return 0.0
        x = np.arange(n, dtype=float)
        y = np.array(values, dtype=float)
        # np.polyfit(x, y, 1) returns [slope, intercept]
        slope, _intercept = np.polyfit(x, y, 1)
        return float(slope)

    @staticmethod
    def detect_escalation_advanced(
        segments: List[Dict[str, Any]],
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Phase 4 escalation detection with two selectable modes:

          'strict'    — delegates to the legacy detect_escalation() rule
                        above (every consecutive step must increase).

          'tolerant'  — (default, config.ESCALATION_MODE) fits a robust
                        linear slope across all segment intensities and
                        allows individual dips up to
                        config.ESCALATION_DIP_TOLERANCE without breaking
                        the trend. Requires ALL of:
                          1. at least config.ESCALATION_MIN_SEGMENTS segments
                          2. slope >= config.ESCALATION_MIN_SLOPE
                          3. net rise (last - first) >= config.ESCALATION_MIN_NET_RISE
                          4. no single-step drop exceeds config.ESCALATION_DIP_TOLERANCE

        This directly resolves the contradiction where Trend said
        "Increasing" but Escalation said "Not Detected" for the same
        conversation — a conversation with one small dip within
        tolerance can now be correctly flagged as escalating.

        Returns a dict with: escalation_detected, reason, intensity_change
        (net rise — same field name as the legacy rule, so
        recommendation_engine.py's existing .get("intensity_change", ...)
        read continues to work unchanged), mode, slope, dip_violations
        (list of {index, drop} for any step that broke tolerance).
        """
        mode = mode or config.ESCALATION_MODE

        if mode == "strict":
            legacy = ConversationAnalyzer.detect_escalation(segments)
            legacy["mode"] = "strict"
            legacy["slope"] = None
            legacy["dip_violations"] = []
            return legacy

        # --- tolerant mode ---
        if len(segments) < config.ESCALATION_MIN_SEGMENTS:
            return {
                "escalation_detected": False,
                "reason": f"At least {config.ESCALATION_MIN_SEGMENTS} segments are required.",
                "intensity_change": 0.0,
                "mode": "tolerant",
                "slope": 0.0,
                "dip_violations": [],
            }

        intensities = [
            float(segment.get("intensity", 0.0)) for segment in segments
        ]

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
    # Unified verdict (Phase 4)
    # ---------------------------------------------------------

    @staticmethod
    def compute_unified_verdict(
        segments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Phase 4: a single conversation-level verdict replacing the
        separate (and sometimes contradictory) Trend + Escalation
        display. One of:

          'escalating'     — clear sustained upward slope, no large dips
          'de-escalating'  — clear sustained downward slope, no large
                              upward spikes
          'fluctuating'    — notable back-and-forth movement without a
                              clear net direction
          'stable'         — small overall movement, no notable spikes
          'insufficient_data' — fewer than 2 segments

        This uses the same slope/dip-tolerance machinery as
        detect_escalation_advanced('tolerant') but is INDEPENDENT of
        config.ESCALATION_MODE — the verdict is always computed the
        tolerant way, since it is meant to be a softer, always-available
        descriptive signal. escalation_detected (used for the "HIGH
        ESCALATION" critical recommendation) remains gated by
        config.ESCALATION_MIN_SEGMENTS and whichever mode is configured;
        the verdict may say "escalating" even for a 2-segment rise that
        would not by itself trigger a HIGH ESCALATION recommendation.
        """
        if len(segments) < 2:
            return {
                "verdict": "insufficient_data",
                "reason": "At least 2 segments are required to assess a trend.",
                "slope": 0.0,
                "net_change": 0.0,
            }

        intensities = [
            float(segment.get("intensity", 0.0)) for segment in segments
        ]

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
        # Fluctuating: notable single-step movement in either direction
        # (beyond a relaxed 1.5x tolerance) without qualifying as a clean
        # escalating/de-escalating trend.
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

        return {
            "verdict": verdict,
            "reason": reason,
            "slope": round(slope, 4),
            "net_change": net_change,
        }

    # ---------------------------------------------------------
    # Speaker diarization
    # ---------------------------------------------------------

    def _apply_diarization(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Attach speaker labels to transcript segments using pyannote.

        Diarization is an enhancement, so a failure does not abort the
        complete multimodal analysis. The error is returned separately
        so the Streamlit dashboard can explain why labels are Unknown.
        """

        try:
            turns = self.diarizer.diarize(audio_path)

            logger.info(
                "Speaker diarization returned %d turns.",
                len(turns),
            )
            logger.info("Diarization turns: %s", turns)

            labeled_segments = self.diarizer.assign_speakers(
                segments,
                turns,
            )

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
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a single transcript segment.

        Returns None (and logs a warning) if the segment is
        malformed or if any stage of the pipeline fails for
        this segment specifically. A failure here does not
        abort analysis of the remaining segments.
        """

        start = float(raw_segment.get("start", 0.0) or 0.0)
        end = float(raw_segment.get("end", 0.0) or 0.0)
        text = str(raw_segment.get("text", "") or "").strip()

        if not text:
            logger.info(
                "Skipping segment %d: empty transcript.",
                index + 1,
            )
            return None

        if end < start:
            logger.warning(
                "Skipping segment %d: end (%.2f) is before start (%.2f).",
                index + 1,
                end,
                start,
            )
            return None

        logger.info(
            "Analyzing conversation segment %d: %.2f-%.2fs",
            index + 1,
            start,
            end,
        )

        # Raw (unwidened) audio slice for this segment's transcript
        # window — used by both voice calibration and Phase 4's v2
        # intensity formula (prosodic arousal). Computed once here,
        # unconditionally, regardless of which downstream features are
        # enabled.
        start_sample = max(0, int(start * sr))
        end_sample = min(len(audio), int(end * sr))
        segment_audio = audio[start_sample:end_sample]

        # -----------------------------------------------------
        # Phase 3: widen the acoustic window for short segments
        # so the emotion model has enough signal. The transcript
        # window (start/end used everywhere else) is unchanged;
        # only the audio slice passed to analyze_segment below is
        # widened. Wrapped defensively: a failure here must never
        # abort the whole pipeline, it should just fall back to
        # the original (unwidened) window.
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Phase 3: voice calibration (prosody sanity rule +
        # temperature scaling, when fitted). Runs on the raw
        # scores returned above; raw values are preserved
        # alongside the calibrated ones for transparency.
        # -----------------------------------------------------
        if config.VOICE_CALIBRATION_ENABLED:
            try:
                calibration = self.voice_calibrator.calibrate_segment(
                    raw_scores=voice_result["scores"],
                    audio_segment=segment_audio,
                    sr=sr,
                )

                voice_result["raw_emotion"] = voice_result["emotion"]
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

        try:
            sentiment_result = self.sentiment_analyzer.analyze(text)
        except Exception:
            logger.exception(
                "Text sentiment analysis failed for segment %d; skipping.",
                index + 1,
            )
            return None

        try:
            fusion_result = self.fusion.fuse(
                voice_result,
                sentiment_result,
            )
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
            "fusion": fusion_result,
        }

        # -----------------------------------------------------
        # Phase 4: intensity — v2 (prosody-based) by default, with
        # a defensive fallback to v1 (legacy, confidence-based) if
        # anything in the new formula fails, so a bug here degrades
        # gracefully rather than aborting analysis of this segment.
        # -----------------------------------------------------
        try:
            if config.INTENSITY_FORMULA_VERSION == "v2":
                word_count = len(text.split())
                seg_duration = max(0.0, end - start)
                intensity_v2, intensity_components = self._calculate_intensity_v2(
                    voice_scores=voice_result.get("scores", {}),
                    text_scores=sentiment_result.get("scores", {}),
                    audio_segment=segment_audio,
                    sr=sr,
                    word_count=word_count,
                    duration_seconds=seg_duration,
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
                callback(stage, current, total) at each pipeline
                stage, so a caller (e.g. a UI) can show progress.
                Exceptions raised inside the callback are logged
                and otherwise ignored.

        Raises:
            ValueError: If the audio is empty/degenerate, or if
                no usable segments could be produced.
        """

        logger.info(
            "Starting conversation analysis: %s",
            audio_path
        )

        t_total_start = time.perf_counter()

        # -----------------------------------------------------
        # Load audio once
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Loading audio", 0, 1
        )

        t_load_start = time.perf_counter()
        audio, sr, duration = (
            load_and_preprocess_audio(
                audio_path
            )
        )
        t_load_elapsed = time.perf_counter() - t_load_start

        if duration is None or duration <= 0 or audio is None or len(audio) == 0:
            raise ValueError(
                "The audio file appears to be empty or could not be decoded."
            )

        # -----------------------------------------------------
        # Speech-to-text
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Transcribing speech", 0, 1
        )

        t_transcribe_start = time.perf_counter()
        transcription = (
            self.speech_to_text.transcribe(
                audio_path
            )
        )
        t_transcribe_elapsed = time.perf_counter() - t_transcribe_start

        raw_segments = transcription.get(
            "segments",
            []
        )

        if not raw_segments:
            raise ValueError(
                "Speech-to-text returned no usable segments."
            )

        analyzed_segments: List[Dict[str, Any]] = []
        total = len(raw_segments)

        # -----------------------------------------------------
        # Analyze each transcript segment
        # -----------------------------------------------------

        # Pre-compute all segment bounds once so each segment's short-
        # segment widening can see its neighbors without re-scanning.
        all_segment_bounds: List[Tuple[float, float]] = [
            (float(s.get("start", 0.0) or 0.0), float(s.get("end", 0.0) or 0.0))
            for s in raw_segments
        ]

        t_segments_start = time.perf_counter()
        for index, raw_segment in enumerate(raw_segments):

            self._report_progress(
                progress_callback,
                f"Analyzing segment {index + 1} of {total}",
                index + 1,
                total,
            )

            segment_result = self._analyze_segment(
                audio, sr, raw_segment, index, all_segment_bounds, duration
            )

            if segment_result is not None:
                analyzed_segments.append(segment_result)
        t_segments_elapsed = time.perf_counter() - t_segments_start

        if not analyzed_segments:
            raise ValueError(
                "No valid conversation segments were produced."
            )

        # -----------------------------------------------------
        # Speaker diarization
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Identifying speakers", 0, 1
        )

        t_diarize_start = time.perf_counter()
        analyzed_segments, diarization_error = self._apply_diarization(
            audio_path, analyzed_segments
        )
        t_diarize_elapsed = time.perf_counter() - t_diarize_start

        # -----------------------------------------------------
        # Emotional trend (legacy, first-vs-last delta)
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Computing emotional trend", 0, 1
        )

        t_analytics_start = time.perf_counter()
        trend_result = (
            self.detect_emotional_trend(
                analyzed_segments
            )
        )

        # -----------------------------------------------------
        # Escalation — legacy strict (kept for ablation/transparency)
        # and Phase 4 advanced (mode-selectable; this is now the
        # primary user-facing escalation result under result["escalation"])
        # -----------------------------------------------------

        escalation_legacy_strict = self.detect_escalation(analyzed_segments)
        escalation_result = self.detect_escalation_advanced(analyzed_segments)

        # -----------------------------------------------------
        # Phase 4: single unified verdict (always computed the
        # tolerant way, independent of config.ESCALATION_MODE — see
        # compute_unified_verdict's docstring)
        # -----------------------------------------------------

        verdict_result = self.compute_unified_verdict(analyzed_segments)

        # -----------------------------------------------------
        # Modality consistency trajectory
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Computing consistency trajectory", 0, 1
        )

        consistency_trajectory = compute_consistency_trajectory(
            analyzed_segments
        )

        # -----------------------------------------------------
        # Review-priority attention score (heuristic only)
        # -----------------------------------------------------

        attention_score = compute_attention_score(
            segments=analyzed_segments,
            escalation_result=escalation_result,
            trajectory=consistency_trajectory,
        )

        # -----------------------------------------------------
        # Modality incongruence & conflict analysis
        # -----------------------------------------------------

        incongruence_result = analyze_conversation_incongruence(
            analyzed_segments
        )
        t_analytics_elapsed = time.perf_counter() - t_analytics_start

        t_total_elapsed = time.perf_counter() - t_total_start

        # Timing breakdown in seconds
        processing_time = {
            "audio_preprocessing_seconds": round(t_load_elapsed, 3),
            "transcription_whisper_seconds": round(t_transcribe_elapsed, 3),
            "segment_emotion_sentiment_fusion_seconds": round(t_segments_elapsed, 3),
            "speaker_diarization_seconds": round(t_diarize_elapsed, 3),
            "analytics_and_trajectory_seconds": round(t_analytics_elapsed, 3),
            "total_processing_seconds": round(t_total_elapsed, 3),
        }

        # -----------------------------------------------------
        # Final result assembly & recommendation
        # -----------------------------------------------------

        result = {
            "audio_path": audio_path,

            "duration_seconds": duration,

            "sample_rate": sr,

            "language": transcription.get(
                "language",
                "unknown"
            ),

            "language_probability": float(
                transcription.get(
                    "language_probability",
                    0.0
                )
            ),

            "full_transcript": transcription.get(
                "text",
                ""
            ),

            "segments": analyzed_segments,

            "emotional_trend": trend_result,

            # Phase 4: "escalation" now holds the ADVANCED result
            # (strict or tolerant per config.ESCALATION_MODE — default
            # tolerant). This is the key both streamlit_app.py and
            # recommendation_engine.py read. The original strict-only
            # result is preserved separately for transparency/ablation.
            "escalation": escalation_result,
            "escalation_legacy_strict": escalation_legacy_strict,

            # Phase 4: single unified verdict — the new primary
            # conversation-level signal (see compute_unified_verdict).
            "verdict": verdict_result,

            "consistency_trajectory": consistency_trajectory.to_dict(),

            "attention_score": attention_score.to_dict(),

            "modality_incongruence": incongruence_result,

            "processing_time": processing_time,

            "skipped_segments": total - len(analyzed_segments),
            "diarization_error": diarization_error,
        }

        # Synthesize actionable recommendation
        recommendation = generate_recommendation(result)
        result["recommendation"] = recommendation.to_dict()

        self._report_progress(
            progress_callback, "Done", total, total
        )

        logger.info(
            "Conversation analysis completed: %d/%d segments analyzed in %.2fs",
            len(analyzed_segments),
            total,
            t_total_elapsed,
        )

        return result