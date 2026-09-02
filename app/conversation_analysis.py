"""
Conversation Analysis Module.

Builds segment-level multimodal analysis from:
- Audio
- Voice emotion
- Speech-to-text
- Text sentiment
- Multimodal fusion

Also provides:
- Emotional trend detection
- Emotional escalation detection
- Segment-level emotional intensity
- Speaker diarization (Speaker A / Speaker B labels)
"""

import logging
from typing import Any, Callable, Dict, List, Optional

from app.audio_processor import load_and_preprocess_audio
from app.emotion_analyzer import VoiceEmotionAnalyzer
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer
from app.fusion import MultimodalFusion
from app.diarization import SpeakerDiarizer

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
    # Intensity
    # ---------------------------------------------------------

    @staticmethod
    def _calculate_intensity(
        segment: Dict[str, Any]
    ) -> float:
        """
        Calculate emotional intensity using multimodal evidence.

        Agreement between voice and text increases confidence.

        Strong disagreement reduces the influence of the
        combined confidence so that one unreliable modality
        does not dominate the final intensity.

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
    # Emotional trend
    # ---------------------------------------------------------

    @staticmethod
    def detect_emotional_trend(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect the overall emotional intensity trend.

        Rules:

        Increasing:
            Last intensity is at least 0.10 higher
            than the first.

        Decreasing:
            Last intensity is at least 0.10 lower
            than the first.

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
    # Escalation
    # ---------------------------------------------------------

    @staticmethod
    def detect_escalation(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect emotional escalation.

        Escalation requires:

        1. At least three segments.
        2. Intensity continuously increases.
        3. Total increase is at least 0.20.

        Example:

            0.40 → 0.65 → 0.90

        will trigger escalation.

        Example:

            0.75 → 0.67 → 0.97

        will NOT trigger escalation because the increase
        is not continuous.
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

        try:
            voice_result = self.emotion_analyzer.analyze_segment(
                audio,
                sr,
                start,
                end,
            )
        except Exception:
            logger.exception(
                "Voice emotion analysis failed for segment %d; skipping.",
                index + 1,
            )
            return None

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

        try:
            segment_result["intensity"] = self._calculate_intensity(
                segment_result
            )
        except Exception:
            logger.exception(
                "Intensity calculation failed for segment %d; defaulting to 0.0.",
                index + 1,
            )
            segment_result["intensity"] = 0.0

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

        # -----------------------------------------------------
        # Load audio once
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Loading audio", 0, 1
        )

        audio, sr, duration = (
            load_and_preprocess_audio(
                audio_path
            )
        )

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

        transcription = (
            self.speech_to_text.transcribe(
                audio_path
            )
        )

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

        for index, raw_segment in enumerate(raw_segments):

            self._report_progress(
                progress_callback,
                f"Analyzing segment {index + 1} of {total}",
                index + 1,
                total,
            )

            segment_result = self._analyze_segment(
                audio, sr, raw_segment, index
            )

            if segment_result is not None:
                analyzed_segments.append(segment_result)

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

        analyzed_segments, diarization_error = self._apply_diarization(
            audio_path, analyzed_segments
        )

        # -----------------------------------------------------
        # Emotional trend
        # -----------------------------------------------------

        self._report_progress(
            progress_callback, "Computing emotional trend", 0, 1
        )

        trend_result = (
            self.detect_emotional_trend(
                analyzed_segments
            )
        )

        # -----------------------------------------------------
        # Escalation
        # -----------------------------------------------------

        escalation_result = (
            self.detect_escalation(
                analyzed_segments
            )
        )

        # -----------------------------------------------------
        # Final result
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

            "escalation": escalation_result,

            "skipped_segments": total - len(analyzed_segments),
            "diarization_error": diarization_error,
        }

        self._report_progress(
            progress_callback, "Done", total, total
        )

        logger.info(
            "Conversation analysis completed: %d/%d segments analyzed",
            len(analyzed_segments),
            total,
        )

        return result