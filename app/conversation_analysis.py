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
"""

import logging
from typing import Any, Dict, List

from app.audio_processor import load_and_preprocess_audio
from app.emotion_analyzer import VoiceEmotionAnalyzer
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer
from app.fusion import MultimodalFusion

logger = logging.getLogger(__name__)


class ConversationAnalyzer:
    """Analyze a conversation segment by segment."""

    def __init__(
        self,
        emotion_analyzer: VoiceEmotionAnalyzer | None = None,
        speech_to_text: SpeechToText | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        fusion: MultimodalFusion | None = None,
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

        logger.info(
            "Conversation analysis pipeline initialized."
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
    # Main analysis
    # ---------------------------------------------------------

    def analyze(
        self,
        audio_path: str
    ) -> Dict[str, Any]:
        """
        Analyze an audio conversation segment by segment.
        """

        logger.info(
            "Starting conversation analysis: %s",
            audio_path
        )

        # -----------------------------------------------------
        # Load audio once
        # -----------------------------------------------------

        audio, sr, duration = (
            load_and_preprocess_audio(
                audio_path
            )
        )

        # -----------------------------------------------------
        # Speech-to-text
        # -----------------------------------------------------

        transcription = (
            self.speech_to_text.transcribe(
                audio_path
            )
        )

        segments = transcription.get(
            "segments",
            []
        )

        if not segments:
            raise ValueError(
                "Speech-to-text returned no usable segments."
            )

        analyzed_segments: List[
            Dict[str, Any]
        ] = []

        # -----------------------------------------------------
        # Analyze each transcript segment
        # -----------------------------------------------------

        for index, segment in enumerate(
            segments
        ):

            start = float(
                segment["start"]
            )

            end = float(
                segment["end"]
            )

            text = segment["text"].strip()

            if not text:
                continue

            logger.info(
                "Analyzing conversation segment %d: "
                "%.2f-%.2fs",
                index + 1,
                start,
                end,
            )

            # -------------------------------------------------
            # Voice emotion
            # -------------------------------------------------

            voice_result = (
                self.emotion_analyzer.analyze_segment(
                    audio,
                    sr,
                    start,
                    end,
                )
            )

            # -------------------------------------------------
            # Text sentiment
            # -------------------------------------------------

            sentiment_result = (
                self.sentiment_analyzer.analyze(
                    text
                )
            )

            # -------------------------------------------------
            # Multimodal fusion
            # -------------------------------------------------

            fusion_result = (
                self.fusion.fuse(
                    voice_result,
                    sentiment_result,
                )
            )

            # -------------------------------------------------
            # Build segment result
            # -------------------------------------------------

            segment_result = {
                "segment_index": index + 1,

                "start_seconds": start,

                "end_seconds": end,

                "transcript": text,

                "voice_emotion": voice_result,

                "text_sentiment": sentiment_result,

                "fusion": fusion_result,
            }

            # -------------------------------------------------
            # Calculate intensity
            # -------------------------------------------------

            segment_result["intensity"] = (
                self._calculate_intensity(
                    segment_result
                )
            )

            analyzed_segments.append(
                segment_result
            )

        if not analyzed_segments:
            raise ValueError(
                "No valid conversation segments were produced."
            )

        # -----------------------------------------------------
        # Emotional trend
        # -----------------------------------------------------

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
        }

        logger.info(
            "Conversation analysis completed: %d segments",
            len(analyzed_segments)
        )

        return result