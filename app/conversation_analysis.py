"""
Conversation Analysis Module.

Builds segment-level multimodal analysis from audio, transcript,
voice emotion, text sentiment, and fusion results.

Also provides emotional trend and escalation detection.
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
        logger.info("Initializing conversation analysis pipeline...")

        self.emotion_analyzer = (
            emotion_analyzer or VoiceEmotionAnalyzer()
        )

        self.speech_to_text = (
            speech_to_text or SpeechToText()
        )

        self.sentiment_analyzer = (
            sentiment_analyzer or SentimentAnalyzer()
        )

        self.fusion = fusion or MultimodalFusion()

        logger.info("Conversation analysis pipeline initialized.")

    @staticmethod
    def _calculate_intensity(segment: Dict[str, Any]) -> float:
        """
        Calculate a simple emotional intensity score.

        Uses the fusion confidence as the primary signal and gives
        additional weight to clearly negative emotional states.

        Returns:
            Intensity score between 0.0 and 1.0.
        """

        fusion_result = segment["fusion"]

        confidence = float(
            fusion_result.get("combined_confidence", 0.0)
        )

        audio_emotion = segment["voice_emotion"]["emotion"].lower()
        text_sentiment = segment["text_sentiment"]["sentiment"].lower()

        negative_audio = audio_emotion in {
            "angry",
            "sad",
        }

        negative_text = text_sentiment == "negative"

        if negative_audio and negative_text:
            intensity = min(1.0, confidence + 0.10)

        elif negative_audio or negative_text:
            intensity = min(1.0, confidence + 0.05)

        else:
            intensity = confidence

        return round(float(intensity), 4)

    @staticmethod
    def detect_emotional_trend(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect the overall emotional intensity trend.

        Rules:
        - Increasing: later average intensity is meaningfully higher.
        - Decreasing: later average intensity is meaningfully lower.
        - Stable: change is small.

        This is intentionally simple and explainable.
        """

        if len(segments) < 2:
            return {
                "trend": "insufficient_data",
                "intensity_scores": [],
                "change": 0.0,
            }

        intensities = [
            float(segment.get("intensity", 0.0))
            for segment in segments
        ]

        first = intensities[0]
        last = intensities[-1]
        change = round(last - first, 4)

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

    @staticmethod
    def detect_escalation(
        segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Detect emotional escalation.

        Escalation is flagged when:
        1. There are at least three segments.
        2. Emotional intensity increases from segment to segment.
        3. The total increase from the first to final segment is
           at least 0.20.

        This rule is transparent and intentionally conservative.
        """

        if len(segments) < 3:
            return {
                "escalation_detected": False,
                "reason": "At least 3 segments are required.",
                "intensity_change": 0.0,
            }

        intensities = [
            float(segment.get("intensity", 0.0))
            for segment in segments
        ]

        consecutive_increase = all(
            intensities[i] > intensities[i - 1]
            for i in range(1, len(intensities))
        )

        total_change = round(
            intensities[-1] - intensities[0],
            4,
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

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze an audio conversation segment by segment.
        """

        logger.info(
            "Starting conversation analysis: %s",
            audio_path,
        )

        # Load audio once.
        audio, sr, duration = load_and_preprocess_audio(
            audio_path
        )

        # Generate timestamped transcript segments.
        transcription = self.speech_to_text.transcribe(
            audio_path
        )

        segments = transcription["segments"]

        if not segments:
            raise ValueError(
                "Speech-to-text returned no usable segments."
            )

        analyzed_segments: List[Dict[str, Any]] = []

        for index, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])
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

            # Voice emotion for this audio segment.
            voice_result = self.emotion_analyzer.analyze_segment(
                audio,
                sr,
                start,
                end,
            )

            # Text sentiment for this transcript segment.
            sentiment_result = self.sentiment_analyzer.analyze(
                text
            )

            # Multimodal fusion for this segment.
            fusion_result = self.fusion.fuse(
                voice_result,
                sentiment_result,
            )

            segment_result = {
                "segment_index": index + 1,
                "start_seconds": start,
                "end_seconds": end,
                "transcript": text,
                "voice_emotion": voice_result,
                "text_sentiment": sentiment_result,
                "fusion": fusion_result,
            }

            # Calculate intensity from actual analysis results.
            segment_result["intensity"] = (
                self._calculate_intensity(segment_result)
            )

            analyzed_segments.append(segment_result)

        if not analyzed_segments:
            raise ValueError(
                "No valid conversation segments were produced."
            )

        # Detect overall emotional trend.
        trend_result = self.detect_emotional_trend(
            analyzed_segments
        )

        # Detect escalation.
        escalation_result = self.detect_escalation(
            analyzed_segments
        )

        result = {
            "audio_path": audio_path,
            "duration_seconds": duration,
            "sample_rate": sr,
            "language": transcription["language"],
            "language_probability": transcription[
                "language_probability"
            ],
            "full_transcript": transcription["text"],
            "segments": analyzed_segments,
            "emotional_trend": trend_result,
            "escalation": escalation_result,
        }

        logger.info(
            "Conversation analysis completed: %d segments",
            len(analyzed_segments),
        )

        return result