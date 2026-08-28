"""
Multimodal Fusion Module.

Combines voice emotion and text sentiment using transparent,
configurable, rule-based logic.
"""

import logging
from typing import Any, Dict

from config import (
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


class MultimodalFusion:
    """Combine voice emotion and text sentiment."""

    def __init__(
        self,
        audio_weight: float = DEFAULT_AUDIO_WEIGHT,
        text_weight: float = DEFAULT_TEXT_WEIGHT,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ):
        if audio_weight < 0 or text_weight < 0:
            raise ValueError(
                "Fusion weights cannot be negative."
            )

        total_weight = audio_weight + text_weight

        if total_weight <= 0:
            raise ValueError(
                "At least one fusion weight must be positive."
            )

        self.audio_weight = audio_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.confidence_threshold = confidence_threshold

        logger.info(
            "Fusion configured: audio_weight=%.2f, "
            "text_weight=%.2f",
            self.audio_weight,
            self.text_weight,
        )

    # ---------------------------------------------------------
    # Emotion normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_emotion(emotion: str) -> str:
        """
        Normalize different emotion labels into standard names.

        Wav2Vec2 SUPERB may return labels such as:
            neu
            hap
            ang
            sad

        The fusion module converts them to:
            neutral
            happy
            angry
            sad
        """

        emotion = str(emotion).strip().lower()

        emotion_map = {
            "neu": "neutral",
            "neutral": "neutral",

            "hap": "happy",
            "happy": "happy",

            "ang": "angry",
            "angry": "angry",

            "sad": "sad",
        }

        return emotion_map.get(
            emotion,
            emotion,
        )

    # ---------------------------------------------------------
    # Emotional direction
    # ---------------------------------------------------------

    @classmethod
    def _emotion_direction(cls, emotion: str) -> str:
        """
        Map voice emotion to broad emotional direction.

        Returns:
            positive
            negative
            neutral
        """

        normalized_emotion = cls._normalize_emotion(
            emotion
        )

        if normalized_emotion in {
            "angry",
            "sad",
        }:
            return "negative"

        if normalized_emotion == "happy":
            return "positive"

        return "neutral"

    # ---------------------------------------------------------
    # Sentiment normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_sentiment(sentiment: str) -> str:
        """
        Normalize text sentiment labels.
        """

        sentiment = str(sentiment).strip().lower()

        sentiment_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }

        return sentiment_map.get(
            sentiment,
            sentiment,
        )

    # ---------------------------------------------------------
    # Fusion
    # ---------------------------------------------------------

    def fuse(
        self,
        audio_result: Dict[str, Any],
        text_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine voice emotion and text sentiment.

        Fusion uses:

        1. Confidence-weighted modality scores.
        2. Broad emotional direction comparison.
        3. Explicit agreement/disagreement rules.

        Returns:
            Dictionary containing the multimodal interpretation.
        """

        # -----------------------------------------------------
        # Read voice result
        # -----------------------------------------------------

        raw_audio_emotion = audio_result.get(
            "emotion",
            "neutral",
        )

        audio_emotion = self._normalize_emotion(
            raw_audio_emotion
        )

        audio_confidence = float(
            audio_result.get(
                "confidence",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Read text result
        # -----------------------------------------------------

        raw_text_sentiment = text_result.get(
            "sentiment",
            "neutral",
        )

        text_sentiment = self._normalize_sentiment(
            raw_text_sentiment
        )

        text_confidence = float(
            text_result.get(
                "confidence",
                0.0,
            )
        )

        # -----------------------------------------------------
        # Determine emotional directions
        # -----------------------------------------------------

        audio_direction = self._emotion_direction(
            audio_emotion
        )

        text_direction = text_sentiment

        # -----------------------------------------------------
        # Agreement
        # -----------------------------------------------------

        agreement = (
            audio_direction == text_direction
        )

        # -----------------------------------------------------
        # Confidence checks
        # -----------------------------------------------------

        high_audio_confidence = (
            audio_confidence
            >= self.confidence_threshold
        )

        high_text_confidence = (
            text_confidence
            >= self.confidence_threshold
        )

        # -----------------------------------------------------
        # Mismatch
        # -----------------------------------------------------

        mismatch = (
            not agreement
            and high_audio_confidence
            and high_text_confidence
        )

        # -----------------------------------------------------
        # Combined confidence
        # -----------------------------------------------------

        combined_confidence = (
            self.audio_weight * audio_confidence
            + self.text_weight * text_confidence
        )

        # -----------------------------------------------------
        # Interpretation
        # -----------------------------------------------------

        if agreement:

            if audio_direction == "negative":

                interpretation = (
                    "Multimodal negative emotional signal."
                )

            elif audio_direction == "positive":

                interpretation = (
                    "Multimodal positive emotional signal."
                )

            else:

                interpretation = (
                    "Multimodal neutral emotional signal."
                )

        elif mismatch:

            interpretation = (
                "Voice and text indicate different "
                "emotional directions."
            )

        else:

            interpretation = (
                "Weak or low-confidence modality disagreement."
            )

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        result = {
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_confidence,

            "text_sentiment": text_sentiment,
            "text_confidence": text_confidence,

            "audio_direction": audio_direction,
            "text_direction": text_direction,

            "agreement": agreement,
            "mismatch": mismatch,

            "combined_confidence": combined_confidence,

            "interpretation": interpretation,
        }

        logger.info(
            "Fusion result: "
            "emotion=%s, sentiment=%s, "
            "audio_direction=%s, text_direction=%s, "
            "agreement=%s, mismatch=%s, "
            "combined_confidence=%.4f",
            audio_emotion,
            text_sentiment,
            audio_direction,
            text_direction,
            agreement,
            mismatch,
            combined_confidence,
        )

        return result