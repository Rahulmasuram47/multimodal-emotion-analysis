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
            raise ValueError("Fusion weights cannot be negative.")

        total_weight = audio_weight + text_weight

        if total_weight <= 0:
            raise ValueError("At least one fusion weight must be positive.")

        self.audio_weight = audio_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.confidence_threshold = confidence_threshold

        logger.info(
            "Fusion configured: audio_weight=%.2f, text_weight=%.2f",
            self.audio_weight,
            self.text_weight,
        )

    @staticmethod
    def _emotion_direction(emotion: str) -> str:
        """Map voice emotion to a broad emotional direction."""

        emotion = emotion.lower()

        if emotion in {"angry", "sad"}:
            return "negative"

        if emotion == "happy":
            return "positive"

        return "neutral"

    def fuse(
        self,
        audio_result: Dict[str, Any],
        text_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Combine voice emotion and text sentiment.

        The fusion uses:
        1. Confidence-weighted modality scores.
        2. Broad emotional direction comparison.
        3. Explicit agreement/disagreement rules.

        Returns:
            Dictionary containing the multimodal interpretation.
        """

        audio_emotion = audio_result["emotion"].lower()
        audio_confidence = float(audio_result["confidence"])

        text_sentiment = text_result["sentiment"].lower()
        text_confidence = float(text_result["confidence"])

        audio_direction = self._emotion_direction(audio_emotion)
        text_direction = text_sentiment

        # Compare broad emotional direction.
        agreement = audio_direction == text_direction

        # A mismatch occurs when both modalities are sufficiently
        # confident but point in different emotional directions.
        high_audio_confidence = (
            audio_confidence >= self.confidence_threshold
        )

        high_text_confidence = (
            text_confidence >= self.confidence_threshold
        )

        mismatch = (
            not agreement
            and high_audio_confidence
            and high_text_confidence
        )

        # Confidence-weighted combined strength.
        combined_confidence = (
            self.audio_weight * audio_confidence
            + self.text_weight * text_confidence
        )

        if agreement:
            if audio_direction == "negative":
                interpretation = "Multimodal negative emotional signal."
            elif audio_direction == "positive":
                interpretation = "Multimodal positive emotional signal."
            else:
                interpretation = "Multimodal neutral emotional signal."
        elif mismatch:
            interpretation = (
                "Voice and text indicate different emotional directions."
            )
        else:
            interpretation = (
                "Weak or low-confidence modality disagreement."
            )

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
            "Fusion result: agreement=%s, mismatch=%s, "
            "combined_confidence=%.4f",
            agreement,
            mismatch,
            combined_confidence,
        )

        return result