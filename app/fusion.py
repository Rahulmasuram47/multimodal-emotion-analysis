"""
Multimodal Fusion Module.

Combines voice emotion and text sentiment using transparent,
configurable, rule-based logic.

Features:
  - Reliability-aware weighting: if audio_result/text_result carry a
    "voice_reliability" / "text_reliability" key, the fusion weights are
    adjusted proportionally before combining confidences. Absent these
    keys, reliability defaults to 1.0 for both modalities, which
    renormalizes to the exact static weights.
  - A declarative rule table producing a descriptive (never diagnostic)
    "pattern_label" plus an evidence trail of which rule(s) fired and why.
  - Voice arousal from config.VOICE_EMOTION_AROUSAL_MAP. Text arousal is
    explicitly reported as "unknown" — this module never guesses one.
  - Abstention when combined reliability is below a threshold.
  - Counterfactual sensitivity statements.
"""

import logging
from typing import Any, Dict, List, Tuple

import config
from config import (
    DEFAULT_AUDIO_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _clamp01(value: float) -> float:
    """Clamp to [0, 1]."""
    return max(0.0, min(1.0, float(value)))


class MultimodalFusion:
    """Combine voice emotion and text sentiment."""

    def __init__(
        self,
        audio_weight: float = DEFAULT_AUDIO_WEIGHT,
        text_weight: float = DEFAULT_TEXT_WEIGHT,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        abstention_threshold: float = getattr(config, "FUSION_ABSTENTION_THRESHOLD", 0.35),
    ):
        if audio_weight < 0 or text_weight < 0:
            raise ValueError("Fusion weights cannot be negative.")

        total_weight = audio_weight + text_weight

        if total_weight <= 0:
            raise ValueError("At least one fusion weight must be positive.")

        self.audio_weight = audio_weight / total_weight
        self.text_weight = text_weight / total_weight
        self.confidence_threshold = confidence_threshold
        self.abstention_threshold = abstention_threshold

        logger.info(
            "Fusion configured: audio_weight=%.2f, text_weight=%.2f, abstention_threshold=%.2f",
            self.audio_weight,
            self.text_weight,
            self.abstention_threshold,
        )

    # ---------------------------------------------------------
    # Emotion normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_emotion(emotion: str) -> str:
        """Normalize emotion labels (neu/hap/ang/sad -> neutral/happy/angry/sad)."""
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

        return emotion_map.get(emotion, emotion)

    # ---------------------------------------------------------
    # Emotional direction
    # ---------------------------------------------------------

    @classmethod
    def _emotion_direction(cls, emotion: str) -> str:
        """Map voice emotion to broad direction: positive / negative / neutral."""
        normalized_emotion = cls._normalize_emotion(emotion)

        if normalized_emotion in {"angry", "sad"}:
            return "negative"

        if normalized_emotion == "happy":
            return "positive"

        return "neutral"

    # ---------------------------------------------------------
    # Voice arousal
    # ---------------------------------------------------------

    @staticmethod
    def _voice_arousal(normalized_emotion: str) -> float:
        """Arousal for a single normalized voice label, from
        config.VOICE_EMOTION_AROUSAL_MAP. Falls back to 0.5 for any label
        not in the map, so this never raises on an unexpected label."""
        return float(config.VOICE_EMOTION_AROUSAL_MAP.get(normalized_emotion, 0.5))

    # ---------------------------------------------------------
    # Sentiment normalization
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_sentiment(sentiment: str) -> str:
        """Normalize text sentiment labels."""
        sentiment = str(sentiment).strip().lower()

        sentiment_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
        }

        return sentiment_map.get(sentiment, sentiment)

    # ---------------------------------------------------------
    # Reliability-aware weighting
    # ---------------------------------------------------------

    def _effective_weights(
        self,
        audio_reliability: float = 1.0,
        text_reliability: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        """Compute effective (renormalized) fusion weights given each
        modality's reliability in [0, 1].

        When both reliabilities are 1.0 this returns exactly
        (self.audio_weight, self.text_weight).
        """
        audio_reliability = kwargs.get("audio_rel", audio_reliability)
        text_reliability = kwargs.get("text_rel", text_reliability)

        raw_audio = self.audio_weight * _clamp01(audio_reliability)
        raw_text = self.text_weight * _clamp01(text_reliability)
        total = raw_audio + raw_text

        if total <= 0:
            # Both modalities judged fully unreliable — fall back to the
            # static base weights rather than dividing by zero.
            return self.audio_weight, self.text_weight

        return raw_audio / total, raw_text / total

    # ---------------------------------------------------------
    # Declarative rule table
    # ---------------------------------------------------------

    def _evaluate_pattern_rules(
        self,
        audio_direction: str,
        text_direction: str,
        audio_arousal: float,
        agreement: bool,
        mismatch: bool,
    ) -> Tuple[str, str, List[str]]:
        """Evaluate the declarative pattern rules and return
        (pattern_label, explanation, rule_ids_fired).

        Every rule is DESCRIPTIVE of the acoustic/linguistic pattern
        observed, never a diagnostic or psychological claim about the
        speaker.
        """
        is_high_arousal = audio_arousal >= config.FUSION_HIGH_AROUSAL_THRESHOLD

        if agreement:
            if audio_direction == "negative" and is_high_arousal:
                return (
                    "high-arousal negative pattern",
                    "Voice and text both point negative, and the voice's acoustic "
                    "arousal is high (anger-like pattern rather than sadness-like).",
                    ["agreement_negative_high_arousal"],
                )
            if audio_direction == "negative":
                return (
                    "low-arousal negative pattern",
                    "Voice and text both point negative, with low acoustic arousal "
                    "(sadness-like pattern rather than anger-like).",
                    ["agreement_negative_low_arousal"],
                )
            if audio_direction == "positive":
                return (
                    "positive pattern",
                    "Voice and text both point positive.",
                    ["agreement_positive"],
                )
            return (
                "neutral pattern",
                "Voice and text both point neutral.",
                ["agreement_neutral"],
            )

        if mismatch:
            if text_direction == "positive" and audio_direction == "negative":
                return (
                    "incongruence: positive text, negative-toned voice",
                    "Text sentiment is positive but the acoustic tone is negative. "
                    "Voice and text point in different directions. Review the audio and context. "
                    "This is an analytical review signal, not a psychological determination.",
                    ["mismatch_positive_text_negative_voice"],
                )
            if text_direction == "negative" and audio_direction == "positive":
                return (
                    "incongruence: negative text, positive-toned voice",
                    "Text sentiment is negative but the acoustic tone is positive. "
                    "Voice and text point in different directions. Review the audio and context. "
                    "This is an analytical review signal, not a psychological determination.",
                    ["mismatch_negative_text_positive_voice"],
                )
            return (
                "incongruence: directional mismatch",
                f"Voice ({audio_direction}) and text ({text_direction}) point in "
                f"different directions with sufficient confidence in both.",
                ["mismatch_generic"],
            )

        return (
            "weak or low-confidence signal",
            "Voice and text directions differ, but at least one modality's "
            "confidence is below the threshold for a reliable comparison.",
            ["weak_disagreement"],
        )

    # ---------------------------------------------------------
    # Counterfactual sensitivity
    # ---------------------------------------------------------

    def compute_counterfactuals(
        self,
        audio_emotion: str,
        audio_confidence: float,
        text_sentiment: str,
        text_confidence: float,
        audio_arousal: float,
        combined_reliability: float,
        current_pattern: str,
    ) -> List[str]:
        """Compute counterfactual sensitivity: what perturbation would
        change this result? Tests small perturbations against the rules:
          1.  Text confidence threshold crossing.
          1b. Voice confidence threshold crossing (a confident mismatch
              needs BOTH modalities above the threshold).
          2.  Voice/text direction shift.
          3.  Reliability abstention threshold.
        """
        cfs: List[str] = []

        audio_dir = self._emotion_direction(self._normalize_emotion(audio_emotion))
        text_dir = self._normalize_sentiment(text_sentiment)
        agreement = audio_dir == text_dir
        high_audio = audio_confidence >= self.confidence_threshold

        # 1. Text confidence perturbation
        if not agreement and high_audio:
            if text_confidence < self.confidence_threshold:
                new_pat, _, _ = self._evaluate_pattern_rules(
                    audio_direction=audio_dir,
                    text_direction=text_dir,
                    audio_arousal=audio_arousal,
                    agreement=False,
                    mismatch=True,
                )
                if new_pat != current_pattern:
                    cfs.append(
                        f"If text confidence rose from {text_confidence:.2f} to {self.confidence_threshold:.2f}, "
                        f"pattern would flip from '{current_pattern}' to '{new_pat}'."
                    )
            else:
                new_pat, _, _ = self._evaluate_pattern_rules(
                    audio_direction=audio_dir,
                    text_direction=text_dir,
                    audio_arousal=audio_arousal,
                    agreement=False,
                    mismatch=False,
                )
                if new_pat != current_pattern:
                    cfs.append(
                        f"If text confidence fell below {self.confidence_threshold:.2f} (currently {text_confidence:.2f}), "
                        f"pattern would soften to '{new_pat}'."
                    )

        # 1b. Voice confidence perturbation (mirror of the text case above).
        # A confident mismatch hinges on BOTH modalities clearing the
        # threshold, so report how close the voice side is to dropping out.
        # Skipped when the system is abstaining, because the current pattern
        # is then the abstention label and a "soften" statement would be wrong.
        if (
            not agreement
            and high_audio
            and text_confidence >= self.confidence_threshold
            and combined_reliability >= self.abstention_threshold
        ):
            new_pat, _, _ = self._evaluate_pattern_rules(
                audio_direction=audio_dir,
                text_direction=text_dir,
                audio_arousal=audio_arousal,
                agreement=False,
                mismatch=False,
            )
            if new_pat != current_pattern:
                cfs.append(
                    f"If voice confidence fell below {self.confidence_threshold:.2f} "
                    f"(currently {audio_confidence:.2f}), "
                    f"pattern would soften to '{new_pat}'."
                )

        # 2. Direction shift
        if not agreement:
            new_pat, _, _ = self._evaluate_pattern_rules(
                audio_direction=text_dir,
                text_direction=text_dir,
                audio_arousal=audio_arousal,
                agreement=True,
                mismatch=False,
            )
            cfs.append(
                f"If vocal tone aligned with text sentiment ({text_dir}), "
                f"pattern would become '{new_pat}'."
            )
        else:
            opp_dir = (
                "positive" if audio_dir == "negative"
                else ("negative" if audio_dir == "positive" else "neutral")
            )
            if opp_dir != audio_dir and high_audio:
                hyp_mismatch = text_confidence >= self.confidence_threshold
                new_pat, _, _ = self._evaluate_pattern_rules(
                    audio_direction=audio_dir,
                    text_direction=opp_dir,
                    audio_arousal=audio_arousal,
                    agreement=False,
                    mismatch=hyp_mismatch,
                )
                cfs.append(
                    f"If text sentiment shifted from {text_dir} to {opp_dir}, "
                    f"pattern would become '{new_pat}'."
                )

        # 3. Abstention boundary
        if combined_reliability < self.abstention_threshold:
            cfs.append(
                f"If combined reliability rose above {self.abstention_threshold:.2f} (currently {combined_reliability:.2f}), "
                "the system would produce an active classification instead of abstaining."
            )
        else:
            margin = round(combined_reliability - self.abstention_threshold, 2)
            cfs.append(
                f"Abstention buffer: combined reliability ({combined_reliability:.2f}) is {margin:.2f} above "
                f"the abstention cut-off ({self.abstention_threshold:.2f})."
            )

        return cfs

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

        Uses reliability-weighted confidences, broad emotional direction
        comparison, and an explicit declarative rule table that also
        reports which rule fired and why.
        """

        # ---- Voice ----
        audio_emotion = self._normalize_emotion(audio_result.get("emotion", "neutral"))
        audio_confidence = float(audio_result.get("confidence", 0.0))
        audio_reliability = _clamp01(audio_result.get("voice_reliability", 1.0))
        audio_direction = self._emotion_direction(audio_emotion)
        audio_arousal = self._voice_arousal(audio_emotion)
        text_arousal = "unknown"

        # Check if text modality is unavailable
        is_text_unavailable = (
            text_result is None
            or bool(text_result.get("text_unavailable", False))
            or text_result.get("sentiment") is None
        )

        if is_text_unavailable:
            effective_audio_weight = 1.0
            effective_text_weight = 0.0
            text_sentiment = "unavailable"
            text_confidence = 0.0
            text_reliability = 0.0
            text_direction = "unavailable"
            agreement = False
            mismatch = False
            combined_confidence = audio_confidence

            pattern_label = "voice-only (text unavailable)"
            pattern_explanation = (
                "Text sentiment is unavailable (transcript was non-English, repetitive, or empty). "
                "Operating in voice-only analysis mode."
            )
            rule_ids_fired = ["voice_only_text_unavailable"]
            interpretation = "Voice-only analysis; transcript unavailable or excluded."

            combined_reliability = round(audio_reliability, 4)
            is_abstention = combined_reliability < self.abstention_threshold

            if is_abstention:
                pattern_label = "Insufficient reliability. Review audio"
                pattern_explanation = (
                    f"Combined modality reliability ({combined_reliability:.2f}) is below the "
                    f"abstention threshold ({self.abstention_threshold:.2f}). Voice or text signal "
                    "quality is insufficient for a confident classification. Manual review is recommended."
                )
                rule_ids_fired = ["abstention_low_reliability"]
                interpretation = "Insufficient reliability. Review audio"

            # Counterfactual sensitivity for voice-only mode
            counterfactuals = []
            if combined_reliability < self.abstention_threshold:
                counterfactuals.append(
                    f"If voice reliability rose above {self.abstention_threshold:.2f} (currently {combined_reliability:.2f}), "
                    "the system would produce an active classification instead of abstaining."
                )
            else:
                margin = round(combined_reliability - self.abstention_threshold, 2)
                counterfactuals.append(
                    f"Abstention buffer: voice reliability ({combined_reliability:.2f}) is {margin:.2f} above "
                    f"the abstention cut-off ({self.abstention_threshold:.2f})."
                )
            counterfactuals.append(
                "If reliable text transcript becomes available, multimodal cross-modal arbitration will engage."
            )
        else:
            # ---- Text ----
            text_sentiment = self._normalize_sentiment(text_result.get("sentiment", "neutral"))
            text_confidence = float(text_result.get("confidence", 0.0))
            text_reliability = _clamp01(text_result.get("text_reliability", 1.0))
            text_direction = text_sentiment

            # ---- Agreement / mismatch ----
            agreement = audio_direction == text_direction
            high_audio_confidence = audio_confidence >= self.confidence_threshold
            high_text_confidence = text_confidence >= self.confidence_threshold
            mismatch = not agreement and high_audio_confidence and high_text_confidence

            # ---- Effective weights and combined confidence ----
            effective_audio_weight, effective_text_weight = self._effective_weights(
                audio_reliability, text_reliability
            )

            combined_confidence = (
                effective_audio_weight * audio_confidence
                + effective_text_weight * text_confidence
            )

            # ---- Legacy interpretation (kept for backward compatibility) ----
            if agreement:
                if audio_direction == "negative":
                    interpretation = "Multimodal negative emotional signal."
                elif audio_direction == "positive":
                    interpretation = "Multimodal positive emotional signal."
                else:
                    interpretation = "Multimodal neutral emotional signal."
            elif mismatch:
                interpretation = "Voice and text indicate different emotional directions."
            else:
                interpretation = "Weak or low-confidence modality disagreement."

            # ---- Declarative pattern rules ----
            pattern_label, pattern_explanation, rule_ids_fired = self._evaluate_pattern_rules(
                audio_direction=audio_direction,
                text_direction=text_direction,
                audio_arousal=audio_arousal,
                agreement=agreement,
                mismatch=mismatch,
            )

            # ---- Combined reliability & abstention ----
            combined_reliability = round(
                effective_audio_weight * audio_reliability
                + effective_text_weight * text_reliability,
                4,
            )

            is_abstention = combined_reliability < self.abstention_threshold

            if is_abstention:
                pattern_label = "Insufficient reliability. Review audio"
                pattern_explanation = (
                    f"Combined modality reliability ({combined_reliability:.2f}) is below the "
                    f"abstention threshold ({self.abstention_threshold:.2f}). Voice or text signal "
                    "quality is insufficient for a confident classification. Manual review is recommended."
                )
                rule_ids_fired = ["abstention_low_reliability"]
                interpretation = "Insufficient reliability. Review audio"

            # ---- Counterfactuals ----
            counterfactuals = self.compute_counterfactuals(
                audio_emotion=audio_emotion,
                audio_confidence=audio_confidence,
                text_sentiment=text_sentiment,
                text_confidence=text_confidence,
                audio_arousal=audio_arousal,
                combined_reliability=combined_reliability,
                current_pattern=pattern_label,
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
            "audio_arousal": audio_arousal,
            "text_arousal": text_arousal,
            "audio_reliability": audio_reliability,
            "text_reliability": text_reliability,
            "base_weights": {
                "audio_weight": self.audio_weight,
                "text_weight": self.text_weight,
            },
            "effective_weights": {
                "audio_weight": round(effective_audio_weight, 4),
                "text_weight": round(effective_text_weight, 4),
            },
            "pattern_label": pattern_label,
            "pattern_explanation": pattern_explanation,
            "rule_ids_fired": rule_ids_fired,
            "abstention": is_abstention,
            "combined_reliability": combined_reliability,
            "counterfactuals": counterfactuals,
        }

        logger.info(
            "Fusion result: emotion=%s, sentiment=%s, audio_direction=%s, text_direction=%s, "
            "agreement=%s, mismatch=%s, combined_confidence=%.4f, pattern=%s",
            audio_emotion,
            text_sentiment,
            audio_direction,
            text_direction,
            agreement,
            mismatch,
            combined_confidence,
            pattern_label,
        )

        return result