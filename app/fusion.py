"""
Multimodal Fusion Module.

Combines voice emotion and text sentiment using transparent,
configurable, rule-based logic.

Phase 5 additions (all purely additive — every original output key is
unchanged in both name and, when no reliability signal is present, value):

  - Reliability-aware weighting: if audio_result/text_result carry a
    "voice_reliability" / "text_reliability" key (from Phase 3's
    short-segment widening or low-ASR-confidence flagging), the fusion
    weights are adjusted proportionally before combining confidences.
    Absent these keys, reliability defaults to 1.0 for both modalities,
    which renormalizes to the exact original static weights — so
    combined_confidence is IDENTICAL to the pre-Phase-5 formula whenever
    no reliability signal exists (this is what keeps existing tests
    passing unchanged).
  - A declarative rule table producing a descriptive (never diagnostic)
    "pattern_label" plus an evidence trail of which rule(s) fired and why
    — the "why this result?" data the Phase 5 UI panel will display.
  - Voice arousal (from config.VOICE_EMOTION_AROUSAL_MAP, Phase 4). Text
    arousal is explicitly reported as "unknown" — RoBERTa sentiment gives
    no arousal signal, and this module never guesses one.
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
        self.abstention_threshold = abstention_threshold

        logger.info(
            "Fusion configured: audio_weight=%.2f, "
            "text_weight=%.2f, abstention_threshold=%.2f",
            self.audio_weight,
            self.text_weight,
            self.abstention_threshold,
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
    # Phase 5: Voice arousal (expected-value, from config map)
    # ---------------------------------------------------------

    @staticmethod
    def _voice_arousal(normalized_emotion: str) -> float:
        """Arousal level for a single (already-normalized) voice emotion
        label, from config.VOICE_EMOTION_AROUSAL_MAP (Phase 4). Falls back
        to 0.5 (mid-arousal) for any label not in the map, so this never
        raises on an unexpected label.
        """
        return float(config.VOICE_EMOTION_AROUSAL_MAP.get(normalized_emotion, 0.5))

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
    # Phase 5: Reliability-aware weighting
    # ---------------------------------------------------------

    def _effective_weights(
        self,
        audio_reliability: float = 1.0,
        text_reliability: float = 1.0,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        if "audio_rel" in kwargs:
            audio_reliability = kwargs["audio_rel"]
        if "text_rel" in kwargs:
            text_reliability = kwargs["text_rel"]

        """Compute effective (renormalized) fusion weights given each
        modality's reliability in [0, 1].

        When both reliabilities are 1.0 (the default when no Phase 3/
        Phase 5 reliability signal is present on either input dict), this
        returns exactly (self.audio_weight, self.text_weight) — i.e. the
        pre-Phase-5 static weights, unchanged.
        """
        raw_audio = self.audio_weight * max(0.0, min(1.0, audio_reliability))
        raw_text = self.text_weight * max(0.0, min(1.0, text_reliability))
        total = raw_audio + raw_text

        if total <= 0:
            # Both modalities judged fully unreliable — fall back to the
            # static base weights rather than dividing by zero or
            # silently zeroing out combined_confidence.
            return self.audio_weight, self.text_weight

        return raw_audio / total, raw_text / total

    # ---------------------------------------------------------
    # Phase 5: Declarative rule table
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
        speaker — wording is deliberately hedged ("-like", "may indicate")
        per the project's mandatory heuristic-only framing.
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
        """Compute counterfactual sensitivity: what parameter perturbation would change this result?

        Deterministically tests small perturbations against the rules:
          1. Text confidence threshold crossing.
          2. Voice direction shift.
          3. Reliability abstention threshold.
        """
        cfs: List[str] = []

        audio_dir = self._emotion_direction(self._normalize_emotion(audio_emotion))
        text_dir = self._normalize_sentiment(text_sentiment)
        agreement = (audio_dir == text_dir)
        high_audio = (audio_confidence >= self.confidence_threshold)

        # 1. Text confidence perturbation
        if not agreement and high_audio:
            if text_confidence < self.confidence_threshold:
                hyp_mismatch = True
                new_pat, _, _ = self._evaluate_pattern_rules(
                    audio_direction=audio_dir,
                    text_direction=text_dir,
                    audio_arousal=audio_arousal,
                    agreement=False,
                    mismatch=hyp_mismatch,
                )
                if new_pat != current_pattern:
                    cfs.append(
                        f"If text confidence rose from {text_confidence:.2f} to {self.confidence_threshold:.2f}, "
                        f"pattern would flip from '{current_pattern}' to '{new_pat}'."
                    )
            else:
                hyp_mismatch = False
                new_pat, _, _ = self._evaluate_pattern_rules(
                    audio_direction=audio_dir,
                    text_direction=text_dir,
                    audio_arousal=audio_arousal,
                    agreement=False,
                    mismatch=hyp_mismatch,
                )
                if new_pat != current_pattern:
                    cfs.append(
                        f"If text confidence fell below {self.confidence_threshold:.2f} (currently {text_confidence:.2f}), "
                        f"pattern would soften to '{new_pat}'."
                    )

        # 2. Voice tone shift
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
            opp_dir = "positive" if audio_dir == "negative" else ("negative" if audio_dir == "positive" else "neutral")
            if opp_dir != audio_dir and high_audio:
                hyp_mismatch = (text_confidence >= self.confidence_threshold)
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

        Fusion uses:

        1. Confidence-weighted modality scores (Phase 5: weighted by
           each modality's reliability, when provided — see
           _effective_weights).
        2. Broad emotional direction comparison.
        3. Explicit agreement/disagreement rules (Phase 5: expanded into
           a declarative rule table — see _evaluate_pattern_rules — that
           also reports which rule fired and why, for transparency).

        Returns:
            Dictionary containing the multimodal interpretation. Every
            key present before Phase 5 remains present with the same
            meaning; Phase 5 only adds new keys (arousal, reliability,
            effective weights, pattern_label, evidence).
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

        # Phase 5: reliability signal from Phase 3 (short-segment
        # widening / calibration). Defaults to 1.0 — i.e. fully reliable,
        # matching pre-Phase-5 behavior — when absent.
        audio_reliability = float(audio_result.get("voice_reliability", 1.0))

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

        # Phase 5: reliability signal from Phase 3 ASR confidence
        # (attached by ConversationAnalyzer as "text_reliability" when a
        # segment's speech-to-text was flagged low-confidence). Defaults
        # to 1.0 when absent, matching pre-Phase-5 behavior.
        text_reliability = float(text_result.get("text_reliability", 1.0))

        # -----------------------------------------------------
        # Determine emotional directions
        # -----------------------------------------------------

        audio_direction = self._emotion_direction(
            audio_emotion
        )

        text_direction = text_sentiment

        # Phase 5: voice arousal (expected value from the Phase 4
        # arousal map). Text arousal is explicitly "unknown" — RoBERTa
        # sentiment carries no arousal signal, and this is never guessed.
        audio_arousal = self._voice_arousal(audio_emotion)
        text_arousal = "unknown"

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
        # Phase 5: Effective (reliability-weighted) fusion weights.
        # Collapses to (self.audio_weight, self.text_weight) exactly
        # when both reliabilities are 1.0.
        # -----------------------------------------------------

        effective_audio_weight, effective_text_weight = self._effective_weights(
            audio_reliability, text_reliability
        )

        # -----------------------------------------------------
        # Combined confidence (Phase 5: now uses effective weights)
        # -----------------------------------------------------

        combined_confidence = (
            effective_audio_weight * audio_confidence
            + effective_text_weight * text_confidence
        )

        # -----------------------------------------------------
        # Interpretation (legacy — kept byte-identical for backward
        # compatibility with any code/tests reading "interpretation")
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
        # Phase 5: declarative pattern rules (richer, evidence-backed
        # version of the interpretation above)
        # -----------------------------------------------------

        pattern_label, pattern_explanation, rule_ids_fired = self._evaluate_pattern_rules(
            audio_direction=audio_direction,
            text_direction=text_direction,
            audio_arousal=audio_arousal,
            agreement=agreement,
            mismatch=mismatch,
        )

        # -----------------------------------------------------
        # Combined reliability & Abstention check (Prompt G)
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Counterfactual sensitivity (Prompt G)
        # -----------------------------------------------------
        counterfactuals = self.compute_counterfactuals(
            audio_emotion=audio_emotion,
            audio_confidence=audio_confidence,
            text_sentiment=text_sentiment,
            text_confidence=text_confidence,
            audio_arousal=audio_arousal,
            combined_reliability=combined_reliability,
            current_pattern=pattern_label,
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

            # ---- Phase 5 additions (all new keys, nothing removed) ----
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

            # ---- Prompt G additions ----
            "abstention": is_abstention,
            "combined_reliability": combined_reliability,
            "counterfactuals": counterfactuals,
        }

        logger.info(
            "Fusion result: "
            "emotion=%s, sentiment=%s, "
            "audio_direction=%s, text_direction=%s, "
            "agreement=%s, mismatch=%s, "
            "combined_confidence=%.4f, pattern=%s",
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