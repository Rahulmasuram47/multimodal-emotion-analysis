"""
Modality Incongruence and Conflict Detection Module.

Identifies potential situations where acoustic emotion and text sentiment
strongly disagree (e.g. positive words spoken in an angry tone).

CRITICAL SCIENTIFIC & SAFETY TERMINOLOGY:
- Uses "Modality Incongruence" or "Potential Sarcasm / Modality Conflict".
- NEVER claims reliable sarcasm detection.
- NEVER claims lie detection, deception detection, or truth detection.
- NEVER claims psychological diagnosis or assessment of internal emotional state.
- All thresholds are heuristic and require empirical validation against labelled data.
"""

from dataclasses import dataclass, asdict
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Configurable heuristic thresholds
DEFAULT_MIN_CONFIDENCE_THRESHOLD: float = 0.40
DEFAULT_HIGH_CONFIDENCE_THRESHOLD: float = 0.65

MANDATORY_HEURISTIC_NOTICE: str = (
    "This system identifies disagreement between computational speech and text signals. "
    "It does not reliably determine sarcasm, deception, or a person's internal emotional state. "
    "Modality incongruence indicates that acoustic tone and linguistic sentiment point in different "
    "directions under computational models, which may be caused by sarcasm, passive-aggressive tone, "
    "humor, contextual nuances, acoustic noise, or model misclassification."
)


@dataclass
class IncongruenceResult:
    """Structured result of modality incongruence analysis."""

    text_label: str
    text_confidence: float
    acoustic_label: str
    acoustic_confidence: float
    conflict_detected: bool
    conflict_strength: str  # "none", "low", "medium", "high"
    conflict_type: str      # e.g., "positive_text_angry_voice", "none", etc.
    explanation: str
    heuristic_notice: str = MANDATORY_HEURISTIC_NOTICE

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_acoustic_emotion(emotion: Optional[str]) -> str:
    """Normalize acoustic emotion string to standard lowercase token."""
    if not emotion:
        return "neutral"
    norm = str(emotion).strip().lower()
    mapping = {
        "neu": "neutral",
        "neutral": "neutral",
        "hap": "happy",
        "happy": "happy",
        "ang": "angry",
        "angry": "angry",
        "sad": "sad",
    }
    return mapping.get(norm, norm)


def normalize_text_sentiment(sentiment: Optional[str]) -> str:
    """Normalize text sentiment string to standard lowercase token."""
    if not sentiment:
        return "neutral"
    norm = str(sentiment).strip().lower()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
    }
    return mapping.get(norm, norm)


def detect_modality_incongruence(
    acoustic_label: Optional[str],
    acoustic_confidence: float,
    text_label: Optional[str],
    text_confidence: float,
    min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> IncongruenceResult:
    """
    Detect whether acoustic emotion and text sentiment exhibit potential conflict.

    Deterministic logic:
    - If either acoustic or text confidence is below min_confidence_threshold:
      No high-confidence conflict can be declared (marked "none" or "low" if slight mismatch).
    - Strong opposing pairs:
        Positive text + Angry acoustic
        Positive text + Sad acoustic
        Negative text + Happy acoustic
    - When both confidences >= high_confidence_threshold: conflict_strength = "high"
    - When both confidences >= min_confidence_threshold and at least one < high_confidence_threshold:
      conflict_strength = "medium"
    - When one confidence < min_confidence_threshold: conflict_strength = "low" or "none"
    - When modalities are in the same direction (e.g., positive text + happy acoustic,
      negative text + angry/sad acoustic, neutral + neutral): conflict_detected = False, conflict_strength = "none"
    """
    ac_norm = normalize_acoustic_emotion(acoustic_label)
    tx_norm = normalize_text_sentiment(text_label)

    ac_conf = max(0.0, min(1.0, float(acoustic_confidence or 0.0)))
    tx_conf = max(0.0, min(1.0, float(text_confidence or 0.0)))

    # Define polarities
    # Acoustic: happy=positive, angry/sad=negative, neutral=neutral
    # Text: positive=positive, negative=negative, neutral=neutral
    is_opposing = False
    conflict_type = "none"

    if tx_norm == "positive" and ac_norm == "angry":
        is_opposing = True
        conflict_type = "positive_text_angry_voice"
    elif tx_norm == "positive" and ac_norm == "sad":
        is_opposing = True
        conflict_type = "positive_text_sad_voice"
    elif tx_norm == "negative" and ac_norm == "happy":
        is_opposing = True
        conflict_type = "negative_text_happy_voice"
    elif tx_norm == "neutral" and ac_norm in ("angry", "happy", "sad"):
        conflict_type = f"neutral_text_{ac_norm}_voice"
    elif ac_norm == "neutral" and tx_norm in ("positive", "negative"):
        conflict_type = f"{tx_norm}_text_neutral_voice"
    else:
        conflict_type = "none"

    # Evaluation of conflict detection and strength
    if not is_opposing:
        explanation = (
            f"No strong modality conflict detected. Linguistic sentiment is '{tx_norm}' "
            f"and acoustic emotion is '{ac_norm}', showing emotional congruence or neutral alignment."
        )
        return IncongruenceResult(
            text_label=tx_norm,
            text_confidence=round(tx_conf, 4),
            acoustic_label=ac_norm,
            acoustic_confidence=round(ac_conf, 4),
            conflict_detected=False,
            conflict_strength="none",
            conflict_type=conflict_type,
            explanation=explanation,
        )

    # Opposing directions exist: check confidences
    if ac_conf < min_confidence_threshold or tx_conf < min_confidence_threshold:
        conflict_detected = False
        conflict_strength = "low"
        explanation = (
            f"Potential directional disagreement between text ('{tx_norm}') and acoustic tone ('{ac_norm}'), "
            f"but at least one modality has low confidence (voice: {ac_conf:.1%}, text: {tx_conf:.1%}). "
            f"Incongruence cannot be reliably established."
        )
    elif ac_conf >= high_confidence_threshold and tx_conf >= high_confidence_threshold:
        conflict_detected = True
        conflict_strength = "high"
        explanation = (
            f"Strong disagreement between acoustic and textual signals. "
            f"The speaker's verbal sentiment is strongly {tx_norm} ({tx_conf:.1%}), but vocal delivery "
            f"is strongly {ac_norm} ({ac_conf:.1%}). This may indicate potential sarcasm, "
            f"passive-aggressive communication, contextual mismatch, or model uncertainty."
        )
    else:
        conflict_detected = True
        conflict_strength = "medium"
        explanation = (
            f"Moderate disagreement between acoustic and textual signals. "
            f"The verbal sentiment is {tx_norm} ({tx_conf:.1%}) while acoustic tone is {ac_norm} ({ac_conf:.1%}). "
            f"Possible sarcasm, mixed emotional expression, or classification ambiguity."
        )

    return IncongruenceResult(
        text_label=tx_norm,
        text_confidence=round(tx_conf, 4),
        acoustic_label=ac_norm,
        acoustic_confidence=round(ac_conf, 4),
        conflict_detected=conflict_detected,
        conflict_strength=conflict_strength,
        conflict_type=conflict_type,
        explanation=explanation,
    )


def analyze_conversation_incongruence(
    segments: List[Dict[str, Any]],
    min_confidence_threshold: float = DEFAULT_MIN_CONFIDENCE_THRESHOLD,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluate modality incongruence across conversation segments.

    Returns summary metrics and per-segment incongruence analyses.
    """
    if not segments:
        return {
            "status": "NO_SEGMENTS",
            "total_segments": 0,
            "conflict_segments_count": 0,
            "high_conflict_count": 0,
            "medium_conflict_count": 0,
            "has_incongruence": False,
            "overall_assessment": "No speech segments available for incongruence analysis.",
            "heuristic_notice": MANDATORY_HEURISTIC_NOTICE,
            "segment_results": [],
        }

    segment_results: List[Dict[str, Any]] = []
    conflict_count = 0
    high_count = 0
    medium_count = 0

    for seg in segments:
        v_res = seg.get("voice_emotion", {}) or {}
        t_res = seg.get("text_sentiment", {}) or {}
        v_emo = v_res.get("emotion")
        v_cnf = v_res.get("confidence", 0.0)
        t_sent = t_res.get("sentiment")
        t_cnf = t_res.get("confidence", 0.0)

        res = detect_modality_incongruence(
            acoustic_label=v_emo,
            acoustic_confidence=v_cnf,
            text_label=t_sent,
            text_confidence=t_cnf,
            min_confidence_threshold=min_confidence_threshold,
            high_confidence_threshold=high_confidence_threshold,
        )

        res_dict = res.to_dict()
        res_dict["segment_index"] = seg.get("segment_index")
        res_dict["start_seconds"] = seg.get("start_seconds")
        res_dict["end_seconds"] = seg.get("end_seconds")
        res_dict["transcript"] = seg.get("transcript", "")
        segment_results.append(res_dict)

        if res.conflict_detected:
            conflict_count += 1
            if res.conflict_strength == "high":
                high_count += 1
            elif res.conflict_strength == "medium":
                medium_count += 1

    total = len(segments)
    has_inc = conflict_count > 0

    if high_count > 0:
        overall = f"High-confidence modality incongruence identified in {high_count} of {total} segment(s)."
    elif medium_count > 0:
        overall = f"Moderate modality incongruence identified in {medium_count} of {total} segment(s)."
    else:
        overall = "No substantial modality conflict detected across conversation segments."

    return {
        "status": "OK",
        "total_segments": total,
        "conflict_segments_count": conflict_count,
        "high_conflict_count": high_count,
        "medium_conflict_count": medium_count,
        "has_incongruence": has_inc,
        "overall_assessment": overall,
        "heuristic_notice": MANDATORY_HEURISTIC_NOTICE,
        "segment_results": segment_results,
    }
