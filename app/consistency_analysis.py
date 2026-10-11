"""
Modality Consistency Trajectory and Explainable Attention Score.

This module analyses the relationship between voice emotion and text sentiment
across conversation segments using data that has already been computed by the
multimodal fusion pipeline.  It adds two complementary layers of post-hoc
analysis on top of the core pipeline output:

1. MODALITY CONSISTENCY TRAJECTORY
   Classifies whether voice and text modalities tell a consistent or divergent
   emotional story across the conversation.

   Segment-level classifications:
       agreement        – both modalities point in the same emotional direction.
       divergence       – modalities disagree with high confidence in both channels.
       low_confidence   – at least one modality has insufficient confidence;
                          relationship is inconclusive.

   Conversation-level trajectory:
       mostly_consistent  – agreement_rate >= 0.70
       mostly_divergent   – agreement_rate <  0.30
       mixed              – otherwise
       insufficient_data  – fewer than MIN_TRAJECTORY_SEGMENTS usable segments

   IMPORTANT WORDING RULES:
   - This analysis NEVER implies that either modality represents a person's
     true or actual internal emotional state.
   - It is NOT lie detection, truth detection, or concealment detection.
   - The terms "agreement" and "divergence" refer only to whether the two
     computational signals point in the same direction — nothing more.

2. REVIEW-PRIORITY ATTENTION SCORE
   A single transparent heuristic number in [0.0, 1.0] indicating how much
   computational attention a conversation warrants for manual review.

   Formula (documented explicitly):
       attention_score = W_MISMATCH   × mismatch_rate
                       + W_INTENSITY  × average_intensity
                       + W_ESCALATION × escalation_indicator

   Default weights (heuristic; NOT scientifically validated):
       W_MISMATCH   = 0.40
       W_INTENSITY  = 0.35
       W_ESCALATION = 0.25

   The escalation_indicator is 1.0 if escalation was detected, else 0.0.

   This score is a COMPUTATIONAL REVIEW-PRIORITY HEURISTIC ONLY.
   It is NOT:
       - a risk score
       - a danger score
       - a distress indicator
       - a psychological assessment
       - a clinical measure

   Users must exercise professional judgement when interpreting this number.
   The weights are configurable.  Any changes from the defaults are the
   responsibility of the user and should be documented.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Trajectory thresholds (transparent and documented)
# ---------------------------------------------------------

#: Minimum segments required before a trajectory can be reported.
MIN_TRAJECTORY_SEGMENTS: int = 2

#: agreement_rate threshold above which trajectory is "mostly_consistent".
MOSTLY_CONSISTENT_THRESHOLD: float = 0.70

#: agreement_rate threshold below which trajectory is "mostly_divergent".
MOSTLY_DIVERGENT_THRESHOLD: float = 0.30

# ---------------------------------------------------------
# Attention-score weights (heuristic defaults — NOT scientifically validated)
# ---------------------------------------------------------

#: Weight applied to the mismatch rate component.
DEFAULT_W_MISMATCH: float = 0.40

#: Weight applied to the average intensity component.
DEFAULT_W_INTENSITY: float = 0.35

#: Weight applied to the escalation indicator component.
DEFAULT_W_ESCALATION: float = 0.25


# ---------------------------------------------------------
# Segment consistency classification
# ---------------------------------------------------------

SEGMENT_RELATION_AGREEMENT = "agreement"
SEGMENT_RELATION_DIVERGENCE = "divergence"
SEGMENT_RELATION_LOW_CONFIDENCE = "low_confidence"


def classify_segment_relationship(segment: Dict[str, Any]) -> str:
    """
    Classify the voice–text relationship for one already-computed segment.

    Reads from segment["fusion"]:
        agreement  – produced by MultimodalFusion.fuse()
        mismatch   – produced by MultimodalFusion.fuse()

    Returns one of:
        "agreement"       – both modalities point in the same direction.
        "divergence"      – high-confidence disagreement between modalities.
        "low_confidence"  – disagreement exists but confidence is insufficient
                            to classify as strong divergence.
    """
    fusion: Dict[str, Any] = segment.get("fusion", {})
    is_agreement: bool = bool(fusion.get("agreement", False))
    is_mismatch: bool = bool(fusion.get("mismatch", False))

    if is_agreement:
        return SEGMENT_RELATION_AGREEMENT
    elif is_mismatch:
        return SEGMENT_RELATION_DIVERGENCE
    else:
        return SEGMENT_RELATION_LOW_CONFIDENCE


# ---------------------------------------------------------
# Consistency trajectory
# ---------------------------------------------------------

@dataclass
class ConsistencyTrajectory:
    """Modality consistency trajectory across the full conversation."""

    trajectory: str
    """
    One of: "mostly_consistent", "mostly_divergent", "mixed", "insufficient_data".
    """

    total_segments: int
    agreement_count: int
    divergence_count: int
    low_confidence_count: int

    agreement_rate: float
    """Fraction of segments classified as agreement (0.0 – 1.0)."""

    mismatch_rate: float
    """Fraction of segments classified as divergence (0.0 – 1.0)."""

    per_segment_relationships: List[Dict[str, Any]] = field(default_factory=list)
    """Per-segment relationship classification and direction details."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory": self.trajectory,
            "total_segments": self.total_segments,
            "agreement_count": self.agreement_count,
            "divergence_count": self.divergence_count,
            "low_confidence_count": self.low_confidence_count,
            "agreement_rate": self.agreement_rate,
            "mismatch_rate": self.mismatch_rate,
            "per_segment_relationships": self.per_segment_relationships,
        }


def compute_consistency_trajectory(
    segments: List[Dict[str, Any]],
) -> ConsistencyTrajectory:
    """
    Compute the modality consistency trajectory across analysed segments.

    Terminology:
        - "agreement" means both voice and text signals point in the same
          emotional direction for a segment.
        - "divergence" means they point in different directions with
          sufficient confidence in both.
        - "low_confidence" means there is a directional difference but at
          least one modality has insufficient confidence to be conclusive.

    This analysis makes NO claim about which modality (if either) reflects
    a person's actual internal state.

    Args:
        segments: List of segment result dicts produced by ConversationAnalyzer.

    Returns:
        ConsistencyTrajectory dataclass.
    """
    if len(segments) < MIN_TRAJECTORY_SEGMENTS:
        return ConsistencyTrajectory(
            trajectory="insufficient_data",
            total_segments=len(segments),
            agreement_count=0,
            divergence_count=0,
            low_confidence_count=0,
            agreement_rate=0.0,
            mismatch_rate=0.0,
            per_segment_relationships=[],
        )

    per_seg: List[Dict[str, Any]] = []
    agreement_count = 0
    divergence_count = 0
    low_conf_count = 0

    for seg in segments:
        relation = classify_segment_relationship(seg)
        fusion = seg.get("fusion", {})

        per_seg.append({
            "segment_index": seg.get("segment_index"),
            "start_seconds": seg.get("start_seconds"),
            "end_seconds": seg.get("end_seconds"),
            "voice_direction": fusion.get("audio_direction"),
            "text_direction": fusion.get("text_direction"),
            "relationship": relation,
        })

        if relation == SEGMENT_RELATION_AGREEMENT:
            agreement_count += 1
        elif relation == SEGMENT_RELATION_DIVERGENCE:
            divergence_count += 1
        else:
            low_conf_count += 1

    total = len(segments)
    agreement_rate = round(float(agreement_count) / float(total), 4)
    mismatch_rate = round(float(divergence_count) / float(total), 4)

    if agreement_rate >= MOSTLY_CONSISTENT_THRESHOLD:
        trajectory = "mostly_consistent"
    elif agreement_rate < MOSTLY_DIVERGENT_THRESHOLD:
        trajectory = "mostly_divergent"
    else:
        trajectory = "mixed"

    logger.info(
        "Consistency trajectory: %s (agreement_rate=%.3f, mismatch_rate=%.3f, n=%d)",
        trajectory,
        agreement_rate,
        mismatch_rate,
        total,
    )

    return ConsistencyTrajectory(
        trajectory=trajectory,
        total_segments=total,
        agreement_count=agreement_count,
        divergence_count=divergence_count,
        low_confidence_count=low_conf_count,
        agreement_rate=agreement_rate,
        mismatch_rate=mismatch_rate,
        per_segment_relationships=per_seg,
    )


# ---------------------------------------------------------
# Review-priority attention score
# ---------------------------------------------------------

ATTENTION_SCORE_FORMULA: str = (
    "attention_score = W_MISMATCH × mismatch_rate"
    " + W_INTENSITY × average_intensity"
    " + W_ESCALATION × escalation_indicator"
)


@dataclass
class AttentionScore:
    """
    Explainable review-priority attention score for a conversation.

    HEURISTIC ONLY — not a risk, danger, distress, or clinical assessment.
    All weights are defaults; users may configure them as required.
    """

    attention_score: float
    """Final score in [0.0, 1.0]."""

    mismatch_contribution: float
    """W_MISMATCH × mismatch_rate component."""

    intensity_contribution: float
    """W_INTENSITY × average_intensity component."""

    escalation_contribution: float
    """W_ESCALATION × escalation_indicator component."""

    mismatch_rate: float
    average_intensity: float
    escalation_detected: bool

    w_mismatch: float
    w_intensity: float
    w_escalation: float

    formula: str = ATTENTION_SCORE_FORMULA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_score": self.attention_score,
            "mismatch_contribution": self.mismatch_contribution,
            "intensity_contribution": self.intensity_contribution,
            "escalation_contribution": self.escalation_contribution,
            "mismatch_rate": self.mismatch_rate,
            "average_intensity": self.average_intensity,
            "escalation_detected": self.escalation_detected,
            "weights": {
                "w_mismatch": self.w_mismatch,
                "w_intensity": self.w_intensity,
                "w_escalation": self.w_escalation,
            },
            "formula": self.formula,
            "notice": (
                "This is a computational review-priority heuristic only. "
                "It is NOT a risk score, danger score, distress indicator, "
                "psychological assessment, or clinical measure. "
                "Weights are heuristic defaults and are not scientifically validated."
            ),
        }


def compute_attention_score(
    segments: List[Dict[str, Any]],
    escalation_result: Optional[Dict[str, Any]] = None,
    trajectory: Optional[ConsistencyTrajectory] = None,
    w_mismatch: float = DEFAULT_W_MISMATCH,
    w_intensity: float = DEFAULT_W_INTENSITY,
    w_escalation: float = DEFAULT_W_ESCALATION,
) -> AttentionScore:
    """
    Compute a transparent review-priority attention score.

    Formula (explicitly documented):
        attention_score = w_mismatch   × mismatch_rate
                        + w_intensity  × average_intensity
                        + w_escalation × escalation_indicator

    Where:
        mismatch_rate        – fraction of segments where voice and text diverge.
        average_intensity    – mean emotional intensity across all segments.
        escalation_indicator – 1.0 if escalation was detected, else 0.0.

    Default weights (heuristic; NOT scientifically validated):
        w_mismatch   = 0.40
        w_intensity  = 0.35
        w_escalation = 0.25

    This score is a COMPUTATIONAL HEURISTIC ONLY.
    It is NOT a risk, danger, distress, or psychological measure.

    Args:
        segments:          Segment result dicts from ConversationAnalyzer.
        escalation_result: Escalation dict from ConversationAnalyzer.detect_escalation().
        trajectory:        Pre-computed ConsistencyTrajectory (optional; avoids recomputing).
        w_mismatch:        Weight for mismatch_rate component.
        w_intensity:       Weight for average_intensity component.
        w_escalation:      Weight for escalation_indicator component.

    Returns:
        AttentionScore dataclass with full decomposition.
    """
    # Mismatch rate — use pre-computed trajectory if available
    if trajectory is not None:
        mismatch_rate = trajectory.mismatch_rate
    elif len(segments) == 0:
        mismatch_rate = 0.0
    else:
        mismatch_count = sum(
            1 for s in segments
            if classify_segment_relationship(s) == SEGMENT_RELATION_DIVERGENCE
        )
        mismatch_rate = round(float(mismatch_count) / float(max(1, len(segments))), 4)

    # Average intensity
    if segments:
        intensities = [float(s.get("intensity", 0.0)) for s in segments]
        average_intensity = round(sum(intensities) / len(intensities), 4)
    else:
        average_intensity = 0.0

    # Escalation indicator
    if escalation_result is not None:
        escalation_detected = bool(escalation_result.get("escalation_detected", False))
    else:
        escalation_detected = False

    escalation_indicator = 1.0 if escalation_detected else 0.0

    # Component contributions (each is non-negative)
    mismatch_contribution = round(w_mismatch * mismatch_rate, 6)
    intensity_contribution = round(w_intensity * average_intensity, 6)
    escalation_contribution = round(w_escalation * escalation_indicator, 6)

    raw_score = mismatch_contribution + intensity_contribution + escalation_contribution
    attention_score = round(min(1.0, max(0.0, raw_score)), 4)

    logger.info(
        "Attention score: %.4f "
        "(mismatch_contribution=%.4f, intensity_contribution=%.4f, escalation_contribution=%.4f)",
        attention_score,
        mismatch_contribution,
        intensity_contribution,
        escalation_contribution,
    )

    return AttentionScore(
        attention_score=attention_score,
        mismatch_contribution=mismatch_contribution,
        intensity_contribution=intensity_contribution,
        escalation_contribution=escalation_contribution,
        mismatch_rate=mismatch_rate,
        average_intensity=average_intensity,
        escalation_detected=escalation_detected,
        w_mismatch=w_mismatch,
        w_intensity=w_intensity,
        w_escalation=w_escalation,
    )
