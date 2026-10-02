"""
Actionable Recommendation Engine Module.

Translates complex multimodal analytics (voice emotion, text sentiment,
incongruence/conflict, escalation dynamics, consistency trajectory,
and attention score) into clear, operational decision-support recommendations.

CRITICAL SCIENTIFIC & REGULATORY BOUNDARIES:
- This is a HEURISTIC DECISION-SUPPORT TOOL ONLY.
- It is NOT an autonomous decision maker.
- It is NOT a risk assessment, psychological assessment, or clinical diagnosis.
- It does NOT detect lies, deception, truth, or a speaker's actual inner feelings.
- Human review and professional domain judgment are mandatory.
"""

from dataclasses import dataclass, asdict, field
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANDATORY_RECOMMENDATION_DISCLAIMER: str = (
    "Decision-support heuristic only. This recommendation is generated computationally "
    "from acoustic tone, text sentiment, conflict metrics, and intensity patterns. "
    "It does not constitute a psychological assessment, risk evaluation, or clinical diagnosis, "
    "and should never replace human review or professional judgment."
)

# Status Levels
STATUS_LOW_STABLE = "LOW / STABLE"
STATUS_MODERATE = "MODERATE"
STATUS_HIGH_ESCALATION = "HIGH ESCALATION"
STATUS_HIGH_INCONGRUENCE = "HIGH INCONGRUENCE"
STATUS_LOW_CONFIDENCE = "LOW CONFIDENCE"


@dataclass
class ActionableRecommendation:
    """Structured actionable recommendation output."""

    status: str
    status_level: str  # "info", "success", "warning", "critical"
    recommended_action: str
    reason: str
    contributing_factors: List[str] = field(default_factory=list)
    disclaimer: str = MANDATORY_RECOMMENDATION_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_recommendation(analysis_result: Dict[str, Any]) -> ActionableRecommendation:
    """
    Generate an actionable recommendation from conversation analysis results.

    Evaluates:
    - Overall confidence across modalities
    - Escalation status and intensity trend
    - Modality incongruence / conflict severity
    - Review-priority attention score
    - Dominant emotions and sentiments

    Rules:
    1. LOW CONFIDENCE: If confidence across segments or modalities is very low
       (< 0.40) or data is insufficient, recommend manual review due to low reliability.
    2. HIGH ESCALATION: If continuous escalation is detected or intensity is high and increasing,
       recommend active de-escalation, supervisor review, or intervention.
    3. HIGH INCONGRUENCE: If high modality conflict / sarcasm / incongruence is present,
       recommend contextual investigation of potential tone/text misalignment.
    4. MODERATE: If attention score >= 0.45 or negative dominant emotion without escalation,
       recommend standard queue review or gentle follow-up.
    5. LOW / STABLE: If emotions are positive/neutral, trend is stable/decreasing, and conflict is low,
       no immediate intervention required.
    """
    if not analysis_result:
        return ActionableRecommendation(
            status=STATUS_LOW_CONFIDENCE,
            status_level="info",
            recommended_action="Conduct manual audio review before drawing conclusions.",
            reason="No conversation analysis data was provided for recommendation synthesis.",
            contributing_factors=["Empty analysis result"],
        )

    segments = analysis_result.get("segments", [])
    if not segments:
        return ActionableRecommendation(
            status=STATUS_LOW_CONFIDENCE,
            status_level="info",
            recommended_action="Review raw audio recording manually.",
            reason="No usable conversation segments were identified in the audio.",
            contributing_factors=["Zero usable speech segments"],
        )

    # 1. Extract core metrics
    escalation_info = analysis_result.get("escalation", {}) or {}
    escalation_detected = bool(escalation_info.get("escalation_detected", False))
    # BUG FIX (Phase 4): this previously read "intensity_increase", a key
    # that detect_escalation()/detect_escalation_advanced() never produce
    # (the real key is "intensity_change") — escalation_change was always
    # silently 0.0 regardless of the actual computed rise. Both the legacy
    # strict result and the Phase 4 advanced result use "intensity_change",
    # so this single key now correctly reads either.
    escalation_change = float(escalation_info.get("intensity_change", 0.0) or 0.0)

    trend_info = analysis_result.get("emotional_trend", {}) or {}
    trend = str(trend_info.get("trend", "stable")).lower()

    # Phase 4: the unified verdict, when present, is used only to enrich
    # the reason text for the LOW/STABLE and MODERATE outcomes below — it
    # does NOT change which status is chosen. The escalation_detected flag
    # above (from config.ESCALATION_MODE) remains the sole gate for the
    # HIGH ESCALATION status, since that status has operational
    # consequences and should stay tied to the deliberately-thresholded
    # rule rather than the softer descriptive verdict.
    verdict_info = analysis_result.get("verdict", {}) or {}
    verdict_label = str(verdict_info.get("verdict", "")).lower()

    attention_info = analysis_result.get("attention_score", {}) or {}
    attention_score = float(attention_info.get("score", 0.0) or 0.0)

    incongruence_info = analysis_result.get("modality_incongruence", {}) or {}
    high_conflict_count = int(incongruence_info.get("high_conflict_count", 0) or 0)
    conflict_segments_count = int(incongruence_info.get("conflict_segments_count", 0) or 0)

    # Compute average voice and text confidences
    voice_confs = []
    text_confs = []
    fused_confs = []
    dominant_emotions = []
    dominant_sentiments = []

    for seg in segments:
        v = seg.get("voice_emotion", {}) or {}
        t = seg.get("text_sentiment", {}) or {}
        f = seg.get("fusion", {}) or {}

        if "confidence" in v and v["confidence"] is not None:
            voice_confs.append(float(v["confidence"]))
        if "confidence" in t and t["confidence"] is not None:
            text_confs.append(float(t["confidence"]))
        if "combined_confidence" in f and f["combined_confidence"] is not None:
            fused_confs.append(float(f["combined_confidence"]))

        if v.get("emotion"):
            dominant_emotions.append(str(v.get("emotion")).lower())
        if t.get("sentiment"):
            dominant_sentiments.append(str(t.get("sentiment")).lower())

    avg_voice_conf = sum(voice_confs) / len(voice_confs) if voice_confs else 0.0
    avg_text_conf = sum(text_confs) / len(text_confs) if text_confs else 0.0
    avg_fused_conf = sum(fused_confs) / len(fused_confs) if fused_confs else 0.0

    contributing_factors = []

    # 2. Rule evaluation

    # Condition A: Low Confidence check
    if avg_fused_conf < 0.40 or avg_voice_conf < 0.35:
        contributing_factors.append(f"Low overall multimodal confidence (fused: {avg_fused_conf:.2f}, voice: {avg_voice_conf:.2f})")
        return ActionableRecommendation(
            status=STATUS_LOW_CONFIDENCE,
            status_level="warning",
            recommended_action="Conduct manual audio review before drawing conclusions.",
            reason="Acoustic or linguistic model confidences fall below operational reliability thresholds (< 0.40). Audio quality, background noise, or vocabulary ambiguity may be affecting prediction accuracy.",
            contributing_factors=contributing_factors,
        )

    # Condition B: High Escalation check
    if escalation_detected or (trend == "increasing" and attention_score >= 0.60):
        if escalation_detected:
            contributing_factors.append(f"Continuous emotional escalation detected (+{escalation_change:.2f} intensity delta)")
        if trend == "increasing":
            contributing_factors.append("Overall conversation intensity trend is upward")
        if attention_score >= 0.60:
            contributing_factors.append(f"Elevated attention score ({attention_score:.2f})")

        return ActionableRecommendation(
            status=STATUS_HIGH_ESCALATION,
            status_level="critical",
            recommended_action="Initiate immediate de-escalation protocol or supervisor review.",
            reason="Emotional intensity shows continuous growth across successive turns, indicating rising conversation tension or emotional escalation.",
            contributing_factors=contributing_factors,
        )

    # Condition C: High Modality Incongruence check
    if high_conflict_count >= 1 or (conflict_segments_count >= 2 and attention_score >= 0.45):
        if high_conflict_count >= 1:
            contributing_factors.append(f"{high_conflict_count} segment(s) with high-confidence tone/sentiment conflict")
        if conflict_segments_count >= 2:
            contributing_factors.append(f"{conflict_segments_count} total incongruent segments detected")

        return ActionableRecommendation(
            status=STATUS_HIGH_INCONGRUENCE,
            status_level="warning",
            recommended_action="Review conversation context for potential sarcasm, passive tension, or transcription mismatch.",
            reason="Strong contradiction detected between spoken acoustic tone (e.g. angry or agitated voice) and literal linguistic sentiment (e.g. positive text), suggesting potential nuance or masking.",
            contributing_factors=contributing_factors,
        )

    # Condition D: Moderate Attention / Negative Sentiment
    neg_emo_count = sum(1 for e in dominant_emotions if e in ("angry", "sad"))
    neg_sent_count = sum(1 for s in dominant_sentiments if s == "negative")

    if attention_score >= 0.45 or neg_emo_count > len(segments) / 2 or neg_sent_count > len(segments) / 2:
        if attention_score >= 0.45:
            contributing_factors.append(f"Moderate attention score ({attention_score:.2f})")
        if neg_emo_count > len(segments) / 2:
            contributing_factors.append("Prevalence of negative voice emotions (angry/sad)")
        if neg_sent_count > len(segments) / 2:
            contributing_factors.append("Prevalence of negative linguistic sentiment")
        if verdict_label == "fluctuating":
            contributing_factors.append("Emotional intensity fluctuated across the conversation without a clear escalating trend")

        return ActionableRecommendation(
            status=STATUS_MODERATE,
            status_level="warning",
            recommended_action="Flag for standard quality assurance or follow-up review.",
            reason="Conversation exhibits notable negative valence or elevated attention heuristic, but without continuous escalation dynamics.",
            contributing_factors=contributing_factors,
        )

    # Condition E: Low / Stable
    contributing_factors.append(f"Stable emotional intensity trend ({trend})")
    contributing_factors.append(f"Low review-priority attention score ({attention_score:.2f})")
    contributing_factors.append(f"Solid multimodal confidence ({avg_fused_conf:.2f})")
    if verdict_label == "de-escalating":
        contributing_factors.append("Conversation intensity was actively de-escalating")

    return ActionableRecommendation(
        status=STATUS_LOW_STABLE,
        status_level="success",
        recommended_action="No immediate intervention required; standard operational monitoring.",
        reason="Conversation exhibits balanced emotional dynamics, negligible modality conflict, and stable intensity throughout.",
        contributing_factors=contributing_factors,
    )