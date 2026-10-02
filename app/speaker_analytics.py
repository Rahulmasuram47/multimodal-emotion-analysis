"""
Speaker-Aware Analytics Module (Phase 6).

Builds per-speaker breakdowns from an already-analyzed conversation:
  - talk-time share per speaker
  - per-speaker emotional trend and unified verdict (reusing the exact
    same ConversationAnalyzer logic used for the whole conversation,
    applied to just that speaker's segments in original order)
  - a heuristic "possible escalation initiator" — the speaker of the
    segment with the largest single-step intensity rise, reported only
    as a descriptive pattern, never as attribution of fault or intent

Only meaningful with 2+ distinct identified speakers (see
config.SPEAKER_ANALYTICS_MIN_SPEAKERS). With fewer, callers should skip
this entirely and rely on the existing conversation-level analytics.

All functions here are pure and stateless — they take already-computed
segment dicts (with "speaker", "intensity", etc. already attached) and
do not touch models, audio, or the filesystem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import config
from app.conversation_analysis import ConversationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SpeakerAnalytics:
    """Per-speaker analytics for one identified speaker."""
    speaker: str
    segment_count: int
    talk_time_seconds: float
    talk_time_pct: float
    average_intensity: float
    trend: Dict[str, Any] = field(default_factory=dict)
    verdict: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "segment_count": self.segment_count,
            "talk_time_seconds": self.talk_time_seconds,
            "talk_time_pct": self.talk_time_pct,
            "average_intensity": self.average_intensity,
            "trend": self.trend,
            "verdict": self.verdict,
        }


def group_segments_by_speaker(
    segments: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group segments by their assigned speaker label, preserving
    original conversation order within each speaker's list. Segments
    labeled "Unknown" or with no speaker are grouped under "Unknown" and
    should typically be excluded from per-speaker analytics by the
    caller (get_speaker_labels_with_enough_data handles this).
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for seg in segments:
        speaker = str(seg.get("speaker", "") or "Unknown").strip() or "Unknown"
        groups.setdefault(speaker, []).append(seg)
    return groups


def get_identified_speakers(segments: List[Dict[str, Any]]) -> List[str]:
    """Distinct speaker labels, excluding "Unknown", in first-appearance order."""
    seen: List[str] = []
    for seg in segments:
        speaker = str(seg.get("speaker", "") or "").strip()
        if speaker and speaker != "Unknown" and speaker not in seen:
            seen.append(speaker)
    return seen


def compute_speaker_analytics(
    segments: List[Dict[str, Any]],
    total_duration_seconds: float,
) -> List[SpeakerAnalytics]:
    """Compute per-speaker analytics for every identified (non-Unknown)
    speaker. Returns an empty list if fewer than
    config.SPEAKER_ANALYTICS_MIN_SPEAKERS distinct speakers are present.
    """
    identified = get_identified_speakers(segments)
    if len(identified) < config.SPEAKER_ANALYTICS_MIN_SPEAKERS:
        return []

    groups = group_segments_by_speaker(segments)
    results: List[SpeakerAnalytics] = []

    for speaker in identified:
        speaker_segments = groups.get(speaker, [])
        if not speaker_segments:
            continue

        talk_time = sum(
            max(0.0, float(s.get("end_seconds", 0.0)) - float(s.get("start_seconds", 0.0)))
            for s in speaker_segments
        )
        talk_time_pct = (
            round(100.0 * talk_time / total_duration_seconds, 1)
            if total_duration_seconds > 0
            else 0.0
        )

        intensities = [float(s.get("intensity", 0.0)) for s in speaker_segments]
        avg_intensity = round(sum(intensities) / len(intensities), 4) if intensities else 0.0

        # Reuse the exact same trend/verdict logic used conversation-wide,
        # applied to just this speaker's segments in their original order.
        trend = ConversationAnalyzer.detect_emotional_trend(speaker_segments)
        verdict = ConversationAnalyzer.compute_unified_verdict(speaker_segments)

        results.append(
            SpeakerAnalytics(
                speaker=speaker,
                segment_count=len(speaker_segments),
                talk_time_seconds=round(talk_time, 2),
                talk_time_pct=talk_time_pct,
                average_intensity=avg_intensity,
                trend=trend,
                verdict=verdict,
            )
        )

    return results


def find_possible_escalation_initiator(
    segments: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Heuristic only: identify the speaker of the segment with the
    largest single-step intensity rise (compared to the immediately
    preceding segment in the whole conversation, regardless of speaker).

    This is a DESCRIPTIVE pattern-level signal, not an attribution of
    fault, intent, or responsibility — a rise can be a reaction to what
    the other speaker just said, not necessarily its cause. Returns None
    if fewer than 2 segments exist, or if no rise exceeds
    config.SPEAKER_ESCALATION_INITIATOR_MIN_RISE.
    """
    if len(segments) < 2:
        return None

    identified = get_identified_speakers(segments)
    if len(identified) < config.SPEAKER_ANALYTICS_MIN_SPEAKERS:
        return None

    best_rise = 0.0
    best_index: Optional[int] = None

    for i in range(1, len(segments)):
        prev_intensity = float(segments[i - 1].get("intensity", 0.0))
        curr_intensity = float(segments[i].get("intensity", 0.0))
        rise = curr_intensity - prev_intensity
        if rise > best_rise:
            best_rise = rise
            best_index = i

    if best_index is None or best_rise < config.SPEAKER_ESCALATION_INITIATOR_MIN_RISE:
        return None

    seg = segments[best_index]
    return {
        "speaker": str(seg.get("speaker", "Unknown")),
        "segment_index": seg.get("segment_index"),
        "start_seconds": seg.get("start_seconds"),
        "intensity_rise": round(best_rise, 4),
        "transcript": str(seg.get("transcript", "")).strip(),
    }