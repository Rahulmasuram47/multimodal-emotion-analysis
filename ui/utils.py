"""
Formatting and statistical utility helpers for the UI components.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to a finite float with fallback."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def emotion_icon(emotion: str) -> str:
    """Return a descriptive icon for emotion/sentiment categories."""
    key = str(emotion).lower().strip()
    icons = {
        "angry": "😠",
        "ang": "😠",
        "sad": "😟",
        "happy": "😊",
        "hap": "😊",
        "neutral": "😐",
        "neu": "😐",
        "positive": "↗",
        "negative": "↘",
        "increasing": "↗",
        "decreasing": "↘",
        "stable": "→",
    }
    return icons.get(key, "🎙️")


def direction_label(direction: str) -> str:
    """Format an emotional direction label for clean display."""
    if not direction:
        return "Unknown"
    return str(direction).replace("_", " ").title()


def _dominant_label(weighted: Dict[str, float]) -> str:
    """Label with the highest total weight (ties resolved alphabetically
    so the result is deterministic)."""
    if not weighted:
        return "unknown"
    return sorted(weighted.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def derive_segment_summary(segments: List[Dict[str, Any]]) -> Tuple[str, str, int, int]:
    """
    Derive summary statistics strictly from actual segment outputs.

    Returns (dominant_voice, dominant_text, agreement_count, mismatch_count).

    The dominant label is chosen by summed model confidence rather than a
    plain head-count. With a plain count, a 1-vs-1 tie between a
    low-confidence 'sad' (0.37) and a confident 'happy' (0.65) used to
    resolve to whichever label appeared first.

    The tuple shape is unchanged so existing callers keep working. For the
    number of weak / low-confidence segments use derive_weak_count().
    """
    voice_weight: Dict[str, float] = defaultdict(float)
    text_weight: Dict[str, float] = defaultdict(float)
    agreement_count = 0
    mismatch_count = 0

    for segment in segments:
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}

        voice_label = str(voice.get("emotion", "")).strip().lower()
        text_label = str(sentiment.get("sentiment", "")).strip().lower()

        if voice_label:
            # tiny epsilon so a 0.0-confidence label still counts as a vote
            voice_weight[voice_label] += safe_float(voice.get("confidence"), 0.0) + 1e-6
        if text_label:
            text_weight[text_label] += safe_float(sentiment.get("confidence"), 0.0) + 1e-6

        if fusion.get("agreement", False):
            agreement_count += 1
        elif fusion.get("mismatch", False):
            mismatch_count += 1

    return (
        _dominant_label(voice_weight),
        _dominant_label(text_weight),
        agreement_count,
        mismatch_count,
    )


def derive_weak_count(segments: List[Dict[str, Any]]) -> int:
    """Number of segments that neither agree nor are a confident mismatch
    (i.e. weak / low-confidence disagreement). Use this next to the
    agree/mismatch counts so '1/2 Agree · Mismatches 0' is not confusing:
    agree + mismatch + weak == total segments."""
    weak = 0
    for segment in segments:
        fusion = segment.get("fusion", {}) or {}
        if not fusion.get("agreement", False) and not fusion.get("mismatch", False):
            weak += 1
    return weak