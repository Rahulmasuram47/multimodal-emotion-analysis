"""
Formatting and statistical utility helpers for the UI components.
"""

from typing import Any, Dict, List, Tuple
import pandas as pd


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def derive_segment_summary(segments: List[Dict[str, Any]]) -> Tuple[str, str, int, int]:
    """
    Derive summary statistics strictly from actual segment outputs.
    Preserves real pipeline outputs without hallucinating or inventing data.
    """
    voice_labels = []
    text_labels = []
    agreement_count = 0
    mismatch_count = 0

    for segment in segments:
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}

        voice_label = str(voice.get("emotion", "")).strip().lower()
        text_label = str(sentiment.get("sentiment", "")).strip().lower()

        if voice_label:
            voice_labels.append(voice_label)
        if text_label:
            text_labels.append(text_label)

        if fusion.get("agreement", False):
            agreement_count += 1
        elif fusion.get("mismatch", False):
            mismatch_count += 1

    dominant_voice = (
        pd.Series(voice_labels).value_counts().idxmax()
        if voice_labels
        else "unknown"
    )

    dominant_text = (
        pd.Series(text_labels).value_counts().idxmax()
        if text_labels
        else "unknown"
    )

    return dominant_voice, dominant_text, agreement_count, mismatch_count
