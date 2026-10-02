"""
Formatting and statistical utility helpers for the UI components.
"""

import math
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to a finite float with fallback."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


# ---------------------------------------------------------------------------
# Item 9: Canonical emotion-code → display-label map (single source of truth)
# ---------------------------------------------------------------------------
_EMOTION_CODE_MAP: Dict[str, str] = {
    "neu": "Neutral",
    "neutral": "Neutral",
    "hap": "Happy",
    "happy": "Happy",
    "ang": "Angry",
    "angry": "Angry",
    "sad": "Sad",
    "positive": "Positive",
    "negative": "Negative",
    "unknown": "Unknown",
}


def emotion_code_label(code: str) -> str:
    """Map a raw model code (neu/hap/ang/sad) or full word to its display label.

    Used by UI chips, key-finding text, JSON/CSV exports — single source of truth.
    Falls back to title-cased input for codes not in the map.
    """
    key = str(code).strip().lower()
    return _EMOTION_CODE_MAP.get(key, key.title())


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


# ---------------------------------------------------------------------------
# Item 7: Single HTML-block helper that prevents Streamlit code-block bug
# ---------------------------------------------------------------------------

def html_block(raw: str) -> str:
    """Dedent and strip leading whitespace on every line so that
    Streamlit's markdown renderer never sees 4+ leading spaces on any line
    (which triggers Markdown code-block rendering instead of HTML).
    """
    dedented = textwrap.dedent(raw).strip()
    return "\n".join(line.strip() for line in dedented.splitlines() if line.strip())



# ---------------------------------------------------------------------------
# Item 11: OLS slope — SINGLE source of truth used by verdict, chart, exports
# ---------------------------------------------------------------------------

def compute_ols_slope(intensities: List[float]) -> Tuple[float, float, float]:
    """Fit a 1-degree OLS line to ``intensities`` and return (slope, intercept, net_change).

    * slope        — intensity rise per turn (same value used by verdict, chart legend, exports)
    * intercept    — y-intercept of the fitted line
    * net_change   — fitted_y[-1] - fitted_y[0]  (avoids end-point noise)

    Returns (0.0, mean, 0.0) when len(intensities) < 2 so callers can always
    unpack safely without a try/except.
    """
    n = len(intensities)
    if n < 2:
        mean = float(np.mean(intensities)) if intensities else 0.0
        return 0.0, mean, 0.0
    x = np.arange(1, n + 1, dtype=float)
    y = np.array(intensities, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    fitted_y = slope * x + intercept
    net_change = float(fitted_y[-1] - fitted_y[0])
    return float(slope), float(intercept), net_change


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


# ---------------------------------------------------------------------------
# Item 10: "predominantly" vs "mixed" key-finding helper
# ---------------------------------------------------------------------------

def predominantly_label(
    label_counts: Dict[str, int],
    threshold: Optional[float] = None,
) -> Tuple[bool, str, str]:
    """Return (is_predominantly, dominant_label, mixed_description).

    ``label_counts`` maps emotion/sentiment labels to how many turns they appear in.
    ``threshold`` defaults to config.PREDOMINANTLY_THRESHOLD (0.60).

    If one class has >= threshold fraction of turns → is_predominantly=True,
    dominant_label = that class (display form), mixed_description = "".

    Otherwise is_predominantly=False, dominant_label = the top class (display form),
    mixed_description = "X in turn N, Y in turn M" listing per-turn labels.
    """
    import config as _config

    if threshold is None:
        threshold = getattr(_config, "PREDOMINANTLY_THRESHOLD", 0.60)

    if not label_counts:
        return False, "Unknown", ""

    total = sum(label_counts.values())
    if total == 0:
        return False, "Unknown", ""

    sorted_labels = sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top_label, top_count = sorted_labels[0]
    top_fraction = top_count / total

    display_top = emotion_code_label(top_label)
    if top_fraction >= threshold:
        return True, display_top, ""

    # Build per-turn listing: "Sad in turn 1, Happy in turn 2"
    parts: List[str] = []
    turn_num = 1
    for label, count in sorted_labels:
        display = emotion_code_label(label)
        for _ in range(count):
            parts.append(f"{display} in turn {turn_num}")
            turn_num += 1
    mixed_desc = ", ".join(parts)
    return False, display_top, mixed_desc