"""
Verdict card rendering and key finding banner generation.
"""

import html
from typing import Any, Dict, List
import streamlit as st


def build_key_finding(
    dominant_voice: str,
    dominant_text: str,
    segments: List[Dict[str, Any]],
    trend_label: str,
    escalation_detected: bool,
) -> str:
    """Create a plain-English summary strictly from actual pipeline outputs."""
    voice = dominant_voice.title() if dominant_voice != "unknown" else "Unknown"
    text = dominant_text.title() if dominant_text != "unknown" else "Unknown"

    if escalation_detected:
        dynamic = "Continuous escalation was detected."
    elif trend_label.lower() == "increasing":
        dynamic = "Overall emotional intensity increased, but continuous escalation was not detected."
    elif trend_label.lower() == "decreasing":
        dynamic = "Overall emotional intensity decreased across the analyzed conversation."
    else:
        dynamic = "Overall emotional intensity remained relatively stable."

    if segments:
        first = str((segments[0].get("fusion") or {}).get("audio_direction", "")).lower()
        last = str((segments[-1].get("fusion") or {}).get("audio_direction", "")).lower()
        if first and last and first != last:
            transition = f"The dominant multimodal direction moved from {first} toward {last}."
        else:
            transition = "The multimodal emotional direction was broadly consistent across the conversation."
    else:
        transition = "No segment-level transition could be determined."

    return (
        f"Voice is predominantly {voice.lower()} while text sentiment is predominantly "
        f"{text.lower()}. {dynamic} {transition}"
    )


def render_verdict_card(finding: str) -> None:
    """Render prominent key finding banner."""
    st.markdown(
        f"""
        <div class="modality-card fusion-accent" style="margin-bottom:1rem;">
            <div class="card-meta-label">KEY FINDING</div>
            <div style="font-size:0.98rem;font-weight:600;color:var(--text);line-height:1.55;">
                {html.escape(finding)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
