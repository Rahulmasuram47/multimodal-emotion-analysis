"""
Verdict card rendering and key finding banner generation.
"""

import html
from typing import Any, Dict, List, Optional
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


def render_verdict_hero(
    verdict: Optional[Dict[str, Any]],
    finding: str,
    attention_score: Optional[Dict[str, Any]] = None,
    recommendation: Optional[Dict[str, Any]] = None,
) -> None:
    """Render high-impact Verdict Hero panel per Prompt H specification:
    verdict, reason, attention gauge, recommendation.
    """
    from ui.utils import safe_float

    v_dict = verdict or {}
    v_label = str(v_dict.get("verdict", "insufficient_data")).lower()
    v_reason = str(v_dict.get("reason", finding)).strip()
    v_slope = safe_float(v_dict.get("slope", 0.0))
    v_net = safe_float(v_dict.get("net_change", 0.0))

    att_score = attention_score or {}
    att_val = safe_float(att_score.get("score", 0.0))
    att_pct = max(0.0, min(1.0, att_val)) * 100
    att_level = str(att_score.get("level", "low")).upper()
    att_notice = str(att_score.get("notice", "Analytical review priority metric only."))

    if att_level == "HIGH":
        gauge_color = "var(--negative)"
        gauge_bg = "rgba(248, 113, 113, 0.15)"
    elif att_level == "MEDIUM":
        gauge_color = "var(--warning)"
        gauge_bg = "rgba(251, 191, 36, 0.15)"
    else:
        gauge_color = "var(--positive)"
        gauge_bg = "rgba(74, 222, 128, 0.15)"

    v_icons = {
        "escalating": "📈",
        "de-escalating": "📉",
        "stable": "⚖️",
        "fluctuating": "〰️",
        "insufficient_data": "ℹ️",
    }
    v_icon = v_icons.get(v_label, "❔")
    v_title = v_label.replace("_", " ").title()

    v_color = {
        "escalating": "var(--negative)",
        "fluctuating": "var(--warning)",
        "de-escalating": "var(--positive)",
        "stable": "var(--positive)",
        "insufficient_data": "var(--muted)",
    }.get(v_label, "var(--muted)")

    hero_col, gauge_col = st.columns([2, 1])

    with hero_col:
        st.markdown(
            f"""
            <div class="modality-card" style="border-left: 4px solid {v_color}; height: 100%;">
                <div class="card-meta-label">UNIFIED CONVERSATION VERDICT</div>
                <div style="display:flex; align-items:baseline; gap:0.6rem; margin-top:0.2rem; margin-bottom:0.4rem;">
                    <div style="font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:1.35rem; color:{v_color};">
                        {v_icon} {html.escape(v_title)}
                    </div>
                    <div style="font-size:0.75rem; color:var(--muted); font-family:'IBM Plex Mono',monospace;">
                        slope: {v_slope:+.3f}/turn · net: {v_net:+.2f}
                    </div>
                </div>
                <div style="font-size:0.85rem; color:var(--text); line-height:1.55; margin-bottom:0.5rem;">
                    {html.escape(v_reason)}
                </div>
                <div style="font-size:0.78rem; color:var(--muted); border-top:1px dashed var(--border); padding-top:0.4rem;">
                    <strong>Summary Finding:</strong> {html.escape(finding)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with gauge_col:
        rec_data = recommendation or {}
        rec_title = str(rec_data.get("title", "Review Recommended")).upper()
        rec_color = (
            "var(--negative)"
            if "HIGH" in rec_title or "CRITICAL" in rec_title
            else ("var(--warning)" if "REVIEW" in rec_title else "var(--positive)")
        )
        st.markdown(
            f"""
            <div class="modality-card" style="height: 100%;">
                <div class="card-meta-label">REVIEW ATTENTION GAUGE</div>
                <div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:0.2rem; margin-bottom:0.3rem;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:1.35rem; font-weight:700; color:{gauge_color};">
                        {att_pct:.0f}%
                    </span>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; font-weight:700; background:{gauge_bg}; color:{gauge_color}; padding:2px 8px; border-radius:4px;">
                        {att_level} PRIORITY
                    </span>
                </div>
                <div style="background:rgba(255,255,255,0.08); height:8px; border-radius:4px; overflow:hidden; margin-bottom:0.5rem;">
                    <div style="background:{gauge_color}; width:{att_pct:.1f}%; height:100%; border-radius:4px;"></div>
                </div>
                <div style="font-size:0.75rem; color:var(--text); margin-bottom:0.3rem;">
                    <strong>Action:</strong> <span style="color:{rec_color}; font-weight:600;">{html.escape(rec_title)}</span>
                </div>
                <div style="font-size:0.75rem; color:var(--muted-dark); line-height:1.35;">
                    ⚠️ {html.escape(att_notice)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
