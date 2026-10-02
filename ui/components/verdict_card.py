"""
Verdict card rendering and key finding banner generation.
"""

import html as _html
from typing import Any, Dict, List, Optional
import streamlit as st

from ui.utils import (
    compute_ols_slope,
    emotion_code_label,
    html_block,
    predominantly_label,
    safe_float,
)


def build_key_finding(
    dominant_voice: str,
    dominant_text: str,
    segments: List[Dict[str, Any]],
    trend_label: str,
    escalation_detected: bool,
) -> str:
    """Create a plain-English summary strictly from actual pipeline outputs.

    Item 10: uses "predominantly X" only when >= PREDOMINANTLY_THRESHOLD of
    turns share the same label; otherwise describes the mixed distribution.
    Item 9: maps raw codes (hap/neu) to display labels via emotion_code_label().
    """
    # Build per-turn voice and text label counts for the "predominantly" test
    voice_per_turn: Dict[str, int] = {}
    text_per_turn: Dict[str, int] = {}
    for idx, seg in enumerate(segments):
        v_raw = (seg.get("voice_emotion") or {}).get("emotion")
        if not v_raw or v_raw == "unknown":
            v_raw = dominant_voice
        t_raw = (seg.get("text_sentiment") or {}).get("sentiment")
        if not t_raw or t_raw == "unknown":
            t_raw = dominant_text
        v_label = str(v_raw).lower()
        t_label = str(t_raw).lower()
        voice_per_turn[v_label] = voice_per_turn.get(v_label, 0) + 1
        text_per_turn[t_label] = text_per_turn.get(t_label, 0) + 1


    # Voice description
    if segments:
        v_is_pred, v_top, v_mixed = predominantly_label(voice_per_turn)
        if v_is_pred:
            voice_desc = f"Voice is predominantly {v_top}"
        else:
            voice_desc = f"Voice is mixed ({v_mixed})"
    else:
        v_top = emotion_code_label(dominant_voice)
        voice_desc = f"Voice is predominantly {v_top}"

    # Text description
    if segments:
        t_is_pred, t_top, t_mixed = predominantly_label(text_per_turn)
        if t_is_pred:
            text_desc = f"text sentiment is predominantly {t_top}"
        else:
            text_desc = f"text sentiment is mixed ({t_mixed})"
    else:
        t_top = emotion_code_label(dominant_text)
        text_desc = f"text sentiment is predominantly {t_top}"

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

    return f"{voice_desc} while {text_desc}. {dynamic} {transition}"


def render_verdict_card(finding: str) -> None:
    """Render prominent key finding banner."""
    st.markdown(
        html_block(f"""
<div class="modality-card fusion-accent" style="margin-bottom:1rem;">
<div class="card-meta-label">KEY FINDING</div>
<div style="font-size:0.98rem;font-weight:600;color:var(--text);line-height:1.55;">
{_html.escape(finding)}
</div>
</div>
"""),
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

    Item 11: slope/net come from the verdict dict (computed once by
    conversation_analysis.py via compute_ols_slope). When < 3 turns,
    verdict["verdict"] == "insufficient_data" and slope/net are shown
    as n/a.

    Item 12: the Action text is derived from the SAME attention_score
    level/score that drives the gauge percentage — single source of truth.
    """
    v_dict = verdict or {}
    v_label = str(v_dict.get("verdict", "insufficient_data")).lower()
    v_reason = str(v_dict.get("reason", finding)).strip()

    # Item 11: show slope/net only when the verdict carries real data
    insufficient = v_label == "insufficient_data"
    if insufficient:
        slope_text = "n/a"
        net_text = "n/a"
    else:
        v_slope = safe_float(v_dict.get("slope", 0.0))
        v_net = safe_float(v_dict.get("net_change", 0.0))
        slope_text = f"{v_slope:+.3f}/turn"
        net_text = f"{v_net:+.2f}"

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

    # Item 12: derive action text from the same level/score (single SSOT)
    if att_level == "HIGH" or att_pct >= 66:
        action_text = "REVIEW RECOMMENDED"
        action_color = "var(--negative)"
    elif att_level == "MEDIUM" or att_pct >= 33:
        action_text = "MONITOR"
        action_color = "var(--warning)"
    else:
        action_text = "LOW PRIORITY"
        action_color = "var(--positive)"

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
            html_block(f"""
<div class="modality-card" style="border-left: 4px solid {v_color}; height: 100%;">
<div class="card-meta-label">UNIFIED CONVERSATION VERDICT</div>
<div style="display:flex; align-items:baseline; gap:0.6rem; margin-top:0.2rem; margin-bottom:0.4rem;">
<div style="font-family:'IBM Plex Mono',monospace; font-weight:700; font-size:1.35rem; color:{v_color};">
{v_icon} {_html.escape(v_title)}
</div>
<div style="font-size:0.75rem; color:var(--muted); font-family:'IBM Plex Mono',monospace;">
slope: {slope_text} · net: {net_text}
</div>
</div>
<div style="font-size:0.85rem; color:var(--text); line-height:1.55; margin-bottom:0.5rem;">
{_html.escape(v_reason)}
</div>
<div style="font-size:0.78rem; color:var(--muted); border-top:1px dashed var(--border); padding-top:0.4rem;">
<strong>Summary Finding:</strong> {_html.escape(finding)}
</div>
</div>
"""),
            unsafe_allow_html=True,
        )

    with gauge_col:
        st.markdown(
            html_block(f"""
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
<strong>Action:</strong> <span style="color:{action_color}; font-weight:600;">{_html.escape(action_text)}</span>
</div>
<div style="font-size:0.75rem; color:var(--muted-dark); line-height:1.35;">
⚠️ {_html.escape(att_notice)}
</div>
</div>
"""),
            unsafe_allow_html=True,
        )
