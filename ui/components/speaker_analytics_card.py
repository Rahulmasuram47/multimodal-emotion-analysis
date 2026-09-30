"""
Speaker-aware analytics card (Phase 6) — per-speaker trend/verdict,
talk-time share, and the heuristic "possible escalation initiator"
signal. Only rendered when 2+ distinct speakers are identified.
"""

import html
from typing import Any, Dict, List, Optional

import streamlit as st

from ui.utils import emotion_icon, safe_float

_VERDICT_ICONS: Dict[str, str] = {
    "escalating": "📈",
    "de-escalating": "📉",
    "fluctuating": "🔀",
    "stable": "➖",
    "insufficient_data": "❔",
}


def render_speaker_analytics_card(
    speaker_analytics: List[Dict[str, Any]],
    initiator: Optional[Dict[str, Any]],
) -> None:
    """Render per-speaker breakdown cards plus the escalation-initiator
    note, when applicable. Caller should only invoke this when
    speaker_analytics is non-empty (2+ identified speakers) — an empty
    list renders nothing.
    """
    if not speaker_analytics:
        return

    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🗣️ Speaker-Aware Analytics</div>
            <div class="section-badge">PER-SPEAKER TRENDS & TALK-TIME</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(speaker_analytics))
    for col, sa in zip(cols, speaker_analytics):
        speaker = str(sa.get("speaker", "Unknown"))
        talk_pct = safe_float(sa.get("talk_time_pct", 0.0))
        talk_sec = safe_float(sa.get("talk_time_seconds", 0.0))
        avg_intensity = safe_float(sa.get("average_intensity", 0.0))
        verdict = sa.get("verdict", {}) or {}
        verdict_label = str(verdict.get("verdict", "insufficient_data")).lower()
        verdict_reason = str(verdict.get("reason", "")).strip()
        icon = _VERDICT_ICONS.get(verdict_label, "❔")
        display_label = verdict_label.replace("_", " ").title()

        with col:
            st.markdown(
                f"""
                <div class="modality-card">
                    <div class="card-meta-label">{html.escape(speaker)}</div>
                    <div class="card-highlight-value" style="color: var(--fused);">
                        <span>{icon}</span>
                        <span>{html.escape(display_label)}</span>
                    </div>
                    <div class="card-subtext">
                        Talk time: <strong>{talk_sec:.1f}s ({talk_pct:.0f}%)</strong><br>
                        Avg. intensity: <strong>{avg_intensity:.0%}</strong><br>
                        {html.escape(verdict_reason)}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if initiator:
        speaker = str(initiator.get("speaker", "Unknown"))
        rise = safe_float(initiator.get("intensity_rise", 0.0))
        transcript = str(initiator.get("transcript", "")).strip()
        st.markdown(
            f"""
            <div style="background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--negative); border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.6rem; font-size: 0.8rem; color: var(--text);">
                <strong style="font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; color: var(--negative);">📈 Possible Escalation Point:</strong>
                The largest single-step intensity rise (+{rise:.2f}) occurs at a turn by <strong>{html.escape(speaker)}</strong>: "{html.escape(transcript[:120])}{'...' if len(transcript) > 120 else ''}"
                <div style="margin-top: 0.4rem; font-size: 0.72rem; color: var(--muted);">
                    This is a descriptive pattern only — a rise in intensity may be a reaction to the prior turn, not necessarily its cause. It is not an attribution of fault or intent.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )