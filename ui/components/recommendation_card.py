"""
Actionable decision-support recommendation card rendering.
"""

import html
from typing import Any, Dict
import streamlit as st

from app.recommendation_engine import MANDATORY_RECOMMENDATION_DISCLAIMER


def render_recommendation_card(rec_data: Dict[str, Any]) -> None:
    """Render operational recommendation box with status badge and contributing signals."""
    if not rec_data:
        return

    rec_status = rec_data.get("status", "LOW / STABLE")
    rec_level = rec_data.get("status_level", "info")
    rec_action = rec_data.get("recommended_action", "")
    rec_reason = rec_data.get("reason", "")
    rec_factors = rec_data.get("contributing_factors", []) or []
    rec_disclaimer = rec_data.get("disclaimer", MANDATORY_RECOMMENDATION_DISCLAIMER)

    if rec_level == "critical":
        border_color = "var(--negative)"
        badge_bg = "var(--negative-bg)"
        status_icon = "🚨"
    elif rec_level == "warning":
        border_color = "var(--voice)"
        badge_bg = "var(--voice-glow)"
        status_icon = "⚠️"
    elif rec_level == "success":
        border_color = "var(--positive)"
        badge_bg = "var(--positive-bg)"
        status_icon = "✓"
    else:
        border_color = "var(--text-accent)"
        badge_bg = "var(--text-glow)"
        status_icon = "ℹ️"

    factors_html = "".join([f"<li>{html.escape(f)}</li>" for f in rec_factors])
    st.markdown(
        f"""
        <div class="modality-card" style="border-left: 4px solid {border_color}; margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                <div class="card-meta-label" style="color: {border_color}; font-weight: 700;">
                    🎯 OPERATIONAL RECOMMENDATION (DECISION SUPPORT)
                </div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: var(--text); background: {badge_bg}; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid {border_color};">
                    {status_icon} {html.escape(rec_status)}
                </div>
            </div>
            <div style="font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                {html.escape(rec_action)}
            </div>
            <div style="font-size: 0.8rem; color: var(--muted); line-height: 1.5; margin-bottom: 0.4rem;">
                <strong>Rationale:</strong> {html.escape(rec_reason)}
            </div>
            {f'<div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 0.4rem;"><strong>Contributing Signals:</strong><ul style="margin: 0.2rem 0 0 1.2rem; padding: 0;">{factors_html}</ul></div>' if factors_html else ''}
            <div style="font-size: 0.68rem; color: var(--muted-dark); border-top: 1px solid var(--border); padding-top: 0.4rem; line-height: 1.4;">
                <strong>Mandatory Heuristic Notice:</strong> {html.escape(rec_disclaimer)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
