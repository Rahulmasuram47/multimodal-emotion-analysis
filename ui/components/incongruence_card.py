"""
Modality incongruence conflict card rendering with safety disclaimers.
"""

import html
import streamlit as st

from app.incongruence_analysis import (
    MANDATORY_HEURISTIC_NOTICE,
    detect_modality_conflict,
)


def render_incongruence_card(
    acoustic_label: str,
    acoustic_confidence: float,
    text_label: str,
    text_confidence: float,
    title: str = "MODALITY INCONGRUENCE & CONFLICT ANALYSIS",
) -> None:
    """Render a prominent conflict card with level, signals, explanation, and safety disclaimer."""
    conflict_info = detect_modality_conflict(
        acoustic_label=acoustic_label,
        acoustic_confidence=acoustic_confidence,
        text_label=text_label,
        text_confidence=text_confidence,
    )

    conflict_level = conflict_info["conflict_level"]
    badge_class = f"conflict-{conflict_level}"
    level_display = conflict_level.upper()

    signals_html = ""
    for sig in conflict_info["signals_involved"]:
        signals_html += f'<span class="conflict-signal-tag">{html.escape(sig)}</span>'

    st.markdown(
        f"""
        <div class="incongruence-card">
            <div class="incongruence-header">
                <span class="incongruence-title">{html.escape(title)}</span>
                <span class="conflict-badge {badge_class}">CONFLICT: {level_display}</span>
            </div>
            <div style="font-size: 0.78rem; color: var(--muted); line-height: 1.45; margin-bottom: 0.4rem;">
                {html.escape(conflict_info['explanation'])}
            </div>
            {f'<div class="conflict-signals-row">{signals_html}</div>' if signals_html else ''}
            <div class="incongruence-disclaimer">
                <strong>Mandatory Scientific Disclaimer:</strong> {html.escape(conflict_info['disclaimer'])}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
