"""
Acoustic Inspection page orchestrator (Tab 2).
"""

from typing import Any, Dict, List
import streamlit as st

from ui.charts.inspection import render_acoustic_inspection
from ui.utils import safe_float


def render_acoustic_page(
    audio_bytes: bytes,
    segments: List[Dict[str, Any]],
    duration: float,
) -> None:
    """Orchestrate Tab 2 Acoustic Inspection."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🎛️ Acoustic Feature Inspection</div>
            <div class="section-badge">SYNCHRONIZED MULTI-CHANNEL VISUALIZATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    turn_boundaries = [
        safe_float(s.get("start_seconds", 0.0))
        for s in segments
        if safe_float(s.get("start_seconds", 0.0)) > 0
    ]

    render_acoustic_inspection(
        audio_bytes=audio_bytes,
        turn_boundaries=turn_boundaries,
        duration=duration,
    )
