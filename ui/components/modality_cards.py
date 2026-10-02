"""
Modality cards and dual-modality channel meters rendering.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.utils import emotion_icon, safe_float


def render_channel_meters_html(
    voice_label: str,
    voice_confidence: float,
    text_label: str,
    text_confidence: float,
    fused_confidence: float,
) -> str:
    """Generate HTML for dual-modality and fused confidence meters."""
    voice_pct = max(0.0, min(1.0, safe_float(voice_confidence))) * 100
    text_pct = max(0.0, min(1.0, safe_float(text_confidence))) * 100
    fused_pct = max(0.0, min(1.0, safe_float(fused_confidence))) * 100

    return f"""
    <div style="display: flex; flex-direction: column; gap: 0.45rem; margin-top: 0.6rem; padding: 0.65rem 0.85rem; background: var(--panel-2); border-radius: 6px; border: 1px solid var(--border);">
        <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; font-family: 'IBM Plex Mono', monospace;">
                <span>🎙️ ACOUSTIC (VOICE) · {html.escape(str(voice_label).upper())}</span>
                <span style="color: var(--voice); font-weight: 700;">{voice_pct:.1f}%</span>
            </div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill voice-fill" style="width: {voice_pct:.1f}%;"></div>
            </div>
        </div>
        <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; font-family: 'IBM Plex Mono', monospace;">
                <span>💬 LINGUISTIC (TEXT) · {html.escape(str(text_label).upper())}</span>
                <span style="color: var(--text-accent); font-weight: 700;">{text_pct:.1f}%</span>
            </div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill text-fill" style="width: {text_pct:.1f}%;"></div>
            </div>
        </div>
        <div>
            <div style="display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-bottom: 0.2rem; font-family: 'IBM Plex Mono', monospace;">
                <span>⚡ MULTIMODAL FUSED CONFIDENCE</span>
                <span style="color: var(--fused); font-weight: 700;">{fused_pct:.1f}%</span>
            </div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill fused-fill" style="width: {fused_pct:.1f}%;"></div>
            </div>
        </div>
    </div>
    """


def render_modality_cards(
    dominant_voice: str,
    dominant_text: str,
    segments: List[Dict[str, Any]],
    agree_count: int,
    mismatch_count: int,
) -> None:
    """Render side-by-side Voice, Text, and Alignment snapshot cards."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🎚️ Modality Snapshot & Fusion Arbitration</div>
            <div class="section-badge">VOICE VS TEXT ALIGNMENT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snap_col1, snap_col2, snap_col3 = st.columns(3)

    with snap_col1:
        st.markdown(
            f"""
            <div class="modality-card voice-accent">
                <div class="card-meta-label">Acoustic Modality (Voice)</div>
                <div class="card-highlight-value" style="color: var(--voice);">
                    <span>{emotion_icon(dominant_voice)}</span>
                    <span>{html.escape(dominant_voice.title())}</span>
                </div>
                <div class="card-subtext">
                    Dominant acoustic emotion extracted via Wav2Vec2-SUPERB acoustic emotion classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap_col2:
        st.markdown(
            f"""
            <div class="modality-card text-accent">
                <div class="card-meta-label">Linguistic Modality (Text)</div>
                <div class="card-highlight-value" style="color: var(--text-accent);">
                    <span>{emotion_icon(dominant_text)}</span>
                    <span>{html.escape(dominant_text.title())}</span>
                </div>
                <div class="card-subtext">
                    Dominant textual sentiment classified via CardiffNLP RoBERTa transformer embeddings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap_col3:
        total_segs = max(1, len(segments))
        agree_pct = (agree_count / total_segs) * 100
        st.markdown(
            f"""
            <div class="modality-card fusion-accent">
                <div class="card-meta-label">Cross-Modal Alignment</div>
                <div class="card-highlight-value" style="color: var(--fused);">
                    <span>⚡ {agree_count}/{total_segs} Agree</span>
                </div>
                <div class="card-subtext">
                    Modality agreement: <strong style="color: var(--positive);">{agree_pct:.0f}%</strong> ·
                    Mismatches: <strong style="color: var(--voice);">{mismatch_count}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
