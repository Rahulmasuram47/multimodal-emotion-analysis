"""
Modality cards and dual-modality channel meters rendering.
"""

import html
from typing import Any, Dict, List, Optional
import streamlit as st

from ui.utils import emotion_code_label, emotion_icon, safe_float


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
                <span title="Reliability-weighted average of acoustic and text confidences">⚡ MULTIMODAL EVIDENCE STRENGTH</span>
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
    voice_reliability: Optional[float] = None,
    text_reliability: Optional[float] = None,
    combined_reliability: Optional[float] = None,
    abstention: bool = False,
) -> None:
    """Render side-by-side Voice, Text, and Alignment snapshot cards with reliability chips."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🎚️ Modality Snapshot & Fusion Arbitration</div>
            <div class="section-badge">VOICE VS TEXT ALIGNMENT & RELIABILITY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Compute average reliabilities across segments if not explicitly passed
    if voice_reliability is None and segments:
        v_rels = [safe_float((s.get("fusion") or {}).get("audio_reliability", 1.0)) for s in segments]
        voice_reliability = sum(v_rels) / len(v_rels) if v_rels else 1.0
    elif voice_reliability is None:
        voice_reliability = 1.0

    if text_reliability is None and segments:
        t_rels = [safe_float((s.get("fusion") or {}).get("text_reliability", 1.0)) for s in segments]
        text_reliability = sum(t_rels) / len(t_rels) if t_rels else 1.0
    elif text_reliability is None:
        text_reliability = 1.0

    if combined_reliability is None and segments:
        c_rels = [safe_float((s.get("fusion") or {}).get("combined_reliability", 1.0)) for s in segments]
        combined_reliability = sum(c_rels) / len(c_rels) if c_rels else 1.0
    elif combined_reliability is None:
        combined_reliability = 1.0

    snap_col1, snap_col2, snap_col3 = st.columns(3)

    with snap_col1:
        st.markdown(
            f"""
            <div class="modality-card voice-accent">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                    <div class="card-meta-label">Acoustic Modality (Voice)</div>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(251,191,36,0.15); color:var(--voice); padding:2px 6px; border-radius:4px; font-weight:700;">
                        REL: {voice_reliability:.2f}
                    </span>
                </div>
                <div class="card-highlight-value" style="color: var(--voice);">
                    <span>{emotion_icon(dominant_voice)}</span>
                    <span>{html.escape(emotion_code_label(dominant_voice))}</span>
                </div>
                <div class="card-subtext">
                    Dominant acoustic emotion extracted via Wav2Vec2-SUPERB acoustic classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap_col2:
        st.markdown(
            f"""
            <div class="modality-card text-accent">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                    <div class="card-meta-label">Linguistic Modality (Text)</div>
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(56,189,248,0.15); color:var(--text-accent); padding:2px 6px; border-radius:4px; font-weight:700;">
                        REL: {text_reliability:.2f}
                    </span>
                </div>
                <div class="card-highlight-value" style="color: var(--text-accent);">
                    <span>{emotion_icon(dominant_text)}</span>
                    <span>{html.escape(emotion_code_label(dominant_text))}</span>
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
        abstain_badge = (
            """<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(248,113,113,0.2); color:var(--negative); padding:2px 6px; border-radius:4px; font-weight:700;">ABSTAINED</span>"""
            if abstention
            else f"""<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(226,232,240,0.15); color:var(--fused); padding:2px 6px; border-radius:4px; font-weight:700;">REL: {combined_reliability:.2f}</span>"""
        )
        st.markdown(
            f"""
            <div class="modality-card fusion-accent">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;">
                    <div class="card-meta-label">Cross-Modal Alignment</div>
                    {abstain_badge}
                </div>
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
