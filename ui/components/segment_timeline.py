"""
Segment-level timeline breakdown and turn-by-turn expander accordion.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.components.incongruence_card import render_incongruence_card
from ui.components.modality_cards import render_channel_meters_html
from ui.utils import direction_label, emotion_icon, safe_float


def render_segment_timeline(segments: List[Dict[str, Any]]) -> None:
    """Render the per-turn expander list with fine-grained multimodal details."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🔍 Segment-Level Multimodal Breakdown</div>
            <div class="section-badge">TURN-BY-TURN INFERENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not segments:
        st.info("No conversational segments extracted from this recording.")
        return

    for segment in segments:
        idx = segment.get("segment_index", "?")
        start = safe_float(segment.get("start_seconds", 0.0))
        end = safe_float(segment.get("end_seconds", 0.0))
        text = str(segment.get("transcript", "")).strip()
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}
        intensity = safe_float(segment.get("intensity", 0.0))
        speaker = str(segment.get("speaker", "")).strip()

        v_emotion = str(voice.get("emotion", "unknown")).lower()
        v_conf = safe_float(voice.get("confidence", 0.0))
        t_sentiment = str(sentiment.get("sentiment", "unknown")).lower()
        t_conf = safe_float(sentiment.get("confidence", 0.0))
        f_conf = safe_float(fusion.get("combined_confidence", 0.0))

        agreement = bool(fusion.get("agreement", False))
        mismatch = bool(fusion.get("mismatch", False))
        interpretation = str(fusion.get("interpretation", "")).strip()
        audio_dir = direction_label(fusion.get("audio_direction", "unknown"))
        text_dir = direction_label(fusion.get("text_direction", "unknown"))

        title_prefix = f"Turn {idx:02d}" if isinstance(idx, int) else f"Turn {idx}"
        speaker_tag = f" · {speaker}" if speaker and speaker != "Unknown" else ""

        expander_title = (
            f"{title_prefix}{speaker_tag} · [{start:.2f}s → {end:.2f}s] · "
            f"Voice: {v_emotion.upper()} | Text: {t_sentiment.upper()} | Intensity: {intensity:.0%}"
        )

        with st.expander(expander_title, expanded=True):
            if text:
                speaker_prefix = f"{html.escape(speaker)}: " if speaker and speaker != "Unknown" else ""
                st.markdown(
                    f"""
                    <div class="segment-transcript-box">
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-accent); font-weight: 700; text-transform: uppercase;">Transcript:</span> "{speaker_prefix}{html.escape(text)}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            card_col1, card_col2, card_col3 = st.columns(3)

            with card_col1:
                st.markdown(
                    f"""
                    <div class="segment-card voice">
                        <div class="card-meta-label" style="color: var(--voice);">🎙️ VOICE EMOTION</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            {emotion_icon(v_emotion)} {html.escape(v_emotion.upper())}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Confidence: <strong style="color: var(--voice);">{v_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col2:
                st.markdown(
                    f"""
                    <div class="segment-card text">
                        <div class="card-meta-label" style="color: var(--text-accent);">💬 TEXT SENTIMENT</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            {emotion_icon(t_sentiment)} {html.escape(t_sentiment.upper())}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Confidence: <strong style="color: var(--text-accent);">{t_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col3:
                st.markdown(
                    f"""
                    <div class="segment-card fusion">
                        <div class="card-meta-label" style="color: var(--fused);">⚡ MULTIMODAL FUSION</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            INTENSITY: {intensity:.1%}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Combined Conf: <strong style="color: var(--fused);">{f_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                render_channel_meters_html(
                    voice_label=v_emotion,
                    voice_confidence=v_conf,
                    text_label=t_sentiment,
                    text_confidence=t_conf,
                    fused_confidence=f_conf,
                ),
                unsafe_allow_html=True,
            )

            if agreement:
                badge_class = "ok"
                badge_text = "✓ Voice & Text Modalities Agree"
            elif mismatch:
                badge_class = "warn"
                badge_text = "⚠ Voice / Text Modality Mismatch Detected"
            else:
                badge_class = "info"
                badge_text = "ℹ Low-Confidence Disagreement"

            st.markdown(
                f"""
                <div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.4rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
                        <span class="status-badge {badge_class}">{badge_text}</span>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted);">
                            Voice Dir: <strong style="color: var(--text);">{html.escape(audio_dir)}</strong> &nbsp;|&nbsp;
                            Text Dir: <strong style="color: var(--text);">{html.escape(text_dir)}</strong>
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--muted); line-height: 1.4;">
                        <strong>Interpretation:</strong> {html.escape(interpretation) if interpretation else 'Standard multimodal arbitration.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            render_incongruence_card(
                acoustic_label=v_emotion,
                acoustic_confidence=v_conf,
                text_label=t_sentiment,
                text_confidence=t_conf,
                title=f"{title_prefix} MODALITY INCONGRUENCE",
            )
