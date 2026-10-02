"""
Segment-level timeline breakdown and turn-by-turn expander accordion.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.components.incongruence_card import render_incongruence_card
from ui.components.modality_cards import render_channel_meters_html
from ui.utils import direction_label, emotion_icon, safe_float


def _render_why_this_result(fusion: Dict[str, Any]) -> None:
    """Phase 5: 'Why this result?' expander — shows the declarative
    pattern rule that fired, the evidence behind it, and the effective
    (reliability-weighted) fusion weights actually used for this segment.

    Reads keys added by app/fusion.py's fuse() in Phase 5. Uses .get()
    with safe defaults throughout so this never breaks on an older
    cached result dict that predates Phase 5.
    """
    pattern_label = str(fusion.get("pattern_label", "")).strip()
    pattern_explanation = str(fusion.get("pattern_explanation", "")).strip()
    rule_ids = fusion.get("rule_ids_fired", []) or []

    audio_arousal = fusion.get("audio_arousal")
    text_arousal = fusion.get("text_arousal", "unknown")

    audio_reliability = safe_float(fusion.get("audio_reliability", 1.0))
    text_reliability = safe_float(fusion.get("text_reliability", 1.0))

    base_weights = fusion.get("base_weights", {}) or {}
    effective_weights = fusion.get("effective_weights", {}) or {}
    base_audio_w = safe_float(base_weights.get("audio_weight", 0.0))
    base_text_w = safe_float(base_weights.get("text_weight", 0.0))
    eff_audio_w = safe_float(effective_weights.get("audio_weight", base_audio_w))
    eff_text_w = safe_float(effective_weights.get("text_weight", base_text_w))

    weights_shifted = (
        abs(eff_audio_w - base_audio_w) > 0.001
        or abs(eff_text_w - base_text_w) > 0.001
    )

    if not pattern_label:
        return  # nothing to show for an older/incomplete fusion dict

    with st.expander("🔎 Why this result?", expanded=False):
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700; color: var(--fused); text-transform: uppercase; margin-bottom: 0.3rem;">
                Pattern: {html.escape(pattern_label)}
            </div>
            <div style="font-size: 0.78rem; color: var(--text); line-height: 1.5; margin-bottom: 0.6rem;">
                {html.escape(pattern_explanation)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if rule_ids:
            rule_ids_str = ", ".join(html.escape(str(r)) for r in rule_ids)
            st.markdown(
                f"""
                <div style="font-size: 0.68rem; color: var(--muted); margin-bottom: 0.6rem;">
                    Rule ID(s) fired: <code>{rule_ids_str}</code>
                </div>
                """,
                unsafe_allow_html=True,
            )

        arousal_display = f"{safe_float(audio_arousal):.2f}" if isinstance(audio_arousal, (int, float)) else "n/a"
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;font-size:0.72rem;color:var(--muted);margin-bottom:0.5rem;">
                <div>Voice arousal: <strong style="color:var(--text);">{arousal_display}</strong></div>
                <div>Text arousal: <strong style="color:var(--text);">{html.escape(str(text_arousal))}</strong> (not modeled — RoBERTa gives no arousal signal)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        weight_note = (
            " (adjusted from base due to segment reliability)"
            if weights_shifted
            else " (equal to base weights — both modalities fully reliable)"
        )
        st.markdown(
            f"""
            <div style="font-size:0.72rem;color:var(--muted);">
                <div style="margin-bottom:0.2rem;">
                    Base weights: audio {base_audio_w:.2f} / text {base_text_w:.2f}
                </div>
                <div>
                    Effective weights used: <strong style="color:var(--text);">audio {eff_audio_w:.2f} / text {eff_text_w:.2f}</strong>{weight_note}
                </div>
                <div style="margin-top:0.3rem;">
                    Reliability — voice: {audio_reliability:.2f}, text: {text_reliability:.2f}
                    {' (widened short-segment window)' if audio_reliability < 1.0 else ''}
                    {' (low ASR confidence)' if text_reliability < 1.0 else ''}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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

            # Phase 5: "Why this result?" — declarative rule trace and
            # effective (reliability-weighted) fusion weights for this
            # specific segment.
            _render_why_this_result(fusion)

            render_incongruence_card(
                acoustic_label=v_emotion,
                acoustic_confidence=v_conf,
                text_label=t_sentiment,
                text_confidence=t_conf,
                title=f"{title_prefix} MODALITY INCONGRUENCE",
            )