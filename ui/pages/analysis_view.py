"""
Conversation & Multimodal Analysis page orchestrator (Tab 1).
Follows exact Prompt H results page hierarchy:
  (a) Verdict Hero: verdict, reason, attention gauge, recommendation
  (b) Three Modality Cards with reliability chips
  (c) Timeline: agreement ribbon, valence-arousal map, intensity slope/tolerance, speaker comparison
  (d) Tabs: segments, acoustic inspection, why-this-result, validation
"""

import html
from typing import Any, Dict, List, Optional
import streamlit as st

from app.speaker_analytics import compute_speaker_analytics, find_possible_escalation_initiator
from ui.charts.dynamics import (
    render_agreement_strip,
    render_intensity_trajectory_chart,
    render_speaker_comparison_chart,
    render_valence_arousal_map,
)
from ui.components.modality_cards import render_modality_cards
from ui.components.segment_timeline import render_segment_timeline
from ui.components.speaker_analytics_card import render_speaker_analytics_card
from ui.pages.validation_view import render_multimodal_validation_section
from ui.components.verdict_card import build_key_finding, render_verdict_hero
from ui.components.waveform_player import render_wavesurfer_player
from ui.pages.acoustic_view import render_acoustic_page
from ui.pages.why_result_view import render_why_result_page
from ui.utils import html_block, safe_float


def render_analysis_page(
    result: Dict[str, Any],
    audio_bytes: bytes,
    audio_format: str,
    duration: float,
    language: str,
    language_prob: float,
    segments: List[Dict[str, Any]],
    skipped_segments: int,
    dominant_voice: str,
    dominant_text: str,
    agree_count: int,
    mismatch_count: int,
    trend: Dict[str, Any],
    trend_label: str,
    trend_delta: float,
    escalation: Dict[str, Any],
    escalation_detected: bool,
    escalation_reason: str,
    verdict: Optional[Dict[str, Any]] = None,
) -> None:
    """Orchestrate Conversation & Multimodal Analysis results page in Prompt H hierarchy."""
    # -------------------------------------------------------------
    # Item 18: Success banner — show once, results tab only, one icon,
    # with elapsed time of the CURRENT run.
    # -------------------------------------------------------------
    if st.session_state.get("_show_success_banner") and st.session_state.get("_last_run_elapsed") is not None:
        elapsed = safe_float(st.session_state.get("_last_run_elapsed"))
        st.success(f"Analysis completed in {elapsed:.2f}s")

    # -------------------------------------------------------------
    # (a) Verdict Hero: verdict, reason, attention gauge, recommendation
    # -------------------------------------------------------------

    finding = build_key_finding(
        dominant_voice=dominant_voice,
        dominant_text=dominant_text,
        segments=segments,
        trend_label=trend_label,
        escalation_detected=escalation_detected,
    )
    render_verdict_hero(
        verdict=verdict,
        finding=finding,
        attention_score=result.get("attention_score", {}),
        recommendation=result.get("recommendation", {}),
    )

    # -------------------------------------------------------------
    # (b) Three Modality Cards with reliability chips
    # -------------------------------------------------------------
    v_rels = [safe_float((s.get("fusion") or {}).get("audio_reliability", 1.0)) for s in segments]
    t_rels = [safe_float((s.get("fusion") or {}).get("text_reliability", 1.0)) for s in segments]
    c_rels = [safe_float((s.get("fusion") or {}).get("combined_reliability", 1.0)) for s in segments]
    v_rel_avg = sum(v_rels) / len(v_rels) if v_rels else 1.0
    t_rel_avg = sum(t_rels) / len(t_rels) if t_rels else 1.0
    c_rel_avg = sum(c_rels) / len(c_rels) if c_rels else 1.0
    is_abstain = any((s.get("fusion") or {}).get("abstention", False) for s in segments)

    render_modality_cards(
        dominant_voice=dominant_voice,
        dominant_text=dominant_text,
        segments=segments,
        agree_count=agree_count,
        mismatch_count=mismatch_count,
        voice_reliability=v_rel_avg,
        text_reliability=t_rel_avg,
        combined_reliability=c_rel_avg,
        abstention=is_abstain,
    )

    # -------------------------------------------------------------
    # (c) Timeline & Conversation Dynamics
    # -------------------------------------------------------------
    render_agreement_strip(segments)
    render_valence_arousal_map(segments)
    render_intensity_trajectory_chart(segments, verdict=verdict, escalation=escalation)

    speaker_analytics = [
        sa.to_dict() for sa in compute_speaker_analytics(segments, duration)
    ]
    if speaker_analytics and len(speaker_analytics) >= 2:
        render_speaker_comparison_chart(speaker_analytics)
        initiator = find_possible_escalation_initiator(segments)
        render_speaker_analytics_card(speaker_analytics, initiator)

    # Audio Playback (WaveSurfer) & Complete Transcript
    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 1.2rem;">
<div class="section-title">🎙️ Audio Recording & Transcript</div>
<div class="section-badge">WHISPER RECOGNITION & WAVESURFER.JS</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    audio_col, transcript_col = st.columns([1, 1])

    with audio_col:
        with st.container(key="audio_player_box"):
            render_wavesurfer_player(
                audio_bytes=audio_bytes,
                audio_format=audio_format,
                segments=segments,
                duration=duration,
            )

    with transcript_col:
        full_transcript = str(result.get("full_transcript", "")).strip()
        if full_transcript:
            escaped_transcript = html.escape(full_transcript).replace("\n", "<br>")
            st.markdown(
                html_block(f"""
<div class="transcript-container">
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 700; color: var(--text-accent); text-transform: uppercase; margin-bottom: 0.4rem; letter-spacing: 0.08em;">
COMPLETE SPEECH TRANSCRIPT
</div>
<div style="font-size: 0.82rem; color: var(--text); line-height: 1.55;">
{escaped_transcript}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )
        else:
            st.warning("No speech transcript detected in audio stream.")

    # -------------------------------------------------------------
    # (d) Tabs: segments, acoustic inspection, why-this-result, validation
    # -------------------------------------------------------------
    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 1.5rem;">
<div class="section-title">📑 Detailed Analysis & Evidence Tabs</div>
<div class="section-badge">SEGMENTS · ACOUSTIC · WHY THIS RESULT · VALIDATION</div>
</div>
"""),
        unsafe_allow_html=True,
    )


    tab_segs, tab_acoust, tab_why, tab_valid, tab_annot = st.tabs([
        "💬 Segments",
        "🎛️ Acoustic Inspection",
        "🔎 Why This Result?",
        "📊 Multimodal Validation",
        "🏷️ Annotation Helper",
    ])

    with tab_segs:
        render_segment_timeline(segments)

    with tab_acoust:
        render_acoustic_page(audio_bytes=audio_bytes, segments=segments, duration=duration)

    with tab_why:
        render_why_result_page(result=result, segments=segments)

    with tab_valid:
        render_multimodal_validation_section()

    with tab_annot:
        from ui.pages.annotation_view import render_annotation_page
        render_annotation_page()