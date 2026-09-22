"""
Conversation & Multimodal Analysis page orchestrator (Tab 1).
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.components.incongruence_card import render_incongruence_card
from ui.components.metric_cards import (
    render_dynamics_card,
    render_overview_metrics,
    render_speaker_metrics,
)
from ui.components.modality_cards import render_modality_cards
from ui.components.recommendation_card import render_recommendation_card
from ui.components.segment_timeline import render_segment_timeline
from ui.components.verdict_card import build_key_finding, render_verdict_card
from ui.components.waveform_player import render_wavesurfer_player
from ui.utils import safe_float


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
) -> None:
    """Orchestrate Tab 1 Conversation & Multimodal Analysis widgets."""
    # 1. Key Finding banner
    finding = build_key_finding(
        dominant_voice=dominant_voice,
        dominant_text=dominant_text,
        segments=segments,
        trend_label=trend_label,
        escalation_detected=escalation_detected,
    )
    render_verdict_card(finding)

    # 2. Overall Modality Incongruence Card
    v_top_conf = safe_float(result.get("voice_emotion", {}).get("confidence", 0.0))
    t_top_conf = safe_float(result.get("text_sentiment", {}).get("confidence", 0.0))
    if not v_top_conf and segments:
        v_top_conf = safe_float(segments[0].get("voice_emotion", {}).get("confidence", 0.0))
    if not t_top_conf and segments:
        t_top_conf = safe_float(segments[0].get("text_sentiment", {}).get("confidence", 0.0))

    render_incongruence_card(
        acoustic_label=dominant_voice,
        acoustic_confidence=v_top_conf or 0.85,
        text_label=dominant_text,
        text_confidence=t_top_conf or 0.88,
        title="OVERALL CONVERSATION MODALITY INCONGRUENCE",
    )

    # 3. Actionable Recommendation Card
    rec_data = result.get("recommendation", {}) or {}
    render_recommendation_card(rec_data)

    # 4. Overview Metrics (6 cards)
    proc_time = result.get("processing_time", {}) or {}
    tot_time = proc_time.get("total_processing_seconds")
    latency_val = f"{tot_time:.2f}s" if tot_time is not None else "Near-RT"

    render_overview_metrics(
        language=language,
        language_prob=language_prob,
        duration=duration,
        segments=segments,
        skipped_segments=skipped_segments,
        trend_label=trend_label,
        trend_delta=trend_delta,
        escalation_detected=escalation_detected,
        latency_val=latency_val,
    )

    # 5. Modality Cards & Alignment
    render_modality_cards(
        dominant_voice=dominant_voice,
        dominant_text=dominant_text,
        segments=segments,
        agree_count=agree_count,
        mismatch_count=mismatch_count,
    )

    # 6. Speaker Analysis
    speaker_labels = [
        str(s.get("speaker", "")).strip()
        for s in segments
        if str(s.get("speaker", "")).strip() and str(s.get("speaker", "")).strip() != "Unknown"
    ]
    unique_speakers = list(dict.fromkeys(speaker_labels))
    diarization_error = str(result.get("diarization_error", "") or "").strip()

    render_speaker_metrics(
        unique_speakers=unique_speakers,
        speaker_labels=speaker_labels,
        diarization_error=diarization_error,
    )

    # 7. Audio Playback (WaveSurfer) & Full Transcript
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🎙️ Audio Recording & Transcript</div>
            <div class="section-badge">WHISPER RECOGNITION & WAVESURFER.JS</div>
        </div>
        """,
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
                f"""
                <div class="transcript-container">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; font-weight: 700; color: var(--text-accent); text-transform: uppercase; margin-bottom: 0.4rem; letter-spacing: 0.08em;">
                        COMPLETE SPEECH TRANSCRIPT
                    </div>
                    {escaped_transcript}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("No speech transcript detected in audio stream.")

    # 8. Emotional Dynamics & Escalation Chart
    render_dynamics_card(
        trend=trend,
        escalation=escalation,
        trend_label=trend_label,
        escalation_detected=escalation_detected,
        escalation_reason=escalation_reason,
    )

    # 9. Conversation Segments Breakdown
    render_segment_timeline(segments)
