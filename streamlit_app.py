"""
Multimodal Voice & Conversation Emotion Analysis System
Thin Streamlit Orchestrator (< 250 lines).
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional
import streamlit as st

from app.audio_processor import AudioProcessingError, AudioValidationError
from app.conversation_analysis import ConversationAnalyzer
from app.microphone import (
    FALLBACK_MICROPHONE_MESSAGE,
    categorize_pipeline_error,
    is_microphone_available,
    save_audio_buffer,
)
from ui.components.export_bar import render_export_bar
from ui.components.header import render_header, render_pipeline_steps, render_sidebar
from ui.components.stepper import render_workflow_stepper
from ui.components.transparency_panels import (
    render_fusion_explainer,
    render_model_transparency_section,
    render_professional_limitations_section,
    render_system_notes,
)
from ui.pages.acoustic_view import render_acoustic_page
from ui.pages.analysis_view import render_analysis_page
from ui.pages.annotation_view import render_annotation_page
from ui.pages.validation_view import render_multimodal_validation_section
from ui.state import get_active_uploader_keys, init_session_state, reset_session
from ui.styles_loader import load_css
from ui.utils import derive_segment_summary, safe_float

# 1. PAGE CONFIGURATION & STYLES
st.set_page_config(
    page_title="Multimodal Emotion Analysis System",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
init_session_state()


@st.cache_resource(show_spinner=False)
def load_analyzer() -> ConversationAnalyzer:
    """Load and initialize the complete ML pipeline once per process."""
    return ConversationAnalyzer()


# 2. BRAND HEADER & SIDEBAR
render_header()
render_sidebar()
render_pipeline_steps()

# 3. AUDIO INGESTION (UPLOAD & MICROPHONE)
st.markdown(
    """
    <div class="section-header-bar">
        <div class="section-title">📂 Audio Ingestion & Analysis</div>
        <div class="section-badge">SUPPORTED: WAV · FLAC · MP3 · OGG · M4A</div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploader_key, mic_key = get_active_uploader_keys()
upload_tab, mic_tab = st.tabs(["📂 Upload File", "🎙️ Record Microphone"])

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload audio recording for multimodal emotion analysis",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        key=uploader_key,
        help="Upload WAV (recommended), FLAC, MP3, OGG, or M4A audio file (max 50 MB).",
    )

with mic_tab:
    mic_audio = None
    if is_microphone_available():
        st.caption("Record live audio directly from your browser microphone.")
        mic_audio = st.audio_input("Record audio", key=mic_key)
    else:
        st.info(f"ℹ️ {FALLBACK_MICROPHONE_MESSAGE}")

# Reconcile audio inputs
active_audio = None
audio_format = "audio/wav"
current_source = None

if mic_audio is not None:
    active_audio = mic_audio
    audio_format = "audio/wav"
    current_source = "mic"
elif uploaded_file is not None:
    active_audio = uploaded_file
    suffix = Path(uploaded_file.name).suffix.lower()
    format_map = {".wav": "audio/wav", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".flac": "audio/flac", ".m4a": "audio/mp4"}
    audio_format = format_map.get(suffix, "audio/wav")
    current_source = "upload"
elif st.query_params.get("sample"):
    sample_name = str(st.query_params.get("sample"))
    candidate = Path("sample_audio") / (sample_name if sample_name.endswith(".wav") else f"{sample_name}.wav")
    if candidate.exists():
        with open(candidate, "rb") as f:
            sample_bytes = f.read()
        active_audio = io.BytesIO(sample_bytes)
        active_audio.name = candidate.name
        audio_format = "audio/wav"
        current_source = "sample"

if current_source != st.session_state.get("_last_active_source"):
    st.session_state["_last_active_source"] = current_source
    if active_audio is not None:
        reset_session()

# Analysis trigger & action buttons
btn_col1, btn_col2, _ = st.columns([1.5, 1, 3])
auto_analyze = st.query_params.get("auto_analyze") in ("true", "1")
with btn_col1:
    analyze_clicked = st.button("ANALYZE AUDIO ▶", type="primary", use_container_width=True, disabled=(active_audio is None)) or (auto_analyze and active_audio is not None and not st.session_state.get("analysis_result"))
with btn_col2:
    if st.button("RESET SESSION ⟲", use_container_width=True):
        reset_session()
        st.rerun()

# 4. PIPELINE EXECUTION
if analyze_clicked and active_audio is not None:
    audio_bytes = active_audio.read()
    if hasattr(active_audio, "seek"):
        active_audio.seek(0)

    with st.status("Running Multimodal Emotion Analysis Pipeline...", expanded=True) as status:
        try:
            status.write("Initializing ConversationAnalyzer...")
            analyzer = load_analyzer()
            temp_path = save_audio_buffer(audio_bytes, suffix=Path(getattr(active_audio, "name", "record.wav")).suffix or ".wav")
            status.write("Executing neural speech, acoustic emotion, text sentiment & fusion models...")
            result = analyzer.analyze(temp_path)
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

            st.session_state["analysis_result"] = result
            st.session_state["audio_bytes"] = audio_bytes
            st.session_state["audio_format"] = audio_format
            status.update(label="Analysis Completed Successfully!", state="complete", expanded=False)
        except Exception as exc:
            title, guidance = categorize_pipeline_error(exc)
            st.error(f"**{title}**\n\n{guidance}")
            st.session_state["analysis_result"] = None
            status.update(label="Analysis Failed", state="error", expanded=True)

# 5. DASHBOARD VIEWS (ACTIVE OR EMPTY STATE)
result = st.session_state.get("analysis_result")
audio_bytes = st.session_state.get("audio_bytes")
audio_format = st.session_state.get("audio_format", "audio/wav")

if result:
    overall = result.get("overall", {}) or {}
    segments = result.get("segments", []) or []
    audio_meta = result.get("audio_metadata", {}) or {}
    transcript_info = result.get("transcript_language", {}) or {}
    trend = result.get("emotional_trend", {}) or {}
    escalation = result.get("escalation_analysis", {}) or {}

    duration = safe_float(audio_meta.get("duration_seconds", 0.0))
    language = transcript_info.get("language", "en").upper()
    language_prob = safe_float(transcript_info.get("probability", 1.0))
    skipped_segments = int(result.get("skipped_segments_count", 0))
    trend_label = trend.get("trend", "stable").lower()
    trend_delta = safe_float(trend.get("delta", 0.0))
    escalation_detected = bool(escalation.get("escalation_detected", False))
    escalation_reason = str(escalation.get("reason", "")).strip()

    dom_voice, dom_text, agree_count, mismatch_count = derive_segment_summary(segments)

    tab_analysis, tab_acoustic, tab_validation, tab_annotation = st.tabs([
        "💬 Conversation & Multimodal Analysis",
        "🎛️ Acoustic Inspection",
        "📊 Multimodal Validation",
        "🏷️ Annotation Helper",
    ])

    with tab_analysis:
        render_analysis_page(
            result=result,
            audio_bytes=audio_bytes,
            audio_format=audio_format,
            duration=duration,
            language=language,
            language_prob=language_prob,
            segments=segments,
            skipped_segments=skipped_segments,
            dominant_voice=dom_voice,
            dominant_text=dom_text,
            agree_count=agree_count,
            mismatch_count=mismatch_count,
            trend=trend,
            trend_label=trend_label,
            trend_delta=trend_delta,
            escalation=escalation,
            escalation_detected=escalation_detected,
            escalation_reason=escalation_reason,
        )

    with tab_acoustic:
        render_acoustic_page(audio_bytes=audio_bytes, segments=segments, duration=duration)

    with tab_validation:
        render_multimodal_validation_section()

    with tab_annotation:
        render_annotation_page()

    # Explainability & System Transparency Panels
    render_fusion_explainer()
    render_system_notes()
    render_model_transparency_section()
    render_professional_limitations_section()

    # Report Export Bar
    render_export_bar(result)

elif active_audio is None:
    # Show annotation helper + validation as top-level tabs even without an analysis result
    tab_empty_validation, tab_empty_annotation = st.tabs([
        "📊 Multimodal Validation",
        "🏷️ Annotation Helper",
    ])
    with tab_empty_validation:
        render_workflow_stepper()
        st.markdown("<br>", unsafe_allow_html=True)
        render_multimodal_validation_section()
    with tab_empty_annotation:
        render_annotation_page()

# 6. DASHBOARD FOOTER
st.markdown(
    """
    <div class="dashboard-footer">
        MULTIMODAL VOICE & CONVERSATION EMOTION ANALYSIS SYSTEM &nbsp;·&nbsp;
        B.TECH MINI-PROJECT AI ANALYTICS ENGINE &nbsp;·&nbsp;
        CPU INFERENCE OPTIMIZED
    </div>
    """,
    unsafe_allow_html=True,
)
