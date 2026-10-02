"""
Multimodal Voice & Conversation Emotion Analysis System
Thin Streamlit Orchestrator.
"""
from dotenv import load_dotenv

load_dotenv()

import io
import time
from pathlib import Path

import streamlit as st

from app.audio_processor import AudioProcessingError, AudioValidationError  # noqa: F401
from app.conversation_analysis import ConversationAnalyzer
from app.fusion import MultimodalFusion
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
from ui.pages.acoustic_view import render_acoustic_page  # noqa: F401
from ui.pages.analysis_view import render_analysis_page
from ui.pages.annotation_view import render_annotation_page
from ui.pages.validation_view import render_multimodal_validation_section
from ui.state import (
    clear_results,
    get_active_uploader_keys,
    get_fusion_settings,
    init_session_state,
    reset_session,
)
from ui.styles_loader import load_css
from ui.utils import derive_segment_summary, safe_float

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SAMPLE_DIR = Path("sample_audio")

# 1. PAGE CONFIGURATION & STYLES
st.set_page_config(
    page_title="Multimodal Emotion Analysis System",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()
init_session_state()


@st.cache_resource(show_spinner="Loading models (first run only, this can take a while)...")
def load_analyzer(whisper_model_size: str = "base", whisper_language: str = "auto") -> ConversationAnalyzer:
    """Load and initialize the complete ML pipeline once per configuration."""
    return ConversationAnalyzer(whisper_model_size=whisper_model_size, whisper_language=whisper_language)


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
upload_tab, mic_tab, sample_tab = st.tabs(["📂 Upload File", "🎙️ Record Microphone", "🎵 Pick Sample Audio"])

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

with sample_tab:
    sample_options = {
        "rising_tone.wav": "Rising Tone (Escalating Multimodal Dynamics)",
        "happy_sample.wav": "Happy Clip (Positive Emotion Alignment)",
        "angry_sample.wav": "Angry Clip (High-Arousal Negative Emotion)",
        "sad_sample.wav": "Sad Clip (Low-Arousal Negative Emotion)",
        "neutral_sample.wav": "Neutral Discussion (Balanced Dialogue)",
        "real_voice.wav": "Real Conversational Voice Recording",
    }
    selected_sample = st.selectbox(
        "Choose benchmark sample clip:",
        options=list(sample_options.keys()),
        format_func=lambda x: f"{x} — {sample_options[x]}",
        key="sample_clip_selector",
    )
    if st.button("LOAD SAMPLE CLIP 📥", use_container_width=True):
        st.session_state["_selected_sample_clip"] = selected_sample
        clear_results()
        st.rerun()

# Reconcile audio inputs
active_audio = None
audio_format = "audio/wav"
current_source = None

if mic_audio is not None:
    active_audio = mic_audio
    current_source = "mic"
elif uploaded_file is not None:
    active_audio = uploaded_file
    suffix = Path(uploaded_file.name).suffix.lower()
    format_map = {".wav": "audio/wav", ".mp3": "audio/mp3", ".ogg": "audio/ogg", ".flac": "audio/flac", ".m4a": "audio/mp4"}
    audio_format = format_map.get(suffix, "audio/wav")
    current_source = "upload"
elif st.session_state.get("_selected_sample_clip") or st.query_params.get("sample"):
    raw_name = str(st.session_state.get("_selected_sample_clip") or st.query_params.get("sample"))
    # SECURITY FIX: strip any directory components so a crafted ?sample=../../x
    # query parameter cannot read files outside sample_audio/.
    safe_name = Path(raw_name).name
    if not safe_name.endswith(".wav"):
        safe_name += ".wav"
    candidate = SAMPLE_DIR / safe_name
    if candidate.exists():
        active_audio = io.BytesIO(candidate.read_bytes())
        active_audio.name = candidate.name
        current_source = "sample"
        st.caption(f"Loaded demo sample: `{candidate.name}`")

if current_source != st.session_state.get("_last_active_source"):
    st.session_state["_last_active_source"] = current_source
    st.session_state["_auto_analyzed"] = False
    if active_audio is not None:
        clear_results()

# Analysis trigger & action buttons
btn_col1, btn_col2, _ = st.columns([1.5, 1, 3])
auto_analyze = st.query_params.get("auto_analyze") in ("true", "1")
with btn_col1:
    clicked = st.button(
        "ANALYZE AUDIO ▶", type="primary", use_container_width=True, disabled=(active_audio is None)
    )
    # Auto-analyze only once per source, so a failed run cannot loop forever.
    auto_trigger = (
        auto_analyze
        and active_audio is not None
        and not st.session_state.get("analysis_result")
        and not st.session_state.get("_auto_analyzed", False)
    )
    analyze_clicked = clicked or auto_trigger
with btn_col2:
    if st.button("RESET SESSION ⟲", use_container_width=True):
        reset_session()
        st.rerun()

# 4. PIPELINE EXECUTION
if analyze_clicked and active_audio is not None:
    if auto_trigger:
        st.session_state["_auto_analyzed"] = True

    audio_bytes = active_audio.read()
    if hasattr(active_audio, "seek"):
        active_audio.seek(0)

    if not audio_bytes:
        st.error("The selected audio is empty. Please upload or record again.")
    elif len(audio_bytes) > MAX_UPLOAD_BYTES:
        st.error("Audio file exceeds the 50 MB limit.")
    else:
        t_pipeline_start = time.perf_counter()
        temp_path = None
        with st.status("Executing Multimodal Emotion Analysis Pipeline...", expanded=True) as status:
            try:
                whisper_size = st.session_state.get("whisper_model_size", "base")
                whisper_lang = st.session_state.get("whisper_language", "auto")
                status.write(f"[0.00s] Initializing ConversationAnalyzer (Whisper {whisper_size}, lang={whisper_lang})...")
                analyzer = load_analyzer(whisper_model_size=whisper_size, whisper_language=whisper_lang)

                fusion_settings = get_fusion_settings()
                analyzer.fusion = MultimodalFusion(
                    audio_weight=fusion_settings["audio_weight"],
                    text_weight=fusion_settings["text_weight"],
                    confidence_threshold=fusion_settings["confidence_threshold"],
                )
                temp_path = save_audio_buffer(
                    audio_bytes,
                    suffix=Path(getattr(active_audio, "name", "record.wav")).suffix or ".wav",
                )

                def progress_cb(msg: str, current: int, total: int) -> None:
                    elapsed = time.perf_counter() - t_pipeline_start
                    status.write(f"[{elapsed:.2f}s] {msg} ({current}/{total})")

                status.write(
                    f"[{time.perf_counter() - t_pipeline_start:.2f}s] "
                    "Executing acoustic Wav2Vec2, Whisper STT & RoBERTa models..."
                )
                result = analyzer.analyze(temp_path, progress_callback=progress_cb)

                st.session_state["analysis_result"] = result
                st.session_state["audio_bytes"] = audio_bytes
                st.session_state["audio_format"] = audio_format
                t_total = time.perf_counter() - t_pipeline_start
                st.session_state["_last_run_elapsed"] = t_total
                st.session_state["_show_success_banner"] = True
                status.update(label=f"Analysis Completed Successfully in {t_total:.2f}s!", state="complete", expanded=False)

            except Exception as exc:
                title, guidance = categorize_pipeline_error(exc)
                st.error(f"**{title}**\n\n{guidance}")
                st.session_state["analysis_result"] = None
                status.update(label="Analysis Failed", state="error", expanded=True)
            finally:
                if temp_path:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception:
                        pass

# 5. DASHBOARD VIEWS (ACTIVE OR EMPTY STATE)
result = st.session_state.get("analysis_result")
audio_bytes = st.session_state.get("audio_bytes")
audio_format = st.session_state.get("audio_format", "audio/wav")

if result:
    segments = result.get("segments", []) or []
    # ConversationAnalyzer.analyze() puts duration_seconds, language and
    # language_probability at the top level of the result dict.
    audio_meta = result
    trend = result.get("emotional_trend", {}) or {}
    escalation = result.get("escalation", {}) or {}
    verdict = result.get("verdict", {}) or {}

    duration = safe_float(audio_meta.get("duration_seconds", 0.0))
    language = str(result.get("language") or "en").upper()
    language_prob = safe_float(result.get("language_probability", 1.0))
    skipped_segments = int(result.get("skipped_segments", 0) or 0)
    trend_label = str(trend.get("trend") or "stable").lower()
    trend_delta = safe_float(trend.get("delta", 0.0))
    escalation_detected = bool(escalation.get("escalation_detected", False))
    escalation_reason = str(escalation.get("reason", "")).strip()

    dom_voice, dom_text, agree_count, mismatch_count = derive_segment_summary(segments)

    if result.get("no_speech_detected"):
        st.info("ℹ️ **No speech detected.** " + result.get("no_speech_message", "The recording appears to be silence or non-speech audio."))

    voice_collapse = result.get("voice_collapse")
    if voice_collapse and voice_collapse.get("detected"):
        st.warning(f"⚠️ **Voice model shows low discrimination on this input:** {voice_collapse.get('warning', 'Treat voice labels as low reliability.')}")

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
        verdict=verdict,
    )

    # Explainability & System Transparency Panels
    render_fusion_explainer()
    render_system_notes()
    render_model_transparency_section()
    render_professional_limitations_section()

    # Report Export Bar
    render_export_bar(result)

else:
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