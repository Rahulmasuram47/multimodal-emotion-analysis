"""
Header, brand banner, pipeline stages, and sidebar rendering components.
"""

import streamlit as st

from app.diarization import get_diarization_status
from ui.state import get_fusion_settings, reset_fusion_settings


def render_header() -> None:
    """Render the top brand banner, beacon, title, and architecture chips."""
    diar_status = get_diarization_status()
    diar_error = st.session_state.get("diarization_error", "")
    if diar_error:
        speaker_id_text = "Error"
        speaker_chip_style = "color: var(--negative);"
    elif diar_status.get("token_present", False):
        speaker_id_text = "Active (Pyannote)"
        speaker_chip_style = "color: var(--positive);"
    else:
        speaker_id_text = "Off (no token)"
        speaker_chip_style = "color: var(--muted);"

    st.markdown(
        f"""
        <div class="brand-container">
            <div class="brand-top-row">
                <div class="brand-title-group">
                    <div class="brand-beacon"></div>
                    <h1 class="main-title">MULTIMODAL VOICE & CONVERSATION EMOTION ANALYSIS SYSTEM</h1>
                </div>
                <div class="system-badge">AI ANALYTICS ENGINE · V1.0</div>
            </div>
            <div class="subtitle">Voice + Speech + Text → Unified Conversation Emotion Analysis</div>
            <div class="header-chips-row">
                <div class="header-chip">🎙️ <strong>Voice Emotion:</strong> Wav2Vec2-SUPERB ER</div>
                <div class="header-chip">📝 <strong>Speech Recognition:</strong> Whisper STT</div>
                <div class="header-chip">💬 <strong>Text Sentiment:</strong> RoBERTa</div>
                <div class="header-chip">🗣️ <strong>Speaker ID:</strong> <span style="{speaker_chip_style}">{speaker_id_text}</span></div>
                <div class="header-chip">⚡ <strong>Fusion:</strong> Multi-Evidence Arbitration</div>
                <div class="header-chip">📈 <strong>Dynamics:</strong> Trend & Escalation Detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_steps() -> None:
    """Render the 3-step start workflow and live system strip (Prompt H)."""
    import config
    diar_status = get_diarization_status()
    whisper_size = st.session_state.get("whisper_model_size", getattr(config, "STT_MODEL_SIZE", "base"))
    whisper_lang = st.session_state.get("whisper_language", getattr(config, "STT_LANGUAGE", "auto"))

    diar_text = "Active" if diar_status["token_present"] else "Off (no token)"
    diar_color = "var(--positive)" if diar_status["token_present"] else "var(--muted)"

    st.markdown(
        f"""
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:0.75rem; margin-bottom:1.0rem;">
            <div class="pipeline-step" style="border-top: 3px solid var(--text-accent);">
                <div class="step-number" style="color: var(--text-accent);">STEP 01</div>
                <div class="step-name">Select or Record Audio</div>
                <div class="step-modality">Upload WAV/MP3, record live mic, or pick a benchmark sample clip</div>
            </div>
            <div class="pipeline-step" style="border-top: 3px solid var(--voice);">
                <div class="step-number" style="color: var(--voice);">STEP 02</div>
                <div class="step-name">Multimodal Extraction</div>
                <div class="step-modality">Wav2Vec2 ER + Faster-Whisper ({whisper_size}) + RoBERTa NLP on CPU</div>
            </div>
            <div class="pipeline-step" style="border-top: 3px solid var(--positive);">
                <div class="step-number" style="color: var(--positive);">STEP 03</div>
                <div class="step-name">Arbitrate & Review</div>
                <div class="step-modality">Explainable verdict hero, agreement ribbon & counterfactuals</div>
            </div>
        </div>

        <div style="display:flex; flex-wrap:wrap; gap:0.5rem; align-items:center; background:var(--panel-2); border:1px solid var(--border); border-radius:6px; padding:0.45rem 0.85rem; margin-bottom:1.0rem; font-family:'IBM Plex Mono',monospace; font-size:0.75rem;">
            <span style="color:var(--muted); font-weight:700;">SYSTEM STATUS:</span>
            <span style="color:var(--voice);">🎙️ Wav2Vec2: Ready</span>
            <span style="color:var(--muted);">·</span>
            <span style="color:var(--text);">📝 Whisper: INT8 ({whisper_size}, {whisper_lang})</span>
            <span style="color:var(--muted);">·</span>
            <span style="color:var(--text-accent);">💬 RoBERTa: Ready</span>
            <span style="color:var(--muted);">·</span>
            <span style="color:{diar_color};">🗣️ Pyannote: {diar_text}</span>
            <span style="color:var(--muted);">·</span>
            <span style="color:var(--fused);">⚡ Fusion: Rule-Based</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render technical specifications, diarization status, and (Phase 5) fusion settings in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
                <div style="width: 8px; height: 8px; border-radius: 50%; background: var(--voice); box-shadow: 0 0 8px var(--voice);"></div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.85rem; color: var(--text);">
                    AI ANALYSIS CONSOLE
                </div>
            </div>
            <div style="font-size: 0.72rem; color: var(--muted); margin-bottom: 1.25rem;">
                B.Tech Mini-Project AI Pipeline
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-heading">Appearance</div>', unsafe_allow_html=True)
        current_theme = st.session_state.get("theme", "dark")
        selected_theme = st.radio(
            "Interface Theme",
            ["Dark", "Light"],
            index=0 if current_theme == "dark" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="theme_radio_selector",
        )
        if selected_theme.lower() != current_theme:
            st.session_state["theme"] = selected_theme.lower()
            st.rerun()

        st.markdown('<div class="sidebar-heading">Models & Architecture</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Acoustic Emotion</span>
                <span class="sidebar-spec-val" style="color: var(--voice);">Wav2Vec2-SUPERB ER</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Speech-to-Text</span>
                <span class="sidebar-spec-val" style="color: var(--text);">Whisper STT</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Text Sentiment</span>
                <span class="sidebar-spec-val" style="color: var(--text-accent);">RoBERTa NLP</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Speaker ID</span>
                <span class="sidebar-spec-val" style="color: var(--positive);">Pyannote Diarization</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Fusion Strategy</span>
                <span class="sidebar-spec-val" style="color: var(--fused);">Weighted Rule-Based</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Dynamics Engine</span>
                <span class="sidebar-spec-val">Trend & Escalation</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------
        # Diarization status: reflects configured / not configured / error
        # without exposing tokens, previews, or credentials.
        # -----------------------------------------------------
        diar_status = get_diarization_status()
        diar_error = st.session_state.get("diarization_error", "")

        st.markdown('<div class="sidebar-heading">Diarization Status</div>', unsafe_allow_html=True)
        if diar_error:
            st.markdown(
                f"""
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-key">Diarization</span>
                    <span class="sidebar-spec-val" style="color: var(--critical);">✗ Error ({html.escape(str(diar_error))})</span>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 0.3rem; margin-bottom: 1rem; line-height: 1.3;">
                    Diarization pipeline encountered an error.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif diar_status.get("token_present", False):
            st.markdown(
                """
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-key">Diarization</span>
                    <span class="sidebar-spec-val" style="color: var(--positive);">✓ Configured</span>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 0.3rem; margin-bottom: 1rem;">
                    Hugging Face credentials found. Diarization will attempt to run on each analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="sidebar-spec-row">
                    <span class="sidebar-spec-key">Diarization</span>
                    <span class="sidebar-spec-val" style="color: var(--text-muted);">✗ Not configured</span>
                </div>
                <div style="font-size: 0.68rem; color: var(--text-muted); margin-top: 0.3rem; margin-bottom: 1rem;">
                    No token found. Speakers will show as "Unknown". See docs/DIARIZATION_SETUP.md to enable speaker separation.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="sidebar-heading">Audio Input Specs</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Supported</span>
                <span class="sidebar-spec-val">WAV, FLAC, MP3, OGG, M4A</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Target Rate</span>
                <span class="sidebar-spec-val">16,000 Hz Mono</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Recommended</span>
                <span class="sidebar-spec-val" style="color: var(--voice);">WAV (PCM 16-bit)</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Hardware Engine</span>
                <span class="sidebar-spec-val">CPU Inference</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # -----------------------------------------------------
        # Speech Recognition Settings (Whisper size & language)
        # -----------------------------------------------------
        import config
        st.markdown('<div class="sidebar-heading">Speech Recognition Settings</div>', unsafe_allow_html=True)
        st.selectbox(
            "Whisper model size",
            options=["base", "small"],
            index=0 if st.session_state.get("whisper_model_size", config.STT_MODEL_SIZE) == "base" else 1,
            key="whisper_model_size",
            help="base (fast, ~140MB) or small (higher accuracy, ~460MB). INT8 CPU inference.",
        )
        st.selectbox(
            "Transcription language",
            options=["auto", "en"],
            index=0 if st.session_state.get("whisper_language", config.STT_LANGUAGE) == "auto" else 1,
            format_func=lambda x: "Auto-detect language" if x == "auto" else "Force English (en)",
            key="whisper_language",
            help="Auto-detect spoken language, or force English transcription.",
        )

        # -----------------------------------------------------
        # Phase 5: Fusion Settings (session-scoped; applied on the
        # NEXT "ANALYZE AUDIO" click, not instantly — see
        # ui/state.py::get_fusion_settings docstring for why).
        # -----------------------------------------------------
        st.markdown('<div class="sidebar-heading">Fusion Settings</div>', unsafe_allow_html=True)
        st.caption("Applied on your next ANALYZE AUDIO run, not instantly.")

        settings = get_fusion_settings()
        # FIX: no `value=` here — the widget's value is driven entirely
        # by st.session_state["fusion_audio_weight"], which
        # init_session_state() seeds once with the config default and
        # reset_fusion_settings() overwrites on reset. Passing both
        # `value=` and a pre-existing session_state[key] triggers
        # Streamlit's "widget created with a default value but also had
        # its value set via the Session State API" warning (soon to be
        # a hard error).
        new_audio_weight = st.slider(
            "Audio weight",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="fusion_audio_weight",
            help="Text weight is automatically 1 minus this value.",
        )
        st.caption(f"Text weight: {1.0 - new_audio_weight:.2f}")

        st.slider(
            "Agreement confidence threshold",
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            key="fusion_confidence_threshold",
            help="Minimum confidence, in both modalities, required to flag a mismatch.",
        )

        if st.button("Reset fusion settings to defaults", use_container_width=True):
            reset_fusion_settings()
            st.rerun()