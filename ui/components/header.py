"""
Header, brand banner, pipeline stages, and sidebar rendering components.
"""

import streamlit as st


def render_header() -> None:
    """Render the top brand banner, beacon, title, and architecture chips."""
    st.markdown(
        """
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
                <div class="header-chip">🗣️ <strong>Speaker ID:</strong> Pyannote Diarization</div>
                <div class="header-chip">⚡ <strong>Fusion:</strong> Multi-Evidence Arbitration</div>
                <div class="header-chip">📈 <strong>Dynamics:</strong> Trend & Escalation Detection</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_steps() -> None:
    """Render the 7-stage architectural pipeline workflow cards."""
    st.markdown(
        """
        <div class="pipeline-container">
            <div class="pipeline-step">
                <div class="step-number">STAGE 01</div>
                <div class="step-name">Audio Input</div>
                <div class="step-modality">WAV / MP3 Preprocessing</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--voice);">
                <div class="step-number" style="color: var(--voice);">STAGE 02</div>
                <div class="step-name">Voice Emotion</div>
                <div class="step-modality">Acoustic Wav2Vec2</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--text-accent);">
                <div class="step-number" style="color: var(--text-accent);">STAGE 03</div>
                <div class="step-name">Speech-to-Text</div>
                <div class="step-modality">Whisper Transcription</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--text-accent);">
                <div class="step-number" style="color: var(--text-accent);">STAGE 04</div>
                <div class="step-name">Text Sentiment</div>
                <div class="step-modality">RoBERTa Classifier</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--neutral);">
                <div class="step-number" style="color: var(--neutral);">STAGE 05</div>
                <div class="step-name">Speaker ID</div>
                <div class="step-modality">Pyannote Diarization</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--fused);">
                <div class="step-number" style="color: var(--fused);">STAGE 06</div>
                <div class="step-name">Multimodal Fusion</div>
                <div class="step-modality">Decision Arbitration</div>
            </div>
            <div class="pipeline-step" style="border-top: 2px solid var(--positive);">
                <div class="step-number" style="color: var(--positive);">STAGE 07</div>
                <div class="step-name">Conversation Analytics</div>
                <div class="step-modality">Trend & Escalation</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render technical specifications in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; font-weight: 700; color: var(--text); padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);">
                ENGINE SPECIFICATIONS
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-heading">Neural Models</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Acoustic Emotion</span>
                <span class="sidebar-spec-val">Wav2Vec2-SUPERB</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Speech-to-Text</span>
                <span class="sidebar-spec-val">Faster-Whisper (base)</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Text Sentiment</span>
                <span class="sidebar-spec-val">CardiffNLP RoBERTa</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Diarization</span>
                <span class="sidebar-spec-val">Pyannote Community-1</span>
            </div>
            <div class="sidebar-spec-row">
                <span class="sidebar-spec-key">Fusion Strategy</span>
                <span class="sidebar-spec-val">Decision-Level Arbitration</span>
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
