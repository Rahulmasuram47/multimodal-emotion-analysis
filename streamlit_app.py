"""
Streamlit dashboard for the Multimodal Emotion Analysis System.

Run from the project root:

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
"""

import os
import tempfile
from pathlib import Path

import streamlit as st

from app.conversation_analysis import ConversationAnalyzer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal Emotion Analysis",
    page_icon="🎙️",
    layout="wide",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    .segment-label {
        font-size: 0.85rem;
        opacity: 0.65;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .transcript-box {
        padding: 0.9rem;
        border-radius: 0.6rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        margin-bottom: 1rem;
    }

    .footer {
        text-align: center;
        opacity: 0.55;
        margin-top: 2rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD ML PIPELINE ONCE
# =========================================================

@st.cache_resource
def load_analyzer():
    """Load the complete ML pipeline once."""
    return ConversationAnalyzer()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def emotion_icon(emotion: str) -> str:
    """Return a simple icon for an emotion."""

    emotion = emotion.lower()

    icons = {
        "angry": "😠",
        "sad": "😟",
        "happy": "😊",
        "neutral": "😐",
        "neu": "😐",
        "ang": "😠",
        "hap": "😊",
    }

    return icons.get(
        emotion,
        "🎙️"
    )


def direction_label(direction: str) -> str:
    """Format emotional direction for display."""

    if not direction:
        return "Unknown"

    return str(
        direction
    ).replace(
        "_",
        " "
    ).title()


def show_fusion_status(fusion):
    """
    Display the final fusion interpretation.

    IMPORTANT:
    The directions displayed here come directly from
    fusion.py.

    Streamlit does NOT independently determine whether
    emotions agree or disagree.
    """

    agreement = bool(
        fusion.get(
            "agreement",
            False
        )
    )

    mismatch = bool(
        fusion.get(
            "mismatch",
            False
        )
    )

    confidence = float(
        fusion.get(
            "combined_confidence",
            0.0
        )
    )

    audio_direction = direction_label(
        fusion.get(
            "audio_direction",
            "unknown"
        )
    )

    text_direction = direction_label(
        fusion.get(
            "text_direction",
            "unknown"
        )
    )

    interpretation = fusion.get(
        "interpretation",
        ""
    )

    # -----------------------------------------------------
    # Agreement
    # -----------------------------------------------------

    if agreement:

        st.success(
            "✓ Voice and text agree"
        )

    # -----------------------------------------------------
    # Mismatch
    # -----------------------------------------------------

    elif mismatch:

        st.warning(
            "⚠️ Voice/Text mismatch detected"
        )

    # -----------------------------------------------------
    # Weak disagreement
    # -----------------------------------------------------

    else:

        st.info(
            "ℹ️ Weak or low-confidence modality disagreement"
        )

    # -----------------------------------------------------
    # Direction information
    # -----------------------------------------------------

    st.caption(
        f"Voice direction: **{audio_direction}** · "
        f"Text direction: **{text_direction}** · "
        f"Fusion confidence: "
        f"**{confidence:.1%}**"
    )

    if interpretation:

        st.caption(
            interpretation
        )


# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(
    result,
    audio_bytes=None,
    audio_format="audio/wav",
):
    """Display complete analysis results."""

    # =====================================================
    # ANALYSIS OVERVIEW
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Analysis Overview'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    # -----------------------------------------------------
    # Language
    # -----------------------------------------------------

    with col1:

        language = str(
            result.get(
                "language",
                "Unknown"
            )
        ).upper()

        st.metric(
            "LANGUAGE",
            language
        )

    # -----------------------------------------------------
    # Duration
    # -----------------------------------------------------

    with col2:

        duration = float(
            result.get(
                "duration_seconds",
                0.0
            )
        )

        st.metric(
            "DURATION",
            f"{duration:.2f}s"
        )

    # -----------------------------------------------------
    # Segments
    # -----------------------------------------------------

    with col3:

        segments = result.get(
            "segments",
            []
        )

        st.metric(
            "SEGMENTS",
            len(segments)
        )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    with col4:

        escalation = result.get(
            "escalation",
            {}
        )

        detected = bool(
            escalation.get(
                "escalation_detected",
                False
            )
        )

        st.metric(
            "ESCALATION",
            "Detected"
            if detected
            else "Not detected"
        )

    # =====================================================
    # AUDIO
    # =====================================================

    if audio_bytes:

        st.markdown(
            '<div class="section-title">'
            '🔊 Audio'
            '</div>',
            unsafe_allow_html=True
        )

        st.audio(
            audio_bytes,
            format=audio_format
        )

    # =====================================================
    # FULL TRANSCRIPT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📝 Full Transcript'
        '</div>',
        unsafe_allow_html=True
    )

    transcript = result.get(
        "full_transcript",
        ""
    )

    if transcript:

        st.info(
            transcript
        )

    else:

        st.warning(
            "No transcript available."
        )

    language_probability = float(
        result.get(
            "language_probability",
            0.0
        )
    )

    st.caption(
        f"Detected language: "
        f"{language} · "
        f"Language confidence: "
        f"{language_probability:.1%}"
    )

    # =====================================================
    # EMOTIONAL TREND
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📈 Emotional Trend'
        '</div>',
        unsafe_allow_html=True
    )

    trend = result.get(
        "emotional_trend",
        {}
    )

    trend_col1, trend_col2, trend_col3 = (
        st.columns(3)
    )

    # -----------------------------------------------------
    # Trend
    # -----------------------------------------------------

    with trend_col1:

        trend_name = (
            str(
                trend.get(
                    "trend",
                    "unknown"
                )
            )
            .replace(
                "_",
                " "
            )
            .title()
        )

        st.metric(
            "OVERALL TREND",
            trend_name
        )

    # -----------------------------------------------------
    # Change
    # -----------------------------------------------------

    with trend_col2:

        change = float(
            trend.get(
                "change",
                0.0
            )
        )

        st.metric(
            "INTENSITY CHANGE",
            f"{change:+.4f}"
        )

    # -----------------------------------------------------
    # Escalation
    # -----------------------------------------------------

    with trend_col3:

        st.metric(
            "ESCALATION",
            "Detected"
            if detected
            else "Not detected"
        )

    # -----------------------------------------------------
    # Intensity chart
    # -----------------------------------------------------

    scores = trend.get(
        "intensity_scores",
        []
    )

    if scores:

        st.line_chart(
            {
                "Emotional Intensity": scores
            }
        )

    escalation_reason = escalation.get(
        "reason",
        ""
    )

    if escalation_reason:

        st.caption(
            "Escalation analysis: "
            + escalation_reason
        )

    # =====================================================
    # CONVERSATION SEGMENTS
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎙️ Conversation Segments'
        '</div>',
        unsafe_allow_html=True
    )

    for segment in segments:

        # -------------------------------------------------
        # Basic segment data
        # -------------------------------------------------

        index = segment.get(
            "segment_index",
            "?"
        )

        start = float(
            segment.get(
                "start_seconds",
                0.0
            )
        )

        end = float(
            segment.get(
                "end_seconds",
                0.0
            )
        )

        transcript = segment.get(
            "transcript",
            ""
        )

        voice = segment.get(
            "voice_emotion",
            {}
        )

        sentiment = segment.get(
            "text_sentiment",
            {}
        )

        fusion = segment.get(
            "fusion",
            {}
        )

        intensity = float(
            segment.get(
                "intensity",
                0.0
            )
        )

        # -------------------------------------------------
        # Segment title
        # -------------------------------------------------

        title = (
            f"Segment {index} · "
            f"{start:.2f}s → {end:.2f}s"
        )

        with st.expander(
            title,
            expanded=True
        ):

            # =============================================
            # TRANSCRIPT
            # =============================================

            st.markdown(
                '<div class="segment-label">'
                'Transcript'
                '</div>',
                unsafe_allow_html=True
            )

            st.write(
                transcript
            )

            # =============================================
            # METRICS
            # =============================================

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            # -------------------------------------------------
            # Voice emotion
            # -------------------------------------------------

            with c1:

                emotion = str(
                    voice.get(
                        "emotion",
                        "unknown"
                    )
                )

                voice_confidence = float(
                    voice.get(
                        "confidence",
                        0.0
                    )
                )

                st.metric(
                    "🎙️ Voice Emotion",
                    (
                        emotion_icon(
                            emotion
                        )
                        + " "
                        + emotion.title()
                    ),
                    f"{voice_confidence:.1%}"
                )

            # -------------------------------------------------
            # Text sentiment
            # -------------------------------------------------

            with c2:

                sentiment_name = str(
                    sentiment.get(
                        "sentiment",
                        "unknown"
                    )
                )

                sentiment_confidence = float(
                    sentiment.get(
                        "confidence",
                        0.0
                    )
                )

                st.metric(
                    "📝 Text Sentiment",
                    sentiment_name.title(),
                    f"{sentiment_confidence:.1%}"
                )

            # -------------------------------------------------
            # Fusion
            # -------------------------------------------------

            with c3:

                fusion_confidence = float(
                    fusion.get(
                        "combined_confidence",
                        0.0
                    )
                )

                st.metric(
                    "🔗 Fusion Confidence",
                    f"{fusion_confidence:.1%}"
                )

            # -------------------------------------------------
            # Intensity
            # -------------------------------------------------

            with c4:

                st.metric(
                    "📊 Intensity",
                    f"{intensity:.1%}"
                )

            # =============================================
            # FUSION STATUS
            # =============================================

            show_fusion_status(
                fusion
            )


# =========================================================
# MAIN APPLICATION
# =========================================================

st.markdown(
    '<div class="main-title">'
    '🎙️ Multimodal Emotion Analysis'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze spoken conversations using voice emotion, '
    'speech-to-text, text sentiment, multimodal fusion, '
    'and emotional trend analysis.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🎧 Analyze Audio'
    '</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=[
        "wav",
        "flac",
        "mp3",
        "ogg",
        "m4a",
    ],
    help=(
        "Upload a clear speech recording. "
        "WAV files are recommended for testing."
    ),
)


# =========================================================
# FILE INFORMATION
# =========================================================

if uploaded_file is not None:

    file_size_mb = (
        uploaded_file.size
        / (1024 * 1024)
    )

    st.caption(
        f"Selected file: **{uploaded_file.name}** · "
        f"{file_size_mb:.2f} MB"
    )


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_clicked = st.button(
    "🔎 Analyze Audio",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)


# =========================================================
# RUN ANALYSIS
# =========================================================

if (
    analyze_clicked
    and uploaded_file is not None
):

    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".wav"
    )

    temp_path = None

    try:

        # -------------------------------------------------
        # Save uploaded audio temporarily
        # -------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_audio:

            temp_audio.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_audio.name

        # -------------------------------------------------
        # Run pipeline
        # -------------------------------------------------

        with st.spinner(
            "Running complete multimodal analysis..."
        ):

            analyzer = load_analyzer()

            result = analyzer.analyze(
                temp_path
            )

        st.success(
            "Analysis completed successfully."
        )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        display_results(
            result,
            audio_bytes=uploaded_file.getvalue(),
            audio_format=(
                uploaded_file.type
                or "audio/wav"
            ),
        )

    except Exception as exc:

        st.error(
            "❌ Analysis failed."
        )

        st.exception(
            exc
        )

    finally:

        # -------------------------------------------------
        # Remove temporary file
        # -------------------------------------------------

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except OSError:

                pass


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'Multimodal Emotion Analysis System · '
    'Streamlit Dashboard'
    '</div>',
    unsafe_allow_html=True
)