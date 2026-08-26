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


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multimodal Emotion Analysis",
    page_icon="🎙️",
    layout="wide",
)


# ---------------------------------------------------------
# Load ML pipeline only once
# ---------------------------------------------------------

@st.cache_resource
def load_analyzer():
    """Load the complete ML pipeline once."""
    return ConversationAnalyzer()


# ---------------------------------------------------------
# Fusion display
# ---------------------------------------------------------

def show_fusion_status(fusion):
    agreement = fusion.get("agreement", False)
    mismatch = fusion.get("mismatch", False)

    confidence = float(
        fusion.get("combined_confidence", 0.0)
    )

    if mismatch:
        st.warning(
            f"⚠️ Voice/Text mismatch detected — "
            f"combined confidence: {confidence:.1%}"
        )

    elif agreement:
        st.success(
            f"✅ Voice and text agree — "
            f"combined confidence: {confidence:.1%}"
        )

    else:
        st.info(
            f"Combined confidence: {confidence:.1%}"
        )

    interpretation = fusion.get("interpretation")

    if interpretation:
        st.caption(interpretation)


# ---------------------------------------------------------
# Display analysis results
# ---------------------------------------------------------

def display_results(result):

    st.header("📊 Analysis Results")

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Language",
            result.get("language", "Unknown").upper()
        )

    with col2:
        st.metric(
            "Duration",
            f"{result.get('duration_seconds', 0):.2f}s"
        )

    with col3:
        st.metric(
            "Segments",
            len(result.get("segments", []))
        )

    with col4:
        probability = float(
            result.get(
                "language_probability",
                0.0
            )
        )

        st.metric(
            "Language Confidence",
            f"{probability:.1%}"
        )

    # -----------------------------------------------------
    # Transcript
    # -----------------------------------------------------

    st.subheader("📝 Full Transcript")

    st.info(
        result.get(
            "full_transcript",
            "No transcript available."
        )
    )

    # -----------------------------------------------------
    # Emotional Trend
    # -----------------------------------------------------

    trend = result.get(
        "emotional_trend",
        {}
    )

    st.subheader("📈 Emotional Trend")

    trend_col1, trend_col2, trend_col3 = st.columns(3)

    with trend_col1:

        trend_name = (
            trend.get(
                "trend",
                "unknown"
            )
            .replace("_", " ")
            .title()
        )

        st.metric(
            "Trend",
            trend_name
        )

    with trend_col2:

        change = float(
            trend.get(
                "change",
                0.0
            )
        )

        st.metric(
            "Intensity Change",
            f"{change:+.4f}"
        )

    with trend_col3:

        escalation = result.get(
            "escalation",
            {}
        )

        detected = escalation.get(
            "escalation_detected",
            False
        )

        st.metric(
            "Escalation",
            "Detected" if detected else "Not detected"
        )

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

    escalation = result.get(
        "escalation",
        {}
    )

    if escalation.get("reason"):

        st.caption(
            f"Escalation analysis: "
            f"{escalation['reason']}"
        )

    # -----------------------------------------------------
    # Segment Analysis
    # -----------------------------------------------------

    st.subheader("🎙️ Conversation Segments")

    segments = result.get(
        "segments",
        []
    )

    for segment in segments:

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

        title = (
            f"Segment "
            f"{segment.get('segment_index', '?')} "
            f"| {start:.2f}s → {end:.2f}s"
        )

        with st.expander(
            title,
            expanded=True
        ):

            st.write(
                f"**Transcript:** "
                f"{segment.get('transcript', '')}"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:

                st.metric(
                    "Voice Emotion",
                    str(
                        voice.get(
                            "emotion",
                            "unknown"
                        )
                    ).title(),
                    f"{float(voice.get('confidence', 0.0)):.1%}"
                )

            with c2:

                st.metric(
                    "Text Sentiment",
                    str(
                        sentiment.get(
                            "sentiment",
                            "unknown"
                        )
                    ).title(),
                    f"{float(sentiment.get('confidence', 0.0)):.1%}"
                )

            with c3:

                st.metric(
                    "Multimodal Confidence",
                    f"{float(fusion.get('combined_confidence', 0.0)):.1%}"
                )

            with c4:

                st.metric(
                    "Intensity",
                    f"{intensity:.1%}"
                )

            show_fusion_status(
                fusion
            )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

st.title(
    "🎙️ Multimodal Emotion Analysis"
)

st.write(
    "Analyze spoken audio using voice emotion, "
    "speech-to-text, text sentiment, multimodal "
    "fusion, and conversation-level emotional trends."
)

st.divider()

# ---------------------------------------------------------
# Upload audio
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📁 Upload an audio file",
    type=[
        "wav",
        "flac",
        "mp3",
        "ogg",
        "m4a"
    ],
    help=(
        "For the first test, use "
        "sample_audio/real_voice.wav"
    )
)

# ---------------------------------------------------------
# Analyze button
# ---------------------------------------------------------

analyze_clicked = st.button(
    "🔎 Analyze Audio",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True
)


# ---------------------------------------------------------
# Run analysis
# ---------------------------------------------------------

if analyze_clicked and uploaded_file is not None:

    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".wav"
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_audio:

            temp_audio.write(
                uploaded_file.getbuffer()
            )

            temp_path = temp_audio.name

        with st.spinner(
            "Running complete multimodal analysis..."
        ):

            analyzer = load_analyzer()

            result = analyzer.analyze(
                temp_path
            )

        display_results(
            result
        )

    except Exception as exc:

        st.error(
            "❌ Analysis failed."
        )

        st.exception(
            exc
        )

    finally:

        if temp_path:

            try:
                os.remove(
                    temp_path
                )

            except OSError:
                pass


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Multimodal Emotion Analysis System | "
    "Streamlit Dashboard"
)