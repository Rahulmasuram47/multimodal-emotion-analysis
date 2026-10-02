"""
System transparency, fusion explainability, model specifications, and limitations panels.
"""

from typing import Any, Dict
import streamlit as st

SYSTEM_NOTICES_AND_LIMITATIONS: Dict[str, Any] = {
    "system_notice": (
        "This application is a decision-support heuristic designed to extract computational "
        "acoustic and linguistic features. Outputs are technical indicators, not clinical, "
        "psychological, psychiatric, or deception assessments."
    ),
    "heuristic_nature": "Outputs are algorithmic heuristics intended strictly for advisory decision-support.",
    "sarcasm_and_deception": (
        "The system measures computational modality disagreement (incongruence); "
        "it does NOT detect sarcasm, lies, truth, deception, concealment, or a speaker's inner psychological state."
    ),
    "human_in_the_loop": (
        "All automated outputs, recommendations, and metrics are strictly advisory. "
        "Final decisions, interventions, and assessments must always be made by qualified human professionals exercising domain judgment."
    ),
    "operational_limitations": [
        "Absence of Ground Truth: Performance metrics reflect runtime heuristics until empirical evaluation is conducted on labelled corpora.",
        "Pretrained Cross-Corpus Shift: Models were pretrained on external datasets; acoustic and domain shifts may affect accuracy.",
        "Acoustic Discretization: Categorizes segments into discrete labels (neutral, happy, angry, sad); complex affective states may not be fully represented.",
        "Sarcasm & Deception Non-Detection: Modality incongruence does not indicate dishonesty or deception.",
        "Audio Overlap & Acoustics: Overlapping speech, background noise, microphone differences, and reverberation alter acoustic contours.",
        "Linguistic & Dialect Diversity: Speech recognition and sentiment vary with accents, dialects, and speaking styles.",
        "Escalation Heuristic: Escalation indicates continuous acoustic/linguistic intensity rise, not psychological crisis.",
        "Attention Score Heuristic: Attention score is a triage prioritization metric, not a risk or danger index.",
        "Mandatory Human-in-the-Loop: Automated inferences must be supervised and validated by qualified personnel.",
    ],
}


def render_fusion_explainer() -> None:
    """Render explainable fusion architecture breakdown panel."""
    with st.expander("⚡ Explainable Multimodal Fusion Architecture", expanded=False):
        st.markdown(
            """
            ### Weighted Rule-Based Decision Arbitration
            The system combines acoustic features and text sentiment vectors through **decision-level fusion**:

            1. **Voice Emotion Prediction**: `superb/wav2vec2-base-superb-er` predicts discrete emotional categories (*neutral, happy, angry, sad*) with confidence probabilities.
            2. **Text Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest` classifies transcription semantics into (*positive, neutral, negative*).
            3. **Cross-Modal Directional Mapping**:
               - Voice categories map to directional valence: *Happy* → `increasing`, *Angry/Sad* → `decreasing`, *Neutral* → `stable`.
               - Text sentiments map directly: *Positive* → `increasing`, *Negative* → `decreasing`, *Neutral* → `stable`.
            4. **Arbitration Logic**:
               - **Agreement**: If acoustic direction equals linguistic direction, fused confidence is reinforced:
                 $$\\text{Conf}_{\\text{fused}} = \\min(1.0, 0.50 \\times \\text{Conf}_{\\text{voice}} + 0.50 \\times \\text{Conf}_{\\text{text}} + 0.10)$$
               - **Mismatch**: When directions conflict, the modality with higher confidence arbitrates the final decision, tagged with a mismatch warning.
            """,
            unsafe_allow_html=True,
        )


def render_system_notes() -> None:
    """Render operational deployment notes."""
    with st.expander("📝 Engineering Notes & Hardware Optimization", expanded=False):
        st.markdown(
            """
            - **CPU-First Execution**: Optimized for standard multi-core laptop CPU processors without CUDA hardware accelerators.
            - **INT8 Quantization**: Whisper Speech-to-Text inference runs in 8-bit integer quantization, minimizing working memory to under 2 GB.
            - **WAV Ingestion**: Audio converted to standardized 16,000 Hz mono PCM channels to match Wav2Vec2 transformer receptive fields.
            """
        )


def render_model_transparency_section() -> None:
    """Render neural model specifications and citation details."""
    with st.expander("🔬 Model Transparency & Engineering Specifications", expanded=False):
        st.markdown(
            """
            | Pipeline Stage | Model Architecture | Training Corpus | Quantization / Compute | Source |
            |---|---|---|---|---|
            | **Voice Emotion** | Wav2Vec2-Base-SUPERB | IEMOCAP Affective DB | PyTorch FP32 CPU | `superb/wav2vec2-base-superb-er` |
            | **Speech-to-Text** | Faster-Whisper Base | Multilingual Web Speech | CTranslate2 INT8 CPU | `openai/whisper-base` |
            | **Text Sentiment** | RoBERTa-Base | TweetEval Benchmark | PyTorch FP32 CPU | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
            | **Diarization** | Pyannote Segmentation 3.0 | VoxCeleb / DIHARD | ONNX / Torch CPU | `pyannote/speaker-diarization-community-1` |
            """
        )


def render_professional_limitations_section() -> None:
    """Render operational limitations and human-in-the-loop requirement."""
    with st.expander("⚠️ Operational Limitations & Human-in-the-Loop Requirement", expanded=False):
        st.markdown(
            """
            1. **Absence of Ground Truth**: Performance metrics reflect runtime heuristics until empirical evaluation is conducted on labelled corpora.
            2. **Pretrained Cross-Corpus Shift**: Models were pretrained on external datasets; acoustic and domain shifts may affect accuracy.
            3. **Acoustic Discretization**: Categorizes segments into discrete labels; complex affective states may not be fully represented.
            4. **Sarcasm & Deception Non-Detection**: Modality incongruence does not indicate dishonesty or deception.
            5. **Audio Overlap & Acoustics**: Overlapping speech, background noise, microphone differences, and reverberation alter acoustic contours.
            6. **Linguistic & Dialect Diversity**: Speech recognition and sentiment vary with accents, dialects, and speaking styles.
            7. **Mandatory Human-in-the-Loop**: Automated inferences must be supervised and validated by qualified personnel.
            """
        )
