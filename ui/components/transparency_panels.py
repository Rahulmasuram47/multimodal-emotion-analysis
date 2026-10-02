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
            ### Dynamic Reliability-Weighted Rule Arbitration
            The system combines acoustic features and text sentiment vectors through **transparent decision-level fusion**:

            1. **Voice Emotion Prediction**: `superb/wav2vec2-base-superb-er` predicts discrete emotional categories (*neutral, happy, angry, sad*) with confidence probabilities and prosodic arousal $A_{\\text{prosody}}$.
            2. **Text Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest` classifies transcription semantics into (*positive, neutral, negative*) with confidence probabilities.
            3. **Cross-Modal Directional Mapping**:
               - Voice categories map to valence $[-1, 1]$ and direction: *Happy* → `increasing`, *Angry/Sad* → `decreasing`, *Neutral* → `stable`.
               - Text sentiments map directly: *Positive* → `increasing`, *Negative* → `decreasing`, *Neutral* → `stable`.
            4. **Reliability-Aware Dynamic Weighting**:
               - Segment weights dynamically adapt to acoustic signal-to-noise / segment duration ($r_{\\text{audio}}$) and speech-to-text log-probability ($r_{\\text{text}}$):
                 $$w_{\\text{eff, audio}} = \\frac{w_{\\text{base, audio}} \\cdot r_{\\text{audio}}}{w_{\\text{base, audio}} \\cdot r_{\\text{audio}} + w_{\\text{base, text}} \\cdot r_{\\text{text}}}$$
            5. **Declarative Rule Arbitration**:
               - **R1 Agreement Reinforcement**: Concordant voice and text reinforce confidence with an agreement bonus.
               - **R2/R3 Asymmetric Modality Dominance**: Modality with superior effective weight and confidence arbitrates the classification.
               - **R4 Neutral Mediation**: Mild or neutral cues mediate extreme unilateral classifications.
               - **R5 Modality Incongruence Flagging**: Conflicting directions trigger a review signal for human auditing.
               - **R6 Reliability Gating / Abstention**: If combined reliability falls below threshold, the system flags *"Insufficient reliability — manual review recommended"*.
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
            1. **Absence of Broad In-Domain Ground Truth**: Model weights were trained on external benchmarks (IEMOCAP, TweetEval); cross-corpus evaluation against local ground truth is reported in the Validation Tab.
            2. **Pretrained Cross-Corpus Acoustic Shift**: Pretrained acoustic priors may demonstrate domain bias; voice diagnostics and collapse monitoring flag low discrimination.
            3. **Discrete Affective Mapping**: Emotion is discretized into 4 acoustic categories (*neutral, happy, angry, sad*); continuous valence-arousal mapping provides richer gradient nuance.
            4. **Sarcasm & Deception Non-Detection**: Modality incongruence indicates computational disagreement between text and acoustic classifiers; it does NOT assess honesty, deception, or internal states.
            5. **Audio Overlap & Acoustics**: Overlapping speech, reverberation, and background noise degrade diarization and STT word error rates.
            6. **Linguistic & Dialect Diversity**: Speech recognition and text sentiment vary with accents, dialects, and speaking idioms.
            7. **Mandatory Human-in-the-Loop**: All automated inferences are advisory heuristics; decisions must be validated by human evaluators.
            """
        )
