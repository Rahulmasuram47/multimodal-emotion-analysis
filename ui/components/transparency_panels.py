import json
from pathlib import Path
from typing import Any, Dict
import streamlit as st

import config

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
    """Render explainable fusion architecture breakdown panel strictly reflecting app/fusion.py."""
    with st.expander("⚡ Explainable Multimodal Fusion Architecture", expanded=False):
        st.markdown(
            r"""
            ### Dynamic Reliability-Weighted Rule Arbitration
            The system combines acoustic features and text sentiment vectors through **transparent, rule-based decision-level fusion**:

            1. **Voice Emotion Prediction**: `superb/wav2vec2-base-superb-er` outputs discrete class probabilities (*neutral, happy, angry, sad*) with acoustic confidence $c_{\text{audio}}$ and acoustic arousal $A_{\text{audio}}$ (from `config.VOICE_EMOTION_AROUSAL_MAP`).
            2. **Text Sentiment Analysis**: `cardiffnlp/twitter-roberta-base-sentiment-latest` classifies transcript semantics into (*positive, neutral, negative*) with text confidence $c_{\text{text}}$.
            3. **Cross-Modal Directional Mapping**:
               - Voice categories map to broad directions: *Happy* → `positive`, *Angry/Sad* → `negative`, *Neutral* → `neutral`.
               - Text sentiments map directly: *Positive* → `positive`, *Negative* → `negative`, *Neutral* → `neutral`.
            4. **Reliability-Aware Dynamic Weighting**:
               - Effective weights adapt to acoustic quality / duration / probability margin ($r_{\text{audio}}$) and speech-to-text reliability ($r_{\text{text}}$):
            """
        )
        st.latex(r"w_{\text{eff, audio}} = \frac{w_{\text{base, audio}} \cdot r_{\text{audio}}}{w_{\text{base, audio}} \cdot r_{\text{audio}} + w_{\text{base, text}} \cdot r_{\text{text}}}")
        st.latex(r"w_{\text{eff, text}} = \frac{w_{\text{base, text}} \cdot r_{\text{text}}}{w_{\text{base, audio}} \cdot r_{\text{audio}} + w_{\text{base, text}} \cdot r_{\text{text}}}")
        st.markdown(
            r"""
            5. **Combined Confidence (Evidence Strength)**:
               - Strictly a reliability-weighted convex combination:
            """
        )
        st.latex(r"c_{\text{combined}} = w_{\text{eff, audio}} \cdot c_{\text{audio}} + w_{\text{eff, text}} \cdot c_{\text{text}}")
        st.markdown(
            r"""
               - Bounded in $[\min(c_{\text{audio}}, c_{\text{text}}), \max(c_{\text{audio}}, c_{\text{text}})]$. Modality disagreement (weak or strong) receives **no artificial boost**; uncertainty in either modality directly attenuates the combined confidence.
            6. **Declarative Pattern Rules** (derived strictly from `app/fusion.py` rule IDs):
               - `agreement_negative_high_arousal`: Voice and text both negative with high acoustic arousal ($A_{\text{audio}} \ge 0.60$).
               - `agreement_negative_low_arousal`: Voice and text both negative with low acoustic arousal ($A_{\text{audio}} < 0.60$).
               - `agreement_positive`: Voice and text both indicate positive direction.
               - `agreement_neutral`: Voice and text both indicate neutral direction.
               - `mismatch_positive_text_negative_voice`: Positive text with negative voice tone ($c_{\text{audio}}, c_{\text{text}} \ge \theta$).
               - `mismatch_negative_text_positive_voice`: Negative text with positive voice tone ($c_{\text{audio}}, c_{\text{text}} \ge \theta$).
               - `mismatch_generic`: Conflicting directions with both confidences $\ge \theta$.
               - `weak_disagreement`: Opposing or differing directions with at least one modality confidence below threshold $\theta$.
               - `abstention_low_reliability`: Triggered when combined reliability $r_{\text{combined}} < \tau_{\text{abstain}}$ (default $0.35$).
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
    """Render neural model specifications and citation details derived from config."""
    whisper_size = st.session_state.get("whisper_model_size", getattr(config, "STT_MODEL_SIZE", "base"))
    voice_model = getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er")
    text_model = getattr(config, "TEXT_SENTIMENT_MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")
    diar_model = getattr(config, "DIARIZATION_MODEL_NAME", "pyannote/speaker-diarization-community-1")

    with st.expander("🔬 Model Transparency & Engineering Specifications", expanded=False):
        st.markdown(
            f"""
            | Pipeline Stage | Model Architecture | Training Corpus | Quantization / Compute | Source ID |
            |---|---|---|---|---|
            | **Voice Emotion** | Wav2Vec2-Base-SUPERB | IEMOCAP Affective DB | PyTorch FP32 CPU | `{voice_model}` |
            | **Speech-to-Text** | Faster-Whisper ({whisper_size}) | Multilingual Web Speech | CTranslate2 INT8 CPU | `openai/whisper-{whisper_size}` |
            | **Text Sentiment** | RoBERTa-Base | TweetEval Benchmark | PyTorch FP32 CPU | `{text_model}` |
            | **Diarization** | Pyannote Community-1 | VoxCeleb / DIHARD | PyTorch CPU | `{diar_model}` |
            """
        )


def _has_ground_truth_eval_results() -> bool:
    """Check whether local evaluation metrics file exists with valid annotated results."""
    metrics_path = Path("evaluation/results/evaluation_metrics.json")
    if not metrics_path.exists():
        return False
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return bool(data.get("total_samples", 0) >= getattr(config, "EVAL_MIN_N", 5) and data.get("has_ground_truth", False))
    except Exception:
        return False


def render_professional_limitations_section() -> None:
    """Render operational limitations and human-in-the-loop requirement."""
    if _has_ground_truth_eval_results():
        val_status = "cross-corpus evaluation against local ground truth is reported in the Validation Tab."
    else:
        val_status = "No local evaluation results yet (requires ≥ 5 annotated ground-truth recordings)."

    with st.expander("⚠️ Operational Limitations & Human-in-the-Loop Requirement", expanded=False):
        st.markdown(
            f"""
            1. **Absence of Broad In-Domain Ground Truth**: Model weights were trained on external benchmarks (IEMOCAP, TweetEval); {val_status}
            2. **Pretrained Cross-Corpus Acoustic Shift**: Pretrained acoustic priors may demonstrate domain bias; voice diagnostics and collapse monitoring flag low discrimination.
            3. **Discrete Affective Mapping**: Emotion is discretized into 4 acoustic categories (*neutral, happy, angry, sad*); continuous valence-arousal mapping provides richer gradient nuance.
            4. **Sarcasm & Deception Non-Detection**: Modality incongruence indicates computational disagreement between text and acoustic classifiers; it does NOT assess honesty, deception, or internal states.
            5. **Audio Overlap & Acoustics**: Overlapping speech, reverberation, and background noise degrade diarization and STT word error rates.
            6. **Linguistic & Dialect Diversity**: Speech recognition and text sentiment vary with accents, dialects, and speaking idioms.
            7. **Mandatory Human-in-the-Loop**: All automated inferences are advisory heuristics; decisions must be validated by human evaluators.
            """
        )

