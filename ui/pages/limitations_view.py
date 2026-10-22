"""
System Limitations & Operational Scope View.
Comprehensive documentation of algorithmic boundaries, model assumptions, and human-in-the-loop requirements.
"""

import streamlit as st
import config as _lim_config
from ui.utils import html_block

_VOICE_ARCH_MAP = {"emotion2vec": "emotion2vec+ base", "superb": "Wav2Vec2-SUPERB ER"}
_VOICE_CORPUS_MAP = {"emotion2vec": "Bilingual (ZH/EN) Affective Speech Corpus", "superb": "IEMOCAP Affective DB"}


def render_limitations_page() -> None:
    """Render the official system limitations, algorithmic bounds, and ethical notices."""
    _configured = getattr(_lim_config, "VOICE_BACKEND", "superb")
    _voice_arch = _VOICE_ARCH_MAP.get(_configured, _configured)
    _voice_corpus = _VOICE_CORPUS_MAP.get(_configured, "unknown corpus")

    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 0.5rem;">
    <div class="section-title">⚠️ Operational Scope & System Limitations</div>
    <div class="section-badge">DECISION-SUPPORT HEURISTICS · HUMAN-IN-THE-LOOP MANDATE</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
### 1. Nature of System Outputs (Algorithmic Heuristics)
All outputs produced by this system—including emotion labels, sentiment polarities, agreement flags, attention scores, and escalation trajectories—are **computational heuristics**. They represent pattern correlations extracted by machine learning models and must **never be interpreted as objective psychological truth**.

---

### 2. Explicit Non-Detection of Deception, Sarcasm, or Concealment
> **Crucial Analytical Boundary**:
> When the system flags **Modality Disagreement** (e.g., positive text combined with negative acoustic tone), this indicates only that the linguistic classifier and the acoustic classifier arrived at differing directional categories.
>
> **The system does NOT detect**:
> - Lies, deceit, or falsehoods.
> - Sarcasm, irony, or cynicism.
> - Hidden intentions or concealed motives.
> - A speaker's genuine affective or psychological state.
>
> Incongruence is an analytical indicator for human conversational review, not evidence of dishonesty.

---

### 3. Model Pretraining Bounds & Cross-Corpus Shift
- **Voice Emotion Classification (`{_voice_arch}`)**: Pretrained on {_voice_corpus}. Natural conversational speech exhibits lower acoustic arousal contrast, leading to potential misclassifications.
- **Speech-to-Text Transcription (`Faster-Whisper`)**: INT8 CPU transcription can degrade on heavily accented speech, rapid dialogue, technical jargon, or overlapping speakers.
- **Text Sentiment Analysis (`RoBERTa`)**: Pretrained on social media text benchmarks (TweetEval). Conversational spoken transcripts may contain disfluencies, filler words, or informal syntax that alter classifier probabilities.
- **Speaker Diarization (`Pyannote`)**: Performance depends on acoustic environment, microphone positioning, and absence of severe cross-talk.

---

### 4. Acoustic Discretization & Affective Nuance
Emotion is categorized into four discrete categorical labels (*neutral, happy, angry, sad*). Human emotional experience is continuous, multi-dimensional, and context-dependent. Nuanced affective states such as curiosity, fatigue, skepticism, or empathy are mapped to the nearest discrete category or labeled neutral.

---

### 5. Acoustic Artifacts, Noise & Dialects
- **Acoustic Distortions**: Background noise, reverberation, microphone clipping, and distance variations distort acoustic pitch and spectral centroids.
- **Linguistic and Dialectal Diversity**: Prosodic norms (pitch range, syllable timing) vary across dialects, cultures, and native languages. A high-energy conversational style should not be conflated with anger.

---

### 6. Mandatory Human-in-the-Loop Requirement
**Automated inferences must remain advisory.** Any decisions, interventions, or operational workflows must be reviewed, supervised, and approved by qualified human domain specialists exercising professional judgment in the relevant context.
"""
    )
