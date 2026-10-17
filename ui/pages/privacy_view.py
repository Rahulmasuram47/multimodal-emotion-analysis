"""
Privacy Policy & Data Handling View.
Articulates data governance, local processing guarantees, and educational/research scope.
"""

import streamlit as st
from ui.utils import html_block


def render_privacy_page() -> None:
    """Render the official privacy policy and data governance overview."""
    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 0.5rem;">
    <div class="section-title">🔒 Privacy Policy & Data Handling Architecture</div>
    <div class="section-badge">EDUCATIONAL & RESEARCH USE ONLY · LOCAL INFERENCE</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
### 1. Educational & Academic Research Scope Only
This system is an **academic research prototype** developed for multimodal emotion analysis and conversation dynamics study.

> **Strict Non-Application Notice**:
> This software is **NOT** designed, certified, or intended for:
> - Clinical psychiatric, psychological, or medical diagnostics.
> - Employment, hiring, recruitment, promotion, or workplace performance evaluation.
> - Employee surveillance, monitoring, or disciplinary assessment.
> - Legal interrogation, polygraph testing, credibility scoring, or deception detection.
> - Any high-stakes decision affecting an individual's rights, livelihood, health, or freedom.

---

### 2. Audio Processing & Data Retention
- **Local CPU Processing**: All acoustic feature extraction, transcription, and sentiment classification execute entirely within the local host environment. No audio signals or transcripts are transmitted to third-party APIs.
- **Ephemeral Storage**: Uploaded files and recorded microphone buffers are held in temporary storage strictly for the duration of inference. The pipeline unlinks (deletes) all temporary audio buffers immediately upon completion or failure.
- **No Involuntary Retention**: Audio recordings and transcripts are never persisted to a server database or external cache without explicit user export.
- **Session Isolation**: Session states exist in browser memory and reset completely when the session is refreshed, reset, or closed.

---

### 3. Speech Transcripts & Logging Hygiene
- **No Content Logging**: Transcripts, text sentiments, and acoustic features are processed in memory and are **never logged at INFO or DEBUG levels** to system stdout/stderr.
- **Anonymized Analytics**: The system extracts computational metrics (turn-level duration, arousal, sentiment polarity) without storing personally identifiable metadata.

---

### 4. Credentials & Secret Governance
- **Server-Side Token Isolation**: Gated model authentication (`HF_TOKEN`) is managed exclusively server-side via environment variables or secret managers.
- **Zero Token Leakage**: Credentials are never echoed in UI displays, client-side session states, error dialogs, or analytical export bundles.

---

### 5. Inquiries & Academic Contact
For questions regarding the research methodology, data handling protocols, or reproducibility benchmarks, please consult the project repository documentation.
"""
    )
