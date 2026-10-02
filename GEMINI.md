# Multimodal Voice & Conversation Emotion Analysis System — Workspace Rules

## HARD CONSTRAINTS (Submitted Abstract; Never Violate)
1. **Front-end stays Streamlit**: No React/FastAPI replacement. HTML/CSS/JS may only be embedded through Streamlit components or injected CSS.
2. **Named model families stay**:
   - `Wav2Vec2` (`superb/wav2vec2-base-superb-er`) for voice emotion.
   - `Faster-Whisper` (`base` or `small` model, INT8 CPU) for Speech-to-Text.
   - `RoBERTa` (`cardiffnlp/twitter-roberta-base-sentiment-latest`) for text sentiment.
   - `Pyannote` (`pyannote/speaker-diarization-community-1`) for speaker diarization.
3. **Fusion stays transparent, weighted, rule-based and configurable**: Learned fusion (scikit-learn Logistic Regression) stays an experimental comparison baseline only, never the production path.
4. **Only two modalities**: Voice/audio and text. No video or facial emotion recognition.
5. **Never claim deception or intent detection**: The system never claims to detect sarcasm, lies, deception, truth, or actual psychological states. Modality disagreement is strictly an "analytical signal for review". Keep all safety disclaimers.
6. **Hardware & Environment Target**: Windows 10/11, Python 3.11, CPU-only inference. Must run comfortably within 8 GB RAM.
7. **Absolute Data Integrity**: Never fabricate data, labels, metrics, or model outputs. Never hardcode demonstration results. If data is missing or unannotated, the UI and reports explicitly say so.

## ENGINEERING RULES
- **Plan first, then implement**: List files you will touch before editing.
- **Centralized Configuration**: Everything tunable goes in `config.py` or a dedicated config file, never magic numbers scattered in application logic.
- **Test-Driven Rigor**: Every behavior change needs tests. `python -m pytest -q` must pass before finishing any phase.
- **Secret & Weight Protection**: Never commit model weights, tokens (`.env`, `HF_TOKEN`), or personal audio recordings.
- **Documentation Cadence**: Update `PROGRESS.md` at the end of every phase with what changed, why, and what was measured.
- **Immutable Specification**: Do NOT edit `PROJECT_SPEC.md` (it is locked and read-only).
- **Escalate Conflicts**: If a requirement conflicts with these constraints, stop and ask instead of choosing silently.
