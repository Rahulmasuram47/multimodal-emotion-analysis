# Comprehensive Verification Test Plan

This document outlines end-to-end and unit verification scenarios for the **Multimodal Voice & Conversation Emotion Analysis System**, tracing each scenario to expected code behavior and empirical test status.

---

## 1. Test Verification Matrix

| Scenario | Input / Test Fixture | Expected Behavior (from Code) | Verification Status |
|---|---|---|---|
| **English Happy** | `sample_audio/happy_sample.wav` | Whisper STT transcribes positive text; Wav2Vec2/emotion2vec predicts `happy`; Multimodal fusion flags `agreement_positive`. | **PASS** (`test_tier3_validation.py`) |
| **English Sad** | `sample_audio/sad_sample.wav` | Whisper STT transcribes negative/neutral text; acoustic classifier predicts `sad`; Fusion flags `agreement_negative_low_arousal` or mismatch. | **PASS** (`test_tier3_validation.py`) |
| **English Neutral** | `sample_audio/neutral_sample.wav` | Whisper STT transcribes balanced dialogue; acoustic classifier predicts `neutral`; Fusion flags `agreement_neutral`. | **PASS** (`test_tier3_validation.py`) |
| **English Angry** | `sample_audio/angry_sample.wav` | Whisper STT transcribes negative/escalated text; acoustic classifier predicts `angry`; High arousal ($A \ge 0.60$); flags `agreement_negative_high_arousal`. | **PASS** (`test_tier3_validation.py`) |
| **CREMA-D Happy Clip** | `data/crema_d/*_HAP_*.wav` | Audio resampled to 16 kHz mono; predictions cached by SHA-256; mapped to `happy`. | **PASS** (`test_crema_d.py`) |
| **CREMA-D Sad Clip** | `data/crema_d/*_SAD_*.wav` | Resampled to 16 kHz mono; predictions cached; mapped to `sad`. | **PASS** (`test_crema_d.py`) |
| **CREMA-D Neutral Clip** | `data/crema_d/*_NEU_*.wav` | Resampled to 16 kHz mono; predictions cached; mapped to `neutral`. | **PASS** (`test_crema_d.py`) |
| **CREMA-D Angry Clip** | `data/crema_d/*_ANG_*.wav` | Resampled to 16 kHz mono; predictions cached; mapped to `angry`. | **PASS** (`test_crema_d.py`) |
| **Noisy Audio** | Audio with additive synthetic noise | Resampled cleanly; low acoustic reliability attenuates $w_{\text{eff, audio}}$, preventing spurious mismatch flags. | **PASS** (`test_robustness.py`) |
| **Silent Audio** | 0-amplitude or all-zero WAV buffer | Detected by silence/energy check; displays friendly banner "No speech detected" without crashing or traceback. | **PASS** (`test_speech_vad_resilience.py`) |
| **1-Second Clip** | Boundary 1.0s speech audio | Meets `VOICE_MIN_SEGMENT_SEC = 1.0`; processed normally through acoustic and STT pipeline. | **PASS** (`test_phase4e.py`) |
| **Sub-Second Clip (< 1.0s)** | 0.4s – 0.8s audio turn | Tagged with `⚠️ Short Segment` warning chip; excluded from trend OLS regression to prevent noisy slopes. | **PASS** (`test_phase4e.py`, `test_web_hygiene.py`) |
| **Corrupt File** | Corrupted header / random non-audio bytes | `AudioValidationError` caught; friendly alert "Unsupported or Corrupted Audio Format" displayed without traceback. | **PASS** (`test_web_hygiene.py`) |
| **Oversized File (> 50 MB)** | File exceeding `MAX_AUDIO_FILE_SIZE_MB` | Validation rejects file immediately; displays "Audio File Size Exceeded" without uploading to memory. | **PASS** (`test_web_hygiene.py`) |
| **Mic Permission Denied** | Headless or browser permission denied | Streamlit falls back cleanly; displays `FALLBACK_MICROPHONE_MESSAGE` advising file upload workflow. | **PASS** (`test_microphone_workflow.py`) |
| **Two-Speaker Audio** | Multi-speaker recording with `HF_TOKEN` | Pyannote assigns speaker labels (`SPEAKER_00`, `SPEAKER_01`); timeline renders speaker comparison charts. | **PASS** (`test_diarization.py`) |
| **Missing HF Token** | Unset `HF_TOKEN` or `HF_TOKEN=""` | Sidebar shows `✗ Not configured`; diarization safely disabled; speaker labeled `Unknown`; pipeline runs normally. | **PASS** (`test_diarization_badges.py`) |
| **HF_HUB_OFFLINE=1** | Network disconnected with cached models | Transformers load from local disk cache (`~/.cache/huggingface/`); no outbound network requests executed. | **PASS** (`test_ui_wiring.py`) |
| **Theme Switching** | Dark / Light theme toggle | CSS token override switches background, card panels, text colors; passes WCAG AA contrast (ratio $\ge 4.5:1$). | **PASS** (`test_ui_contracts.py`) |
| **Export Integrity** | JSON / CSV report generation | Timeline displays collapsed repetitive tokens, but exports retain 100% raw, uncollapsed original transcripts. | **PASS** (`test_tier3_regression.py`) |
| **Repetition Collapse** | Repetitive transcript (e.g. "Thank you. " $\times 40$) | Timeline collapses repeated runs into `... ×N (repetitive transcript)`; "very, very good" does NOT collapse. | **PASS** (`test_tier5_polish.py`) |
| **No Raw HTML Leakage** | UI card and meter rendering | `html_block()` sanitizes all rendered templates; zero `<div` or unrendered tags visible on page. | **PASS** (`test_tier5_polish.py`) |
| **Fresh Clone + Pytest** | Mocked unit test suite | Runs without datasets, audio files, or network tokens; executes in $< 20$ seconds. | **PASS** (482 passed, 13 skipped) |
| **Latency Spot Check** | Cold start vs. warm turn | Cold start model load: 1.2–3.5s; Warm turn on 3.0s clip: 0.95s ($\text{RTF} \approx 0.32$). | **PASS** (`evaluation/results/latency.md`) |
| **Robustness Spot Check** | SNR mixing (20/10/0 dB) & 8 kHz roundtrip | SNR calculation verified mathematically within 0.5 dB in unit test; full audio dataset run marked not yet measured. | **PASS (Tooling verified)** (`test_robustness.py`) |
