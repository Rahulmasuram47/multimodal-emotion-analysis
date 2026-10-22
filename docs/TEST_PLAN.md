# Comprehensive Verification Test Plan

This document outlines end-to-end and unit verification scenarios for the **Multimodal Voice & Conversation Emotion Analysis System**, tracing each scenario to expected code behavior and empirical test status.

> [!NOTE]
> **Historical Artifact Notice**: Screenshots, UI captures, and benchmark logs recorded prior to Phase 12 reflect the earlier Wav2Vec2-SUPERB fallback output. The active production backend is `emotion2vec`.

---

## 1. Test Verification Matrix

| Scenario | Input / Test Fixture | Expected Behavior (from Code) | Verification Status |
|---|---|---|---|
| **English Happy** | `sample_audio/happy_sample.wav` | Whisper STT transcribes positive text; emotion2vec predicts `happy`; Multimodal fusion flags `agreement_positive`. | **PASS** (`test_tier3_validation.py`) |
| **English Sad** | `sample_audio/sad_sample.wav` | Whisper STT transcribes negative/neutral text; acoustic classifier predicts `sad`; Fusion flags `agreement_negative_low_arousal` or mismatch. | **PASS** (`test_tier3_validation.py`) |
| **English Neutral** | `sample_audio/neutral_sample.wav` | Whisper STT transcribes balanced dialogue; acoustic classifier predicts `neutral`; Fusion flags `agreement_neutral`. | **PASS** (`test_tier3_validation.py`) |
| **English Angry** | `sample_audio/angry_sample.wav` | Whisper STT transcribes negative/escalated text; acoustic classifier predicts `angry`; High arousal ($A \ge 0.60$); flags `agreement_negative_high_arousal`. | **PASS** (`test_tier3_validation.py`) |
| **CREMA-D Dev Baseline** | 400 dev clips (9 actors) | Zero-shot emotion2vec evaluation; accuracy 94.00%, UAR 95.19%. | **PASS** (`test_crema_d.py`, `evaluation/results/crema_d_dev_baseline.json`) |
| **CREMA-D One-Shot Test** | 1,673 test clips (31 actors) | Single held-out pass with frozen head decision; accuracy 93.31%, UAR 94.88% (94.84% on answered). | **PASS** (`evaluation/results/crema_d_benchmark.json`) |
| **Latency Benchmark API** | `scripts/benchmark_latency.py --check-api` | Inspects all class methods and arities without loading model weights; confirms `SpeechToText.transcribe`, `SentimentAnalyzer.analyze`, `MultimodalFusion.fuse`. | **PASS** (`test_phase17c_bench_calib.py`) |
| **Calibration --run Guard** | `evaluation/calibration.py` | Without `--run`, prints help and writes 0 files; with `--run`, evaluates temperature scaling and ECE/NLL. | **PASS** (`test_phase17c_bench_calib.py`) |
| **Noisy Audio** | Audio with additive synthetic noise | Resampled cleanly; low acoustic reliability attenuates $w_{\text{eff, audio}}$, preventing spurious mismatch flags. | **PASS** (`test_robustness.py`) |
| **Silent Audio** | 0-amplitude or all-zero WAV buffer | Detected by silence/energy check; displays friendly banner "No speech detected" without crashing or traceback. | **PASS** (`test_speech_vad_resilience.py`) |
| **1-Second Clip** | Boundary 1.0s speech audio | Meets `VOICE_MIN_SEGMENT_SEC = 1.0`; processed normally through acoustic and STT pipeline. | **PASS** (`test_phase4e.py`) |
| **Sub-Second Clip (< 1.0s)** | 0.4s – 0.8s audio turn | Tagged with `⚠️ Short Segment` warning chip; excluded from trend OLS regression to prevent noisy slopes. | **PASS** (`test_phase4e.py`, `test_web_hygiene.py`) |
| **Corrupt File** | Corrupted header / random non-audio bytes | `AudioValidationError` caught; friendly alert "Unsupported or Corrupted Audio Format" displayed without traceback. | **PASS** (`test_web_hygiene.py`) |
| **Oversized File (> 50 MB)** | File exceeding `MAX_AUDIO_FILE_SIZE_MB` | Validation rejects file immediately; displays "Audio File Size Exceeded" without uploading to memory. | **PASS** (`test_web_hygiene.py`) |
| **Mic Permission Denied** | Headless or browser permission denied | Streamlit falls back cleanly; displays `FALLBACK_MICROPHONE_MESSAGE` advising file upload workflow. | **PASS** (`test_microphone_workflow.py`) |
| **Two-Speaker Audio** | Multi-speaker recording with `HF_TOKEN` | Pyannote assigns speaker labels (`SPEAKER_00`, `SPEAKER_01`); merges same-speaker turns with gap $\le 0.25$s. | **PASS** (`test_diarization.py`) |
| **Missing HF Token** | Unset `HF_TOKEN` or `HF_TOKEN=""` | Sidebar shows `✗ Not configured`; diarization safely disabled; speaker labeled `Unknown`; pipeline runs normally. | **PASS** (`test_diarization_badges.py`) |
| **HF_HUB_OFFLINE=1** | Network disconnected with cached models | Transformers load from local disk cache (`~/.cache/huggingface/`); no outbound network requests executed. | **PASS** (`test_ui_wiring.py`) |
| **Theme Switching** | Dark / Light theme toggle | CSS token override switches background, card panels, text colors; passes WCAG AA contrast (ratio $\ge 4.5:1$). | **PASS** (`test_ui_contracts.py`) |
| **Export Integrity** | JSON / CSV report generation | Timeline displays collapsed repetitive tokens, but exports retain 100% raw, uncollapsed original transcripts. | **PASS** (`test_tier3_regression.py`) |
| **Repetition Collapse** | Repetitive transcript (e.g. "Thank you. " $\times 40$) | Timeline collapses repeated runs into `... ×N (repetitive transcript)`; "very, very good" does NOT collapse. | **PASS** (`test_tier5_polish.py`) |
| **No Raw HTML Leakage** | UI card and meter rendering | `html_block()` sanitizes all rendered templates; zero `<div` or unrendered tags visible on page. | **PASS** (`test_tier5_polish.py`) |
| **Unit Test Suite** | Full mocked pytest suite | Executes completely offline on CPU without tokens or dataset downloads in $< 2.5$ minutes. | **PASS** (521 passed) |
