# 5-Minute Evaluation Demonstration Script

This script provides an exact, step-by-step 5-minute walkthrough for demonstrating the **Multimodal Voice & Conversation Emotion Analysis System** during academic presentations, viva examinations, and peer reviews.

---

## ⏱️ Pre-Demo Warm-Up (Crucial Step)

> [!IMPORTANT]
> **Pre-Warm the ML Pipeline**:
> On CPU-only environments, the first inference run instantiates PyTorch transformer weights, Faster-Whisper CTranslate2 models, and RoBERTa pipelines into memory. This initial cold start can take **60 to 90 seconds**.
> Subsequent runs reuse cached memory resources via `@st.cache_resource`, dropping execution time to **15 to 30 seconds**.
> **Action**: Before beginning your presentation, click "Analyze a clip ▶" once on `sample_audio/neutral_sample.wav` so all models reside warm in RAM.

---

## 🎙️ Demonstration Walkthrough (5 Minutes)

### Minute 0:00 – 1:00: Problem Statement & Multimodal Rationale
- **Spoken Talking Points**:
  - *"In spoken human communication, emotion is split across two channels: how something is said (prosody, vocal pitch, acoustic energy) and what is said (the literal lexical semantics). Unimodal systems miss cross-channel incongruence."*
  - Point to the top brand banner displaying the three specialized models:
    - **Voice Emotion**: Wav2Vec2 / emotion2vec acoustic transformer.
    - **Speech-to-Text**: Faster-Whisper (INT8 quantized CPU inference).
    - **Text Sentiment**: Twitter-RoBERTa sentiment classifier.
    - **Speaker ID**: Pyannote diarization (active when `HF_TOKEN` is configured).

### Minute 1:00 – 2:00: Ingestion & Primary Analysis
- **Action**:
  1. In the "📂 Audio Ingestion & Analysis" panel, switch to the **"🎵 Pick Sample Audio"** tab.
  2. Select `rising_tone.wav` (Escalating Multimodal Dynamics) or `happy_sample.wav`.
  3. Click **"LOAD SAMPLE CLIP 📥"**.
  4. Point out the primary action button above the fold: click **"Analyze a clip ▶"**.
- **Spoken Talking Points**:
  - Show the live execution status bar reporting turn-by-turn stage milestones.

### Minute 2:00 – 3:00: Interpreting Results & Modality Cards
- **Action**: Direct attention to the top **Verdict Hero**:
  - **Verdict Summary**: Point out the dominant acoustic and linguistic classifications.
  - **Attention Score**: Explain that the score is a *triage heuristic* for review priority, not a psychological risk metric.
  - **Three Modality Cards**:
    - *Voice Emotion*: Shows predicted discrete emotion, acoustic confidence $c_{\text{audio}}$, and acoustic arousal.
    - *Text Sentiment*: Shows RoBERTa polarity and text confidence $c_{\text{text}}$.
    - *Fused Affect*: Shows reliability-weighted direction and combined confidence $c_{\text{combined}}$.

### Minute 3:00 – 4:00: Cross-Modal Arbitration & Dynamic Recompute
- **Action**:
  1. Scroll down to the **Detailed Analysis & Evidence Tabs**:
     - **💬 Segments**: Point out turn-level agreement tags, ASR confidence chips, and repetitive phrase collapse.
     - **🎛️ Acoustic Inspection**: Highlight MFCC coefficients (MFCC 2–13) and pitch contours.
  2. In the Sidebar, adjust the **"Audio weight"** slider from `0.50` to `0.70`.
  3. Click the **"⚡ Recompute Fusion"** button.
- **Spoken Talking Points**:
  - *"Notice that recomputing fusion happens instantaneously (< 5 ms) because the system re-arbitrates stored segment features without re-running heavy neural networks."*

### Minute 4:00 – 5:00: Ethical Guardrails, Export & Fallback
- **Action**:
  1. Open the **"⚠️ Operational Limitations & Human-in-the-Loop Requirement"** panel.
  2. Reiterate that modality disagreement is an *analytical signal for human review*; the system strictly never claims to detect lies, deceit, or sarcasm.
  3. In the sidebar, toggle the **Interface Theme** from Dark to Light to show WCAG AA compliant responsive contrast.
  4. Click **"EXPORT JSON REPORT 📥"** in the export bar to show clean, reproducible analytical records.
- **Offline / Hardware Fallback**:
  - If presenting on constrained or restricted conference hardware where CPU inference cannot run live, use the **pre-recorded high-resolution screen capture** (as no synthetic `DEMO_MODE` is hardcoded into the pipeline per data integrity rules).
