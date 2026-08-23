# Multimodal Voice & Conversation Emotion Analysis System — Project Spec

You are the lead software engineer helping me build my B.Tech 3rd-year mini project.

**IMPORTANT:** This is a college mini-project, NOT a research-grade reproduction of an existing paper. The system must be practical, understandable, demonstrable, and runnable on a normal Windows laptop.

Build the project incrementally and carefully. Do NOT over-engineer it. This file is the full spec — re-read it at the start of every milestone. Do not rely only on chat history for requirements; this file is the source of truth.

---

## 1. Actual Project Objective

Analyze a conversation using TWO modalities:

1. Voice/audio characteristics
2. Spoken text/transcript

Analyze both independently, then combine their results into a multimodal emotional analysis.

```
Audio / Conversation
        |
        +----------------------+
        |                      |
        v                      v
Audio Processing          Speech-to-Text
        |                      |
        v                      v
Voice Emotion Analysis   Text Sentiment Analysis
        |                      |
        +----------+-----------+
                   |
                   v
          Multimodal Fusion
                   |
                   v
        Conversation Analysis
                   |
                   v
              Dashboard
```

---

## 2. Locked-In Model Choices (do not substitute without discussion)

Don't leave these to the agent's judgement — pick once, document once:

- **Voice emotion:** `superb/wav2vec2-base-superb-er` (HuggingFace) — 4 classes: neutral / happy / angry / sad. Fits the "keep emotion categories manageable" requirement.
- **Speech-to-text:** `faster-whisper` (NOT vanilla `openai-whisper`) — meaningfully faster on CPU, which matters on a normal laptop. Use the `base` or `small` model size.
- **Text sentiment:** `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3-class (positive/neutral/negative), matches the required output categories directly. (`distilbert-base-uncased-finetuned-sst-2-english` is an acceptable 2-class fallback if the above is slow to download.)

If any of these models turn out to be impractical on this machine (too slow, too large, download issues), stop and tell me before switching — don't silently swap models.

---

## 3. Required Core Features

### A. Audio / Voice Analysis
Extract relevant acoustic features (pitch, energy/intensity, MFCC, speaking rate, etc.) and produce a voice-based emotion prediction using the model in Section 2.

DO NOT recreate the old TensorFlow 1.x architecture from the David Yoon repository. DO NOT require Python 2.7, TensorFlow 1.4, or CUDA 8. Use modern Python-compatible libraries.

### B. Speech-to-Text
Convert spoken audio into text using `faster-whisper` (Section 2). Display the transcript in the dashboard and pass it to text sentiment analysis. Handle transcription failures gracefully (see Section 12).

### C. Text Sentiment Analysis
At minimum: positive / negative / neutral sentiment with a score, using the model in Section 2. Keep this understandable for a college mini-project — no custom-trained NLP architecture.

### D. Multimodal Fusion
Combine audio emotion output and text sentiment output. Fusion logic must be transparent, explainable, and rule-based/weighted — NOT a claimed novel deep-learning fusion architecture. Weights must be configurable (Section 14), not hardcoded magic numbers.

Example:
```
Audio: Angry = 0.72
Text: Negative = 0.81
Combined: High emotional negativity / likely anger or escalation
```

### E. Conversation Analysis
Divide the conversation into segments. For each segment store: timestamp, transcript, voice emotion, text sentiment, combined emotion, confidence/scores. This enables emotional-change tracking over time.

### F. Emotion Change Over Time
Visualize emotional progression across segments in the dashboard. All plotted values must come from actual analysis results — never fabricated.

### G. Voice-Text Mismatch
Detect when voice emotion and text sentiment disagree (e.g., text positive, voice angry). Use wording like "Possible voice-text emotional mismatch" or "Audio and text signals show different emotional tendencies." Never claim this proves someone is lying.

### H. Emotional Escalation
Detect increasing emotional intensity across segments, based on actual score trends only — never invented. Report as "Possible emotional escalation detected."

### I. Speaker-Wise Analysis
Support speaker-wise analysis where practical, but do not let diarization complexity block the core project. If full diarization isn't practical in the time available, architect the system so it can be added later cleanly. Never fake speaker identities or claim speaker-wise results without an actual basis for the assignment.

---

## 4. Dashboard

**Technology:** Streamlit.

The dashboard should let the user:
1. Upload an audio file
2. Start analysis
3. See audio info (duration, sample rate)
4. See the generated transcript
5. See voice emotion result
6. See text sentiment result
7. See combined multimodal result
8. See emotion/sentiment scores
9. See emotion changes over time
10. See possible voice-text mismatch
11. See possible emotional escalation
12. See speaker-wise information when available

Keep it simple and functional. No React frontend unless there's a genuine reason.

---

## 5. Technology Requirements

- **OS:** Windows
- **Python:** 3.11.x
- **Virtual environment:** use the existing `.venv` in the project folder. Do not recreate or destroy it unnecessarily.
- **Project folder:** `multimodal-emotion-analysis`

### Windows-specific landmines (check these explicitly in Milestone 1, don't discover them later)
- **PyTorch:** install the **CPU-only** wheel unless a working CUDA setup is verified on this machine first. GPU wheels silently fail or bloat the install on a laptop without a supported GPU.
- **ffmpeg:** `faster-whisper` and `librosa` may need `ffmpeg` on PATH. Check for it during Milestone 1's environment verification and install/document it if missing — don't find out mid-pipeline.

### Recommended libraries
librosa, numpy, pandas, scikit-learn, soundfile, transformers, torch (CPU), streamlit, faster-whisper, matplotlib or plotly.

Do NOT install every library blindly — before installing anything heavy, explain why it's necessary. Avoid unnecessary dependencies.

---

## 6. Project Must Be Lightweight

Explicitly NOT required: TensorFlow 1.x, Python 2.7, CUDA 8/CUDNN 6, custom RNN architectures, reproducing MDRE/MDREA, reproducing Luong attention exactly, training transformers from scratch, building a new research architecture, massive custom datasets, distributed systems, cloud infra, Kubernetes, Docker (unless genuinely necessary), microservices, complicated React frontend, unnecessary databases.

The goal is a working multimodal analysis system, not a research paper reproduction.

---

## 7. Relation to Existing Research (for honesty in the report/viva)

Existing repos researched include `david-yoon/multimodal-speech-emotion` (old, TF1.4/Python2.7/CUDA8 — reference only, not to be copied), plus other IEMOCAP/MELD/wav2vec2/BERT-based multimodal fusion projects. Use these only as conceptual reference. Do not copy large sections of source code. Distinguish this project through practical system integration and conversation-level analysis, not a claimed new algorithm.

---

## 8. What We Can Truthfully Claim

Do NOT claim: "first ever," "we invented a new algorithm," "scientifically novel fusion," "we outperform existing research."

Can truthfully claim:
1. End-to-end practical pipeline from audio to conversation analysis
2. Integration of STT, voice emotion, and text sentiment analysis
3. Multimodal fusion of voice and text results
4. Conversation-level emotional trend visualization
5. Possible voice-text emotional mismatch detection
6. Possible emotional escalation detection
7. Speaker-wise analysis where technically supported
8. A simple interactive dashboard suitable for real-world demonstration

Only claim what the implemented system actually supports.

---

## 9. Dataset Strategy

Do not commit copyrighted datasets or large audio files to GitHub. Possible datasets for development/testing: RAVDESS, TESS, CREMA-D, IEMOCAP, MELD — but don't auto-download all of them; pick the smallest practical option. Support user-provided audio files. `.gitignore` must exclude data/model files.

---

## 10. Code Structure

```
multimodal-emotion-analysis/
│
├── .venv/
├── app/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── audio_emotion.py
│   ├── speech_to_text.py
│   ├── text_sentiment.py
│   ├── fusion.py
│   ├── conversation_analysis.py
│   ├── speaker_analysis.py
│   └── visualization.py
├── models/
├── data/
├── tests/
├── sample_audio/
├── app.py
├── requirements.txt
├── README.md
├── PROGRESS.md
├── .gitignore
└── config.py
```

Modify only if there's a clearly better lightweight organization. No unnecessary files.

---

## 11. Software Design (module responsibilities)

- **audio_processor.py** — load/validate/resample/normalize audio, compute duration, extract acoustic features
- **audio_emotion.py** — run voice emotion model, return labels + confidence, clean interface
- **speech_to_text.py** — transcribe audio, return transcript, handle errors clearly
- **text_sentiment.py** — analyze transcript, return sentiment + scores
- **fusion.py** — combine audio + text results, produce multimodal result, detect modality disagreement
- **conversation_analysis.py** — segment-level analysis, trend/escalation detection, summary stats
- **speaker_analysis.py** — speaker-wise analysis only when real speaker info is available; never invent speakers
- **visualization.py** — charts/timelines, agreement/disagreement visuals
- **app.py** — Streamlit interface, file upload, pipeline orchestration, results display

---

## 12. Error Handling

Must not crash with a raw traceback on common user mistakes: unsupported audio file, corrupted audio, empty audio, very long audio, missing model, missing dependency, speech recognition failure, no transcript, low-confidence prediction. Show useful messages in the Streamlit UI.

---

## 13. Performance

Designed for a normal student laptop. Avoid unnecessarily large models. If a model is computationally expensive, explain it, use a lightweight variant, make it configurable. Load models once and reuse — never retrain/reload on every Analyze click.

---

## 14. Configuration

Centralize: supported audio extensions, max audio duration, fusion weights, model names, confidence thresholds. No scattered magic numbers.

---

## 15. Logging

Simple, useful logs only, e.g.:
```
[INFO] Audio loaded
[INFO] Extracting features
[INFO] Running audio emotion model
[INFO] Transcribing audio
[INFO] Running text sentiment analysis
[INFO] Performing multimodal fusion
[INFO] Generating conversation analysis
```
Not excessive.

---

## 16. Testing

Basic tests for: audio validation, feature extraction, sentiment analysis, fusion logic, mismatch detection, escalation detection. Use mocked/sample inputs — no huge dataset downloads required to run tests.

---

## 17. README Contents

Title, problem statement, objective, features, architecture, pipeline, technologies used, installation, how to run, how to test with audio, expected output, limitations, existing-work/research inspiration, what's different about this implementation, future enhancements, team/project info placeholder. No false claims.

---

## 18. GitHub

Do NOT commit: `.venv`, datasets, large audio files, model checkpoints, API keys, passwords, `.env` files, temp/cache files. Keep `.gitignore` correct. Meaningful commit messages, roughly matching the milestone list in Section 20.

---

## 19. Development Method — READ THIS BEFORE EVERY MILESTONE

**Do NOT build everything at once. Work in small, verified steps.**

At each stage:
1. Explain what you're about to create.
2. Create only the required files.
3. Give exact terminal commands.
4. Run/check the code.
5. Fix errors.
6. Explain the result.
7. **Update `PROGRESS.md`** with 2-4 lines: what was built this milestone, what model/library choices were made or changed, and any open issues.
8. Stop and wait for me to confirm before moving to the next milestone.

Do not silently make major architectural decisions. If a dependency or model choice has a significant tradeoff not already covered in Section 2, explain it before implementing.

`PROGRESS.md` exists so that in a long session (or a new session) you can re-read what's already been decided instead of re-deriving it or asking me to re-explain the project.

---

## 20. Milestones (with explicit done-criteria)

**Milestone 1 — Project Setup + Audio Test**
Verify Python version, `.venv`, project directory, VS Code, Git, GitHub repo, `.gitignore`, README, `requirements.txt`, and that `ffmpeg` is on PATH. Install only minimum audio-processing dependencies. Create a test script that loads a sample audio file, prints duration + sample rate, extracts MFCC, prints feature dimensions.
**Done when:** the test script runs with no errors and prints duration, sample rate, and MFCC shape, and I've confirmed it works on my machine.

**Milestone 2 — Audio Processing Module**
Build `audio_processor.py` per Section 11.
**Done when:** it handles a valid file, an unsupported format, and a corrupted file without crashing, and I've tested all three.

**Milestone 3 — Audio Emotion Module**
Build `audio_emotion.py` using the model from Section 2.
**Done when:** it returns emotion label + confidence scores on a real sample file, and the output looks sane to me.

**Milestone 4 — Speech-to-Text**
Build `speech_to_text.py` using `faster-whisper`.
**Done when:** it produces a correct-looking transcript on a real sample file within a reasonable time on this laptop.

**Milestone 5 — Text Sentiment**
Build `text_sentiment.py` using the model from Section 2.
**Done when:** it returns sentiment + score on a few test sentences I give you, including at least one obviously positive and one obviously negative.

**Milestone 6 — Multimodal Fusion**
Build `fusion.py` with an explicit, documented mismatch/agreement rule (not just "compare both").
**Done when:** you can show me the exact rule in plain English, and it correctly flags at least one deliberately mismatched test case (e.g., positive text spoken angrily).

**Milestone 7 — Conversation Analysis**
Build `conversation_analysis.py` — segmenting, trend detection, escalation detection.
**Done when:** it produces a segment-by-segment table with real values on a multi-segment sample, and escalation detection fires correctly on a test case with rising intensity.

**Milestone 8 — Dashboard**
Build `app.py` (Streamlit) wiring everything together per Section 4.
**Done when:** `streamlit run app.py` runs the full pipeline end-to-end on an uploaded file and shows all 12 items from Section 4.

**Milestone 9 — Testing & Documentation**
Basic tests (Section 16) + final README (Section 17).
**Done when:** tests pass with sample/mocked inputs and the README is accurate to what's actually implemented.

---

## 21. Rules for You as the Coding Agent

1. Do not over-engineer this project.
2. Do not reproduce David Yoon's TensorFlow 1.4 project.
3. Do not use Python 2.7.
4. Do not require obsolete TensorFlow/CUDA versions.
5. Do not add facial emotion recognition — core project is voice/audio + text only.
6. Do not add unrelated features.
7. Do not claim scientific novelty that can't be proven.
8. Do not fabricate emotion predictions, confidence values, speaker identities, or analysis — ever, including for demo purposes.
9. Do not hard-code fake dashboard data for the final system.
10. Use actual outputs from the implemented models/pipeline.
11. Keep the system understandable enough that I can explain every major component during a project review/viva.
12. Prefer simple, reliable solutions over complicated research architectures.
13. If a pretrained model is used, clearly identify it in the documentation.
14. Respect model/dataset licenses.
15. Do not copy entire repositories into this project.
16. Do not modify unrelated files without explaining why.
17. Never delete the existing `.gitignore`, README, or `.venv`.
18. Before installing packages, inspect the current environment and avoid duplicates.
19. If something can't be implemented reliably in the lightweight version, state the limitation clearly rather than faking it.
20. The final application must run from one command: `streamlit run app.py`.
21. Do not proceed past a milestone's done-criteria (Section 20) without my explicit confirmation.

---

## 22. Expected Final Demo

Open the Streamlit app → upload conversation/audio file → click Analyze → system transcribes, predicts voice emotion, analyzes text sentiment, fuses both → dashboard shows transcript, audio emotion, text sentiment, multimodal result, scores, emotion timeline, possible mismatch, possible escalation, speaker-wise info when available.

---

## 23. Final Project Philosophy

Lightweight. Modern. Practical. Explainable. Modular. Demonstrable. Honest about limitations.

**Start now with Milestone 1 only. First inspect the existing project structure and environment. Do not start Milestone 2 until Milestone 1's done-criteria are verified by me.**