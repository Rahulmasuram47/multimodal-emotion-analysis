# 🎙️ Multimodal Voice & Conversation Emotion Analysis

A B.Tech mini-project that analyzes spoken conversations using **voice
emotion, speech-to-text, text sentiment, multimodal fusion, and
conversation-level emotional analysis**, presented through an
interactive Streamlit dashboard.

---

## 📌 Problem Statement

Emotion in spoken conversation is carried through two separate
channels: *how* something is said (tone, pitch, energy) and *what* is
said (the words themselves). Most simple sentiment tools look only at
text, and most emotion-recognition demos look only at audio — neither
alone gives a complete picture, and the two can genuinely disagree
(e.g. a flat, resigned "I'm fine" spoken sadly). This project builds a
practical system that analyzes both channels independently and then
combines them into a single, explainable multimodal result at the
level of a full conversation, not just a single clip.

---

## 🎯 Objective

Given an audio recording of speech, the system should:

1. Transcribe the speech to text.
2. Predict voice emotion directly from the audio.
3. Predict text sentiment from the transcript.
4. Combine both into a transparent, rule-based multimodal result.
5. Track how that combined emotional signal changes across the
   conversation, and flag possible mismatches or escalation.
6. Present all of this through a simple, functional dashboard.

---

## 🚀 Project Pipeline

```text
Audio Input
    ↓
Audio Processing
    ↓
Voice Emotion Analysis
    ↓
Speech-to-Text
    ↓
Text Sentiment Analysis
    ↓
Multimodal Fusion
    ↓
Conversation Analysis
    ↓
Streamlit Dashboard
```

---

## 🏗️ Architecture

```text
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

Audio is loaded once per file and shared across the voice-emotion and
speech-to-text stages. Each transcript segment produced by
speech-to-text becomes one unit of analysis: it gets its own voice
emotion prediction (sliced from the same audio by timestamp), its own
text sentiment prediction, and its own fused result. Conversation
Analysis then looks across all segments to compute trend and
escalation.

---

## ✨ Features

### 🎧 Audio Processing

* Audio file validation (existence, supported format, non-empty,
  not corrupted).
* Mono conversion.
* Resampling to 16 kHz.
* Peak normalization.
* Duration validation (min/max bounds).
* Acoustic feature extraction: MFCC, RMS energy, pitch/F0
  (via YIN), zero-crossing rate, spectral centroid.

### 🎙️ Voice Emotion Analysis

Uses `superb/wav2vec2-base-superb-er` — 4 classes: **neutral, happy,
angry, sad**. Supports both whole-file prediction and segment-level
prediction sliced by timestamp from the same preprocessed audio.

### 📝 Speech-to-Text

Uses `faster-whisper` (`base` model, CPU, int8). Produces a full
transcript, automatic language detection with a confidence score, and
timestamped transcript segments (with voice-activity filtering to
skip silence).

### 💬 Text Sentiment Analysis

Uses `cardiffnlp/twitter-roberta-base-sentiment-latest` — 3 classes:
**negative, neutral, positive**, each with a confidence score and the
full score distribution.

### 🔀 Multimodal Fusion

Combines voice emotion and text sentiment using an explicit,
documented rule (see [Fusion Rule](#-fusion-rule-plain-english) below)
rather than a learned or "black box" fusion model. Produces:

* Voice/text **agreement** or **mismatch** flag.
* Emotional **direction** for each modality (positive / negative /
  neutral).
* Confidence-weighted **combined confidence**.
* A plain-language **interpretation** string.

Default fusion weights: Audio `0.50` / Text `0.50` — configurable in
`config.py`, not hardcoded in the fusion logic.

### 📈 Conversation Analysis

* Segments the conversation using Whisper's own segment boundaries.
* Computes a per-segment **emotional intensity** score from the
  fusion result (agreement boosts intensity, mismatch or weak
  disagreement discounts it — see `_calculate_intensity`).
* Detects an overall **emotional trend** (increasing / decreasing /
  stable) by comparing the first and last segment's intensity.
* Detects **possible emotional escalation**: requires at least three
  segments, a continuously rising intensity across every consecutive
  pair, and a total rise of at least `0.20`. Conversations with fewer
  than three usable segments are explicitly reported as
  "not enough data" rather than skipped silently.

### 🖥️ Streamlit Dashboard

* Audio upload and in-browser playback.
* One-click end-to-end analysis (models are loaded once via
  `st.cache_resource` and reused across runs).
* Full transcript, detected language, and language confidence.
* Emotional trend metric, intensity-change value, and an intensity
  line chart across segments.
* Escalation status with the underlying reason.
* Per-segment expandable cards: transcript, voice emotion (with
  confidence), text sentiment (with confidence), fusion confidence,
  intensity, and the agreement/mismatch/interpretation caption.
* Graceful error handling — a failed analysis shows an error message
  in the UI instead of crashing the app.

---

## 🔀 Fusion Rule (Plain English)

1. Voice emotion is mapped to a broad direction: `angry`/`sad` →
   **negative**, `happy` → **positive**, `neutral` → **neutral**.
2. Text sentiment (already negative/neutral/positive) is used as its
   own direction directly.
3. **Agreement** = the two directions match.
4. **Mismatch** = the two directions do *not* match, **and** both
   modalities individually clear the confidence threshold
   (`CONFIDENCE_THRESHOLD = 0.4` by default). This avoids flagging a
   "mismatch" when one modality's prediction is too uncertain to
   trust in the first place.
5. If the directions disagree but at least one modality is
   low-confidence, it's reported as **weak/low-confidence
   disagreement** — a distinct, softer category from a confident
   mismatch.
6. **Combined confidence** is a weighted average of the two
   modalities' individual confidences, using the configured
   audio/text weights.

The system never claims a mismatch proves deception or hidden intent
— only that the two signals point in different directions.

---

## 🧠 Models

| Component          | Model                                              |
| ------------------ | -------------------------------------------------- |
| Voice Emotion      | `superb/wav2vec2-base-superb-er`                   |
| Speech-to-Text     | `faster-whisper` (`base`, CPU, int8)               |
| Text Sentiment     | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment Fallback | `distilbert-base-uncased-finetuned-sst-2-english`  |

All three are pretrained, publicly available HuggingFace models used
as-is (no custom training). See [Existing Work](#-existing-work--research-inspiration)
below for how this relates to prior research.

---

## 📁 Project Structure

```text
multimodal-emotion-analysis/
│
├── app/
│   ├── __init__.py
│   ├── audio_processor.py
│   ├── emotion_analyzer.py
│   ├── speech_to_text.py
│   ├── sentiment_analyzer.py
│   ├── fusion.py
│   └── conversation_analysis.py
│
├── sample_audio/
│   ├── real_voice.wav
│   ├── sample_16k.wav
│   ├── test_corrupted.wav
│   └── test_empty.wav
│
├── tests/
│   ├── test_audio_setup.py
│   ├── test_audio_processor.py
│   ├── test_fusion.py
│   ├── test_conversation_analysis.py
│   └── test_sentiment_analyzer.py
│
├── config.py
├── PROJECT_SPEC.md
├── PROGRESS.md
├── README.md
├── requirements.txt
└── streamlit_app.py
```

---

## ⚙️ Environment

```text
Python 3.11.9
```

A virtual environment is recommended.

### Create the environment

```powershell
py -3.11 -m venv .venv
```

### Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For testing:

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
```

**Windows note:** the PyTorch wheel installed should be the CPU-only
build unless a working CUDA setup has been separately verified on the
machine. `ffmpeg` should be on PATH for broader audio-format support
and for `faster-whisper`; native WAV/FLAC works without it.

---

## ▶️ Run the Dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
```

The dashboard will be available at:

```text
http://localhost:8501
```

Upload an audio file such as:

```text
sample_audio/real_voice.wav
```

and click **Analyze Audio**.

---

## 🧪 How to Test

Run the full automated test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The suite currently includes:

* `test_audio_processor.py` — 5 tests (valid audio, unsupported
  format, corrupted file, empty file, feature extraction).
* `test_fusion.py` — 3 tests (agreement, high-confidence mismatch,
  neutral agreement).
* `test_conversation_analysis.py` — 6 tests (trend
  increasing/decreasing/stable, escalation fires/doesn't
  fire/insufficient segments).
* `test_sentiment_analyzer.py` — 5 tests (positive, negative,
  neutral, empty-text rejection, score-dictionary integrity).

To test manually with your own audio, drop a `.wav`/`.flac`/`.mp3`/
`.ogg`/`.m4a` file into `sample_audio/` and upload it through the
dashboard, or point any of the `app/` module's `analyze()` /
`transcribe()` methods at its path directly from a Python shell.

---

## 📊 Expected Output

For a short spoken sample, the system produces:

* Detected language and language confidence.
* Audio duration and segment count.
* A full transcript.
* Per-segment voice emotion (label + confidence).
* Per-segment text sentiment (label + confidence).
* Per-segment fusion result (agreement/mismatch, combined confidence,
  interpretation).
* Per-segment emotional intensity.
* An overall emotional trend and intensity-change value.
* An escalation status with the reason for that determination.

For the current sample audio (`sample_audio/real_voice.wav`), the
system detects English, a duration of roughly 13 seconds, and
classifies the sample as **sad** by voice while the transcript
sentiment is **neutral** — a voice/text mismatch, since the recording
is a flatly-spoken test sentence rather than genuinely sad speech.
This is a useful demonstration case for the mismatch feature, not a
claim that the speaker was actually sad.

---

## ⚠️ Limitations

* **Voice emotion model is general-purpose, not fine-tuned for this
  project.** Predictions can vary with speaker characteristics,
  pitch, speaking style, recording conditions, microphone quality,
  and emotional intensity. Results are model estimates, not
  ground-truth human labels.
* **No automated correctness test for speech-to-text transcription.**
  Whisper output has no fixed ground truth to assert against
  automatically for arbitrary audio; transcription accuracy is
  verified manually rather than via pytest.
* **Escalation and trend detection need enough segments.** Fewer than
  three usable segments means escalation is reported as
  "not enough data" rather than a real determination.
* **Speaker-wise analysis is architected for, not fully implemented.**
  Full diarization was out of scope for the timeframe; the system
  never fabricates speaker identities or speaker-wise results without
  a real basis.
* **CPU-only inference.** All three models run on CPU by design (for
  a normal laptop), so processing time scales with audio length.

---

## 📚 Existing Work & Research Inspiration

Multimodal speech-emotion research (e.g. IEMOCAP/MELD-based systems,
and older repositories such as `david-yoon/multimodal-speech-emotion`
built on TensorFlow 1.4 / Python 2.7 / CUDA 8) informed the general
idea of combining audio and text signals for emotion. No code from
these repositories was copied, and none of their specific
architectures (custom RNNs, MDRE/MDREA, exact Luong attention) were
reproduced.

## 🆕 What's Different About This Implementation

This project does not claim a novel algorithm or improved accuracy
over existing research. What it demonstrates instead is **practical
end-to-end system integration**:

* A modern, CPU-friendly pipeline using off-the-shelf pretrained
  models (Wav2Vec2, faster-whisper, RoBERTa) rather than a
  research-grade custom architecture.
* Conversation-level analysis (trend + escalation) built on top of
  segment-level fusion, rather than single-clip classification.
* A transparent, weighted, rule-based fusion layer — every fusion
  decision can be explained in plain English (see
  [Fusion Rule](#-fusion-rule-plain-english)), with no hidden
  learned-fusion step.
* A working interactive dashboard suitable for a live demo, built
  entirely from real pipeline outputs — no hardcoded or fabricated
  demo data anywhere in the display layer.

---

## 🔮 Future Enhancements

* Full speaker diarization for genuine speaker-wise analysis.
* Dashboard UI/UX polish: improved visual design, dedicated
  emotion/sentiment charts, better segment visualization, and
  clearer loading/error states (tracked as the next project phase).
* Configurable/adjustable fusion weights exposed directly in the
  dashboard, rather than only in `config.py`.
* Automated regression testing against a small, deliberately curated
  set of sample recordings with agreed-upon expected labels.

---

## 🗺️ Project Progress

| Milestone                          | Status         |
| ----------------------------------- | -------------- |
| Project Setup & Audio Verification  | ✅ Complete     |
| Audio Processing                    | ✅ Complete     |
| Voice Emotion Analysis              | ✅ Complete     |
| Speech-to-Text                      | ✅ Complete     |
| Text Sentiment Analysis             | ✅ Complete     |
| Multimodal Fusion                   | ✅ Complete     |
| Conversation Analysis               | ✅ Complete     |
| Streamlit Dashboard                 | ✅ Complete     |
| Testing & Documentation             | ✅ Complete     |

See [`PROGRESS.md`](PROGRESS.md) for detailed development history and
[`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the full requirements this
project is built against.

---

## 👤 Team / Project Info

**B.Tech 3rd-year mini project**

| Name | Roll Number |
|------|--------------|
| Sahithi | 1608-24-737-27 |
| Akshitha | 1608-24-737-57 |
| Rahul | 1608-24-737-23 |

**Guide:** Mrs.K.Mounika

---

## 📌 Current Status

**Functional multimodal emotion analysis pipeline: COMPLETE**

**Functional Streamlit dashboard: COMPLETE**

**Automated tests: 19/19 passing**, covering all required categories
from the project spec (audio validation, feature extraction,
sentiment analysis, fusion logic, mismatch detection, escalation
detection)

**Final UI/UX enhancement: NEXT PHASE**