# 🎙️ Multimodal Voice & Conversation Emotion Analysis

A multimodal emotion analysis system that analyzes spoken conversations using **voice emotion, speech-to-text, text sentiment, multimodal fusion, and conversation-level emotional analysis**.

The system processes an audio recording and produces segment-level insights including emotional state, sentiment, confidence, voice/text mismatch, emotional intensity, emotional trends, and escalation detection.

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

## ✨ Features

### 🎧 Audio Processing

* Audio file validation.
* Supported audio format checking.
* Corruption and empty-file detection.
* Mono conversion.
* Resampling to 16 kHz.
* Peak normalization.
* Duration validation.
* Acoustic feature extraction:

  * MFCC
  * RMS energy
  * Pitch/F0
  * Zero-crossing rate
  * Spectral centroid

### 🎙️ Voice Emotion Analysis

Uses:

```text
superb/wav2vec2-base-superb-er
```

Supported emotions:

* Neutral
* Happy
* Angry
* Sad

The system provides emotion predictions and confidence scores for conversation segments.

### 📝 Speech-to-Text

Uses:

```text
faster-whisper
```

The system provides:

* Full transcript.
* Timestamped transcript segments.
* Detected language.
* Language confidence.

### 💬 Text Sentiment Analysis

Uses:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

Supported sentiment classes:

* Negative
* Neutral
* Positive

### 🔀 Multimodal Fusion

Voice emotion and text sentiment are combined to identify:

* Multimodal agreement.
* Voice/text mismatch.
* Emotional direction.
* Combined confidence.
* Interpretable fusion results.

Default fusion weights:

```text
Audio: 0.50
Text:  0.50
```

### 📈 Conversation Analysis

The conversation analyzer provides:

* Segment-level emotion.
* Segment-level sentiment.
* Multimodal fusion.
* Emotional intensity.
* Overall emotional trend.
* Escalation detection.
* Full conversation transcript.

### 🖥️ Streamlit Dashboard

The current functional dashboard provides:

* Audio upload.
* Conversation analysis.
* Transcript display.
* Emotion and sentiment results.
* Confidence scores.
* Emotional trend.
* Escalation status.
* Segment-level analysis.
* Voice/text mismatch detection.
* Emotional intensity.

UI/UX improvements are planned as a separate final polish phase.

---

## 🧠 Models

| Component          | Model                                              |
| ------------------ | -------------------------------------------------- |
| Voice Emotion      | `superb/wav2vec2-base-superb-er`                   |
| Speech-to-Text     | `faster-whisper` base                              |
| Text Sentiment     | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment Fallback | `distilbert-base-uncased-finetuned-sst-2-english`  |

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
│   ├── test_audio_processor.py
│   └── test_audio_setup.py
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

The project currently uses:

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

and click the analysis control in the dashboard.

---

## 🧪 Run Tests

Run the automated test suite with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current verification:

```text
5 passed
```

The current test run produces dependency deprecation warnings, but no test failures.

---

## 📊 Example Analysis

For the current sample audio, the system successfully produces:

* English language detection.
* Approximately 13-second audio duration.
* Timestamped transcript segments.
* Voice emotion predictions.
* Text sentiment predictions.
* Multimodal confidence.
* Voice/text mismatch detection.
* Emotional intensity.
* Emotional trend.
* Escalation analysis.

The current sample is primarily classified as **sad by the voice model** while the transcript sentiment is **neutral**, resulting in a voice/text mismatch.

---

## 🗺️ Project Progress

| Milestone                          | Status         |
| ---------------------------------- | -------------- |
| Project Setup & Audio Verification | ✅ Complete     |
| Audio Processing                   | ✅ Complete     |
| Voice Emotion Analysis             | ✅ Complete     |
| Speech-to-Text                     | ✅ Complete     |
| Text Sentiment Analysis            | ✅ Complete     |
| Multimodal Fusion                  | ✅ Complete     |
| Conversation Analysis              | ✅ Complete     |
| Streamlit Dashboard                | ✅ Complete     |
| Testing & Documentation            | 🟡 In Progress |

See [`PROGRESS.md`](PROGRESS.md) for detailed development progress.

---

## 🔮 Next Phase

The functional backend and dashboard are now established.

The next phase focuses on **dashboard UI/UX enhancement** while preserving the working multimodal analysis pipeline.

Planned improvements include:

* Professional visual design.
* Improved dashboard layout.
* Emotion and sentiment charts.
* Emotional timeline.
* Better segment visualization.
* Confidence indicators.
* Voice/text mismatch visualization.
* Escalation visualization.
* Improved loading and error states.
* Final demo/presentation polish.

The current functional implementation should remain the baseline while UI/UX improvements are introduced.

---

## 📌 Current Status

**Functional multimodal emotion analysis pipeline: COMPLETE**

**Functional Streamlit dashboard: COMPLETE**

**Automated tests: PASSING**

**Testing & documentation: IN PROGRESS**

**Final UI/UX enhancement: NEXT PHASE**
