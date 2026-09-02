# 🎙️ Multimodal Voice & Conversation Emotion Analysis System

A B.Tech mini-project that analyzes spoken conversations using **voice emotion, speech-to-text, text sentiment, multimodal fusion, and conversation-level emotional analysis**, presented through an interactive Streamlit dashboard.

---

## 📌 Problem Statement

Emotion in spoken conversation is carried through two separate channels:

- **How** something is said — tone, pitch, energy, and other vocal characteristics.
- **What** is said — the actual words and their textual sentiment.

Most simple sentiment-analysis tools focus only on text, while many emotion-recognition systems focus only on audio. Neither modality alone provides a complete picture of emotional communication, and the two modalities can genuinely disagree.

For example, a speaker may say **"I'm fine"** while using a sad or resigned tone.

This project builds a practical multimodal system that analyzes both audio and text independently and then combines their outputs into a single explainable result at the level of a conversation rather than only a single audio clip.

---

## 🎯 Objective

Given an audio recording of speech, the system aims to:

1. Transcribe the speech into text.
2. Predict voice emotion directly from the audio.
3. Predict text sentiment from the generated transcript.
4. Combine both modalities using a transparent, rule-based fusion method.
5. Track how the combined emotional signal changes throughout the conversation.
6. Detect possible emotional mismatch and escalation.
7. Present the analysis through an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```text
                    Audio / Conversation
                            |
              +-------------+-------------+
              |                           |
              v                           v
      Audio Processing              Speech-to-Text
              |                           |
              v                           v
    Voice Emotion Analysis       Text Sentiment Analysis
              |                           |
              +-------------+-------------+
                            |
                            v
                    Multimodal Fusion
                            |
                            v
                 Conversation Analysis
                            |
                            v
                     Streamlit Dashboard
```

### Processing Flow

1. **Audio Processing**
   - Validates the uploaded audio.
   - Converts audio to mono.
   - Resamples audio to 16 kHz.
   - Normalizes the audio.
   - Extracts acoustic features.

2. **Voice Emotion Analysis**
   - Analyzes the acoustic signal using a pretrained Wav2Vec2 emotion model.
   - Produces an emotion label and confidence score.

3. **Speech-to-Text**
   - Converts spoken audio into text using faster-whisper.
   - Produces timestamped transcript segments.

4. **Text Sentiment Analysis**
   - Analyzes each transcript segment.
   - Produces negative, neutral, or positive sentiment with confidence.

5. **Multimodal Fusion**
   - Maps voice emotion and text sentiment into comparable emotional directions.
   - Determines agreement, mismatch, or weak disagreement.
   - Calculates confidence-weighted combined confidence.

6. **Conversation Analysis**
   - Analyzes emotional intensity across segments.
   - Detects overall emotional trends.
   - Detects possible emotional escalation.

7. **Streamlit Dashboard**
   - Presents the complete analysis interactively.

---

## ✨ Features

### 🎧 Audio Processing

The audio-processing module provides:

- Audio file validation.
- Supported-format validation.
- Empty and corrupted-file detection.
- Mono conversion.
- Resampling to 16 kHz.
- Peak normalization.
- Minimum and maximum duration validation.
- Acoustic feature extraction:
  - MFCC
  - RMS energy
  - Pitch / F0 using YIN
  - Zero-crossing rate
  - Spectral centroid

---

### 🎙️ Voice Emotion Analysis

The system uses:

```text
superb/wav2vec2-base-superb-er
```

The model supports four emotion classes:

- Neutral
- Happy
- Angry
- Sad

The system supports both:

- Whole-file emotion prediction.
- Segment-level emotion prediction using transcript timestamps.

Segment-level predictions are generated from the corresponding preprocessed audio intervals.

---

### 📝 Speech-to-Text

The system uses **faster-whisper** with the base model in CPU/int8 mode.

The speech-to-text component provides:

- Full speech transcript.
- Automatic language detection.
- Language confidence score.
- Timestamped transcript segments.
- Voice-activity filtering to reduce analysis of silence.

---

### 💬 Text Sentiment Analysis

The primary sentiment model is:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model provides three sentiment classes:

- Negative
- Neutral
- Positive

Each prediction includes:

- Sentiment label.
- Confidence score.
- Complete sentiment score distribution.

---

## 🔀 Multimodal Fusion

Voice emotion and text sentiment are combined using an explicit, documented, **rule-based fusion method** rather than a learned or black-box fusion model.

The fusion layer produces:

- Voice/text agreement or mismatch status.
- Emotional direction for each modality.
- Confidence-weighted combined confidence.
- Plain-language interpretation.

### Default Fusion Weights

```text
Audio: 0.50
Text:  0.50
```

The weights are configurable through `config.py`.

---

## 🔀 Fusion Rule

The multimodal fusion process follows these rules.

### 1. Map Voice Emotion to Emotional Direction

```text
angry  → negative
sad    → negative
happy  → positive
neutral → neutral
```

### 2. Use Text Sentiment Direction Directly

Text sentiment is already represented as:

```text
negative
neutral
positive
```

Therefore, the text sentiment direction can be used directly.

### 3. Agreement

Agreement occurs when the voice-emotion direction and text-sentiment direction match.

Example:

```text
Voice: Happy
Text:  Positive

Result: Agreement
```

### 4. Confident Mismatch

Mismatch occurs when:

- The two emotional directions disagree.
- Both modalities individually exceed the configured confidence threshold.

Default threshold:

```text
CONFIDENCE_THRESHOLD = 0.4
```

Example:

```text
Voice: Sad
Text:  Positive
```

This may be classified as a confident mismatch when both predictions have sufficient confidence.

### 5. Weak Disagreement

If the directions disagree but at least one modality has insufficient confidence, the system classifies the result as a weak or low-confidence disagreement rather than a confident mismatch.

### 6. Combined Confidence

Combined confidence is calculated as the weighted average of the individual modality confidences using the configured audio and text weights.

The fusion approach is intentionally transparent so that the reason for a result can be explained rather than hidden inside a learned fusion model.

> The system does not claim that a voice/text mismatch proves deception, hidden intent, or dishonesty. A mismatch only indicates that the two analyzed signals point in different emotional directions.

---

## 📈 Conversation Analysis

The conversation-analysis layer:

- Uses speech-to-text segment boundaries.
- Performs analysis at the segment level.
- Calculates a per-segment emotional intensity score.
- Considers modality agreement, mismatch, and confidence when calculating intensity.
- Detects an overall emotional trend:
  - Increasing
  - Decreasing
  - Stable
- Detects possible emotional escalation.

### Escalation Detection

Escalation detection requires:

- At least three usable segments.
- Continuously increasing intensity across consecutive segments.
- A total intensity increase of at least `0.20`.

If fewer than three usable segments are available, the system reports insufficient data rather than silently skipping the determination.

---

## 👥 Speaker Diarization

The project includes a dedicated diarization module:

```text
app/diarization.py
```

and corresponding tests:

```text
tests/test_diarization.py
```

The architecture is prepared for speaker identification and speaker-wise analysis.

However, the project does **not fabricate speaker identities** when reliable speaker information is unavailable.

Complete speaker-wise conversation analysis remains a future enhancement.

---

## 🖥️ Streamlit Dashboard

The Streamlit dashboard provides:

- Audio upload.
- In-browser audio playback.
- One-click end-to-end analysis.
- Reset functionality.
- Cached model loading using `st.cache_resource`.
- Full transcript display.
- Detected language.
- Language confidence.
- Audio duration.
- Segment count.
- Overall emotional trend.
- Intensity-change value.
- Emotional intensity line chart.
- Escalation status and reason.
- Voice-emotion results.
- Text-sentiment results.
- Multimodal fusion results.
- Segment-level expandable analysis cards.
- Voice emotion confidence.
- Text sentiment confidence.
- Fusion confidence.
- Emotional intensity.
- Agreement/mismatch status.
- Plain-language interpretation.
- Graceful error handling.
- Exportable analysis results.

The dashboard is built from actual pipeline outputs and does not use fabricated demonstration results.

---

## 🧠 Models

| Component | Model / Configuration |
|---|---|
| Voice Emotion | `superb/wav2vec2-base-superb-er` |
| Speech-to-Text | `faster-whisper` — base, CPU/int8 |
| Text Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Sentiment Fallback | `distilbert-base-uncased-finetuned-sst-2-english` |

All primary models are pretrained and publicly available Hugging Face models used without custom training.

The project focuses on practical end-to-end integration rather than training a new emotion-recognition model.

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
│   ├── diarization.py
│   └── conversation_analysis.py
│
├── sample_audio/
│   └── real_voice.wav
│
├── tests/
│   ├── test_audio_processor.py
│   ├── test_audio_setup.py
│   ├── test_emotion_analyzer.py
│   ├── test_emotion_model.py
│   ├── test_fusion.py
│   ├── test_conversation_analysis.py
│   ├── test_sentiment_analyzer.py
│   └── test_diarization.py
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

### Recommended Environment

```text
Python 3.11.9
```

A virtual environment is recommended.

### Create the Environment

```powershell
py -3.11 -m venv .venv
```

### Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Windows Note

The project is designed for CPU-based inference on a normal laptop.

For broader audio-format support, `ffmpeg` should be available on the system PATH.

Native WAV/FLAC processing can work without ffmpeg in the relevant parts of the pipeline.

---

## ▶️ Run the Dashboard

From the project root:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
```

The dashboard will normally be available at:

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

Run the complete automated test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### Current Test Status

```text
57 passed
0 failed
```

The current test suite covers:

- Audio setup and preprocessing.
- Audio validation.
- Audio feature extraction.
- Fusion logic.
- Agreement detection.
- High-confidence mismatch detection.
- Neutral agreement.
- Conversation trend detection.
- Escalation detection.
- Insufficient-segment handling.
- Sentiment analysis.
- Sentiment score integrity.
- Diarization module behavior.

The current test suite passes successfully with:

```text
57 passed, 4 warnings
```

The warnings are dependency deprecation warnings and do not represent test failures.

---

## 🧪 Manual Testing

Supported audio formats include:

```text
.wav
.flac
.mp3
.ogg
.m4a
```

Audio files can be placed inside `sample_audio/` or uploaded directly through the Streamlit dashboard.

The individual project modules can also be tested directly from a Python environment.

---

## 📊 Expected Output

For a short spoken sample, the system can produce:

- Detected language.
- Language confidence.
- Audio duration.
- Number of transcript segments.
- Full transcript.
- Per-segment voice emotion.
- Voice emotion confidence.
- Per-segment text sentiment.
- Text sentiment confidence.
- Fusion result.
- Agreement/mismatch status.
- Combined confidence.
- Plain-language interpretation.
- Per-segment emotional intensity.
- Overall emotional trend.
- Intensity-change value.
- Escalation status and explanation.

---

## 🎙️ Current Sample Audio

For:

```text
sample_audio/real_voice.wav
```

the system detects English and produces a transcript of approximately 13 seconds of speech.

The current voice-emotion model predicts:

```text
Sad
```

with high confidence for the sample audio, while the transcript sentiment is:

```text
Neutral
```

This provides a useful example of a voice/text disagreement.

However, this result should **not** be interpreted as proof that the speaker was actually sad. The prediction is a model estimate, and the sample was recorded as a normal test sentence.

---

## ⚠️ Limitations

### Voice Emotion Model

The voice-emotion model is general-purpose and is not fine-tuned specifically for this project.

Predictions can vary because of:

- Speaker characteristics.
- Pitch.
- Speaking style.
- Recording conditions.
- Microphone quality.
- Background noise.
- Emotional intensity.

Therefore, model predictions should be treated as estimates rather than ground-truth human labels.

### Speech-to-Text Evaluation

There is no automated correctness test for arbitrary speech-to-text transcription because arbitrary uploaded recordings do not have a fixed ground-truth transcript.

Transcription quality is therefore verified manually during testing.

### Trend and Escalation Detection

Trend and escalation detection depend on having enough usable segments.

Fewer than three usable segments are reported as insufficient data for escalation detection.

### Speaker Diarization

The project contains a diarization module and tests, but complete speaker-wise conversation analysis is not claimed as a fully completed feature.

The system does not fabricate speaker identities when reliable speaker information is unavailable.

### CPU-Only Inference

The primary models are designed to run on CPU so the project can be demonstrated on a normal laptop.

As a result, processing time increases with audio length and model complexity.

---

## 📚 Existing Work & Research Inspiration

Multimodal speech-emotion research, including systems based on datasets such as **IEMOCAP** and **MELD**, informed the general idea of combining audio and text signals for emotion analysis.

Existing research demonstrates that multiple modalities can provide complementary information for understanding emotional communication.

These works provided conceptual inspiration for the project, but no code from external repositories was copied and their specific research architectures were not reproduced.

This project does not claim to reproduce or outperform those research systems.

---

## 🆕 What Is Different About This Implementation?

This project does not claim a novel algorithm or improved accuracy over existing research.

Instead, it demonstrates practical end-to-end system integration:

- A modern CPU-friendly pipeline using pretrained Wav2Vec2, faster-whisper, and RoBERTa models.
- Independent voice-emotion and text-sentiment analysis.
- Segment-level multimodal fusion.
- Conversation-level emotional trend analysis.
- Rule-based emotional escalation detection.
- Transparent weighted fusion where decisions can be explained in plain English.
- A working interactive Streamlit dashboard.
- Real pipeline outputs displayed in the dashboard rather than hardcoded or fabricated demonstration data.
- Automated tests covering the main functional components.

---

## 🔮 Future Enhancements

Possible future improvements include:

- Full production-ready speaker diarization and speaker-wise emotion analysis.
- More advanced speaker-level conversation analytics.
- Configurable fusion weights directly from the dashboard.
- Additional emotion categories.
- Better visualization of emotion and sentiment changes.
- Automated regression testing using a curated labeled audio dataset.
- Performance optimization for longer recordings.
- GPU acceleration where available.
- Improved multilingual speech and sentiment support.
- Further dashboard refinements based on demonstration and user feedback.

---

## 🗺️ Project Progress

| Milestone | Status |
|---|---|
| Project Setup & Audio Verification | ✅ Complete |
| Audio Processing | ✅ Complete |
| Voice Emotion Analysis | ✅ Complete |
| Speech-to-Text | ✅ Complete |
| Text Sentiment Analysis | ✅ Complete |
| Multimodal Fusion | ✅ Complete |
| Conversation Analysis | ✅ Complete |
| Streamlit Dashboard | ✅ Complete |
| Testing & Documentation | ✅ Complete |

See [`PROGRESS.md`](PROGRESS.md) for the detailed development history and [`PROJECT_SPEC.md`](PROJECT_SPEC.md) for the complete project requirements and design specification.

---

## 👤 Team / Project Information

**B.Tech 3rd-Year Mini Project**

| Name | Roll Number |
|---|---|
| Sahithi | 1608-24-737-27 |
| Akshitha | 1608-24-737-57 |
| Rahul | 1608-24-737-23 |

**Internal Guide:** Mrs. K. Mounika

---

## 📌 Current Status

**Multimodal Voice & Conversation Emotion Analysis System**

- ✅ Functional multimodal emotion-analysis pipeline — **COMPLETE**
- ✅ Functional Streamlit dashboard — **COMPLETE**
- ✅ UI/UX enhancement phase — **COMPLETE**
- ✅ Automated test suite — **57/57 tests passing**
- ✅ Project documentation — **COMPLETE**
- ✅ Git stable milestone — **v1.0-functional**

The current implementation is suitable for functional demonstration, testing, and academic evaluation.
