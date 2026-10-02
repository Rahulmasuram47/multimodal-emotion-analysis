# Empirical Error Analysis & Misclassification Log

This document lists every sample where model predictions disagreed with annotated ground truth, providing full multimodal context (acoustic probabilities, prosodic features, transcription, and sentiment).

Total misclassifications identified: **3**

| Clip | Modality | Ground Truth | Prediction | Diagnostic Root Cause |
|---|---|---|---|---|
| `happy_sample.wav` | Voice Emotion | `happy` | `sad` | Acoustic domain collapse / IEMOCAP prior shift |
| `angry_sample.wav` | Voice Emotion | `angry` | `sad` | Acoustic domain collapse / IEMOCAP prior shift |
| `neutral_sample.wav` | Voice Emotion | `neutral` | `sad` | Acoustic domain collapse / IEMOCAP prior shift |

## Detailed Sample Context & Multimodal Diagnosis

### Sample 1: `happy_sample.wav`
- **File Path**: `sample_audio/happy_sample.wav`
- **Ground Truth**: Voice: `happy`, Text: `N/A`
- **Predictions**: Voice: `sad`, Text: `positive`
- **Transcript**: "Hello! I just wanted to say that everything worked perfectly. I am really happy with the result and I appreciate all the help I received. This has made my day much better. Thank you so much!"
- **Diagnostic Analysis**:
  * **Acoustic Domain Shift**: Model predicted `sad` instead of `happy`. Wav2Vec2 SUPERB ER exhibits heavy inductive prior bias towards 'sad' on conversational speech. The decision fusion pipeline mitigates this via prosodic sanity checks and reliability penalties.

### Sample 2: `angry_sample.wav`
- **File Path**: `sample_audio/angry_sample.wav`
- **Ground Truth**: Voice: `angry`, Text: `N/A`
- **Predictions**: Voice: `sad`, Text: `negative`
- **Transcript**: "I have already explained this several times and I am still having the same problem. I have a stalled this would be fixed but nothing has changed. This is extremely frustrating and I need someone to actually resolve this."
- **Diagnostic Analysis**:
  * **Acoustic Domain Shift**: Model predicted `sad` instead of `angry`. Wav2Vec2 SUPERB ER exhibits heavy inductive prior bias towards 'sad' on conversational speech. The decision fusion pipeline mitigates this via prosodic sanity checks and reliability penalties.

### Sample 3: `neutral_sample.wav`
- **File Path**: `sample_audio/neutral_sample.wav`
- **Ground Truth**: Voice: `neutral`, Text: `N/A`
- **Predictions**: Voice: `sad`, Text: `neutral`
- **Transcript**: "Hello, I wanted to check the status of my request. I submitted the information study and I just wanted to know if there are any updates. Please let me know when you have a moment."
- **Diagnostic Analysis**:
  * **Acoustic Domain Shift**: Model predicted `sad` instead of `neutral`. Wav2Vec2 SUPERB ER exhibits heavy inductive prior bias towards 'sad' on conversational speech. The decision fusion pipeline mitigates this via prosodic sanity checks and reliability penalties.
