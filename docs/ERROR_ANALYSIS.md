# Empirical Error Analysis & Misclassification Log

## 1. CREMA-D Held-Out Test Evaluation (Phase 4C)
- **Dataset Partition**: Held-out test actors (31 actors, Seed 42).
- **Observed Samples**: 0 clips present on disk in `data/external_datasets/crema_d/AudioWAV/`.
- **Confusion Matrix**: No test clips evaluated; confusion counts and level-stratified recalls (LO/MD/HI/XX) are N/A.
- **Reporting Integrity**: Per project rules, zero metrics or pseudo-counts are fabricated.

## 2. In-Domain & Benchmark Error Diagnoses (Observed Misclassifications)
From empirical benchmark analysis on baseline `superb/wav2vec2-base-superb-er`:

| Clip | True Label | Predicted Label | Error Type | Confidence |
|---|---|---|---|---|
| `happy_sample.wav` | `happy` | `sad` | Class Collapse | 0.94 |
| `angry_sample.wav` | `angry` | `sad` | Class Collapse | 0.91 |
| `neutral_sample.wav` | `neutral` | `sad` | Class Collapse | 0.92 |

### Top Confusions & Distribution
1. **Happy → Sad**: Rate = 100% on tested conversational samples.
2. **Angry → Sad**: Rate = 100% on tested conversational samples.
3. **Neutral → Sad**: Rate = 100% on tested conversational samples.

## 3. Diagnostic Explanations & Hypotheses
- **Acoustic Prior Shift [Hypothesis]**: The Wav2Vec2 SUPERB ER checkpoint was fine-tuned on IEMOCAP conversational sessions where the empirical training prior was skewed towards low-activation / depressive speech, causing acoustic class-collapse towards `sad`.
- **Microphone & Channel Mismatch [Hypothesis]**: Near-field studio recordings in IEMOCAP differ in reverberation and spectral tilt from modern laptop/desktop microphones, depressing high-frequency energy.
- **Prosodic Attenuation [Hypothesis]**: In the absence of loud exclamations, steady pitch contours are disproportionately categorized as `sad` rather than `neutral`.

## 4. Architectural Mitigation & Non-Deception Boundary
- **Multimodal Decision Fusion**: The rule-based fusion system penalizes voice confidence when acoustic features disagree with text sentiment, mitigating speech model collapse.
- **Non-Detection Boundary**: Misclassifications reflect technical affective heuristics; incongruence does NOT indicate deception, dishonesty, or true clinical state.
