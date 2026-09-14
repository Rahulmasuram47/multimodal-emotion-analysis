# Project Progress Log

## Project: Multimodal Voice & Conversation Emotion Analysis System

A multimodal system that analyzes spoken conversations using:

- Audio processing
- Voice emotion recognition
- Speech-to-text
- Text sentiment analysis
- Multimodal fusion
- Conversation-level emotional analysis
- Speaker diarization support
- Streamlit dashboard visualization

---

# Milestones

## Milestone 1: Project Setup & Audio Verification — COMPLETED

### Built

- Configured the Python virtual environment.
- Standardized project dependencies.
- Updated `.gitignore`.
- Added audio test fixtures.
- Added `tests/test_audio_setup.py`.
- Verified native WAV audio processing.

### Environment

- Python: `3.11.9`

### Audio Libraries

- `librosa==0.11.0`
- `soundfile==0.14.0`
- `numpy==2.2.6`
- `scipy==1.15.3`

### Verification

- Successfully loaded 16 kHz audio.
- Successfully extracted 13 MFCC coefficients.
- Audio processing pipeline verified successfully.

### Notes

Native WAV/FLAC processing works correctly.

FFmpeg is required for broader audio-format support and the
faster-whisper workflow.

---

## Milestone 2: Audio Processing Module — COMPLETED

### Built

- `config.py`
- `app/audio_processor.py`

### Features

- Audio file validation.
- Supported-format checking.
- Corrupted-file detection.
- Empty-file detection.
- Mono conversion.
- 16 kHz resampling.
- Peak normalization.
- Duration validation.
- MFCC extraction.
- RMS energy extraction.
- Pitch/F0 extraction.
- Zero-crossing rate extraction.
- Spectral centroid extraction.

### Error Handling

Implemented:

- `AudioValidationError`
- `AudioProcessingError`

### Verification

Audio processing tests cover:

- Valid audio.
- Unsupported formats.
- Corrupted files.
- Empty files.
- Audio feature extraction.

---

## Milestone 3: Voice Emotion Analysis — COMPLETED

### Model

`superb/wav2vec2-base-superb-er`

### Emotion Classes

- Neutral
- Happy
- Angry
- Sad

### Implementation

Implemented Wav2Vec2-based voice emotion analysis with:

- Full-audio emotion prediction.
- Segment-level emotion prediction.
- Emotion probability scores.
- Prediction confidence.
- Timestamp-aware segment analysis.

### Verification

Successfully analyzed:

`sample_audio/real_voice.wav`

The model returned an emotion prediction and confidence scores.

The sample audio was predicted as **sad** with high confidence.

Segment-level emotion analysis was also verified successfully.

### Important Model Limitation

The current voice-emotion model is a pretrained general-purpose model.

Predictions can vary with:

- Speaker characteristics.
- Voice pitch.
- Speaking style.
- Recording conditions.
- Microphone quality.
- Background noise.
- Emotional intensity.

Therefore, individual voice-emotion predictions should be interpreted
as model estimates rather than ground-truth human labels.

---

## Milestone 4: Speech-to-Text — COMPLETED

### Model

`faster-whisper` using the `base` model.

### Configuration

- CPU inference.
- INT8 computation.
- Automatic language detection.

### Implementation

Added speech-to-text processing with:

- Automatic language detection.
- Language probability.
- Full transcript generation.
- Timestamped transcript segments.
- Voice-activity filtering.

### Verification

Successfully transcribed:

`sample_audio/real_voice.wav`

The system detected English and generated timestamped transcript
segments covering the recorded speech.

### Known Limitation

There is no automated correctness test for arbitrary speech-to-text
transcription because arbitrary uploaded recordings do not have a fixed
ground-truth transcript.

Transcription quality is therefore verified manually during testing.

---

## Milestone 5: Text Sentiment Analysis — COMPLETED

### Model

`cardiffnlp/twitter-roberta-base-sentiment-latest`

### Sentiment Classes

- Negative
- Neutral
- Positive

### Implementation

Added sentiment analysis for transcript segments.

Each prediction returns:

- Sentiment label.
- Sentiment confidence.
- Complete sentiment score distribution.

### Fallback

A fallback sentiment model is available:

`distilbert-base-uncased-finetuned-sst-2-english`

### Verification

Successfully generated sentiment predictions and confidence scores
for tested transcript segments.

---

## Milestone 6: Multimodal Fusion — COMPLETED

### Implementation

Combined:

- Voice emotion.
- Text sentiment.

using configurable fusion weights.

### Default Weights

- Audio: `0.50`
- Text: `0.50`

### Features

- Voice/text agreement detection.
- Voice/text mismatch detection.
- Weak disagreement handling.
- Combined confidence.
- Emotional direction.
- Multimodal interpretation.
- Segment-level fusion.

### Verification

Tested:

- Agreement scenarios.
- High-confidence mismatch scenarios.
- Neutral agreement scenarios.
- Low-confidence disagreement handling.

The fusion pipeline correctly identifies agreement and disagreement
between the two modalities according to the configured rules.

### Design Principle

The fusion logic is intentionally rule-based and explainable.

The system does not use a black-box learned fusion model.

The system also does not interpret a voice/text mismatch as proof of
deception, dishonesty, or hidden intent.

---

## Milestone 7: Conversation Analysis — COMPLETED

### Implementation

Added:

`app/conversation_analysis.py`

### Features

- Timestamped conversation segmentation.
- Segment-level voice emotion.
- Segment-level text sentiment.
- Segment-level multimodal fusion.
- Emotional intensity calculation.
- Emotional trend detection.
- Escalation analysis.
- Full conversation transcript.
- Language detection.
- Language confidence.
- Segment-level analysis results.

### Emotional Trend Detection

The system identifies an overall emotional trend as:

- Increasing.
- Decreasing.
- Stable.

### Escalation Detection

Escalation analysis requires sufficient segment data.

The implementation:

- Requires at least three usable segments.
- Requires continuously increasing intensity across consecutive
  segments.
- Requires a total intensity increase of at least `0.20`.
- Reports insufficient data when fewer than three usable segments are
  available.

### Verification

Conversation-analysis functionality was verified through automated
tests and end-to-end dashboard analysis.

The system successfully produces:

- Conversation transcript.
- Segment timestamps.
- Voice emotion.
- Text sentiment.
- Fusion results.
- Emotional intensity.
- Overall emotional trend.
- Escalation status.

---

## Milestone 8: Speaker Diarization Support — COMPLETED

### Implementation

Added:

`app/diarization.py`

### Tests

Added:

`tests/test_diarization.py`

### Purpose

The diarization module provides architectural support for identifying
different speakers within an audio conversation.

### Design Decision

The system does not fabricate speaker identities when reliable
diarization information is unavailable.

Speaker-wise emotion and conversation analysis therefore remain
dependent on obtaining reliable speaker labels.

### Current Status

- Diarization module implemented.
- Diarization tests implemented.
- Speaker-aware architecture prepared.
- Full production-ready speaker-wise conversation analysis remains a
  future enhancement.

This distinction is intentionally documented so that the project does
not claim functionality that cannot be reliably demonstrated.

---

## Milestone 9: Streamlit Dashboard — COMPLETED

### Implementation

Added:

`streamlit_app.py`

### Dashboard Capabilities

The dashboard supports:

- Audio upload.
- Audio playback.
- End-to-end analysis.
- Reset functionality.
- Cached model loading.
- Full transcript display.
- Language detection.
- Language confidence.
- Audio duration.
- Segment count.
- Emotional trend.
- Intensity change.
- Escalation status.
- Timestamped conversation segments.
- Voice emotion.
- Voice emotion confidence.
- Text sentiment.
- Text sentiment confidence.
- Fusion confidence.
- Emotional intensity.
- Voice/text agreement.
- Voice/text mismatch.
- Multimodal interpretation.
- Segment-level expandable analysis.
- Exportable analysis results.
- Graceful error handling.

### UI/UX Enhancement

The dashboard was refined into a technical, structured interface with:

- Clear project branding.
- Pipeline/workflow visualization.
- Dark technical dashboard design.
- Organized result sections.
- Modality-specific result cards.
- Fusion explanation.
- Conversation-dynamics visualization.
- Emotional intensity chart.
- Segment-level analysis cards.
- Audio playback area.
- Export controls.
- Reset controls.
- Responsive Streamlit layout.

### Dashboard Verification

The dashboard was successfully launched using:

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none

The dashboard was manually verified using the project sample audio and
confirmed to display real pipeline outputs.

No fabricated demonstration results are used.

Milestone 10: Testing & Documentation — COMPLETED
Automated Test Suite

The project currently contains tests covering the major functional
components of the system.

Test coverage includes:

Audio setup.
Audio preprocessing.
Audio validation.
Acoustic feature extraction.
Multimodal fusion.
Agreement detection.
High-confidence mismatch detection.
Neutral agreement.
Conversation trend detection.
Escalation detection.
Insufficient-segment handling.
Sentiment analysis.
Sentiment score integrity.
Diarization module behavior.
Test Files

The current test suite includes:

tests/test_audio_setup.py
tests/test_audio_processor.py
tests/test_fusion.py
tests/test_conversation_analysis.py
tests/test_sentiment_analyzer.py
tests/test_diarization.py
Current Verification

Full automated test suite:

57 passed
0 failed

Latest recorded test execution:

57 passed, 4 warnings in 131.25s

The warnings are third-party dependency deprecation warnings and do not
represent test failures.

Speech-to-Text Testing Limitation

There is no automated correctness test for arbitrary
speech_to_text.py transcription output.

Whisper transcription does not have a fixed expected output for
arbitrary audio files without a ground-truth transcript.

Therefore, speech-to-text functionality is verified through manual
testing using sample audio rather than through a fixed transcript
assertion.

This is documented as a testing limitation rather than an unresolved
software defect.

Current Project Status

The complete functional pipeline is now implemented:

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

Supporting speaker-diarization architecture and tests are also
included.

Completed Components
Component	Status
Project Setup	✅ Complete
Audio Processing	✅ Complete
Voice Emotion Analysis	✅ Complete
Speech-to-Text	✅ Complete
Text Sentiment Analysis	✅ Complete
Multimodal Fusion	✅ Complete
Conversation Analysis	✅ Complete
Speaker Diarization Support	✅ Complete
Streamlit Dashboard	✅ Complete
UI/UX Enhancement	✅ Complete
Automated Testing	✅ Complete
Documentation	✅ Complete
Validation Summary

The current implementation has been validated through:

Automated unit and integration tests.
Sample-audio processing.
Voice-emotion model inference.
Speech-to-text inference.
Sentiment analysis.
Multimodal fusion scenarios.
Conversation trend analysis.
Escalation detection.
Diarization module tests.
Manual Streamlit dashboard verification.

Latest automated test result:

57 passed, 0 failed
Known Limitations

The project is functional but still has several limitations that are
important for academic evaluation.

1. Voice Emotion Accuracy

The pretrained voice-emotion model is not specifically trained on this
project's dataset.

Therefore, predictions are model estimates and may be affected by
speaker characteristics and recording conditions.

2. Speech-to-Text Ground Truth

Arbitrary uploaded audio does not have a predefined transcript, so
automated transcription accuracy cannot be verified without a
ground-truth dataset.

3. Speaker-Wise Analysis

Diarization support is present, but complete production-ready
speaker-wise emotion analysis is not claimed as fully implemented.

4. CPU Performance

The system is designed to operate on CPU so that it can be demonstrated
on a normal laptop.

Longer recordings may therefore require additional processing time.

5. Multimodal Fusion

The fusion method is intentionally rule-based and transparent rather
than learned from a large multimodal training dataset.

Therefore, it should be considered an explainable integration method,
not a newly trained multimodal emotion-recognition model.

Future Development

Possible future enhancements include:

Full production-ready speaker diarization.
Complete speaker-wise emotion analysis.
More advanced speaker-level conversation analytics.
Configurable fusion weights through the dashboard.
Additional emotion categories.
Improved emotional visualization.
Automated regression testing using a curated labeled dataset.
Performance optimization for longer recordings.
GPU acceleration.
Improved multilingual speech and sentiment support.
Additional dashboard refinements.
Documentation Status

The following project documentation is maintained:

README.md — Project overview, architecture, setup, usage,
limitations, and current status.
PROJECT_SPEC.md — Project requirements, technical decisions,
models, and design specification.
PROGRESS.md — Development milestones and implementation history.
Git Milestone

The functional milestone has been tagged as:

v1.0-functional

The project also contains the completed UI/UX enhancement work and the
latest automated test suite.

Final Status

Multimodal Voice & Conversation Emotion Analysis System

✅ Core project setup complete.
✅ Audio processing complete.
✅ Voice emotion analysis complete.
✅ Speech-to-text complete.
✅ Text sentiment analysis complete.
✅ Multimodal fusion complete.
✅ Conversation analysis complete.
✅ Speaker diarization support complete.
✅ Streamlit dashboard complete.
✅ UI/UX enhancement complete.
✅ Automated testing complete.
✅ Documentation complete.
✅ 57/57 automated tests passing.

The project is currently suitable for functional demonstration,
testing, documentation review, and academic evaluation.

---

## Phase 2: Modality Consistency Trajectory & Explainable Attention Score — COMPLETED

### Implementation

#### TASK 1 — Modality Consistency Trajectory

Added: `app/consistency_analysis.py`

The module tracks the relationship between voice emotion and text
sentiment across conversation segments using the project's existing
emotional-direction mapping (positive / negative / neutral).

**Segment-level classifications:**

- `agreement` — both modalities point in the same emotional direction.
- `divergence` (mismatch/divergence) — high-confidence disagreement
  between voice and text directions.
- `low_confidence` (insufficient data) — at least one modality lacks
  sufficient confidence to classify the relationship conclusively.

**Conversation-level trajectory:**

- `mostly_consistent`  — agreement_rate ≥ 0.70
- `mostly_divergent`   — agreement_rate < 0.30
- `mixed`              — 0.30 ≤ agreement_rate < 0.70
- `insufficient_data`  — fewer than 2 usable segments

**Wording rules (strictly enforced):**

- This analysis NEVER implies that either modality represents a
  person's true or actual internal emotional state.
- It is NOT lie detection, truth detection, or concealment detection.
- "Agreement" and "divergence" refer only to whether the two
  computational signals point in the same direction — nothing more.

#### TASK 2 — Review-Priority Attention Score

**Formula (explicitly documented):**

```
attention_score = W_MISMATCH   × mismatch_rate
               + W_INTENSITY  × average_intensity
               + W_ESCALATION × escalation_indicator
```

**Default weights (heuristic; NOT scientifically validated):**

| Weight       | Default | Component                  |
|--------------|---------|----------------------------|
| W_MISMATCH   | 0.40    | Fraction of mismatch segs  |
| W_INTENSITY  | 0.35    | Mean emotional intensity   |
| W_ESCALATION | 0.25    | 1.0 if escalation detected |

Scores are clamped to [0.0, 1.0].  Each component is non-negative.

**This score is a COMPUTATIONAL REVIEW-PRIORITY HEURISTIC ONLY.**
It is NOT:
- a risk score
- a danger score
- a distress indicator
- a psychological assessment
- a clinical measure

The `to_dict()` output embeds the formula, per-component
contributions, and a mandatory disclaimer notice.

**Weights:** Users may configure weights.  Any change from the
defaults is the responsibility of the user and should be documented.
Defaults are labelled as heuristic and are not scientifically
validated.

#### Integration into `ConversationAnalyzer.analyze()`

`app/conversation_analysis.py` was updated to:

1. Import `compute_consistency_trajectory` and `compute_attention_score`
   from `app/consistency_analysis.py`.
2. Call both after escalation detection and before returning the
   final result dict.
3. Expose the results as `consistency_trajectory` and `attention_score`
   keys in the returned dict (serialised via `.to_dict()`).

No models, preprocessing steps, fusion weights, confidence thresholds,
or production inference behaviour were modified.

### Tests

Test file: `tests/test_consistency_analysis.py`

**Coverage (all required scenarios):**

| Scenario | Test(s) |
|---|---|
| Agreement classification | `test_agreement_classification` |
| Mismatch/divergence classification | `test_divergence_classification` |
| Low-confidence / insufficient data | `test_low_confidence_classification` |
| Mostly-consistent trajectory | `test_all_agreement_gives_mostly_consistent` |
| Mostly-divergent trajectory | `test_all_divergence_gives_mostly_divergent` |
| Mixed consistency | `test_mixed_trajectory` |
| Insufficient data (0 segs) | `test_insufficient_data_for_zero_segments` |
| Insufficient data (1 seg) | `test_insufficient_data_for_single_segment` |
| Deterministic attention score | `test_deterministic_output` |
| Score range [0.0, 1.0] | `test_score_in_valid_range` |
| Mismatch contribution = 0 for all-agreement | `test_mismatch_contribution_zero_for_all_agreement` |
| Intensity contribution still > 0 for all-agreement | `test_intensity_contribution_nonzero_when_mismatch_is_zero` |
| Escalation contribution | `test_escalation_contribution_is_nonzero_with_escalation`, `test_escalation_contribution_is_zero_without_escalation` |
| High-mismatch scores higher (equal intensity/escalation) | `test_high_mismatch_scores_higher_than_low_mismatch` |
| Formula addition holds | `test_score_formula_addition_holds` |
| Components non-negative | `test_components_are_non_negative` |
| Notice/formula in to_dict() | `test_attention_score_to_dict_contains_notice` |

**NOTE:** We do NOT test that all-agreement gives zero total attention,
because the intensity component may still contribute.  We DO verify
that `mismatch_contribution` is exactly 0.0 for all-agreement cases.

### Human Decisions Still Required

The following items require human judgement before using in
production:

1. **Attention-score weights** — The defaults (W_MISMATCH=0.40,
   W_INTENSITY=0.35, W_ESCALATION=0.25) are heuristic values.
   They are NOT scientifically validated.  Anyone deploying this
   system must evaluate whether these weights are appropriate for
   their specific use case and population.

2. **Trajectory thresholds** — The boundaries (≥ 0.70 for
   mostly_consistent, < 0.30 for mostly_divergent) are heuristic
   cut-points. They have not been calibrated against a labelled
   dataset.

3. **MIN_TRAJECTORY_SEGMENTS = 2** — The minimum segment count
   required to report a trajectory is a heuristic choice.

4. **Operational interpretation** — All outputs are computational
   signals only.  Human professional judgement is required before
   acting on any attention score or consistency trajectory result.

### Files Changed

| File | Change |
|---|---|
| `app/consistency_analysis.py` | New file (Tasks 1 & 2) |
| `app/conversation_analysis.py` | Added imports + wired trajectory & attention score into `analyze()` |
| `tests/test_consistency_analysis.py` | New file (Task 3 — 16 tests) |
| `PROGRESS.md` | Appended Phase 2 milestone |

### Verification

Full automated test suite passed with 0 failures.

---

## Milestone: Phase 3 — Empirical Evaluation, Learned Fusion Baseline & Modality Incongruence — COMPLETED

### Built & Implemented

1. **Empirical Evaluation Pipeline (`evaluation/run_evaluation.py`)**:
   - Automated evaluation workflow executed via `python -m evaluation.run_evaluation`.
   - Ingests `data/evaluation_dataset_template.csv` and verifies ground truth presence.
   - Computes voice emotion accuracy, macro F1, confusion matrix, and speaker count exact match / MAE.
   - Detects missing annotations (e.g. text sentiment, reference transcripts) and marks them "Pending Data" without fabricating metrics.
   - Produces machine-readable `evaluation/results/latest_evaluation.json` and human-readable `evaluation/results/latest_evaluation.txt`.
   - Preserves all pre-existing evaluation artifacts for backward compatibility.

2. **Learned Fusion Baseline (`evaluation/learned_fusion.py`)**:
   - Implemented an experimental Logistic Regression fusion baseline using `scikit-learn`.
   - Features extracted: 12-dimensional numeric vector combining acoustic emotion probabilities, text sentiment probabilities, confidences, intensity, and agreement/mismatch indicators.
   - Requires at least 10 labelled samples for train/test split; reports "Insufficient labelled data" when insufficient annotations exist.
   - Strictly preserves `app/fusion.py` as primary production fusion:
     > "The weighted rule-based fusion remains the production strategy. Learned fusion is an experimental baseline used for comparison."

3. **Modality Ablation Comparison (`evaluation/ablation.py`)**:
   - Extended ablation comparison framework to include Voice-Only, Text-Only, Rule-Based Fusion, and Learned Fusion.
   - Displays "Pending Data" for unannotated modalities without creating placeholder numbers.

4. **Modality Incongruence & Conflict Detection (`app/incongruence_analysis.py`)**:
   - Deterministic conflict detection for opposing acoustic and text signals (Positive Text + Angry/Sad Voice, Negative Text + Happy Voice).
   - Configurable confidence thresholds (min 0.40, high 0.65) to prevent false-alarm conflicts on low-confidence outputs.
   - Categorized conflict strength: `none`, `low`, `medium`, `high`.
   - Mandatory scientific disclaimers:
     > "Modality incongruence is not equivalent to sarcasm, deception, or psychological state detection."
   - Never claims sarcasm detection, lie/deception detection, or psychological diagnosis.

5. **Dashboard Visibility (`streamlit_app.py`)**:
   - Added dedicated "📊 Multimodal Validation" tab and section displaying dataset status, labelled sample counts, model comparison table, and confusion matrices (or clear insufficient data notices).
   - Added "MODALITY INCONGRUENCE" cards at both overall conversation level and per-turn segment levels with conflict levels, signals, interpretations, and safety disclaimers.

6. **Comprehensive Automated Tests (`tests/`)**:
   - `tests/test_incongruence_analysis.py` (12 tests)
   - `tests/test_learned_fusion.py` (6 tests)
   - `tests/test_evaluation_runner.py` (5 tests)
   - Total test suite expanded to **138 tests**, all passing (100%).

7. **Project Rule Compliance**:
   - `PROJECT_SPEC.md` was verified strictly untouched and read-only.
   - Production fusion weights (`0.50` audio, `0.50` text, threshold `0.40`) remain completely intact.
   - Zero fabricated empirical results.

### Files Changed

| File | Change |
|---|---|
| `app/incongruence_analysis.py` | New file: deterministic conflict detection & safety disclaimers |
| `evaluation/learned_fusion.py` | New file: Logistic Regression baseline on 12-dim multimodal features |
| `evaluation/ablation.py` | Modified: added learned fusion baseline & comparison table |
| `evaluation/run_evaluation.py` | Modified: added latest_evaluation.json/.txt, learned baseline integration |
| `streamlit_app.py` | Modified: added Multimodal Validation section and Incongruence cards |
| `tests/test_incongruence_analysis.py` | New file: 12 tests for conflict detection and safety logic |
| `tests/test_learned_fusion.py` | New file: 6 tests for feature extraction and training fixtures |
| `tests/test_evaluation_runner.py` | New file: 5 tests for runner, pending data, and reporting |
| `README.md` | Modified: documented Phase 3 evaluation, learned baseline, incongruence |
| `PROGRESS.md` | Modified: logged Phase 3 completion |

### Verification

All 138 automated unit tests passed with 0 failures. Production rule-based fusion, models, and `PROJECT_SPEC.md` remain completely protected.

---

## Milestone: Phase 4 — UI/UX Enhancements, Microphone Workflow, Decision-Support Recommendations & System Hardening — COMPLETED

### Built & Implemented

1. **Part 1: Real-Time Browser Microphone Workflow (`app/microphone.py`)**:
   - Integrated browser-native audio recording using `st.audio_input` (Streamlit 1.63.0) capturing 16 kHz audio without requiring external C audio drivers.
   - Dual-tab ingestion layout: `[ Upload Audio ]` and `[ Record from Microphone ]`.
   - Temporary WAV buffer persistence (`save_audio_buffer`) passing directly into `ConversationAnalyzer.analyze()`, ensuring identical pipeline reuse (AudioProcessor → Wav2Vec2 → Whisper → RoBERTa → Fusion → Conversation Analysis).
   - Mandatory environment fallback: When microphone capture is unsupported, restricted, or disabled (`DISABLE_MICROPHONE_CAPTURE=1`), displays non-breaking notice:
     > *"Microphone capture is unavailable in this environment. Please use the audio upload workflow."*
   - Upload workflow remains fully operational under all conditions.

2. **Part 2: UI Wiring, Escalation Bugfix & Export Hardening**:
   - **Escalation Intensity Key Fix**: Corrected key extraction in `streamlit_app.py` from `intensity_increase` to `intensity_change`, fixing the issue where net intensity growth displayed `+0.00` regardless of backend values.
   - **Wired Orphaned Panels**: Explicitly invoked `render_model_transparency_section()` and `render_professional_limitations_section()` in the main dashboard flow under Explainability & Export.
   - **Categorized Pipeline Error Handling**: Replaced generic exception displays with user-facing actionable guidance for corrupted audio formats, silent/empty audio, short audio, and model inference errors, while preserving full technical details in application logging.
   - **Structured JSON Export**: Additively populated `notices_and_limitations` in the downloaded JSON report with decision-support disclaimers, deception non-detection notices, and operational limitations without modifying existing schema fields.

3. **Part 3: Demo-Friendly Primary Workflow & Health Check**:
   - Reorganized the empty landing state into a clear 7-step primary workflow guide:
     `01. Ingest (Upload/Record) → 02. Analyze Audio → 03. Review Results → 04. Acoustic Signals → 05. Conflict & Escalation → 06. Recommendations → 07. Export Report`.
   - Comprehensive health check verifying zero regressions across all phases.
   - Preserved `PROJECT_SPEC.md` as locked and read-only.
   - Preserved production rule-based fusion and locked ML models.

### Files Created & Modified in Phase 4

| File | Change |
|---|---|
| `app/microphone.py` | New file: microphone availability checks, audio buffer persistence, fallback messaging, export helpers |
| `streamlit_app.py` | Modified: dual input tabs, escalation bugfix, panel wiring, categorized error handling, additive export, 7-step workflow landing |
| `tests/test_microphone_workflow.py` | New file: 9 tests for microphone availability, fallbacks, and pipeline routing |
| `tests/test_ui_wiring.py` | New file: 8 tests for escalation key, panel invocations, error categorization, and JSON export |
| `README.md` | Modified: documented Phase 4 features, feature categorization, updated progress and status |
| `PROGRESS.md` | Modified: logged Phase 4 completion |

### Verification Outcome

- **Automated Test Suite**: **162 passed / 0 failed, 3 warnings** across 15 test suites (100% passing).
  - Baseline: 145 passed
  - Phase 4 Part 1: +9 passed (154 total)
  - Phase 4 Part 2: +8 passed (162 total)
  - Zero regressions.
- **Microphone Status**: Verified active and operational via `st.audio_input` in Streamlit 1.63.0; verified fallback message triggers reliably when disabled.
- **Integrity**: `PROJECT_SPEC.md` strictly unchanged (`git diff -- PROJECT_SPEC.md` → 0 lines).