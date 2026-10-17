# Multimodal Voice & Conversation Emotion Analysis System â€” Progress Log

## Current Status & Verification Summary

- **Automated Test Suite**: **270 passed**, 0 failed (verified on Windows 11, Python 3.11).
- **Active Git Branch**: `phase-6-diarization`.
- **Phase 6 Status**: **IN PROGRESS** (speaker-aware analytics, truthfulness badges, and mock tests complete; untagged pending real 2-speaker audio recording run per hard constraint).
- **Hard Constraints**:
  - Front-end: Streamlit only with pure injected CSS (`tokens.css`, `components.css`).
  - Model families: Wav2Vec2 (`superb/wav2vec2-base-superb-er`), Faster-Whisper (INT8 CPU), RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`), Pyannote (`pyannote/speaker-diarization-community-1`).
  - Fusion: Pure rule-based, deterministic, weighted arbitration (no sklearn in production pipeline).
  - Modalities: Voice audio and text only (zero facial/video).
  - Framing: Strictly analytical signals for review (never claiming deception, lie detection, or psychological intent).
  - Absolute data integrity: No fabricated metrics or demonstration results.

---

## v1 Milestones (Historical Baseline)

### Milestone 1: Project Setup & Audio Verification (`v1.0-baseline`)
- Configured Python 3.11 virtual environment and standardized core dependencies (`librosa`, `soundfile`, `numpy`, `scipy`).
- Built `tests/test_audio_setup.py` verifying 16 kHz WAV loading and 13 MFCC coefficient extraction.

### Milestone 2: Audio Processing Module
- Implemented `app/audio_processor.py` for file validation, resampling to 16 kHz mono, peak normalization, and acoustic feature extraction (MFCCs, RMS, F0 pitch via Yin, ZCR, spectral centroid).

### Milestone 3: Voice Emotion Analysis
- Implemented `app/emotion_analyzer.py` wrapping `superb/wav2vec2-base-superb-er` with 4-class emotion mapping (`neutral`, `happy`, `angry`, `sad`).

### Milestone 4: Speech-to-Text
- Implemented `app/speech_to_text.py` using `faster-whisper` (`base` model, INT8 CPU) with VAD filtering and timestamped segment extraction.

### Milestone 5: Text Sentiment Analysis
- Implemented `app/sentiment_analyzer.py` wrapping CardiffNLP Twitter-RoBERTa for 3-class sentiment prediction (`positive`, `neutral`, `negative`).

### Milestone 6: Multimodal Fusion
- Implemented `app/fusion.py` combining audio emotion and text sentiment via confidence-weighted scores and broad direction comparison.

### Milestone 7: Conversation Analysis
- Implemented `app/conversation_analysis.py` integrating audio processing, transcription, emotion analysis, sentiment analysis, and multi-turn trend tracking.

### Milestone 8: Speaker Diarization Support
- Implemented `app/diarization.py` wrapping Pyannote speaker diarization with gated HuggingFace token authentication.

### Milestone 9: Streamlit Dashboard (`v1.0-functional`)
- Created unified multi-tab Streamlit interface in `streamlit_app.py`.
- Verified launch command:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
  ```

### v1 Milestone A: Modality Consistency Trajectory & Attention Score (`v1.2-phase2-complete`)
- Implemented `app/consistency_analysis.py` providing turn-by-turn agreement/divergence classification and review-priority attention score heuristic.

### v1 Milestone B: Empirical Evaluation & Learned Baseline (`v1.3-phase3-complete`)
- Implemented `evaluation/harness.py`, `evaluation/dataset.py`, and `evaluation/metrics.py`.
- Added experimental Logistic Regression fusion baseline for academic comparison only.

### v1 Milestone C: UI/UX Enhancements & Hardening (`v1.4-phase4-complete`)
- Built microphone recording workflow in `app/microphone.py`.
- Added decision-support recommendation engine in `app/recommendation_engine.py`.

---

## v2 Upgrade Phases

### v2 Phase 0: Repo Hygiene, Test Audit & Stabilization
- **What Changed**:
  - Full codebase audit of untested modules under `app/` and `ui/`.
  - Discarded stray temporary files and migrated uncommitted changes to feature branch `phase-6-diarization`.
  - Fixed critical bug where sentiment analysis was nested inside voice calibration's `except` block with an unconditional return `None`.
  - Fixed slider key collisions and session state initialization in Streamlit UI.
  - Closed unclosed code fences in documentation and removed stale test counts.
- **Files Modified/Added**: `app/conversation_analysis.py`, `streamlit_app.py`, `tests/`.
- **Measurements**: Baseline test suite expanded from 183 to 231 passing tests.

### v2 Phase 1: UI Modularization & CSS Externalization (`v1.1-phase1-complete`)
- **What Changed**:
  - Refactored monolithic `streamlit_app.py` into modular components under `ui/components/` and `ui/pages/`.
  - Externalized inline CSS into `ui/styles/tokens.css` and `ui/styles/components.css`.
  - Added strict UI contract tests verifying rendering safety without runtime exceptions.
- **Files Modified/Added**: `ui/components/header.py`, `ui/components/verdict_card.py`, `ui/components/modality_cards.py`, `ui/components/segment_timeline.py`, `ui/styles/tokens.css`.
- **Measurements**: Automated UI contract suite added 18 tests passing.

### v2 Phase 2: Evaluation Tooling & Annotator Merge (`v1.2-phase2-complete`)
- **What Changed**:
  - Implemented multi-annotator resolution workflow in `evaluation/scripts/merge_annotations.py`.
  - Added Cohen's kappa inter-annotator agreement calculations for voice emotion, text sentiment, and escalation.
  - Added Dataset Dashboard and error analysis reporting in `evaluation/results/ERROR_ANALYSIS.md`.
  - Strictly enforced data integrity: unannotated samples are marked unannotated with zero fabricated metrics.
- **Files Modified/Added**: `evaluation/scripts/merge_annotations.py`, `tests/test_annotation_merge.py`, `tests/test_error_analysis.py`.
- **Measurements**: 14 annotation tests passing; zero invented ground-truth labels.

### v2 Phase 3: Voice Model Benchmark & Collapse Diagnostics (`v1.3-phase3-complete`)
- **What Changed**:
  - Implemented `evaluation/voice_diagnostics.py` auditing Wav2Vec2 input format (16 kHz mono float32), feature extractor normalization, per-class mean probabilities, and entropy.
  - Implemented automatic Collapse Detector in `app/voice_calibration.py`: triggers on-screen warning and downweights voice reliability when one class dominates > 85% of predictions.
  - Implemented short-segment acoustic window widening (0.5s padding) with reliability penalization (0.60) for segments < 1.0s.
- **Files Modified/Added**: `app/voice_calibration.py`, `evaluation/voice_diagnostics.py`, `tests/test_voice_calibration.py`, `tests/test_voice_collapse.py`.
- **Measurements**: 22 calibration and collapse detector tests passing.

### v2 Phase 4: Prosody-Based Intensity & Robust Escalation (`v1.4-phase4-complete`)
- **What Changed**:
  - Rebuilt prosody in `app/prosody.py`: computes relative loudness against segment's own 10th/90th dB percentiles, F0 semitone range/variation from Yin, voiced speech ratio, and Whisper word speech rate.
  - Formulated Intensity v2 combining prosodic arousal, acoustic arousal, and text negativity strength.
  - Implemented tolerant escalation detection in `app/conversation_analysis.py`: OLS linear regression slope, dip tolerance (0.08), net rise threshold (0.20).
  - Unified Conversation Verdict: `escalating`, `de-escalating`, `fluctuating`, `stable`.
- **Files Modified/Added**: `app/prosody.py`, `app/conversation_analysis.py`, `tests/test_prosody.py`, `tests/test_conversation_verdicts.py`.
- **Measurements**: 28 prosody and escalation tests passing.

### v2 Phase 5: Transparent Declarative Fusion & Text Reliability (`v1.5-phase5-complete`)
- **What Changed**:
  - Replaced ad-hoc branching with a 10-rule declarative pattern table in `app/fusion.py`.
  - Implemented dynamic weight renormalization: adjusts modality weights according to `voice_reliability` and `text_reliability`.
  - Implemented STT text reliability derivation from faster-whisper `avg_logprob` and `no_speech_prob`.
  - Configurable Whisper model size (`base` / `small`, INT8 CPU) and language selector.
- **Files Modified/Added**: `app/fusion.py`, `app/speech_to_text.py`, `config.py`, `tests/test_fusion_rules.py`, `tests/test_text_reliability.py`.
- **Measurements**: Full suite expanded to 247 passing tests.

### v2 Phase 6: Speaker Diarization & Analytics (`phase-6-diarization`, IN PROGRESS)
- **What Changed**:
  - Truthful status badges: Real-time checks for HuggingFace token and Pyannote runtime state (`Active`, `Off â€” No Token`, `Error`).
  - Implemented multi-speaker analytics: talk-time share, per-speaker intensity trajectories and verdicts, escalation initiator heuristic, and speaker count estimation.
  - Implemented "Uncertain Speaker" assignment for low-overlap segments.
  - Secure credential handling: verified `.env` is gitignored and token never leaks in application logs.
- **Tasks NOT Done**:
  - **Phase 6 Tag Deferred**: Per hard constraint, `v2.0-phase6-complete` will only be tagged after a real 2-speaker audio file is processed through the complete pipeline.
- **Files Modified/Added**: `app/diarization.py`, `app/speaker_analytics.py`, `docs/DIARIZATION_SETUP.md`, `tests/test_diarization.py`, `tests/test_diarization_badges.py`.

---

## Novelty Additions (Prompt G)

1. **Abstention Gating**:
   - Implemented `abstention_threshold: float = 0.35` in `config.py` and `app/fusion.py`.
   - When combined modality reliability drops below threshold, outputs `"Insufficient reliability. Review audio"` instead of forcing an unreliable label.
2. **Deterministic Counterfactual Sensitivity**:
   - Segment-level: Evaluates what text confidence change, tone shift, or reliability increase would alter the rule output.
   - Conversation-level: Evaluates verdict sensitivity against dip tolerance drops and slope variations.
3. **Fusion Rule Audit Table**:
   - Evaluates rule firing frequency and human-label agreement rate across the evaluation dataset.
4. **MFCC Timbral Feature Summary & Modality Consistency Trajectory**:
   - MFCC-13 acoustic inspection summary exposed in Acoustic Inspection tab.
   - Turn-by-turn Modality Consistency Trajectory displayed with colored relationship ribbon.
5. **Empirical Research Questions**:
   - Documented RQ1 (prosodic intensity vs model confidence), RQ2 (reliability-weighted rule fusion vs unweighted), and RQ3 (counterfactual interpretability) in `README.md`.

---

## UI/UX Redesign & Real Visuals (Prompts H & I)

1. **Design System & Semantic Tokens (`ui/styles/tokens.css`)**:
   - Modern semantic colors (`--voice-primary`, `--text-primary`, `--fused-primary`, `--warning`, `--critical`).
   - Dark mode default with full light mode support.
   - Accessible WCAG AA compliance verified via `scripts/check_contrast.py`.
   - Accessible `:focus-visible` outlines and `@media (prefers-reduced-motion)`.
2. **Reordered Results Experience**:
   - (a) Verdict Hero: Verdict badge, narrative explanation, review-priority attention gauge, and recommendations.
   - (b) Modality Cards: Audio Emotion, Text Sentiment, and Multimodal Fusion cards with live reliability chips.
   - (c) Dynamics & Timeline: Agreement strip ribbon, 2D Valence-Arousal trajectory, and Intensity trajectory chart.
   - (d) Detailed Tabs: Segments, Acoustic Inspection (with MFCC-13), Why This Result?, Validation, and Annotation Helper.
3. **Interactive Charts (`ui/charts/dynamics.py`)**:
   - Real Plotly charts for Valence-Arousal 2D dynamics, OLS regression slope with +-0.08 tolerance band, and multi-speaker comparisons.
4. **Offline Resilience**:
   - Vendored WaveSurfer.js under `ui/assets/` for self-contained execution without external CDN dependencies.

---

## Verification & Test Evolution

| Phase | Milestone Description | Passing Tests |
|---|---|---|
| Baseline | Initial project verification | 183 |
| Phase 0-1 | Stabilization, UI modularization, contract tests | 201 |
| Phase 2-3 | Annotation merge, voice diagnostics, collapse detector | 223 |
| Phase 4-5 | Prosody v2, tolerant escalation, declarative rules | 247 |
| Phase 6 + Novelty | Counterfactuals, VAD resilience, MFCC & consistency | **270** |

All **270 tests** pass cleanly with `python -m pytest -q`.

---

## fix/final-polish — Tier 0: Theme System (2026-10-02)

**Branch:** ix/final-polish  **Commit:** 427c03f

### What changed
- **.streamlit/config.toml** — already had explicit [theme] (ase="dark", locked colours, 	oolbarMode="minimal"). Verified no changes needed.
- **ui/styles/tokens.css** — Complete rewrite: added canonical CSS-variable tokens (--surface, --surface-raised, --text-muted, --accent, --voice, --text-modality, --fused, --positive, --warning, --critical). Light theme variant via [data-theme="light"] selector. All old token names preserved as aliases so existing code is unaffected.
- **ui/styles_loader.py** — Now reads st.session_state["theme"] (default "dark") and injects the appropriate CSS variable overrides on every load_css() call, enabling live theme switching without a page reload.
- **ui/styles/layout.css** — .stApp background, sidebar background, sidebar spec rows all converted from hardcoded hex to token variables.
- **ui/styles/components.css** — Brand container, tabs, file uploader, metric cards, buttons, modality cards, status badges, step cards: all colours converted to token variables. Fixed invisible elements in light mode (uploader help icon, Play button, empty-state cards, meter bars).
- **ui/charts/theme.py** — **New file.** get_chart_theme_tokens() + pply_chart_theme(fig) helper. All Plotly charts must use this for consistent dark/light appearance.
- **ui/charts/dynamics.py** + **ui/charts/inspection.py** — Replaced all hardcoded paper_bgcolor/plot_bgcolor/ont.color with pply_chart_theme() calls.
- **ui/components/header.py** — Added Appearance radio (Dark / Light) in sidebar; on change writes st.session_state["theme"] and calls st.rerun(). Removed HF token display entirely — now shows only "? Configured / ? Not configured / ? Error" diarization status.
- **ui/components/waveform_player.py** — Theme-aware iframe colours; height raised from 135 ? 155 px to prevent Play button clipping; dark/light CSS vars injected inline.
- **scripts/check_contrast.py** — **New file.** WCAG 2.1 contrast-ratio checker. Tests 9 text tokens × 3 background surfaces × 2 themes = 54 pairs; all pass AA (= 4.5:1). Exposes un_contrast_audit() -> bool called by 	ests/test_ui_contracts.py.
- **	ests/test_audio_features.py** — Wrapped bare-script body in 	est_extract_acoustic_features() to fix pytest collection error.
- **	ests/test_audio_setup.py** — Added 	est_milestone1_verification() to fix pytest collection error.

### Why
Root cause was Streamlit following the browser/OS light theme while all custom CSS assumed dark — text and cards became invisible in light mode. The fix centralises every colour as a CSS variable and locks the Streamlit theme explicitly.

### Measured
- **Tests before:** 183 passed (prior stable baseline, 2-upgrade)
- **Tests after:** 272 passed, 3 warnings (all new tests added in Phases 2-6 + Tier 0 test fixes)
- **Contrast audit:** 54 / 54 pairs = 4.5 : 1 WCAG AA (dark and light themes)
- python -m pytest -q exit code 0

### Tasks NOT done in this tier
- Tiers 1–N (layout polish, chart accessibility, remaining UI bugs) not yet started.

---

## fix/final-polish — Tier 1: Rendering, Contradictions, and Single Source of Truth (2026-10-02)

**Branch:** ix/final-polish  **Commit:** cb8b9d5

### What changed
- **ui/utils.py** — Added html_block() helper which dedents and strips leading spaces from every line, guaranteeing no line ever contains 4+ leading spaces. Added canonical emotion_code_label() mapping (
eu -> Neutral, hap -> Happy, ng -> Angry, sad -> Sad). Added compute_ols_slope() as the single source of truth for OLS slope and net change. Added predominantly_label() implementing the >= 60% rule.
- **ui/charts/dynamics.py** — Fixed ribbon turn indexing to 1-based Turn 01, Turn 02. Integrated compute_ols_slope() and verdict slope as single source of truth. When < 3 turns: hide OLS fitted slope and tolerance band, display "Too few turns to assess a trend", and omit slope claim from legend.
- **ui/components/verdict_card.py** — Key finding uses "predominantly" only when >= 60% of turns share the class (from config.PREDOMINANTLY_THRESHOLD), otherwise reports mixed distribution (e.g. "Voice is mixed (Sad in turn 1, Happy in turn 2)"). When < 3 turns: slope and net change display "n/a". Review Attention Gauge action text ("LOW PRIORITY", "MONITOR", "REVIEW RECOMMENDED") is derived from the same attention score and level that drives the gauge percentage.
- **ui/components/segment_timeline.py** — Stripped duplicate "Speaker A:" prefix from transcript display. Displayed modality-incongruence page-level disclaimer once. Direction mismatch line is only shown when directions actually differ. Widening note (widened short-segment window) is only shown when segment duration < config.SHORT_SEG_SECONDS AND widening actually happened. Low ASR confidence threshold is now configurable via config.LOW_ASR_CONFIDENCE_THRESHOLD.
- **ui/components/incongruence_card.py** — Added show_disclaimer parameter to suppress redundant repeated disclaimers within turn cards.
- **ui/components/header.py** — Updated application version to config.APP_VERSION (2.0). Unified fusion strategy naming to config.FUSION_STRATEGY_NAME (Multi-Evidence Arbitration). Speaker ID chip displays "Single speaker detected" when 1 speaker is detected, and hides speaker-comparison UI.
- **ui/pages/why_result_view.py** — Replaced redundant per-turn expanders with a single comprehensive audit matrix table (Turn, Time Window, Rule Fired, Pattern Label, Effective Weights, Modality Reliability, Counterfactual Sensitivity).
- **ui/pages/analysis_view.py** — Rendered success banner once at top of Results page with elapsed time of current run.
- **ui/state.py** + **pp/microphone.py** — Added banner state (_last_run_elapsed, _show_success_banner) to reset and clear functions to avoid stale display.
- **ui/components/export_bar.py** — Cleaned transcript of speaker prefixes and exported canonical emotion labels in JSON and CSV reports.
- **pp/conversation_analysis.py** — Passed coustic_window_widened and segment_duration to fusion and segment results.
- **pp/speech_to_text.py** — Wired config.LOW_ASR_CONFIDENCE_THRESHOLD into ASR reliability checks.
- **	ests/test_tier1_regressions.py** — Added 11 regression tests covering items 7-19.

### Why
Addressed rendering inconsistencies, duplicate UI components, code-block markdown glitches, and contradictions between metrics, charts, and text summaries.

### Measured
- **Tests before:** 272 passed, 3 warnings
- **Tests after:** 295 passed, 3 warnings
- All 11 new Tier 1 regression tests passing
- python -m pytest -q exit code 0

---

## fix/final-polish - Tier 3: Validation Tab & Evaluation Dataset SSOT (2026-10-02)

**Branch:** fix/final-polish

### What changed
- **evaluation/dataset.py** - Added `DatasetInventory` dataclass, `get_dataset_inventory()`, and `format_remaining_annotation_checklist()`. `get_dataset_inventory()` serves as the single source of truth for both Annotation Helper and Validation tabs, returning the unified counts and a detailed `gap_explanation` explaining why there are 9 WAV files on disk (7 evaluation recordings + 2 unit test fixtures: `test_corrupted.wav`, `test_empty.wav`) and 7 dataset records (5 with ground truth, 2 pending).
- **evaluation/run_evaluation.py** - Safe metric string formatting with `_fmt_float()` so evaluation reporting never crashes when metric values are `None` on partial datasets. Updated `latest_eval_data` to explicitly provide `"n"` and `"total_samples"` keys for `voice_emotion` and `text_sentiment`. Added stdout print of `format_remaining_annotation_checklist(dataset)` at the end of evaluation runs.
- **ui/pages/validation_view.py** - Integrated `get_dataset_inventory()` in summary banner with explicit gap explanation callout. Fixed `_render_classification_section` to safely extract sample count `n` from multiple possible keys (`evaluated_samples`, `n`, `total_samples`). When $N < \text{EVAL\_MIN\_N}$ (Item 29), displays "âš ï¸� Insufficient data: N=4, need >=5" along with a raw prediction-vs-label table marked "descriptive only, not a metric". Under Speaker Count (Item 30), when all labeled clips are single-speaker ($true\_speaker\_count=1$), suppresses the headline MAE metric card and shows an informational notice stating that MAE is uninformative for multi-speaker separation, accompanied by sample-level counts. Displays the remaining-annotation checklist (Item 31).
- **ui/pages/annotation_view.py** - Integrated `get_dataset_inventory()` in the Dataset Dashboard subtab, aligning the metrics (Audio Files on Disk: 9, Dataset Records: 7, Annotated Clips: 5) and displaying the inventory gap explanation.
- **tests/test_tier3_validation.py** - Added 5 regression tests covering dataset inventory counts, raw prediction table loading, all single-speaker MAE suppression, safe handling of None metrics in report formatting, and remaining-annotation checklist formatting.

### Why
Resolved discrepancy between Annotation Helper (9 files) and Validation tab (7 records), eliminated the bug where $N=4$ was reported as "No annotated ground truth", prevented misleading headline MAE metrics when ground truth lacks speaker diversity, and ensured evaluation runs never crash on partial data.

### Measured
- **Tests before:** 295 passed, 3 warnings
- **Tests after:** 300 passed, 3 warnings
- All 5 new Tier 3 regression tests passing
- python -m pytest -q exit code 0

---

## fix/final-polish - Tier 4: Voice-Model Diagnostics & Reliability Down-Weighting (2026-10-02)

**Branch:** fix/final-polish

### What changed
- **evaluation/voice_diagnostics.py** - Verified `superb/wav2vec2-base-superb-er` model card and feature extractor specification:
  * Ingestion: 16 kHz mono float32, DC offset = 0.0000, max amplitude < 0.999 (no clipping).
  * Feature extractor: `do_normalize=True` verified.
  * Label mapping (`id2label`): `{0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}` matches SUPERB ER official IEMOCAP spec.
  * Expanded audio discovery to audit all evaluation clips in `sample_audio/`, `data/`, and `recordings/`.
- **config.py** - Added centralized thresholds:
  * `LOW_VOICE_CONFIDENCE_THRESHOLD: float = 0.50`
  * `LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY: float = 0.60`
  * `LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD: float = 0.65`
- **app/conversation_analysis.py** - Integrated segment-level low voice confidence check: when segment voice confidence is below `LOW_VOICE_CONFIDENCE_THRESHOLD` (0.50), voice reliability is penalized by `LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY` (0.60), setting `low_voice_reliability = True` and lowering the voice weight in multimodal fusion. Across clips/recordings, when one class occupies $\ge 70\%$ of turns (`VOICE_COLLAPSE_THRESHOLD`), collapse penalty (0.50) is applied, `low_voice_reliability = True` is marked, and fusion re-runs with down-weighted acoustic contribution.
- **ui/components/segment_timeline.py** - Added `âš ï¸� LOW VOICE RELIABILITY` chip in turn header and an explanatory warning callout when segment voice reliability is low, confidence is below threshold, or collapse is detected.
- **tests/test_tier4_voice_diagnostics.py** - Added 4 tests verifying Tier 4 thresholds, ensuring prior correction refuses to fit below $N=15$, verifying that low voice reliability lowers effective voice weight in fusion, and testing collapse penalty behavior.

### Empirical Diagnostics Numbers & Root Cause (Domain Shift vs. Bug)
- **Technical Pipeline Verification**: Ingestion format, normalization, and `id2label` are 100% bug-free.
- **Empirical Diagnostics Metrics** across benchmark audio clips:
  * `sad`: Mean Probability = 79.2%, 6/7 predictions (85.7% share)
  * `neutral`: Mean Probability = 14.1%, 1/7 predictions (14.3% share)
  * `happy`: Mean Probability = 4.1%, 0/7 predictions (0.0% share)
  * `angry`: Mean Probability = 2.6%, 0/7 predictions (0.0% share)
  * Shannon Entropy: $H$ ranges from 0.13 to 0.85 bits across clips (extreme overconcentration of probability mass).
- **Root Cause Conclusion**:
  * **Code Bug? NO.** The model pipeline, tensor dimensions, and labels match the model card exactly.
  * **Domain Shift? YES.** The upstream checkpoint was fine-tuned exclusively on IEMOCAP theatrical/acted recordings. In standard conversational speech, its acoustic representations project heavily onto the "sad" subspace.
- **Strict Compliance**:
  * Did NOT alter model thresholds to force `happy_sample.wav` to predict happy.
  * Prior calibration remains strictly gated at $N \ge 15$ (`VOICE_CALIBRATION_MIN_FIT_SAMPLES`).

### Measured
- **Tests before:** 300 passed, 3 warnings
- **Tests after:** 304 passed, 3 warnings
- All 4 new Tier 4 regression tests passing
- python -m pytest -q exit code 0

---

## Tier 5: Polish — Items 33–36 (branch: fix/final-polish)

### What Changed

**Item 33 — Results Order & Landing Layout**
- Verified: `analysis_view.py` already enforces the correct hierarchy: verdict hero → modality cards → timeline → tabs.
- `render_pipeline_steps()` in `ui/components/header.py` provides the 3-step start card + live system strip, called before the audio ingestion section.
- `render_segment_timeline()` already has chat-style turns, per-speaker color dot, emotion chips (🎙️ voice, 💬 text, ⚡ fused), and intensity mini-bar.

**Item 34 — Charts: titles, consistent colors, n/a states**
- `ui/charts/dynamics.py`: replaced all bare `#hex` color literals in trace/bar definitions with `get_chart_theme_tokens()` values. Color mapping:
  - Voice/intensity line → `tokens["critical"]` (warm-red, consistent across dark/light)
  - OLS trend line → `tokens["accent"]` (blue, consistent)
  - Speaker comparison bars → `tokens["accent"]` + `tokens["critical"]`
- Added `title=dict(...)` to `fig.update_layout()` for all three charts.
- `render_valence_arousal_map`: tightened guard to `< 2` segments; shows "Trajectory n/a — need ≥ 2 turns" annotation for single-segment recordings.
- Fixed two `render_*` section headers (speaker comparison, valence map) to use `html_block()` instead of raw indented HTML strings.

**Item 35 — Responsive caching & fusion-only recompute**
- `load_analyzer()` already `@st.cache_resource`, keyed on `(whisper_model_size, whisper_language)`. Models never reload when only fusion sliders change.
- `extract_acoustic_features()` in `ui/charts/inspection.py` already `@st.cache_data`.
- **New: "⚡ Recompute Fusion" button** appears in `streamlit_app.py` when fusion slider settings differ from the settings that produced the current result (`_last_fusion_settings`). Clicking it re-fuses the stored `voice_emotion` + `text_sentiment` per segment using the new weights — no ML model invocation.
- After each full analysis, `_last_fusion_settings` is saved to session state as the baseline.
- Sidebar caption updated to explain the two-path behavior.

**Item 36 — Optional UI library audit**
- Evaluated: streamlit-shadcn-ui (MIT, but requires React/Node build), streamlit-extras (MIT, but some components need npm), streamlit-option-menu (MIT, works offline).
- Decision: **no new libraries added**. Existing plain CSS token system (`ui/styles/tokens.css`) already provides sufficient theming. Adding any of the three candidates would introduce build-step risk or break the offline constraint.

### Tests
- `tests/test_tier5_polish.py` — 16 tests, **16 passed**.
- Full suite: to be updated after test run completes.

### Files Touched
| File | Change |
|------|--------|
| `ui/charts/dynamics.py` | Token colors, chart titles, n/a guards, html_block headers |
| `streamlit_app.py` | Fusion recompute button, `_last_fusion_settings` save |
| `ui/components/header.py` | Sidebar caption update |
| `tests/test_tier5_polish.py` | New — 16 tests |
| `PROGRESS.md` | This entry |

---

## Tier 2: Correctness of Numbers — Items 20–27 (branch: fix/final-polish)

### What Changed

**Item 20 — Valence-Arousal Map**
- Fixed valence fusion in `ui/charts/dynamics.py`: Fuses text sentiment valence and voice emotion valence using effective modality weights ($V_{\text{fused}} = w_{\text{audio}} V_{\text{voice}} + w_{\text{text}} V_{\text{text}}$).
- Added an affective coordinates formula explanation caption.
- Alternated label positions across turns (`top center`, `bottom center`, `top right`, `bottom left`, etc.) to prevent overlapping markers and labels.

**Item 21 — Combined Confidence & No Weak Disagreement Boost**
- Verified `combined_confidence` formula in `app/fusion.py` as a strict convex combination bounded in $[\min(c_a, c_t), \max(c_a, c_t)]$.
- Removed unsupported claim of an "agreement bonus" in `ui/components/transparency_panels.py` to ensure transparency accurately reflects the code. Weak disagreement receives no artificial boost.

**Item 22 — Intensity v2 Component Exposure**
- In `ui/components/segment_timeline.py`, exposed the exact subcomponents for each segment: Prosody ($w=35\%$), Voice Arousal ($w=35\%$), Text Negativity ($w=30\%$).

**Item 23 — Determinism & Seeding**
- Added `set_seed(42)` in `app/conversation_analysis.py` seeding Python `random`, `numpy.random`, and `torch.manual_seed`.
- Added `temperature=0.0` and `condition_on_previous_text=False` in `app/speech_to_text.py` for deterministic Whisper segment boundaries and log-probabilities.
- Added same-file repeatability regression tests.

**Item 24 — Explainable Multimodal Fusion Architecture**
- In `ui/components/transparency_panels.py`, completely aligned the Declarative Rule Arbitration section with the actual 8 rule IDs from `app/fusion.py` (`agreement_negative_high_arousal`, `agreement_negative_low_arousal`, `agreement_positive`, `agreement_neutral`, `mismatch_positive_text_negative_voice`, `mismatch_negative_text_positive_voice`, `mismatch_generic`, `weak_disagreement`) and `abstention_low_reliability`, preserving LaTeX math.

**Item 25 — Model Table Derived from Configuration**
- Added canonical `DIARIZATION_MODEL_NAME = "pyannote/speaker-diarization-community-1"` to `config.py`.
- Corrected Pyannote identification in `ui/components/transparency_panels.py` to Community-1 PyTorch and dynamically derived model names from `config.py`.

**Item 26 — Conditional Limitations Statement**
- Updated `render_professional_limitations_section` in `ui/components/transparency_panels.py` to check for local ground-truth evaluation metrics ($\ge 5$ annotated files); if not present, displays `"No local evaluation results yet (requires ≥ 5 annotated ground-truth recordings)."`.

**Item 27 — MFCC Chart**
- Fixed MFCC 1 energy dominance by plotting MFCC 2–13 in `ui/charts/inspection.py`.
- Labeled series as "Mean ± Std across frames".
- Applied theme token accent color.
- Added plain-English explanation clarifying that MFCCs capture timbral vocal tract acoustics rather than discrete emotion classifications.

### Tests
- `tests/test_tier2_correctness.py`: 18 tests, all 18 passed.

---

## Phase 4d: UI Display Cleanup & Robustness (branch: current)

### What Changed
- **Raw HTML Code Block Fix**: Wrapped all meter and card markdown templates with `html_block()` helper stripping leading spaces and blank lines, eliminating accidental code-block rendering.
- **Voice-Only Turns**: Suppressed fused evidence meter on voice-only turns; updated "Why" reliability line to "text: n/a (text unavailable)"; omitted empty counterfactual sensitivity header.
- **Rendered Quality Verification**: Verified badge formatting as `[lang: <code> (p=<prob>) · flag: <flags>]`; suppressed 0.0% text row on unavailable text; scoped capped confidence note strictly to capped states.
- **Transcript Sanitization & Phrase Collapse**: Added `sanitize_transcript()` stripping `U+FFFD`/`U+FFFE`/`U+FFFF` before text sentiment; collapsed 1–4 token repetitive phrases repeated $\ge 4$ times in timeline display.
- **Short Segments (< 1.0s)**: Added `VOICE_MIN_SEGMENT_SEC = 1.0` in `config.py`; tagged sub-second turns with warning chip; excluded from trend OLS regression and direction shift summary while keeping segment listing intact.
- **Label Consistency**: Standardized canonical display names across cards and meters without mixed `OTHER (SURPRISED)` titles; raw model labels displayed on separate dedicated line.
- **Backend String Agnosticism**: Grepped and confirmed all UI/app components dynamically resolve model IDs and backend names from `config.py`.

### Tests
- Full test suite: **442 passed**, 0 failed, 1 warning (100.13s).

---

## Phase 5a: Evaluation, CI and Repo Hygiene (branch: current)

### What Changed
- **Evaluation Pipeline Verification**: Verified `evaluation/run_evaluation.py` executes clip-level inference via single `ConversationAnalyzer.analyze()` invocation (live pipeline); `predictions_before_pipeline_fix.csv` already preserved.
- **Evaluation Metadata & Columns**: Added `voice_backend`, `voice_conf_cap`, `whisper_model_size`, `sanity_rule_applied`, `n_segments`, `uncertain_rate`, `asr_flags`, `mean_raw_confidence`, `detected_language` to `SamplePrediction`, sample records, and `evaluation_predictions.csv` with full None-safe fallbacks for older runs.
- **Cache Key Specification**: Updated `compute_sample_cache_key` with SHA-256 of file bytes, `VOICE_BACKEND`, `VOICE_CONF_CAP`, `VOICE_SANITY_RULE_BACKENDS`, `VOICE_CALIBRATION_PROSODY_RULE_ENABLED`, `STT_MODEL_SIZE`, `STT_LANGUAGE`, `VOICE_MIN_SEGMENT_SEC`, fusion weights, ASR guard thresholds, and model IDs.
- **Report & Metrics**: Updated `EVALUATION_REPORT.md` with UTC timestamp and active configuration snapshot; metrics treat `uncertain` as incorrect and report uncertain rate beside accuracy.
- **CI & Hygiene**: Created `.github/workflows/tests.yml` (Python 3.12, CPU PyTorch from index, offline env, `pytest -q -x`), `requirements-ci.txt`, `.gitattributes`, `.env.example`, and complete `.gitignore` ignoring media/model caches.
- **Test Portability**: Added `skipif` decorators to sample-audio dependent and standalone test scripts for offline test execution without missing file errors.

### Tests
- `tests/test_evaluation_phase5a.py`: 4 new mocked unit tests for columns, uncertain rate, and cache invalidation.
- Full test suite: **446 passed**, 0 failed, 1 warning (73.31s).

---

## Phase 4e: Last UI/ASR Display Fixes (branch: current)

### What Changed
- **ASR Language Badge**: `app/conversation_analysis.py` — stamped recording-level `language` and `language_probability` from `transcription` onto each per-segment result dict after `_analyze_segment()`. `ui/components/segment_timeline.py` badge logic now shows `"lang: n/a"` instead of config `"auto"` fallback when language is unknown/undetected.
- **Voice-Only Turn Header Chip**: Turn header chip now uses `t_display_header` (`"n/a"` for voice-only) instead of `t_sentiment_display.upper()` (`"UNKNOWN"`). Chip text is `"💬 n/a"` on voice-only turns.
- **Intensity v2 Text Component Hidden**: Text Negativity span (+ bullet separator) is conditionally built in Python and only emitted when `w_t > 0`; voice-only segments (where fusion sets text weight to 0) no longer show a `"Text Negativity (w=0%): 0.0%"` line.
- **Top Transcript Sanitization**: `ui/pages/analysis_view.py` applies `_collapse_repetitive_tokens()` (which internally calls `sanitize_transcript()`) to `full_transcript` before display. Added `dir="auto"` to the transcript `<div>`. Exports keep raw `full_transcript` unchanged.
- **Raw Model Label CJK Strip & Resilience**: Display strips the CJK prefix before `"/"` (e.g. `"生气/angry"` → `"angry"`), while `<unk>`, no-slash labels, and empty/whitespace inputs never crash or render blank lines; `v_raw_label` and all export fields remain the full bilingual string.

### Tests
- `tests/test_phase4e.py`: 9 comprehensive regression tests covering all Phase 4e requirements.
- Full test suite: **455 passed**, 0 failed, 3 warnings (59.38s).

---

## Phase 9: Project Documentation & Defense Materials (branch: current)

### What Changed
- **Comprehensive Test Plan**: Authored `docs/TEST_PLAN.md` covering all 25 operational scenarios with expected code behaviors and verification status.
- **Academic Demonstration Script**: Created `docs/DEMO_SCRIPT.md` with structured 5-minute presentation flow, cold-start pre-warm instructions, and video capture fallback.
- **Technical Viva Q&A**: Produced `docs/VIVA_QA.md` with code-grounded answers for every model, mathematical formula, confidence cap, evaluation split, and ethical constraint.
- **Changelog**: Documented full development history in `CHANGELOG.md` covering v1 baseline milestones, v2 phases, and explicitly scoped-out features.
- **Privacy & Ethics Charter**: Published `docs/PRIVACY_AND_ETHICS.md` (27 lines) validating 0 outbound network calls, local CPU execution, ephemeral deletion, and prohibited uses.

### Tests
- Full test suite: **482 passed**, 13 skipped, 0 failures (verified across all unit and integration test suites).




