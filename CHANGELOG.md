# Changelog

All notable changes, milestones, and architectural versions of the **Multimodal Voice & Conversation Emotion Analysis System** are documented in this file.

---

## [v2.1.0] - Phases 10 through 18: Voice Upgrade, Evaluation & Empirical Hardening

### Phase 18b: Dependency Pinning (`av==18.1.0`) & STT Test Isolation Docs
- Pinned `av==18.1.0` in `requirements.txt` to fix incompatibility with `faster-whisper 1.2.1` (`av 19.0.2` threw `TypeError: unexpected keyword argument 'metadata_errors'`, stopping analysis at transcription).
- Documented tested runtime versions (Python 3.12.10, torch 2.13.0+cpu, av==18.1.0; unpinned transitive dependencies) in [README.md](file:///c:/Project/multimodal-emotion-analysis/README.md).
- Documented `tests/test_speech_to_text.py` import-time transcription dependency and collection failure risk in [KNOWN_ISSUES.md](file:///c:/Project/multimodal-emotion-analysis/docs/KNOWN_ISSUES.md).

### Phase 18: Documentation Alignment & Empirical Grounding
- Fully aligned [README.md](file:///c:/Project/multimodal-emotion-analysis/README.md), [TEST_PLAN.md](file:///c:/Project/multimodal-emotion-analysis/docs/TEST_PLAN.md), [KNOWN_ISSUES.md](file:///c:/Project/multimodal-emotion-analysis/docs/KNOWN_ISSUES.md), [VIVA_QA.md](file:///c:/Project/multimodal-emotion-analysis/docs/VIVA_QA.md), and [CHANGELOG.md](file:///c:/Project/multimodal-emotion-analysis/CHANGELOG.md) with empirical findings in `evaluation/results/`.
- Documented active `emotion2vec` backend, pre-fix SUPERB screenshot history, sadness weakness on low-arousal clips, and defensible claims table.

### Phase 17e: Latency Narrative Alignment
- Corrected narrative descriptions in `scripts/benchmark_latency.py` to match empirical measurements (text stage 38–55 ms, non-linear voice-stage latency scaling from 0.55s at 5s to 6.73s at 30s).

### Phase 17d: Latency Benchmark Warm-up & Table Formatting
- Added explicit warm-up inference step to `benchmark_latency.py` to eliminate first-run cold-start bias.
- Deduplicated markdown table formatting in latency report generation.

### Phase 17c: Script Execution Guards & Benchmark CLI
- Added `--check-api` CLI flag and tests to `scripts/benchmark_latency.py`.
- Added `--run` safety guard to `evaluation/calibration.py` preventing accidental re-execution of calibration fitting without explicit flags.

### Phase 17a-b: Latency & Hardware Benchmarking
- Profiled CPU runtime performance: cold start 28.31 s; warm median voice latencies 0.551 s (5s, RTF 0.1102), 2.199 s (15s, RTF 0.1466), and 6.734 s (30s, RTF 0.2245).
- Recorded environment specs (Python 3.12.10, PyTorch 2.13.0+cpu, CPU-only, 7.7 GB RAM; peak memory unmeasured).

### Phase 16: Temperature Calibration Analysis
- Optimized temperature scaling parameter ($T=3.1192$) on CREMA-D dev split ($N=1,451$ clips across 28 actors) by minimizing NLL.
- Evaluated held-out test split ($N=1,673$ clips across 31 actors) in report-only mode; ECE reduced from 0.0406 to 0.0207 with NLL reduced from 0.3341 to 0.1768 while preserving predictions.
- Documented severe uncalibrated softmax saturation (1,377 of 1,451 clips $> 0.90$ confidence).

### Phase 15: One-Shot CREMA-D Test Split Evaluation
- Evaluated held-out actor-disjoint test split ($N=1,673$ clips, 31 actors) exactly once with zero post-hoc tuning.
- Achieved 93.31% [91.63%, 94.68%] accuracy and 94.88% [91.86%, 94.86%] UAR with actor-cluster bootstrap 95% CIs.
- Recorded confusion matrix ($N=1,646$ answered, 27 abstained / 1.6%).

### Phase 14: CREMA-D 8-Clip Dev Sanity & Pipeline Validation
- Validated new acoustic pipeline on 8-clip development subset (100% accuracy, 100% UAR) verifying feature extraction and label mappings.

### Phase 13: Conversational Turn Dynamics & Same-Speaker Merge
- Implemented same-speaker turn merge rule ($\le 0.25$s threshold) in conversation analyzer to prevent fragmented speaker turns during brief pauses.

### Phase 12: Softmax Saturation Mitigation (`VOICE_CONF_CAP = 0.65`)
- Introduced conservative voice confidence cap (`VOICE_CONF_CAP = 0.65`) for `emotion2vec` to prevent over-saturated acoustic probabilities from dominating text sentiment during multimodal fusion.

### Phase 11: Backend Transparency & Diagnostic UI
- Added active backend badge and graceful fallback status indicators in the Streamlit UI.
- Hardened FunASR import routines and runtime initialization checks.

### Phase 10: Multi-Backend Voice Emotion Architecture
- Introduced extensible `VoiceEmotionBackend` abstract base class with registry and auto-selection.
- Implemented concrete `Emotion2VecBackend` (FunASR `emotion2vec_plus_base`) and `Wav2Vec2Backend` (`superb/wav2vec2-base-superb-er`) with automated fallback.

---

## [v2.0.0] - Phase 1 through 9 Enhancements

### Phase 9: Project Documentation & Defense Materials
- Created comprehensive verification test plan matrix (`docs/TEST_PLAN.md`).
- Authored 5-minute academic demonstration walkthrough script (`docs/DEMO_SCRIPT.md`).
- Documented rigorous viva defense questions and code-grounded answers (`docs/VIVA_QA.md`).
- Added formal data privacy and research ethics charter (`docs/PRIVACY_AND_ETHICS.md`).

### Phase 8: Streamlit Deployment & Web Hygiene
- Added dedicated `ui/pages/privacy_view.py` and `ui/pages/limitations_view.py` accessible via sidebar navigation and footer.
- Updated `st.set_page_config` with About metadata and standardized primary action button to `"Analyze a clip ▶"`.
- Added `static/robots.txt`, `static/sitemap.xml`, and enabled static file serving in `.streamlit/config.toml`.
- Provided production CPU-optimized `Dockerfile` with non-root security context (not built or tested: Docker unavailable on the development machine).
- Documented host requirements, secrets management (`HF_TOKEN`), and Pyannote gated access in `docs/DEPLOYMENT.md` and `docs/KNOWN_ISSUES.md`.
- Added `tests/test_web_hygiene.py` (all tests passing).

### Phase 6: Natural-Speech Evaluation Tooling
- Implemented natural-speech evaluation tooling under `evaluation/natural_set/` (`make_label_sheet.py`, `score.py`).
- Authored natural speech collection protocol (`docs/NATURAL_SET_PROTOCOL.md`) for 40–60 authentic clips from $\ge 8$ speakers with independent rater consensus.

### Phase 5B: Synthetic Noise Robustness Tooling
- Implemented noise mixing harness in `evaluation/robustness.py` supporting SNR conditions (20 dB, 10 dB, 0 dB) and 8 kHz narrowband telephony round-trips.
- Verified SNR mixing accuracy within 0.5 dB in `tests/test_robustness.py`.

### Phase 5A: Priority Analysis, Error Analysis & Calibration
- Completed detailed error analysis on held-out test split in `docs/ERROR_ANALYSIS.md`.
- Implemented temperature calibration optimization in `evaluation/calibration.py` (minimizing NLL on dev logits).
- Benchmarked CPU runtime latency and Real-Time Factor in `scripts/benchmark_latency.py` and `evaluation/results/latency.md`.

### Phase 4C: One-Shot Held-Out Test Evaluation
- Implemented `evaluation/run_crema_d_test_evaluation.py`, evaluating the held-out test split exactly once without post-hoc tuning.
- Generated `evaluation/results/crema_d_benchmark.md` and `.json` with actor-cluster bootstrap 95% confidence intervals.

### Phase 4B: Candidate Head with Dev-Only Decision
- Implemented logistic regression candidate head on frozen Wav2Vec2 penultimate embeddings in `evaluation/train_candidate_head.py`.
- Formulated pre-declared adoption rule in `evaluation/results/head_decision.json`.
- Dev-only grouped cross-validation outcome: candidate head did not beat dev baseline threshold; decision frozen and head not adopted.

### Phase 4A: CREMA-D Loader, Cache & Dev Baseline
- Implemented `evaluation/crema_d.py` and `evaluation/run_crema_d_baseline.py` with actor-disjoint partitioning (dev vs test).
- Implemented SHA-256 clip-level prediction caching and test-metrics refusal guard.

### Phase 3: Engineering, Security & Privacy Audit
- Enforced 50 MB upload limit and safe HuggingFace authentication/gated error categorization without leaking credentials or tracebacks.
- Inlined WaveSurfer bundle for offline privacy and added Ruff linting to CI.

### Phase 2: Repetition Collapse & Banner Fix
- Implemented regex-based repetition collapse appending `… ×N (repetitive transcript)` for runs $\ge 4$ in UI timeline.
- Corrected voice reliability banner string formatting on low-discrimination turns.

### Phase 1: Repository Architecture Inspection
- Verified environment constraints (Streamlit frontend, CPU inference, < 8GB RAM footprint).

---

## [v1.0.0] - Baseline Architecture Milestones

- **Milestone 1**: Project environment setup and 16 kHz audio verification.
- **Milestone 2**: Audio processor module (mono conversion, resampling, MFCCs, F0 pitch, RMS energy).
- **Milestone 3**: Acoustic emotion classifier using `superb/wav2vec2-base-superb-er`.
- **Milestone 4**: Speech-to-Text module using `faster-whisper` INT8 CPU inference with VAD filtering.
- **Milestone 5**: Text sentiment classifier using `cardiffnlp/twitter-roberta-base-sentiment-latest`.
- **Milestone 6**: Transparent, rule-based multimodal decision fusion.
- **Milestone 7**: Multi-turn conversation analytics, emotional trajectory tracking, and escalation detection.
- **Milestone 8**: Speaker diarization support via `pyannote/speaker-diarization-community-1`.
- **Milestone 9**: Interactive Streamlit dashboard in `streamlit_app.py`.
- **Milestone A**: Modality consistency trajectory and attention score triage heuristic.
- **Milestone B**: Empirical evaluation harness and experimental logistic regression fusion baseline.
- **Milestone C**: In-browser microphone recording workflow and UI hardening.

---

## 🚫 Scoped Out & Deferred Features (Not Done by Design)

- **RAVDESS Evaluation**: Deferred per project rules (cross-corpus acted evaluation secondary to CREMA-D and natural speech protocol).
- **Full Corpus Synthetic Noise Evaluation**: Robustness harness and SNR mixing verified mathematically; full audio dataset benchmark not run to preserve CPU execution budget.
- **Live WebSocket Audio Streaming**: Scoped out; partial chunk streaming degrades ASR accuracy; file-based conversational turn analysis remains primary.
- **Black-Box Post-Hoc XAI (LIME/SHAP)**: Scoped out in favor of transparent, rule-based decision arbitration and explicit counterfactual sensitivity tables.
- **Large-Scale Natural-Speech Audio Collection**: Tooling and scoring protocol created; collecting and annotating large real-world audio datasets deferred pending speaker consent and IRB ethical clearance.
