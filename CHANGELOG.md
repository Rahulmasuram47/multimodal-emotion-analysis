# Changelog

All notable changes, milestones, and architectural versions of the **Multimodal Voice & Conversation Emotion Analysis System** are documented in this file.

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
- Provided production CPU-optimized `Dockerfile` with non-root security context.
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
- Benchmarked CPU runtime latency and Real-Time Factor (RTF $\approx 0.32$) in `scripts/benchmark_latency.py` and `evaluation/results/latency.md`.

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
