# Comprehensive Viva & Technical Defense Q&A

This document provides rigorous, code-grounded answers for technical vivas, examinations, and architectural defense of the **Multimodal Voice & Conversation Emotion Analysis System**.

---

## 1. System Architecture & Model Pipeline

### Q1: What models comprise the active multimodal pipeline, and what preprocessing is applied?
- **Acoustic Preprocessing**: Input audio is validated (format, 0-byte, corruption, and 50 MB / 600s limits), converted to mono, resampled to $16,000\text{ Hz}$ PCM float32, and peak-amplitude normalized in `app/audio_processor.py`.
- **Voice Emotion**: `emotion2vec/emotion2vec_plus_base` via FunASR (active default), with automatic fallback to `superb/wav2vec2-base-superb-er` via HuggingFace transformers if FunASR or model weights are unavailable.
- **Speech-to-Text (STT)**: `faster-whisper` (`base` or `small` model), running under CTranslate2 in INT8 CPU quantization. Features Voice Activity Detection (VAD) filtering, producing timestamped word/segment tokens and transcription log-probabilities.
- **Text Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest`, a RoBERTa-base architecture fine-tuned on TweetEval, producing probabilities over *positive, neutral, negative*.
- **Speaker Diarization**: `pyannote/speaker-diarization-community-1`, clustering voice embeddings across time when an authenticated `HF_TOKEN` is provided.

---

## 2. Mathematical Formulas & Fusion Rules

### Q2: What are the mathematical formulas for dynamic weights, evidence strength, and intensity?
1. **Dynamic Reliability-Weighted Effective Weights**:
   $$w_{\text{eff, audio}} = \frac{w_{\text{base, audio}} \cdot r_{\text{audio}}}{w_{\text{base, audio}} \cdot r_{\text{audio}} + w_{\text{base, text}} \cdot r_{\text{text}}}$$
   $$w_{\text{eff, text}} = \frac{w_{\text{base, text}} \cdot r_{\text{text}}}{w_{\text{base, audio}} \cdot r_{\text{audio}} + w_{\text{base, text}} \cdot r_{\text{text}}}$$
   Where $r_{\text{audio}}$ integrates duration and probability margin, and $r_{\text{text}}$ is derived from Whisper average token log-probabilities.

2. **Combined Confidence (Evidence Strength)**:
   $$c_{\text{combined}} = w_{\text{eff, audio}} \cdot c_{\text{audio}} + w_{\text{eff, text}} \cdot c_{\text{text}}$$
   Evidence strength is strictly a convex combination bounded in $[\min(c_{\text{audio}}, c_{\text{text}}), \max(c_{\text{audio}}, c_{\text{text}})]$. Mismatch receives no artificial boost; uncertainty in either modality directly attenuates the combined confidence.

3. **Conversational Turn Intensity**:
   $$I = 0.50 \cdot A_{\text{audio}} + 0.50 \cdot \text{negativity}_{\text{text}}$$
   Where acoustic arousal $A_{\text{audio}}$ is mapped via `config.VOICE_EMOTION_AROUSAL_MAP` (angry=0.85, happy=0.70, sad=0.35, neutral=0.20), and $\text{negativity}_{\text{text}}$ is RoBERTa negative sentiment probability. On voice-only turns ($w_{\text{eff, text}} = 0$), intensity defaults purely to $A_{\text{audio}}$.

---

## 3. Reliability Gating, Confidence Cap & Sanity Rules

### Q3: Why is `VOICE_CONF_CAP` set to 0.65, and why is the sanity rule SUPERB-only?
- **Confidence Cap (`VOICE_CONF_CAP = 0.65`)**: On the `emotion2vec` backend, the raw softmax distribution is severely over-saturated (e.g., in calibration evaluation, 1,377 of 1,451 clips had raw confidence $> 0.90$). Leaving confidence uncapped allowed faulty acoustic classifications to completely dominate text sentiment during fusion. Capping raw confidence at $0.65$ serves as a conservative heuristic to preserve multimodal balance until post-hoc calibration is integrated into runtime fusion.
- **Why the Sanity Rule is SUPERB-Only**: The SUPERB Wav2Vec2 model occasionally produces extreme probability spikes on short or degraded clips, or suffers acoustic collapse where all classes approach identical probabilities. The heuristic sanity rule was calibrated specifically to SUPERB's logit distribution and must not be applied to `emotion2vec`, which has fundamentally different representation geometry and saturation characteristics.

---

## 4. Evaluation Methodology & CREMA-D Benchmark

### Q4: Why use CREMA-D with an actor-disjoint split and actor-cluster bootstrap?
- **Actor-Disjoint Split**: CREMA-D contains 7,442 clips from 91 ethnically diverse actors. Splitting data randomly by clip causes severe data leakage because the model memorizes speaker-specific vocal traits. An actor-disjoint split ensures actors in the development set never appear in the test set.
- **Actor-Cluster Bootstrap**: Utterances spoken by the same actor are correlated (non-i.i.d.). Standard empirical bootstrap underestimates variance. Actor-cluster bootstrap resamples entire actor clusters with replacement, producing valid 95% confidence intervals reflecting generalization across unseen speakers.

### Q5: What was the Candidate Head adoption protocol, and why train on frozen embeddings?
- **Frozen Embeddings**: Fine-tuning all 95M+ transformer parameters on smaller corpora causes catastrophic forgetting and acoustic overfitting. Extracting frozen penultimate transformer representations and training a lightweight Logistic Regression head adapts classification boundaries while preserving learned acoustic representations.
- **Pre-Declared Frozen Decision Protocol**: In accordance with rigorous scientific practice, the adoption rule was pre-declared in `evaluation/results/head_decision.json` **before** execution: candidate must beat dev baseline mean UAR by $\ge 0.03$ in actor-grouped CV across a majority of folds without degrading sad or neutral recall. The dev-only CV rejected the head; the decision was frozen ("not pursued"); the test split was evaluated exactly **once** without post-hoc tuning, and results were honestly reported showing the candidate head was not adopted.

---

## 5. Calibration, Agreement & Natural Speech

### Q6: Why doesn't temperature calibration change classification accuracy?
- Temperature scaling applies a monotonic transformation: $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$. Dividing all logits $z$ by a positive scalar $T$ preserves logit ranking ($\arg\max_i z_i / T = \arg\max_i z_i$). Therefore, predicted classes and raw accuracy remain identical, while calibration metrics (ECE and NLL) improve.

### Q7: Why wasn't `VOICE_CONF_CAP` altered during calibration?
- Central project constraints strictly forbid altering fusion rules, weights, thresholds, or `VOICE_CONF_CAP` to ensure data integrity and prevent post-hoc hyperparameter gaming. Calibration was evaluated offline as a report-only analysis.

### Q8: What is the relevance of Cohen/Fleiss Kappa and human agreement?
- Human annotator agreement on conversational affect typically yields $\kappa \approx 0.40 - 0.65$. Speech emotion is inherently ambiguous; expecting computational models to achieve $> 85\%$ accuracy against single-rater annotations ignores human baseline variance.

### Q9: What is the Natural Speech Protocol, and why is the target sample size small (40–60 clips)?
- Acted corpora feature exaggerated, theatrical affect not representative of real conversations.
- Protocol (`docs/NATURAL_SET_PROTOCOL.md`) gathers 40–60 authentic 3–15s clips from $\ge 8$ consenting speakers (including Indian English). A small, high-quality corpus with multi-rater independent consensus provides verifiable ecological validity without ethical or consent compromises.

---

## 6. Runtime Performance & Production Hygiene

### Q10: What is the system's CPU latency and Real-Time Factor (RTF)?
- Benchmarked on Intel Core Ultra CPU (Python 3.12.10, PyTorch 2.13.0+cpu, CPU-only, 7.7 GB RAM):
  - **Cold Start (Model Loading)**: 28.31 s (deferred until the first inference call).
  - **Warm Median Voice Latencies**:
    - 5s clip: 0.551 s (RTF = 0.1102)
    - 15s clip: 2.199 s (RTF = 0.1466)
    - 30s clip: 6.734 s (RTF = 0.2245)
  - **Scaling Characteristics**: Voice stage scales non-linearly across the 0–30s range (a 2× duration increase from 15s to 30s yields a 3.1× latency jump).
  - **End-to-End Latency**: Full pipeline including Whisper STT and Pyannote diarization has not yet been measured as an integrated total.
  - **Fusion Recompute**: $< 5\text{ ms}$ (instant rule re-arbitration without neural re-execution).

### Q11: How are edge cases (abstention, voice-only, short segments) handled?
- **Abstention**: If combined reliability $r_{\text{combined}} < 0.35$, the system flags low confidence and abstains from asserting definitive classifications.
- **Voice-Only Fallback**: When transcripts are empty, repetitive, or non-English, text weight drops to $0.0$, operating purely on acoustic features.
- **Short Segments (< 1.0s)**: Flagged with warning chips and excluded from OLS trend regressions due to insufficient acoustic frames.

---

## 7. Scoped-Out Features, Ethical Boundaries & Limitations

### Claims I Can and Cannot Make

| What I CAN Defensibly Claim | What I CANNOT Claim |
|---|---|
| **Reproducible benchmark**: Evaluated on held-out actor-disjoint CREMA-D test split ($N=1,673$, 31 actors; 93.31% acc [91.63%, 94.68%], 94.88% UAR [91.86%, 94.86%]). | **Sarcasm detection**: Modality incongruence indicates acoustic-lexical divergence, never pragmatic irony or human intent. |
| **Transparent rule-based fusion**: Fully explainable dynamic reliability weighting with user-tunable sliders and sensitivity tables. | **Deception / truth detection**: The system cannot and does not predict honesty, lies, deception, or internal psychological states. |
| **Calibration benefit**: Temperature scaling ($T=3.1192$ fitted on DEV) lowers test ECE from 0.0406 to 0.0207 without altering accuracy. | **Clinical emotion diagnosis**: Results reflect acoustic and linguistic classification, not psychiatric or diagnostic assessments. |
| **Multi-backend voice architecture**: Runtime auto-detection (`emotion2vec` with graceful fallback to `Wav2Vec2`) and transparent UI badging. | **Linear $O(T)$ latency scaling**: Latency scales non-linearly across the 0–30s range (RTF increases from 0.1102 at 5s to 0.2245 at 30s). |
| **Conversational timeline analysis**: Speaker diarization, segment-level tracking, and same-speaker turn merging ($\le 0.25$s). | **Universal calibration**: Temperature $T$ was fitted on CREMA-D DEV; calibration is corpus-specific and unproven on natural speech. |
| **Non-linear voice latency profile**: Empirically measured cold start (28.31s) and warm median voice latencies (0.55s at 5s, 6.73s at 30s). | **Cross-corpus generalization**: Without cross-corpus testing, high performance on CREMA-D does not guarantee equivalent real-world accuracy. |

### Q12: Why are there NO claims of sarcasm or deception detection?
- Modality incongruence reflects computational divergence between acoustic and linguistic classifiers. Sarcasm, irony, and deception involve human pragmatic intent and internal mental states that cannot be verified computationally from audio. Claiming lie detection would be scientifically invalid and ethically irresponsible.

### Q13: What happened during the test evaluator backend mismatch, and why do some screenshots show SUPERB?
- **The Backend Mismatch Story**: During test harness setup, the test evaluation script initially executed using the SUPERB `Wav2Vec2` backend because `funasr` had not yet been installed in that specific Python virtual environment.
- **Detection & Resolution**: The issue was detected during dry-run validation (`--dry-run`) prior to running the test split. The environment was promptly corrected by installing `funasr` and verifying backend auto-selection before the single one-shot test split evaluation.
- **Pre-Fix Screenshots**: Screenshots captured prior to the environment fix display the SUPERB backend badge and fallback metrics. The codebase, active runtime, and benchmark results are fully aligned on `emotion2vec`.

### Q14: What is the known sadness classification weakness?
- **Acoustic Misclassification**: The underlying `emotion2vec` foundation model frequently misclassifies subdued sadness as neutral (e.g., in test clip `Sad1.wav`, segment 1 is classified as neutral with $\approx 100\%$ confidence despite the speaker expressing sadness).
- **Origin**: This is a known limitation of the underlying foundation model's representation space on low-arousal speech, not a bug in the fusion engine or pipeline code.
- **Mitigation**: Capping voice confidence (`VOICE_CONF_CAP = 0.65`) prevents such misclassifications from overriding clear textual sentiment cues.

### Q15: Why were RAVDESS, Live Streaming, and XAI deferred or scoped out?
- **RAVDESS**: Secondary acted dataset; deferred in favor of CREMA-D and natural speech protocol.
- **Live Streaming**: Requires streaming chunked ASR (which causes hallucinations on partial utterances) and WebSocket complexity; file-based turn analysis satisfies the project specification.
- **Complex XAI (LIME/SHAP)**: Opaque post-hoc explanations were rejected in favor of transparent, rule-based arbitration with explicit counterfactual sensitivity tables.
- **Docker**: A Dockerfile is provided but was NOT built or tested (Docker is not installed on the development machine); the local virtual environment is the tested path.
