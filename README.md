# Multimodal Voice & Conversation Emotion Analysis System

Academic research prototype analyzing spoken dialogue across acoustic prosody and linguistic semantics via transparent rule-based fusion.
**Scope & Ethics Boundary**: Strictly an analytical review signal. Does **NOT** detect lies, sarcasm, deception, or internal psychological state. Not for clinical, hiring, diagnostic, or surveillance use.

## Active Pipeline & Backend Architecture
- **Voice Emotion**: `emotion2vec/emotion2vec_plus_base` (Apache-2.0 via FunASR) is the active production backend (`config.VOICE_BACKEND = "emotion2vec"`).
- **Speech-to-Text**: `openai/whisper-base` (CTranslate2 INT8 CPU transcription + Silero VAD).
- **Text Sentiment**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (RoBERTa 3-class sentiment).
- **Speaker Diarization**: `pyannote/speaker-diarization-community-1` (optional; gated Hugging Face token).
- **UI Verification**: Active backend is displayed in the sidebar Model Architecture card and header status badge.
- **Backend Fallback**: If FunASR is unavailable, system falls back to `superb/wav2vec2-base-superb-er` with an explicit fallback banner. The test evaluator originally ran SUPERB because it was written without FunASR; it was fixed and verified with `--dry-run` before the single test run.
- **Confidence Cap**: An empirical cap (`VOICE_CONF_CAP = 0.65`) is applied in `app/voice_backends/emotion2vec.py:123` and `app/conversation_analysis.py:796` to mitigate softmax saturation. It is an unvalidated heuristic placeholder.

## Empirical Benchmark Results

Full reports: [crema_d_benchmark.md](evaluation/results/crema_d_benchmark.md), [latency.md](evaluation/results/latency.md), [ERROR_ANALYSIS.md](docs/ERROR_ANALYSIS.md), [calibration.json](evaluation/results/calibration.json).

| Benchmark / Corpus | Condition / Head | N (Clips / Actors) | Accuracy / UAR (95% CI) | Status & Protocol Notes |
|---|---|---|---|---|
| **Self-Recorded Dev** | `superb` fallback | 8 clips (1 actor) | Acc: 12.5% (Wilson: 2.2%–47.1%) | preliminary; self-recorded clips labelled by recording intent |
| **Self-Recorded Dev** | `emotion2vec` active | 8 clips (1 actor) | Acc: 62.5% (Wilson: 30.6%–86.3%) | preliminary; self-recorded clips labelled by recording intent |
| **CREMA-D Dev Baseline** | `emotion2vec` zero-shot | 400 clips (9 actors) | Acc: 94.00% [90.06%, 97.14%]<br>UAR: 95.19% [90.09%, 97.14%] | acted speech; first 100 clips/class; actor-cluster bootstrap CI |
| **CREMA-D Dev CV** | Candidate Head | 400 clips (9 actors) | Decision: not pursued (frozen) | $\Delta\text{UAR} < 0.03$ threshold; frozen before test evaluation |
| **CREMA-D Held-Out Test** | `emotion2vec` (one-shot)| 1,673 clips (31 actors)| Acc: 93.31% [91.63%, 94.68%]<br>UAR: 94.88% [91.86%, 94.86%] | acted speech, actor-disjoint, not natural speech; 94.84% on answered |
| **Natural Speech Set** | Multi-rater consensus | 0 clips | not yet measured | protocol in [NATURAL_SET_PROTOCOL.md](docs/NATURAL_SET_PROTOCOL.md) |

- **Coverage & Abstention**: In test evaluation, the confusion matrix accounts for 1,646 clips while $N=1,673$; 27 clips (~1.6%) were abstained/uncertain (no abstention keys in JSON). Per-class recall and UAR use answered clips only, while accuracy counts all 27 as incorrect (93.31% overall vs 94.84% answered).
- **CI Note**: The test UAR point estimate (94.88%) lies slightly above its reported cluster-bootstrap upper bound (94.86%).
- **Deferred Benchmarks**: RAVDESS and multi-condition synthetic noise robustness are **not done** (no result files exist).

## Latency & Resource Utilization
Values quoted from [evaluation/results/latency.md](evaluation/results/latency.md) (approximate; varied between runs on the same machine, e.g. 5s median was 0.69s in earlier run vs 0.55s in committed report):
- **Environment**: Intel x86_64 CPU, 7.7 GB RAM, Windows 11, Python 3.12.10, PyTorch 2.13.0+cpu (note: older documentation referenced Python 3.11). Peak process memory: not measured.
- **Cold Start**: Total 28.31s (Imports: 9.46s, Neural Loading: 17.08s, First Infer: 1.77s).
- **Warm Inference Medians (5 Repetitions)**:
  - **5.0s clip**: Median 0.551s ($\text{RTF} = 0.1102$) — Voice: 0.514s, Text: 0.0438s, Fusion: 0.00016s.
  - **15.0s clip**: Median 2.199s ($\text{RTF} = 0.1466$) — Voice: 2.157s, Text: 0.0416s, Fusion: 0.00015s.
  - **30.0s clip**: Median 6.734s ($\text{RTF} = 0.2245$) — Voice: 6.675s, Text: 0.0582s, Fusion: 0.00017s.
- **Scope & Limits**: Times voice emotion, text sentiment, and fusion only on a tiled sample clip (`real_voice.wav`). Whisper ASR, VAD, diarization, audio decoding, and UI rendering are NOT included; **end-to-end latency including Whisper and diarization: not yet measured**. RTF increases with duration (0.11 at 5s to 0.22 at 30s); scaling is not linear.

## Post-Hoc Probability Calibration
From [evaluation/results/calibration.json](evaluation/results/calibration.json):
- **DEV Set** ($N=1,451$ clips across 28 actors; cached dev set larger than 400-clip report): $T=3.1192$ fitted on DEV by NLL. ECE: 0.0482 $\rightarrow$ 0.0155 (95% CI: [0.0110, 0.0328], mean 0.0203); NLL: 0.3799 $\rightarrow$ 0.1944.
- **TEST Set (REPORT-ONLY)** ($N=1,673$ clips across 31 actors): $T=3.1192$ fixed from DEV without refitting. Read cached outputs of the single test run and changed nothing. ECE: 0.0406 $\rightarrow$ 0.0207; NLL: 0.3341 $\rightarrow$ 0.1768.
- **Invariance & Status**: Argmax decisions and accuracy are strictly unchanged by temperature scaling. Evaluated on acted speech; $T$ is **NOT** applied in the app.
- **Confidence Saturation**: Raw emotion2vec confidence is saturated: 1,377 of 1,451 dev clips lie above 0.9 confidence (accuracy 96.3%), while 14 clips between 0.6–0.7 had only 35.7% accuracy.

## Operational Limitations
1. **Confidence is Not Correctness**: Sadness is the weakest class in both dev (recall 86.87%, precision 97.73%) and test (recall 87.65%, precision 99.45%), mostly confused with Neutral. On self-recorded clip `Sad1.wav` (segment 1), emotion2vec predicted Neutral at ~100% confidence (`app_vs_baseline.md:48`).
2. **Neutral Text vs Emotional Voice**: Acted clips with fixed neutral sentences register as modality mismatches (review gauge 35%–62%).
3. **Same-Speaker Merge Rule**: Turns separated by $\le 0.25$s are merged without an upper duration cap (`app/diarization.py:196`).
4. **Attention Review Gauge**: The review triage metric is an advisory heuristic unvalidated on clinical or operational outcomes.
5. **Acted vs Natural Speech**: High CREMA-D accuracy (93–94%) does not generalize to conversational speech where prosodic expression is nuanced and sub-second.

## How to Run
Tested on **Python 3.12.10**, **torch 2.13.0+cpu**, and `av==18.1.0` (pinned in `requirements.txt` to prevent `av 19.0.2` breaking `faster-whisper 1.2.1` via `metadata_errors`). Other transitive dependencies are not pinned.

```powershell
python -m venv .venv && .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest tests -q
python -m streamlit run streamlit_app.py
```
