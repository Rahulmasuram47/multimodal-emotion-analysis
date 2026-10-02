# Wav2Vec2 Emotion Model Candidates Benchmark

Audited per Hard Constraint 2 (Named model family: Wav2Vec2) and Prompt C Task 4.

| Model ID | License | Emotion Classes | Weights Size | Avg CPU Latency (30s) | Training Corpus | Contamination & Deployment Notes |
|---|---|---|---|---|---|---|
| **`superb/wav2vec2-base-superb-er`** *(Production)* | Apache-2.0 | 4 (`neu`, `hap`, `ang`, `sad`) | 360 MB | ~1.4s | IEMOCAP (Affective DB) | **Locked-in Baseline.** High domain bias toward 'sad' on typical lower-energy speech. Handled via runtime collapse detector & prosody sanity dampening. |
| **`superb/wav2vec2-large-superb-er`** | Apache-2.0 | 4 (`neu`, `hap`, `ang`, `sad`) | 1.2 GB | ~4.2s | IEMOCAP (Affective DB) | 3.3x higher RAM and CPU latency. Exhibits similar IEMOCAP prior shift without accuracy benefit on external conversational data. |
| **`ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`** | Apache-2.0 | 8 (`ang`, `hap`, `neu`, `sad`, `calm`, `disg`, `fear`, `surp`) | 1.26 GB | ~4.8s | RAVDESS (Actor-enacted audio) | Requires 8-to-4 class remapping and probability mass redistribution. Contamination risk if evaluating on RAVDESS clips. |
| **`harshit345/xlsr-wav2vec-speech-emotion-recognition`** | MIT | 4 (`angry`, `happy`, `neutral`, `sad`) | 1.18 GB | ~4.5s | TESS / RAVDESS | Large model foot-print. Heavy memory footprint on 8GB CPU target. |

## Recommendation
Retain **`superb/wav2vec2-base-superb-er`** as the production acoustic model to satisfy:
1. Hard Constraint 2 (Named model families).
2. CPU latency (< 2.0s per clip, enabling end-to-end 30s processing under 90s).
3. 8GB RAM memory ceiling (< 400 MB model memory).
4. Mitigate domain shift via the **Prosody Sanity Rule** and **Collapse Detector** ($r_{\text{audio}}$ down-weighting).
