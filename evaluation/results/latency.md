# Latency and Real-Time Factor (RTF) Benchmark Report

## Hardware & Runtime Environment
- **Operating System**: Windows 11
- **Processor (CPU)**: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
- **Total Physical RAM**: 7.7 GB (peak process memory: not measured)
- **Python Version**: 3.12.10
- **PyTorch Version**: 2.13.0+cpu
- **Execution Target**: CPU Only (GPU Accelerated: False)
- **Active Voice Backend**: `emotion2vec`
- **Active Voice Model ID**: `emotion2vec/emotion2vec_plus_base`

> [!NOTE]
> All measurements were taken strictly on CPU in compliance with the Windows CPU-only hardware target. No GPU values are inferred or extrapolated.

## Cold Start Latency
- **Total Cold Start Time**: `28.31s`
  - Module Imports: `9.46s`
  - Neural Weight Loading (emotion2vec + RoBERTa + faster-whisper): `17.08s`
  - First Inference Pass: `1.77s`

## Warm Inference & Real-Time Factor (RTF)
Measurements taken as median across 5 iterations per clip duration:

| Audio Duration | Median Latency | RTF (Proc s / Audio s) | Voice Stage | Text Stage | Fusion Stage |
|---|---|---|---|---|---|
| **5.0s** | `0.551s` | `0.1102` | `0.514s` | `0.0438s` | `0.00016s` |
| **15.0s** | `2.199s` | `0.1466` | `2.157s` | `0.0416s` | `0.00015s` |
| **30.0s** | `6.734s` | `0.2245` | `6.675s` | `0.0582s` | `0.00017s` |

## Pipeline Stage Characteristics (derived from table above)
- **Voice Emotion (emotion2vec)**: Accounts for 93% to 99% of total per-clip latency across the tested durations.
- **Text Sentiment (RoBERTa)**: Median `0.0438s` per call (a single fixed sentence; see Scope below).
- **Multimodal Fusion**: Median `0.00016s` per call.
- **RTF range**: `0.1102` (shortest clip) to `0.2245` (longest clip); RTF increases with audio duration.

## Scope and Limits
- **Stages timed**: Voice emotion inference, text sentiment inference, rule-based fusion.
- **Stages NOT timed**: Whisper ASR / speech-to-text, VAD, speaker diarization, audio file decoding, Streamlit UI rendering.
- **Audio inputs**: 5 s, 15 s, and 30 s clips at 16 kHz mono, built by tiling the first usable local sample file from `sample_audio/` (real_voice.wav, neutral_sample.wav, or sample_16k.wav); if none found, a synthetic 440 Hz + noise waveform is generated.
- **Text input**: One fixed sentence per iteration (not representative of ASR output variability).
- **Repetitions**: Median of 5 warm runs per duration.
- **Execution**: CPU only; no GPU acceleration.
- **No end-to-end pipeline time is claimed**; real-world latency includes all untimed stages listed above.
