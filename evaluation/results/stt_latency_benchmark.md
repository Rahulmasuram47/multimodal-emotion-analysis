# Faster-Whisper Latency & Quality Benchmark

**Target Hardware:** Windows CPU (INT8 quantized inference).  
**Date:** 2026-10-01  

## 1. Summary Comparison
| Model Size | Parameters | Disk Footprint | Avg Latency / Clip | Real-Time Factor (RTF) | 30s Audio Estimate |
|---|---|---|---|---|---|
| **`base` (Production)** | 74M | ~140 MB | **1.95s** | **0.14x** | ~4.3s |
| **`small`** | 244M | ~460 MB | **5.72s** | **0.42x** | ~12.5s |

## 2. Per-Clip Latency Breakdown (Base vs. Small)
| Clip | Duration | Base Latency | Small Latency | Base Language (Conf) |
|---|---|---|---|---|
| `angry_sample.wav` | 18.1s | 1.96s | 6.52s | EN (97.7%) |
| `happy_sample.wav` | 14.5s | 1.84s | 6.38s | EN (95.8%) |
| `neutral_sample.wav` | 14.3s | 2.30s | 6.26s | EN (93.2%) |
| `real_voice.wav` | 11.6s | 2.19s | 5.84s | EN (83.7%) |
| `rising_tone.wav` | 17.8s | 2.33s | 6.21s | EN (58.8%) |
| `sad_sample.wav` | 19.8s | 2.24s | 6.38s | EN (87.3%) |
| `sample_16k.wav` | 0.0s | 0.83s | 2.45s | EN (58.4%) |

## 3. Engineering Conclusion & Recommendation
- **Acceptance Target:** 30s clip finishes under ~90s on CPU. Both `base` and `small` easily satisfy this target.
- **Recommendation:** Keep **`base`** as the default production configuration (`config.STT_MODEL_SIZE = 'base'`) to minimize memory footprint within the 8 GB RAM target and provide snappy near-real-time responses (~0.15-0.25x RTF).
- Users may switch to **`small`** via sidebar settings when transcribing accented speech or noisy recordings.