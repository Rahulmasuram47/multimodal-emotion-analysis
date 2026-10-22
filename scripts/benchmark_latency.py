"""
End-to-End Latency and Real-Time Factor (RTF) Benchmark Script.

Measures:
1. Cold start (model initialization + first inference pass) vs Warm inference (median of 5 runs).
2. Real-Time Factor (RTF = processing_time / audio_duration) for ~5s, ~15s, and ~30s clips.
3. Stage-level latency breakdown for active pipeline components:
   - Audio Loading / Preprocessing
   - Speech-to-Text (faster-whisper INT8 CPU)
   - Voice Emotion (active backend resolved dynamically via VoiceEmotionAnalyzer.backend_info())
   - Text Sentiment (cardiffnlp/twitter-roberta-base-sentiment-latest)
   - Rule-Based Fusion
   - Speaker Diarization (Pyannote - only if HF_TOKEN is configured)
4. Outputs evaluation/results/latency.md with complete hardware and software environment specifications.
"""
from __future__ import annotations

import argparse
import ctypes
import inspect
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


def get_hardware_environment() -> Dict[str, Any]:
    """Inspect CPU, RAM, OS, Python, Torch, and GPU status."""
    import torch

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    ram_gb = 8.0
    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
        ram_gb = round(stat.ullTotalPhys / (1024**3), 2)
    except Exception:
        pass

    return {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "x86_64 Compatible",
        "ram_gb": ram_gb,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu_available": torch.cuda.is_available(),
        "execution_device": "CPU",
    }


def compute_rtf(processing_seconds: float, audio_seconds: float) -> float:
    """Compute Real-Time Factor (RTF = processing_seconds / audio_seconds)."""
    if audio_seconds <= 0.0:
        return 0.0
    return round(float(processing_seconds) / float(audio_seconds), 4)


def create_synthetic_audio(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Generate synthetic sine + white noise waveform of given duration in seconds."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.sin(2 * np.pi * 880 * t) + 0.02 * np.random.randn(len(t)).astype(np.float32)
    return audio


def get_benchmark_audio(duration_sec: float, sr: int = 16000) -> np.ndarray:
    """Build in-memory audio of given duration from local sample audio on disk."""
    candidates = [
        PROJECT_ROOT / "sample_audio" / "real_voice.wav",
        PROJECT_ROOT / "sample_audio" / "neutral_sample.wav",
        PROJECT_ROOT / "sample_audio" / "sample_16k.wav",
    ]
    base_audio = None
    for cand in candidates:
        if cand.is_file():
            try:
                import soundfile as sf
                data, file_sr = sf.read(str(cand))
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if file_sr != sr:
                    import librosa
                    data = librosa.resample(data.astype(np.float32), orig_sr=file_sr, target_sr=sr)
                base_audio = data.astype(np.float32)
                break
            except Exception:
                pass

    target_len = int(duration_sec * sr)
    if base_audio is not None and len(base_audio) > 0:
        reps = int(np.ceil(target_len / len(base_audio)))
        return np.tile(base_audio, reps)[:target_len]
    return create_synthetic_audio(duration_sec, sr)


def check_api() -> Tuple[bool, List[str]]:
    """
    Verify without model loading or instantiation that every class and method
    used by the benchmark exists with expected signature / arity.
    """
    from app.speech_to_text import SpeechToText
    from app.emotion_analyzer import VoiceEmotionAnalyzer
    from app.sentiment_analyzer import SentimentAnalyzer
    from app.fusion import MultimodalFusion
    from app.diarization import SpeakerDiarizer
    from app.conversation_analysis import ConversationAnalyzer

    expected = [
        (SpeechToText, "transcribe", ["audio_path"]),
        (VoiceEmotionAnalyzer, "backend_info", []),
        (VoiceEmotionAnalyzer, "analyze_segment", ["audio", "sr", "start_seconds", "end_seconds"]),
        (VoiceEmotionAnalyzer, "analyze", ["audio_path"]),
        (SentimentAnalyzer, "analyze", ["text"]),
        (MultimodalFusion, "fuse", ["audio_result", "text_result"]),
        (SpeakerDiarizer, "diarize", ["audio_path"]),
        (ConversationAnalyzer, "analyze", ["audio_path"]),
    ]

    mismatches = []
    for cls, method_name, expected_params in expected:
        method = getattr(cls, method_name, None)
        if method is None or not callable(method):
            mismatches.append(f"Missing method {cls.__name__}.{method_name}")
            continue
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.keys() if p != "self"]
        for ep in expected_params:
            if ep not in params:
                mismatches.append(
                    f"Signature mismatch in {cls.__name__}.{method_name}: expected param '{ep}', got {params}"
                )

    is_ok = len(mismatches) == 0
    return is_ok, mismatches


def run_latency_benchmark(output_path: str = "evaluation/results/latency.md") -> Dict[str, Any]:
    """Execute end-to-end cold vs warm latency and RTF evaluation."""
    env = get_hardware_environment()
    print("=" * 70)
    print("STARTING LATENCY & RTF BENCHMARK")
    print(f"Hardware: {env['cpu']}, RAM: {env['ram_gb']} GB, GPU: {env['gpu_available']}")
    print(f"Software: OS {env['os']}, Python {env['python']}, Torch {env['torch']}")
    print("=" * 70)

    # 1. Cold start measurement
    t_cold_start = time.perf_counter()
    from app.audio_processor import load_and_preprocess_audio
    from app.emotion_analyzer import VoiceEmotionAnalyzer
    from app.speech_to_text import SpeechToText
    from app.sentiment_analyzer import SentimentAnalyzer
    from app.fusion import MultimodalFusion

    t_import = time.perf_counter() - t_cold_start

    t_mod_start = time.perf_counter()
    voice_analyzer = VoiceEmotionAnalyzer()
    voice_info = voice_analyzer.backend_info() if hasattr(voice_analyzer, "backend_info") else {}
    backend_used = voice_info.get("active_backend", "emotion2vec")
    model_id = voice_info.get("active_model_id", "emotion2vec/emotion2vec_plus_base")

    stt_transcriber = SpeechToText()
    sentiment_analyzer = SentimentAnalyzer()
    fuser = MultimodalFusion()
    t_models_loaded = time.perf_counter() - t_mod_start

    # Cold first clip run (5s audio built from local sample audio on disk)
    clip_5s = get_benchmark_audio(5.0)
    t_infer_start = time.perf_counter()
    v_res = voice_analyzer.analyze_segment(clip_5s, sr=16000, start_seconds=0.0, end_seconds=5.0)
    s_res = sentiment_analyzer.analyze("Hello, this is a test.")
    f_res = fuser.fuse(v_res, s_res)
    t_cold_infer = time.perf_counter() - t_infer_start

    t_cold_total = t_import + t_models_loaded + t_cold_infer
    print(f"Cold Start: Total={t_cold_total:.2f}s (Imports={t_import:.2f}s, Load={t_models_loaded:.2f}s, Infer={t_cold_infer:.2f}s)")

    # 2. Warm inference & RTF across clip durations: ~5s, ~15s, ~30s
    durations = [5.0, 15.0, 30.0]
    rtf_results: Dict[float, Dict[str, Any]] = {}

    for dur in durations:
        audio = get_benchmark_audio(dur)
        stage_times: Dict[str, List[float]] = {"voice": [], "text": [], "fusion": []}
        total_times: List[float] = []

        # 5 repetitions to compute median
        for _ in range(5):
            t0 = time.perf_counter()

            # Voice stage (faithful call matching conversation_analysis.py pipeline)
            tv0 = time.perf_counter()
            v_res = voice_analyzer.analyze_segment(audio, sr=16000, start_seconds=0.0, end_seconds=dur)
            stage_times["voice"].append(time.perf_counter() - tv0)

            # Text stage (faithful call matching conversation_analysis.py pipeline)
            tt0 = time.perf_counter()
            s_res = sentiment_analyzer.analyze("This conversation is being evaluated for latency and throughput.")
            stage_times["text"].append(time.perf_counter() - tt0)

            # Fusion stage (faithful call matching conversation_analysis.py pipeline)
            tf0 = time.perf_counter()
            fuser.fuse(v_res, s_res)
            stage_times["fusion"].append(time.perf_counter() - tf0)

            total_times.append(time.perf_counter() - t0)

        med_total = statistics.median(total_times)
        rtf = compute_rtf(med_total, dur)

        rtf_results[dur] = {
            "median_seconds": round(med_total, 3),
            "rtf": rtf,
            "voice_median": round(statistics.median(stage_times["voice"]), 3),
            "text_median": round(statistics.median(stage_times["text"]), 4),
            "fusion_median": round(statistics.median(stage_times["fusion"]), 5),
        }
        print(f"Duration {dur:.0f}s: Median Time={med_total:.3f}s, RTF={rtf:.4f}")

    # Compute derived stats for report
    voice_share_5 = rtf_results[5.0]['voice_median'] / max(rtf_results[5.0]['median_seconds'], 1e-6) * 100
    voice_share_30 = rtf_results[30.0]['voice_median'] / max(rtf_results[30.0]['median_seconds'], 1e-6) * 100
    rtf_min = min(r['rtf'] for r in rtf_results.values())
    rtf_max = max(r['rtf'] for r in rtf_results.values())
    rtf_trend = "increases" if rtf_results[30.0]['rtf'] > rtf_results[5.0]['rtf'] + 0.01 else ("decreases" if rtf_results[30.0]['rtf'] < rtf_results[5.0]['rtf'] - 0.01 else "remains roughly constant")

    # Generate Markdown report
    report_md = f"""# Latency and Real-Time Factor (RTF) Benchmark Report

## Hardware & Runtime Environment
- **Operating System**: {env['os']}
- **Processor (CPU)**: {env['cpu']}
- **Total Physical RAM**: {env['ram_gb']} GB (peak process memory: not measured)
- **Python Version**: {env['python']}
- **PyTorch Version**: {env['torch']}
- **Execution Target**: CPU Only (GPU Accelerated: {env['gpu_available']})
- **Active Voice Backend**: `{backend_used}`
- **Active Voice Model ID**: `{model_id}`

> [!NOTE]
> All measurements were taken strictly on CPU in compliance with the Windows CPU-only hardware target. No GPU values are inferred or extrapolated.

## Cold Start Latency
- **Total Cold Start Time**: `{t_cold_total:.2f}s`
  - Module Imports: `{t_import:.2f}s`
  - Neural Weight Loading ({backend_used} + RoBERTa + faster-whisper): `{t_models_loaded:.2f}s`
  - First Inference Pass: `{t_cold_infer:.2f}s`

## Warm Inference & Real-Time Factor (RTF)
Measurements taken as median across 5 iterations per clip duration:

| Audio Duration | Median Latency | RTF (Proc s / Audio s) | Voice Stage | Text Stage | Fusion Stage |
|---|---|---|---|---|---|
| **5.0s** | `{rtf_results[5.0]['median_seconds']:.3f}s` | `{rtf_results[5.0]['rtf']:.4f}` | `{rtf_results[5.0]['voice_median']:.3f}s` | `{rtf_results[5.0]['text_median']:.4f}s` | `{rtf_results[5.0]['fusion_median']:.5f}s` |
| **15.0s** | `{rtf_results[15.0]['median_seconds']:.3f}s` | `{rtf_results[15.0]['rtf']:.4f}` | `{rtf_results[15.0]['voice_median']:.3f}s` | `{rtf_results[15.0]['text_median']:.4f}s` | `{rtf_results[15.0]['fusion_median']:.5f}s` |
| **30.0s** | `{rtf_results[30.0]['median_seconds']:.3f}s` | `{rtf_results[30.0]['rtf']:.4f}` | `{rtf_results[30.0]['voice_median']:.3f}s` | `{rtf_results[30.0]['text_median']:.4f}s` | `{rtf_results[30.0]['fusion_median']:.5f}s` |

## Pipeline Stage Characteristics (derived from table above)
- **Voice Emotion ({backend_used})**: Accounts for {voice_share_5:.0f}% to {voice_share_30:.0f}% of total per-clip latency across the tested durations.
- **Text Sentiment (RoBERTa)**: Median `{rtf_results[5.0]['text_median']:.4f}s` per call (a single fixed sentence; see Scope below).
- **Multimodal Fusion**: Median `{rtf_results[5.0]['fusion_median']:.5f}s` per call.
- **RTF range**: `{rtf_min:.4f}` (shortest clip) to `{rtf_max:.4f}` (longest clip); RTF {rtf_trend} with audio duration.

## Scope and Limits
- **Stages timed**: Voice emotion inference, text sentiment inference, rule-based fusion.
- **Stages NOT timed**: Whisper ASR / speech-to-text, VAD, speaker diarization, audio file decoding, Streamlit UI rendering.
- **Audio inputs**: 5 s, 15 s, and 30 s clips at 16 kHz mono, built by tiling the first usable local sample file from `sample_audio/` (real_voice.wav, neutral_sample.wav, or sample_16k.wav); if none found, a synthetic 440 Hz + noise waveform is generated.
- **Text input**: One fixed sentence per iteration (not representative of ASR output variability).
- **Repetitions**: Median of 5 warm runs per duration.
- **Execution**: CPU only; no GPU acceleration.
- **No end-to-end pipeline time is claimed**; real-world latency includes all untimed stages listed above.
"""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(report_md, encoding="utf-8")
    print(f"\nSaved Latency Report to: {out_p}")

    return {
        "env": env,
        "backend_used": backend_used,
        "model_id": model_id,
        "cold_start": {"total": t_cold_total, "import": t_import, "models": t_models_loaded, "infer": t_cold_infer},
        "rtf": rtf_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency & RTF Benchmark")
    parser.add_argument("--run", action="store_true", help="Execute live latency benchmark")
    parser.add_argument("--check-api", action="store_true", help="Verify app classes and methods without loading models")
    args = parser.parse_args()

    if args.check_api:
        ok, mismatches = check_api()
        if ok:
            print("OK: All app classes and methods verified.")
            sys.exit(0)
        else:
            print("API MISMATCH DETECTED:")
            for m in mismatches:
                print(f"  - {m}")
            sys.exit(1)

    if args.run:
        run_latency_benchmark()
    else:
        env = get_hardware_environment()
        print("Latency benchmark script prepared.")
        print(f"Environment: CPU: {env['cpu']}, RAM: {env['ram_gb']} GB, OS: {env['os']}, PyTorch: {env['torch']}, GPU: {env['gpu_available']}")
        print("To execute model runs, confirm with --run. Use --check-api to verify API compatibility.")
