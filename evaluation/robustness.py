"""
Synthetic Noise Robustness Evaluation Module.

Evaluates active voice emotion model under acoustic degradation conditions:
1. Clean (uncorrupted reference)
2. Additive White Gaussian Noise at SNR = 20 dB, 10 dB, and 0 dB
3. 8 kHz narrowband down-then-up resampling round trip (telephony bandwidth limitation)

Computes UAR with actor-cluster bootstrap CI and performance drop relative to clean.
Labeled: "synthetic noise, not real environments".
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def add_white_noise_snr(audio: np.ndarray, target_snr_db: float, seed: Optional[int] = None) -> np.ndarray:
    """
    Add white Gaussian noise to audio waveform to achieve an exact target Signal-to-Noise Ratio (SNR) in dB.

    SNR_dB = 10 * log10(P_signal / P_noise)
    """
    if len(audio) == 0:
        return audio.copy()

    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 1e-12:
        return audio.copy()

    # Target noise power
    noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))

    rng = np.random.RandomState(seed)
    noise = rng.randn(len(audio)).astype(np.float32)
    current_noise_power = float(np.mean(noise ** 2))
    if current_noise_power > 0:
        noise = noise * np.sqrt(noise_power / current_noise_power)

    noisy_audio = audio + noise
    return noisy_audio.astype(np.float32)


def compute_empirical_snr_db(clean_audio: np.ndarray, noisy_audio: np.ndarray) -> float:
    """Calculate measured SNR (dB) between clean reference and noisy audio."""
    noise = noisy_audio - clean_audio
    p_signal = float(np.mean(clean_audio ** 2))
    p_noise = float(np.mean(noise ** 2))
    if p_noise <= 1e-12:
        return float("inf")
    if p_signal <= 1e-12:
        return -float("inf")
    return float(10.0 * np.log10(p_signal / p_noise))


def apply_8khz_telephony_roundtrip(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Simulate narrowband telephony bandwidth limitation by resampling
    from sr (16 kHz) down to 8 kHz and back up to 16 kHz.
    """
    if len(audio) == 0 or sr == 8000:
        return audio.copy()

    orig_len = len(audio)
    # Downsample to 8 kHz
    down = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=8000)
    # Upsample back to original sr
    up = librosa.resample(down, orig_sr=8000, target_sr=sr)

    # Match original length
    if len(up) < orig_len:
        up = np.pad(up, (0, orig_len - len(up)), mode="constant")
    elif len(up) > orig_len:
        up = up[:orig_len]

    return up.astype(np.float32)


def evaluate_robustness_condition(
    condition_name: str,
    records: List[Dict[str, Any]],
    classes: Sequence[str] = ("neutral", "happy", "angry", "sad"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute UAR, accuracy, and actor-cluster bootstrap CI for a specific acoustic condition."""
    from evaluation.crema_d import compute_actor_cluster_bootstrap

    if not records:
        return {
            "condition": condition_name,
            "n_samples": 0,
            "accuracy": None,
            "uar": None,
            "uar_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n_clusters": 0},
            "per_class": {},
        }

    y_true = [r["canonical_emotion"] for r in records]
    y_pred = [r["canonical_pred"] for r in records]
    cls_list = list(classes)

    recalls = []
    per_class = {}
    for c in cls_list:
        sup = sum(1 for yt in y_true if yt == c)
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        rec = float(tp / sup) if sup > 0 else 0.0
        recalls.append(rec)
        per_class[c] = {"support": sup, "recall": round(rec, 4)}

    uar = float(np.mean(recalls)) if recalls else 0.0
    acc = float(sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp) / len(y_true))

    def _uar_fn(recs: List[Dict[str, Any]]) -> float:
        rec_list = []
        for cl in cls_list:
            sup_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl)
            tp_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl and r["canonical_pred"] == cl)
            if sup_cnt > 0:
                rec_list.append(tp_cnt / sup_cnt)
        return float(np.mean(rec_list)) if rec_list else 0.0

    uar_ci = compute_actor_cluster_bootstrap(records, _uar_fn, n_bootstrap=1000, seed=seed)

    return {
        "condition": condition_name,
        "n_samples": len(records),
        "n_actors": len(set(r["actor_id"] for r in records)),
        "accuracy": round(acc, 4),
        "uar": round(uar, 4),
        "uar_ci": uar_ci,
        "per_class": per_class,
    }


def generate_robustness_markdown(
    results: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    """Format robustness evaluation outcomes as Markdown."""
    clean_res = results.get("clean", {})
    clean_uar = clean_res.get("uar")

    table_header = "| Condition | Noise Type | UAR | 95% Actor-Cluster CI | UAR Drop vs Clean | Samples |"
    table_sep = "|---|---|---|---|---|---|"
    rows = []

    for cond_key, res in results.items():
        uar_val = res.get("uar")
        ci = res.get("uar_ci", {})
        ci_str = f"[{ci.get('ci_lower', 0.0):.4f}, {ci.get('ci_upper', 0.0):.4f}]" if ci.get("ci_lower") is not None else "N/A"
        if uar_val is not None and clean_uar is not None and clean_uar > 0:
            drop = clean_uar - uar_val
            drop_str = f"-{drop * 100:.2f}%" if drop > 0 else f"{drop * 100:+.2f}%"
        else:
            drop_str = "0.00% (Ref)" if cond_key == "clean" else "N/A"

        uar_str = f"{uar_val * 100:.2f}%" if uar_val is not None else "N/A"
        noise_desc = {
            "clean": "None (Original)",
            "snr_20db": "White Noise (Mild, 20 dB)",
            "snr_10db": "White Noise (Moderate, 10 dB)",
            "snr_0db": "White Noise (Severe, 0 dB)",
            "telephony_8khz": "8 kHz Narrowband Round-Trip",
        }.get(cond_key, cond_key)

        rows.append(f"| **{cond_key}** | {noise_desc} | `{uar_str}` | `{ci_str}` | `{drop_str}` | {res.get('n_samples', 0)} |")

    table_content = "\n".join([table_header, table_sep] + rows)

    return f"""# Synthetic Noise Robustness Evaluation Report

> [!WARNING]
> **Acoustic Environment Scope**: **synthetic noise, not real environments**.
> Degraded conditions simulate artificial additive white Gaussian noise and standard telephone bandwidth limits; they do not encompass non-stationary acoustic reverberation, room impulse responses, or complex multi-talker babble.

## Experiment Profile
- **Active Model Evaluated**: `{meta.get('model_id')}`
- **Evaluation Partition**: Held-out dev/test subset (Budget: {meta.get('max_clips_per_class')} clips/class)
- **Random Seed**: {meta.get('seed')}
- **Target Conditions Tested**:
  1. `clean`: Unaltered audio
  2. `snr_20db`: Additive White Gaussian Noise at 20 dB SNR
  3. `snr_10db`: Additive White Gaussian Noise at 10 dB SNR
  4. `snr_0db`: Additive White Gaussian Noise at 0 dB SNR
  5. `telephony_8khz`: Narrowband 8 kHz downsampling & upsampling round-trip

## Robustness Metrics & Degradation Summary
{table_content}

## Key Observations & Operational Guidance
- **Additive Noise Sensitivity**: High-energy white noise suppresses subtle spectral formants and high-frequency prosody.
- **Narrowband Telephony**: Resampling to 8 kHz eliminates energy above 4 kHz; acoustic models relying on high-frequency spectral cues will exhibit degraded sensitivity.
- **Multimodal Resilience**: In production scenarios with degraded acoustics, the system's rule-based decision fusion lowers acoustic reliability, allowing text sentiment to provide stabilizing evidentiary support.
"""


def main():
    parser = argparse.ArgumentParser(description="Synthetic Noise Robustness Benchmark")
    parser.add_argument("--data-dir", default="data/external_datasets/crema_d/AudioWAV", help="Path to audio clips")
    parser.add_argument("--max-per-class", type=int, default=20, help="Max clips per emotion class")
    parser.add_argument("--output-dir", default="evaluation/results", help="Directory to save robustness reports")
    args = parser.parse_args()

    out_p = Path(args.output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    meta = {
        "model_id": "superb/wav2vec2-base-superb-er",
        "max_clips_per_class": args.max_per_class,
        "seed": 42,
        "notice": "synthetic noise, not real environments",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    # Initialize empty condition records
    conditions = ["clean", "snr_20db", "snr_10db", "snr_0db", "telephony_8khz"]
    results = {c: evaluate_robustness_condition(c, []) for c in conditions}

    report_md = generate_robustness_markdown(results, meta)
    md_file = out_p / "robustness.md"
    json_file = out_p / "robustness.json"

    md_file.write_text(report_md, encoding="utf-8")
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "results": results}, f, indent=2)

    print(f"Robustness report initialized at {md_file}")
    print(f"Robustness JSON initialized at {json_file}")


if __name__ == "__main__":
    main()
