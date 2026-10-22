"""
Post-Hoc Probability Calibration via Temperature Scaling.

Fits scalar temperature T on DEV logits/scores by minimizing Negative Log-Likelihood (NLL).
Computes 10-bin Expected Calibration Error (ECE) and NLL before and after calibration.
Preserves argmax and accuracy invariant.
Evaluates relationship with VOICE_CONF_CAP (0.65).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.crema_d import (
    CANONICAL_4CLASS,
    compute_actor_cluster_bootstrap,
    compute_cache_key,
    load_crema_dataset,
    load_split_file,
)

logger = logging.getLogger("calibration")


def compute_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Compute Negative Log-Likelihood (NLL) for multiclass logits scaled by temperature."""
    if len(logits) == 0:
        return 0.0
    scaled_logits = logits / max(temperature, 1e-4)
    # Numerical stability shift
    shifted = scaled_logits - np.max(scaled_logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    n_samples = len(labels)
    # Clip probabilities to avoid log(0)
    eps = 1e-12
    correct_probs = np.clip(probs[np.arange(n_samples), labels], eps, 1.0)
    return float(-np.mean(np.log(correct_probs)))


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Expected Calibration Error (ECE) and reliability diagram bin statistics across n_bins.

    Returns:
        (ece_score, list_of_bin_dicts)
    """
    if len(probs) == 0:
        return 0.0, []

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_stats: List[Dict[str, Any]] = []
    total_samples = len(probs)
    weighted_ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

        bin_count = int(np.sum(in_bin))
        if bin_count > 0:
            bin_acc = float(np.mean(accuracies[in_bin]))
            bin_conf = float(np.mean(confidences[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            weighted_ece += (bin_count / total_samples) * bin_error
        else:
            bin_acc = 0.0
            bin_conf = float((bin_lower + bin_upper) / 2.0)
            bin_error = 0.0

        bin_stats.append({
            "bin_index": i,
            "range": [round(float(bin_lower), 2), round(float(bin_upper), 2)],
            "count": bin_count,
            "accuracy": round(bin_acc, 4),
            "confidence": round(bin_conf, 4),
            "calibration_gap": round(bin_error, 4),
        })

    return round(float(weighted_ece), 4), bin_stats


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Find scalar temperature T > 0 that minimizes NLL on validation logits."""
    if len(logits) == 0:
        return 1.0

    def objective(t: float) -> float:
        return compute_nll(logits, labels, temperature=t)

    res = minimize_scalar(objective, bounds=(0.05, 10.0), method="bounded")
    return float(round(res.x, 4))


def load_cached_records(
    clips: List[Any],
    model_id: str,
    cache_dir: Path,
    classes: Sequence[str] = CANONICAL_4CLASS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Load cached inference predictions for given clips.
    Extracts 4-class probability scores and converted logits.
    """
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    records: List[Dict[str, Any]] = []
    inspection_info = {
        "files_checked": 0,
        "files_found": 0,
        "sample_keys": [],
        "sample_shapes": {},
    }

    for clip in clips:
        inspection_info["files_checked"] += 1
        key = compute_cache_key(
            model_id=model_id,
            revision="main",
            target_sr=16000,
            clip_filename=clip.filename,
            labels=classes,
        )
        cache_file = cache_dir / f"{key}.npy"
        if not cache_file.is_file():
            continue

        inspection_info["files_found"] += 1
        try:
            raw_data = np.load(cache_file, allow_pickle=True)
            if hasattr(raw_data, "item") and callable(raw_data.item):
                d = raw_data.item()
            elif isinstance(raw_data, dict):
                d = raw_data
            else:
                d = {"embedding": raw_data}

            if not inspection_info["sample_keys"] and isinstance(d, dict):
                inspection_info["sample_keys"] = sorted(list(d.keys()))
                for k, v in d.items():
                    if hasattr(v, "shape"):
                        inspection_info["sample_shapes"][k] = list(v.shape)
                    elif isinstance(v, dict):
                        inspection_info["sample_shapes"][k] = f"dict({len(v)} keys)"

            logits = None
            # Check for per-class scores dict (from emotion2vec)
            if "scores" in d and isinstance(d["scores"], dict) and d["scores"]:
                scores = d["scores"]
                p = [scores.get(c, 0.0) for c in classes]
                sum_p = sum(p)
                if sum_p > 0:
                    probs = np.array([x / sum_p for x in p], dtype=np.float32)
                    logits = np.log(np.clip(probs, 1e-12, 1.0))
            # Check for raw logits (from superb)
            elif "logits" in d and d["logits"] is not None:
                logits = np.array(d["logits"], dtype=np.float32)

            if logits is not None and clip.canonical_emotion in cls_to_idx:
                records.append({
                    "filename": clip.filename,
                    "actor_id": clip.actor_id,
                    "canonical_emotion": clip.canonical_emotion,
                    "label_idx": cls_to_idx[clip.canonical_emotion],
                    "logits": logits,
                })
        except Exception:
            continue

    return records, inspection_info


def run_calibration_pipeline(
    dev_logits: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    dev_records: Optional[List[Dict[str, Any]]] = None,
    test_records: Optional[List[Dict[str, Any]]] = None,
    cache_info: Optional[Dict[str, Any]] = None,
    voice_conf_cap: float = 0.65,
    seed: int = 42,
    output_json: str | Path = "evaluation/results/calibration.json",
    output_md: str | Path = "evaluation/results/calibration.md",
) -> Dict[str, Any]:
    """Execute calibration pipeline or record clean 'not applicable' report if no usable scores exist."""
    out_json = Path(output_json)
    out_md = Path(output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # Decision line: line 215 determines whether usable scores exist
    if dev_logits is None or len(dev_logits) == 0 or dev_labels is None or len(dev_labels) == 0:
        keys_desc = (cache_info or {}).get("sample_keys", [])
        shapes_desc = (cache_info or {}).get("sample_shapes", {})
        result = {
            "status": "not applicable: no usable scores",
            "reason": (
                f"Cache files inspected contain keys {keys_desc} with shapes {shapes_desc}, "
                "but no usable per-class classification scores or logits were present for the target 4 classes. "
                "Decided at evaluation/calibration.py:215."
            ),
            "deciding_location": "evaluation/calibration.py:215",
            "temperature": 1.0,
            "ece_before": None,
            "ece_after": None,
            "nll_before": None,
            "nll_after": None,
            "reliability_bins": [],
            "voice_conf_cap": voice_conf_cap,
            "seed": seed,
            "n_dev_clips": 0,
            "n_dev_actors": 0,
            "invariance_statement": (
                "Argmax predictions and classification accuracy are strictly invariant under temperature scaling; "
                "temperature calibration re-aligns probability magnitudes but does not alter top-1 decision boundaries."
            ),
            "recommendation": (
                f"Maintain VOICE_CONF_CAP = {voice_conf_cap} as an empirical safety threshold in decision fusion. "
                "Temperature scaling is recommended only as a soft-scaling adjunct, never replacing the decision cap."
            ),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        md_content = f"""# Acted speech; argmax and accuracy are unchanged by temperature scaling

## Evaluation Scope & Dataset Setup
- **Status**: `{result['status']}`
- **Reason**: {result['reason']}
- **Decision File & Line**: `evaluation/calibration.py:215`
- **Seed**: {seed}
- **Number of Clips**: 0
- **Number of Actors**: 0
- **Voice Confidence Cap**: `{voice_conf_cap}`

## Technical Invariance & Recommendation
> [!NOTE]
> **Invariance Guarantee**: {result['invariance_statement']}

- **Safety Recommendation**: {result['recommendation']}
"""
        out_md.write_text(md_content, encoding="utf-8")
        return result

    # Usable scores found: fit temperature T on DEV
    T = fit_temperature(dev_logits, dev_labels)
    n_dev_clips = len(dev_logits)
    dev_actors = set(r["actor_id"] for r in (dev_records or []))
    n_dev_actors = len(dev_actors)

    nll_before = compute_nll(dev_logits, dev_labels, temperature=1.0)
    nll_after = compute_nll(dev_logits, dev_labels, temperature=T)

    probs_before = np.exp(dev_logits - np.max(dev_logits, axis=1, keepdims=True))
    probs_before /= np.sum(probs_before, axis=1, keepdims=True)

    scaled_dev_logits = dev_logits / T
    probs_after = np.exp(scaled_dev_logits - np.max(scaled_dev_logits, axis=1, keepdims=True))
    probs_after /= np.sum(probs_after, axis=1, keepdims=True)

    ece_before, bins_before = compute_ece(probs_before, dev_labels, n_bins=10)
    ece_after, bins_after = compute_ece(probs_after, dev_labels, n_bins=10)

    # Actor-cluster bootstrap CIs for dev metrics if dev_records available
    ece_ci = None
    nll_ci = None
    if dev_records and n_dev_actors >= 2:
        def _ece_after_fn(recs: List[Dict[str, Any]]) -> float:
            sub_logits = np.array([r["logits"] for r in recs]) / T
            sub_labels = np.array([r["label_idx"] for r in recs])
            sub_p = np.exp(sub_logits - np.max(sub_logits, axis=1, keepdims=True))
            sub_p /= np.sum(sub_p, axis=1, keepdims=True)
            e, _ = compute_ece(sub_p, sub_labels, n_bins=10)
            return e

        def _nll_after_fn(recs: List[Dict[str, Any]]) -> float:
            sub_logits = np.array([r["logits"] for r in recs])
            sub_labels = np.array([r["label_idx"] for r in recs])
            return compute_nll(sub_logits, sub_labels, temperature=T)

        ece_ci = compute_actor_cluster_bootstrap(dev_records, _ece_after_fn, n_bootstrap=1000, seed=seed)
        nll_ci = compute_actor_cluster_bootstrap(dev_records, _nll_after_fn, n_bootstrap=1000, seed=seed)

    # Evaluate on cached TEST outputs as REPORT-ONLY (with T fixed from DEV, no refitting)
    test_eval_report = None
    if test_records and len(test_records) > 0:
        test_logits = np.array([r["logits"] for r in test_records])
        test_labels = np.array([r["label_idx"] for r in test_records])
        t_nll_before = compute_nll(test_logits, test_labels, temperature=1.0)
        t_nll_after = compute_nll(test_logits, test_labels, temperature=T)
        t_p_before = np.exp(test_logits - np.max(test_logits, axis=1, keepdims=True))
        t_p_before /= np.sum(t_p_before, axis=1, keepdims=True)
        t_scaled_logits = test_logits / T
        t_p_after = np.exp(t_scaled_logits - np.max(t_scaled_logits, axis=1, keepdims=True))
        t_p_after /= np.sum(t_p_after, axis=1, keepdims=True)
        t_ece_before, t_bins_before = compute_ece(t_p_before, test_labels, n_bins=10)
        t_ece_after, t_bins_after = compute_ece(t_p_after, test_labels, n_bins=10)

        test_eval_report = {
            "evaluation_mode": "REPORT-ONLY (fixed T derived from dev; no refitting)",
            "n_test_clips": len(test_records),
            "n_test_actors": len(set(r["actor_id"] for r in test_records)),
            "nll_before": round(t_nll_before, 4),
            "nll_after": round(t_nll_after, 4),
            "ece_before": round(t_ece_before, 4),
            "ece_after": round(t_ece_after, 4),
            "reliability_bins_calibrated": t_bins_after,
        }

    # Identify bin containing VOICE_CONF_CAP (0.65)
    cap_bin_info = next((b for b in bins_after if b["range"][0] <= voice_conf_cap <= b["range"][1]), None)

    result = {
        "status": "CALIBRATED",
        "temperature": T,
        "seed": seed,
        "n_dev_clips": n_dev_clips,
        "n_dev_actors": n_dev_actors,
        "nll_before": round(nll_before, 4),
        "nll_after": round(nll_after, 4),
        "nll_ci": nll_ci if nll_ci else "no CI",
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "ece_ci": ece_ci if ece_ci else "no CI",
        "reliability_bins_calibrated": bins_after,
        "reliability_bins_uncalibrated": bins_before,
        "voice_conf_cap": voice_conf_cap,
        "cap_bin_info": cap_bin_info,
        "test_report_only": test_eval_report,
        "invariance_statement": (
            "Argmax predictions and classification accuracy are strictly invariant under temperature scaling; "
            "temperature calibration re-aligns probability magnitudes but does not alter top-1 decision boundaries."
        ),
        "recommendation": (
            f"Maintain VOICE_CONF_CAP = {voice_conf_cap} as an empirical safety threshold in decision fusion. "
            f"At T={T:.2f}, probability distributions are softened/aligned, and the {voice_conf_cap} cap prevents over-confident acoustic overrides."
        ),
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Markdown table of reliability bins
    bin_rows = []
    for b in bins_after:
        bin_rows.append(
            f"| `{b['range'][0]:.2f} - {b['range'][1]:.2f}` | {b['count']} | "
            f"`{b['confidence'] * 100:.2f}%` | `{b['accuracy'] * 100:.2f}%` | `{b['calibration_gap'] * 100:.2f}%` |"
        )
    bin_table = "\n".join([
        "| Confidence Range | Clip Count | Mean Confidence | Accuracy | Calibration Gap |",
        "|---|---|---|---|---|",
    ] + bin_rows)

    ece_ci_str = f" (95% Actor-Cluster CI: `[{ece_ci.get('ci_lower', 0):.4f}, {ece_ci.get('ci_upper', 0):.4f}]`)" if ece_ci else " (no CI)"

    test_section = ""
    if test_eval_report:
        test_section = f"""
## Cached Test Split Evaluation (REPORT-ONLY)
> [!NOTE]
> **Split Isolation Compliance**: Temperature $T={T:.2f}$ was fixed exclusively from the DEV split. Test metrics are evaluated as REPORT-ONLY with strictly no refitting or parameter adjustment.

- **Test Clips**: {test_eval_report['n_test_clips']} across {test_eval_report['n_test_actors']} held-out actors
- **Test ECE Before**: `{test_eval_report['ece_before']:.4f}` -> **After**: `{test_eval_report['ece_after']:.4f}`
- **Test NLL Before**: `{test_eval_report['nll_before']:.4f}` -> **After**: `{test_eval_report['nll_after']:.4f}`
"""

    md_content = f"""# Acted speech; argmax and accuracy are unchanged by temperature scaling

## Evaluation Scope & Partition Details
- **Dataset Partition**: DEV split (N = {n_dev_clips} clips across {n_dev_actors} actors, Seed: {seed})
- **Active Backend Model**: `emotion2vec` (4 canonical classes: {CANONICAL_4CLASS})
- **Learned Temperature (T)**: `{T:.4f}` (Dev NLL optimized via bounded scalar search)

## Calibration Metrics (DEV)
- **Negative Log-Likelihood (NLL)**:
  - Before Calibration ($T=1.00$): `{nll_before:.4f}`
  - After Calibration ($T={T:.2f}$): `{nll_after:.4f}`
- **Expected Calibration Error (ECE - 10 Bins)**:
  - Before Calibration ($T=1.00$): `{ece_before:.4f}` (`{ece_before * 100:.2f}%`)
  - After Calibration ($T={T:.2f}$): `{ece_after:.4f}` (`{ece_after * 100:.2f}%`){ece_ci_str}

## Reliability Diagram Bins (Calibrated DEV)
{bin_table}

## Relationship to VOICE_CONF_CAP ({voice_conf_cap})
- **Voice Confidence Cap Placement**: Situated in bin `[{cap_bin_info['range'][0]:.2f}, {cap_bin_info['range'][1]:.2f}]` (Bin Count: {cap_bin_info['count']}).
- **Policy Recommendation**: {result['recommendation']}

{test_section}
## Invariance Statement
> [!IMPORTANT]
> {result['invariance_statement']}
"""
    out_md.write_text(md_content, encoding="utf-8")
    return result


def execute_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    """Execute live calibration loading data and cache from paths."""
    from evaluation.run_crema_d_baseline import inspect_voice_backends
    backend_info = inspect_voice_backends()
    active_model_id = backend_info["active_model_id"]

    split_path = Path(args.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file missing: {split_path}")
    split_data = load_split_file(split_path)
    dev_actors = set(split_data.get("dev_actors", []))
    test_actors = set(split_data.get("test_actors", []))
    seed = split_data.get("seed", 42)

    cache_dir = Path(args.cache_dir)
    all_clips = load_crema_dataset(args.data_dir)

    dev_clips = [c for c in all_clips if c.actor_id in dev_actors and c.canonical_emotion in CANONICAL_4CLASS]
    test_clips = [c for c in all_clips if c.actor_id in test_actors and c.canonical_emotion in CANONICAL_4CLASS]

    dev_records, dev_cache_info = load_cached_records(dev_clips, active_model_id, cache_dir)
    test_records, _ = load_cached_records(test_clips, active_model_id, cache_dir) if test_clips else ([], {})

    dev_logits = np.array([r["logits"] for r in dev_records]) if dev_records else None
    dev_labels = np.array([r["label_idx"] for r in dev_records]) if dev_records else None

    return run_calibration_pipeline(
        dev_logits=dev_logits,
        dev_labels=dev_labels,
        dev_records=dev_records,
        test_records=test_records,
        cache_info=dev_cache_info,
        voice_conf_cap=args.voice_conf_cap,
        seed=seed,
        output_json=args.output_json,
        output_md=args.output_md,
    )


def main():
    parser = argparse.ArgumentParser(description="Post-Hoc Probability Calibration via Temperature Scaling")
    parser.add_argument("--run", action="store_true", help="Execute live calibration routine")
    parser.add_argument("--data-dir", default="data/external_datasets/crema_d/AudioWAV", help="Path to CREMA-D AudioWAV folder")
    parser.add_argument("--split-file", default="evaluation/results/crema_d_split.json", help="Path to split file")
    parser.add_argument("--cache-dir", default="evaluation/results/cache", help="Cache directory")
    parser.add_argument("--output-json", default="evaluation/results/calibration.json", help="Output JSON path")
    parser.add_argument("--output-md", default="evaluation/results/calibration.md", help="Output Markdown path")
    parser.add_argument("--voice-conf-cap", type=float, default=0.65, help="Empirical voice confidence cap")
    args = parser.parse_args()

    # Step 1: Without --run print help and exit WITHOUT writing any file
    if not args.run:
        parser.print_help()
        sys.exit(0)

    res = execute_calibration(args)
    print("Calibration Pipeline Status:", res["status"])
    print(f"Results saved to {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
