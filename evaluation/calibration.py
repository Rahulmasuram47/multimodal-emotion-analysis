"""
Post-Hoc Probability Calibration via Temperature Scaling.

Fits scalar temperature T on DEV logits by minimizing Negative Log-Likelihood (NLL).
Computes 10-bin Expected Calibration Error (ECE) and NLL before and after calibration.
Preserves argmax and accuracy invariant.
Evaluates relationship with VOICE_CONF_CAP (0.65).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

logger = logging.getLogger(__name__)


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


def run_calibration_pipeline(
    dev_logits: Optional[np.ndarray] = None,
    dev_labels: Optional[np.ndarray] = None,
    test_logits: Optional[np.ndarray] = None,
    test_labels: Optional[np.ndarray] = None,
    voice_conf_cap: float = 0.65,
    output_path: str | Path = "evaluation/results/calibration.json",
) -> Dict[str, Any]:
    """Execute calibration pipeline or record clean 'not applicable' report if no logits exist."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if dev_logits is None or len(dev_logits) == 0 or dev_labels is None or len(dev_labels) == 0:
        result = {
            "status": "not applicable: no usable scores",
            "reason": "Active voice model outputs no cached logits from dev partition clips.",
            "temperature": 1.0,
            "ece_before": None,
            "ece_after": None,
            "nll_before": None,
            "nll_after": None,
            "reliability_bins": [],
            "voice_conf_cap": voice_conf_cap,
            "invariance_statement": (
                "Argmax predictions and classification accuracy are strictly invariant under temperature scaling; "
                "temperature calibration re-aligns probability magnitudes but does not alter top-1 decision boundaries."
            ),
            "recommendation": (
                f"Maintain VOICE_CONF_CAP = {voice_conf_cap} as an empirical safety threshold in decision fusion. "
                "Temperature scaling is recommended only as a soft-scaling adjunct, never replacing the decision cap."
            ),
        }
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result

    # Fit temperature T on DEV
    T = fit_temperature(dev_logits, dev_labels)

    # Evaluate on TEST (or DEV fallback if test is missing)
    eval_logits = test_logits if (test_logits is not None and len(test_logits) > 0) else dev_logits
    eval_labels = test_labels if (test_labels is not None and len(test_labels) > 0) else dev_labels

    nll_before = compute_nll(eval_logits, eval_labels, temperature=1.0)
    nll_after = compute_nll(eval_logits, eval_labels, temperature=T)

    probs_before = np.exp(eval_logits - np.max(eval_logits, axis=1, keepdims=True))
    probs_before /= np.sum(probs_before, axis=1, keepdims=True)

    scaled_eval_logits = eval_logits / T
    probs_after = np.exp(scaled_eval_logits - np.max(scaled_eval_logits, axis=1, keepdims=True))
    probs_after /= np.sum(probs_after, axis=1, keepdims=True)

    ece_before, bins_before = compute_ece(probs_before, eval_labels, n_bins=10)
    ece_after, bins_after = compute_ece(probs_after, eval_labels, n_bins=10)

    result = {
        "status": "CALIBRATED",
        "temperature": T,
        "nll_before": round(nll_before, 4),
        "nll_after": round(nll_after, 4),
        "ece_before": round(ece_before, 4),
        "ece_after": round(ece_after, 4),
        "reliability_bins_calibrated": bins_after,
        "reliability_bins_uncalibrated": bins_before,
        "voice_conf_cap": voice_conf_cap,
        "invariance_statement": (
            "Argmax predictions and classification accuracy are strictly invariant under temperature scaling; "
            "temperature calibration re-aligns probability magnitudes but does not alter top-1 decision boundaries."
        ),
        "recommendation": (
            f"Maintain VOICE_CONF_CAP = {voice_conf_cap} as an empirical safety threshold in decision fusion. "
            f"At T={T:.2f}, model probabilities are well-dispersed, but the {voice_conf_cap} cap prevents over-confident acoustic overrides."
        ),
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    res = run_calibration_pipeline()
    print("Calibration Pipeline Status:", res["status"])
    print("Saved to evaluation/results/calibration.json")
