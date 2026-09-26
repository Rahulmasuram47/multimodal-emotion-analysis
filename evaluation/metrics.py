"""
Quantitative Evaluation Metrics for Multimodal Emotion Analysis.

Covers:
- Multiclass classification: accuracy, precision, recall, F1, confusion matrix
- Speech-to-text: Word Error Rate (WER) and Character Error Rate (CER)
- Speaker count comparison: Exact match accuracy and Mean Absolute Error (MAE)
- Binary detection: escalation precision, recall, F1
- Bootstrap 95% confidence intervals for every aggregate metric

All functions that receive ground truth treat blank / None values as EXCLUDED
from computation — they never count as a label.
"""

import random
import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ── Text normalisation ────────────────────────────────────────────────────────

def _normalize_text_for_wer(text: str) -> List[str]:
    """Tokenize and normalize text for WER computation (lowercase, strip punctuation)."""
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return cleaned.strip().split()


def _normalize_text_for_cer(text: str) -> List[str]:
    """Normalize text to a character list for CER (lowercase, strip punctuation, collapse spaces)."""
    if not text:
        return []
    cleaned = re.sub(r"[^\w\s']", "", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return list(cleaned)


# ── Levenshtein DP (shared) ───────────────────────────────────────────────────

def _levenshtein_counts(
    ref: List[str], hyp: List[str]
) -> Tuple[int, int, int, int]:
    """Return (total_edits, substitutions, deletions, insertions) via DP."""
    n_ref, n_hyp = len(ref), len(hyp)
    if n_ref == 0:
        return n_hyp, 0, 0, n_hyp
    if n_hyp == 0:
        return n_ref, 0, n_ref, 0

    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(n_hyp + 1)] for _ in range(n_ref + 1)
    ]
    for i in range(1, n_ref + 1):
        dp[i][0] = (i, 0, i, 0)
    for j in range(1, n_hyp + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                s_cost, s, d, ins = dp[i - 1][j - 1]
                d_cost, ds, dd, dins = dp[i - 1][j]
                i_cost, is_, id_, iins = dp[i][j - 1]
                choice_sub = (s_cost + 1, s + 1, d, ins)
                choice_del = (d_cost + 1, ds, dd + 1, dins)
                choice_ins = (i_cost + 1, is_, id_, iins + 1)
                dp[i][j] = min(choice_sub, choice_del, choice_ins, key=lambda x: x[0])

    return dp[n_ref][n_hyp]


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    values: List[float],
    n_iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Optional[float]]:
    """Compute bootstrap 95% CI for the mean of *values*.

    Returns ``{'mean': float, 'ci_lower': float, 'ci_upper': float, 'n': int}``
    or ``{'mean': None, 'ci_lower': None, 'ci_upper': None, 'n': 0}`` if empty.
    """
    n = len(values)
    if n == 0:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0}

    mean_val = round(sum(values) / n, 4)
    if n == 1:
        return {"mean": mean_val, "ci_lower": mean_val, "ci_upper": mean_val, "n": 1}

    rng = random.Random(seed)
    boot_means: List[float] = []
    for _ in range(n_iterations):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    lo_idx = int((alpha / 2) * n_iterations)
    hi_idx = int((1 - alpha / 2) * n_iterations) - 1
    lo_idx = max(0, min(lo_idx, len(boot_means) - 1))
    hi_idx = max(0, min(hi_idx, len(boot_means) - 1))

    return {
        "mean": mean_val,
        "ci_lower": round(boot_means[lo_idx], 4),
        "ci_upper": round(boot_means[hi_idx], 4),
        "n": n,
    }


# ── WER ───────────────────────────────────────────────────────────────────────

def compute_word_error_rate(
    reference: Optional[str], hypothesis: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Compute WER between a reference transcript and an ASR hypothesis.

    WER = (Substitutions + Deletions + Insertions) / N_reference

    Returns None if reference is None or empty (representing N/A).
    """
    if reference is None:
        return None

    ref_words = _normalize_text_for_wer(reference)
    hyp_words = _normalize_text_for_wer(hypothesis or "")

    n_ref = len(ref_words)
    n_hyp = len(hyp_words)

    if n_ref == 0:
        if n_hyp == 0:
            return {"wer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0,
                    "reference_words": 0, "hypothesis_words": 0}
        return {"wer": 1.0, "substitutions": 0, "deletions": 0, "insertions": n_hyp,
                "reference_words": 0, "hypothesis_words": n_hyp}

    total_cost, s, d, ins = _levenshtein_counts(ref_words, hyp_words)
    wer = round(float(total_cost) / float(n_ref), 4)

    return {
        "wer": wer,
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "reference_words": n_ref,
        "hypothesis_words": n_hyp,
    }


# ── CER ───────────────────────────────────────────────────────────────────────

def compute_character_error_rate(
    reference: Optional[str], hypothesis: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Compute Character Error Rate (CER) between reference and hypothesis.

    CER = (Substitutions + Deletions + Insertions at char level) / N_ref_chars

    Returns None if reference is None.
    """
    if reference is None:
        return None

    ref_chars = _normalize_text_for_cer(reference)
    hyp_chars = _normalize_text_for_cer(hypothesis or "")

    n_ref = len(ref_chars)
    n_hyp = len(hyp_chars)

    if n_ref == 0:
        if n_hyp == 0:
            return {"cer": 0.0, "substitutions": 0, "deletions": 0, "insertions": 0,
                    "reference_chars": 0, "hypothesis_chars": 0}
        return {"cer": 1.0, "substitutions": 0, "deletions": 0, "insertions": n_hyp,
                "reference_chars": 0, "hypothesis_chars": n_hyp}

    total_cost, s, d, ins = _levenshtein_counts(ref_chars, hyp_chars)
    cer = round(float(total_cost) / float(n_ref), 4)

    return {
        "cer": cer,
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "reference_chars": n_ref,
        "hypothesis_chars": n_hyp,
    }


# ── Classification ────────────────────────────────────────────────────────────

def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute accuracy, precision, recall, F1, confusion matrix, and bootstrap CIs.

    Args:
        y_true: Ground truth string labels.
        y_pred: Predicted string labels.
        labels: Explicit set/order of class labels. If None, derived from union.
        n_bootstrap: Bootstrap iterations for CI estimation.
        seed: Random seed for bootstrap.

    Returns:
        Dictionary with overall, per-class metrics, N, and 95% bootstrap CIs.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    if not y_true:
        return {
            "status": "NO_DATA",
            "total_samples": 0,
            "n": 0,
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "accuracy_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
            "f1_macro_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
            "per_class": {},
            "confusion_matrix": {},
            "labels": list(labels or []),
        }

    if labels is not None:
        target_labels = list(labels)
    else:
        target_labels = sorted(list(set(y_true) | set(y_pred)))

    label_to_idx = {lbl: idx for idx, lbl in enumerate(target_labels)}
    num_labels = len(target_labels)

    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    correct = 0
    total = len(y_true)
    per_sample_correct: List[float] = []

    for yt, yp in zip(y_true, y_pred):
        match = float(yt == yp)
        per_sample_correct.append(match)
        if yt in label_to_idx and yp in label_to_idx:
            matrix[label_to_idx[yt]][label_to_idx[yp]] += 1
        if yt == yp:
            correct += 1

    accuracy = round(float(correct) / float(total), 4) if total > 0 else 0.0

    per_class: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    per_sample_f1: List[float] = []

    for i, lbl in enumerate(target_labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(num_labels) if r != i)
        fn = sum(matrix[i][c] for c in range(num_labels) if c != i)
        support = sum(matrix[i])

        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (float(2 * prec * rec) / float(prec + rec) if (prec + rec) > 0 else 0.0)

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)
            # per-sample F1 proxy: 1.0 if correct for this class, else 0.0
            for yt, yp in zip(y_true, y_pred):
                if yt == lbl:
                    per_sample_f1.append(float(yp == lbl))

    macro_precision = round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    macro_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    macro_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0

    cm_dict: Dict[str, Dict[str, int]] = {}
    for i, true_lbl in enumerate(target_labels):
        cm_dict[true_lbl] = {}
        for j, pred_lbl in enumerate(target_labels):
            cm_dict[true_lbl][pred_lbl] = matrix[i][j]

    # Bootstrap CIs
    acc_ci = bootstrap_ci(per_sample_correct, n_iterations=n_bootstrap, seed=seed)
    f1_ci = bootstrap_ci(per_sample_f1, n_iterations=n_bootstrap, seed=seed)

    return {
        "status": "OK",
        "total_samples": total,
        "n": total,
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "accuracy_ci": acc_ci,
        "f1_macro_ci": f1_ci,
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "matrix_grid": matrix,
        "labels": target_labels,
    }


# ── Speaker count ─────────────────────────────────────────────────────────────

def compute_speaker_metrics(
    y_true: Sequence[Optional[int]],
    y_pred: Sequence[Optional[int]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare true vs predicted speaker counts where ground truth is available."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    valid_pairs = [
        (yt, yp)
        for yt, yp in zip(y_true, y_pred)
        if yt is not None and yp is not None
    ]

    if not valid_pairs:
        return {
            "status": "NO_DATA",
            "count": 0,
            "n": 0,
            "exact_match_accuracy": None,
            "mae": None,
            "mae_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
            "per_sample": [],
        }

    exact_matches = sum(1 for yt, yp in valid_pairs if yt == yp)
    total = len(valid_pairs)
    exact_acc = round(float(exact_matches) / float(total), 4)

    abs_errors = [abs(yt - yp) for yt, yp in valid_pairs]
    mae = round(float(sum(abs_errors)) / float(total), 4)
    mae_ci = bootstrap_ci([float(e) for e in abs_errors], n_iterations=n_bootstrap, seed=seed)

    per_sample = [
        {"true_speaker_count": yt, "predicted_speaker_count": yp, "diff": yp - yt}
        for yt, yp in valid_pairs
    ]

    return {
        "status": "OK",
        "count": total,
        "n": total,
        "exact_match_accuracy": exact_acc,
        "mae": mae,
        "mae_ci": mae_ci,
        "per_sample": per_sample,
    }


# ── Escalation / Binary detection ────────────────────────────────────────────

def compute_binary_detection_metrics(
    y_true: Sequence[Optional[str]],
    y_pred: Sequence[Optional[str]],
    positive_class: str = "yes",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute precision, recall, F1 for a binary yes/no detection task.

    Samples where y_true is None are excluded entirely.

    Args:
        y_true: Ground truth binary labels ('yes' | 'no' | None).
        y_pred: Predicted binary labels ('yes' | 'no').
        positive_class: Which class counts as the positive label.
        n_bootstrap: Bootstrap iterations for CI estimation.
        seed: RNG seed.

    Returns:
        Dict with status, n, precision, recall, f1, and CIs.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    valid_pairs = [
        (str(yt).lower(), str(yp).lower() if yp else "no")
        for yt, yp in zip(y_true, y_pred)
        if yt is not None
    ]

    if not valid_pairs:
        return {
            "status": "NO_DATA",
            "n": 0,
            "precision": None,
            "recall": None,
            "f1": None,
            "precision_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
            "recall_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
            "f1_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n": 0},
        }

    pos = positive_class.lower()
    tp = sum(1 for yt, yp in valid_pairs if yt == pos and yp == pos)
    fp = sum(1 for yt, yp in valid_pairs if yt != pos and yp == pos)
    fn = sum(1 for yt, yp in valid_pairs if yt == pos and yp != pos)

    precision = round(float(tp) / float(tp + fp), 4) if (tp + fp) > 0 else 0.0
    recall = round(float(tp) / float(tp + fn), 4) if (tp + fn) > 0 else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0

    # Per-sample binary vectors for bootstrap
    per_sample_prec_num = [float(yt == pos and yp == pos) for yt, yp in valid_pairs]
    per_sample_rec_num = [float(yt == pos and yp == pos) for yt, yp in valid_pairs]
    per_sample_f1v = [
        (2.0 if yt == yp == pos else 0.0) /
        max((1.0 if yt == pos else 0.0) + (1.0 if yp == pos else 0.0), 1)
        for yt, yp in valid_pairs
    ]

    p_ci = bootstrap_ci(per_sample_prec_num, n_iterations=n_bootstrap, seed=seed)
    r_ci = bootstrap_ci(per_sample_rec_num, n_iterations=n_bootstrap, seed=seed)
    f_ci = bootstrap_ci(per_sample_f1v, n_iterations=n_bootstrap, seed=seed)

    return {
        "status": "OK",
        "n": len(valid_pairs),
        "positive_class": positive_class,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "precision_ci": p_ci,
        "recall_ci": r_ci,
        "f1_ci": f_ci,
    }
