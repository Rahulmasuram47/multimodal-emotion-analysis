"""
Quantitative Evaluation Metrics for Multimodal Emotion Analysis.

Covers:
- Multiclass classification: accuracy, precision, recall, F1, confusion matrix
- Speech-to-text: Word Error Rate (WER) with Levenshtein alignment
- Speaker count comparison: Exact match accuracy and Mean Absolute Error (MAE)
"""

import re
import string
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _normalize_text_for_wer(text: str) -> List[str]:
    """Tokenize and normalize text for WER computation (lowercase, strip punctuation)."""
    if not text:
        return []
    # Remove punctuation except internal apostrophes if any
    cleaned = re.sub(r"[^\w\s']", " ", text.lower())
    return cleaned.strip().split()


def compute_word_error_rate(
    reference: Optional[str], hypothesis: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Compute Word Error Rate (WER) between a reference transcript and an ASR hypothesis.

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
            return {
                "wer": 0.0,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "reference_words": 0,
                "hypothesis_words": 0,
            }
        return {
            "wer": 1.0,
            "substitutions": 0,
            "deletions": 0,
            "insertions": n_hyp,
            "reference_words": 0,
            "hypothesis_words": n_hyp,
        }

    # Dynamic programming table for Levenshtein distance on words
    # dp[i][j] stores (cost, substitutions, deletions, insertions)
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(n_hyp + 1)] for _ in range(n_ref + 1)
    ]

    for i in range(1, n_ref + 1):
        dp[i][0] = (i, 0, i, 0)  # i deletions

    for j in range(1, n_hyp + 1):
        dp[0][j] = (j, 0, 0, j)  # j insertions

    for i in range(1, n_ref + 1):
        for j in range(1, n_hyp + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost, s, d, ins = dp[i - 1][j - 1]
                del_cost, ds, dd, dins = dp[i - 1][j]
                ins_cost, is_, id_, iins = dp[i][j - 1]

                choice_sub = (sub_cost + 1, s + 1, d, ins)
                choice_del = (del_cost + 1, ds, dd + 1, dins)
                choice_ins = (ins_cost + 1, is_, id_, iins + 1)

                dp[i][j] = min(choice_sub, choice_del, choice_ins, key=lambda x: x[0])

    total_cost, s, d, ins = dp[n_ref][n_hyp]
    wer = round(float(total_cost) / float(n_ref), 4)

    return {
        "wer": wer,
        "substitutions": s,
        "deletions": d,
        "insertions": ins,
        "reference_words": n_ref,
        "hypothesis_words": n_hyp,
    }


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """
    Compute accuracy, precision, recall, F1, and confusion matrix for categorical predictions.

    Args:
        y_true: Ground truth string labels.
        y_pred: Predicted string labels.
        labels: Explicit set/order of class labels. If None, derived from union of inputs.

    Returns:
        Dictionary with overall and per-class metrics.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}"
        )

    if not y_true:
        return {
            "status": "NO_DATA",
            "total_samples": 0,
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "per_class": {},
            "confusion_matrix": {},
            "labels": list(labels or []),
        }

    # Determine unique labels
    if labels is not None:
        target_labels = list(labels)
    else:
        target_labels = sorted(list(set(y_true) | set(y_pred)))

    label_to_idx = {lbl: idx for idx, lbl in enumerate(target_labels)}
    num_labels = len(target_labels)

    # Build confusion matrix: rows = true, cols = pred
    matrix = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
    correct = 0
    total = len(y_true)

    for yt, yp in zip(y_true, y_pred):
        if yt in label_to_idx and yp in label_to_idx:
            matrix[label_to_idx[yt]][label_to_idx[yp]] += 1
        if yt == yp:
            correct += 1

    accuracy = round(float(correct) / float(total), 4) if total > 0 else 0.0

    # Per-class metrics
    per_class: Dict[str, Dict[str, Any]] = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for i, lbl in enumerate(target_labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(num_labels) if r != i)
        fn = sum(matrix[i][c] for c in range(num_labels) if c != i)
        support = sum(matrix[i])

        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            float(2 * prec * rec) / float(prec + rec)
            if (prec + rec) > 0
            else 0.0
        )

        per_class[lbl] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support,
        }

        # For macro averaging, only include classes present in the ground truth
        if support > 0:
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

    macro_precision = (
        round(sum(precisions) / len(precisions), 4) if precisions else 0.0
    )
    macro_recall = round(sum(recalls) / len(recalls), 4) if recalls else 0.0
    macro_f1 = round(sum(f1s) / len(f1s), 4) if f1s else 0.0

    # Human-readable dictionary confusion matrix: cm[true_label][pred_label] = count
    cm_dict: Dict[str, Dict[str, int]] = {}
    for i, true_lbl in enumerate(target_labels):
        cm_dict[true_lbl] = {}
        for j, pred_lbl in enumerate(target_labels):
            cm_dict[true_lbl][pred_lbl] = matrix[i][j]

    return {
        "status": "OK",
        "total_samples": total,
        "accuracy": accuracy,
        "precision_macro": macro_precision,
        "recall_macro": macro_recall,
        "f1_macro": macro_f1,
        "per_class": per_class,
        "confusion_matrix": cm_dict,
        "matrix_grid": matrix,
        "labels": target_labels,
    }


def compute_speaker_metrics(
    y_true: Sequence[Optional[int]], y_pred: Sequence[Optional[int]]
) -> Dict[str, Any]:
    """
    Compare true vs predicted speaker counts where ground truth is available.
    """
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
            "exact_match_accuracy": None,
            "mae": None,
            "per_sample": [],
        }

    exact_matches = sum(1 for yt, yp in valid_pairs if yt == yp)
    total = len(valid_pairs)
    exact_acc = round(float(exact_matches) / float(total), 4)

    abs_errors = [abs(yt - yp) for yt, yp in valid_pairs]
    mae = round(float(sum(abs_errors)) / float(total), 4)

    per_sample = [
        {"true_speaker_count": yt, "predicted_speaker_count": yp, "diff": yp - yt}
        for yt, yp in valid_pairs
    ]

    return {
        "status": "OK",
        "count": total,
        "exact_match_accuracy": exact_acc,
        "mae": mae,
        "per_sample": per_sample,
    }
