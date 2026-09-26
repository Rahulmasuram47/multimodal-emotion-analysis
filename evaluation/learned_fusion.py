"""
Learned Fusion Baseline Module for Multimodal Emotion Analysis.

Provides an experimental machine learning fusion baseline (Logistic Regression)
for empirical comparison against the production weighted rule-based fusion.

CRITICAL RULES:
- DO NOT replace or alter app/fusion.py (Rule-based fusion is the primary production model).
- Logistic Regression is used strictly as an experimental comparison baseline.
- If insufficient labelled data is available, reports "Insufficient labelled data" without
  fabricating any empirical results or metrics.
- CPU-friendly, interpretable, deterministic.
"""

from dataclasses import dataclass, asdict
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, LeaveOneOut

from config import AUDIO_EMOTION_CLASSES, TEXT_SENTIMENT_CLASSES
from evaluation.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    "prob_audio_neutral",
    "prob_audio_happy",
    "prob_audio_angry",
    "prob_audio_sad",
    "prob_text_positive",
    "prob_text_negative",
    "prob_text_neutral",
    "audio_confidence",
    "text_confidence",
    "intensity",
    "agreement",
    "mismatch",
]

MIN_TRAINING_SAMPLES: int = 10

# When labelled samples fall below this threshold, use Leave-One-Out CV
# instead of a fixed 30% hold-out to maximise use of scarce data.
MIN_LOO_CV_THRESHOLD: int = 30


@dataclass
class LearnedFusionResult:
    """Evaluation result of the experimental learned fusion baseline."""

    status: str  # "OK", "INSUFFICIENT_DATA", "ERROR"
    total_samples: int
    train_samples: int
    test_samples: int
    accuracy: Optional[float] = None
    precision_macro: Optional[float] = None
    recall_macro: Optional[float] = None
    f1_macro: Optional[float] = None
    confusion_matrix: Optional[Dict[str, Dict[str, int]]] = None
    message: str = ""
    feature_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_summary(self) -> str:
        if self.status != "OK":
            return f"Learned Fusion Baseline: {self.message or self.status}"
        return (
            f"Learned Fusion Baseline (Logistic Regression):\n"
            f"  Evaluated Test Samples: {self.test_samples} (Trained on {self.train_samples})\n"
            f"  Accuracy:  {self.accuracy:.4f}\n"
            f"  Precision: {self.precision_macro:.4f}\n"
            f"  Recall:    {self.recall_macro:.4f}\n"
            f"  Macro F1:  {self.f1_macro:.4f}"
        )


def extract_fusion_features(
    audio_result: Dict[str, Any],
    text_result: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """
    Extract a normalized 12-dimensional numeric feature vector from multimodal outputs.

    Features:
    [0..3]  Acoustic emotion probabilities: neutral, happy, angry, sad
    [4..6]  Text sentiment probabilities: positive, negative, neutral
    [7]     Acoustic confidence [0.0, 1.0]
    [8]     Text confidence [0.0, 1.0]
    [9]     Emotional intensity [0.0, 1.0]
    [10]    Agreement indicator (1.0 or 0.0)
    [11]    Mismatch indicator (1.0 or 0.0)
    """
    audio_res = audio_result or {}
    text_res = text_result or {}
    extra_res = extra or {}

    # 1. Acoustic probabilities
    scores = audio_res.get("scores", {}) or {}
    prob_audio_neu = float(scores.get("neutral", scores.get("neu", 0.0)))
    prob_audio_hap = float(scores.get("happy", scores.get("hap", 0.0)))
    prob_audio_ang = float(scores.get("angry", scores.get("ang", 0.0)))
    prob_audio_sad = float(scores.get("sad", 0.0))

    # 2. Text probabilities
    t_probs = text_res.get("probabilities", {}) or {}
    prob_text_pos = float(t_probs.get("positive", 0.0))
    prob_text_neg = float(t_probs.get("negative", 0.0))
    prob_text_neu = float(t_probs.get("neutral", 0.0))

    # 3. Confidences
    audio_conf = float(audio_res.get("confidence", 0.0))
    text_conf = float(text_res.get("confidence", 0.0))

    # 4. Intensity
    intensity = float(extra_res.get("intensity", max(audio_conf, text_conf)))

    # 5. Agreement & Mismatch indicators
    # Map audio to direction
    audio_emo = str(audio_res.get("emotion", "neutral")).lower()
    text_sent = str(text_res.get("sentiment", "neutral")).lower()

    if audio_emo == "happy":
        a_dir = "positive"
    elif audio_emo in ("angry", "sad"):
        a_dir = "negative"
    else:
        a_dir = "neutral"

    t_dir = text_sent if text_sent in ("positive", "negative", "neutral") else "neutral"

    agreement = 1.0 if (a_dir == t_dir and a_dir != "neutral") else 0.0
    mismatch = 1.0 if (
        (a_dir == "positive" and t_dir == "negative")
        or (a_dir == "negative" and t_dir == "positive")
    ) else 0.0

    return [
        round(prob_audio_neu, 4),
        round(prob_audio_hap, 4),
        round(prob_audio_ang, 4),
        round(prob_audio_sad, 4),
        round(prob_text_pos, 4),
        round(prob_text_neg, 4),
        round(prob_text_neu, 4),
        round(audio_conf, 4),
        round(text_conf, 4),
        round(intensity, 4),
        agreement,
        mismatch,
    ]


class LearnedFusionBaseline:
    """
    Experimental Logistic Regression classifier for multimodal fusion comparison.

    Strictly separate from the primary production rule-based fusion.
    """

    def __init__(self, max_iter: int = 1000, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.is_fitted: bool = False
        self.classes_: Optional[List[str]] = None

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[str]) -> "LearnedFusionBaseline":
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        self.classes_ = list(self.model.classes_)
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> List[str]:
        if not self.is_fitted:
            raise RuntimeError("LearnedFusionBaseline is not fitted. Call fit() first.")
        X_arr = np.array(X, dtype=np.float32)
        preds = self.model.predict(X_arr)
        return [str(p) for p in preds]

    def predict_proba(self, X: Sequence[Sequence[float]]) -> Dict[str, List[float]]:
        if not self.is_fitted:
            raise RuntimeError("LearnedFusionBaseline is not fitted. Call fit() first.")
        X_arr = np.array(X, dtype=np.float32)
        probs = self.model.predict_proba(X_arr)
        return {
            cls_name: [round(float(p), 4) for p in probs[:, idx]]
            for idx, cls_name in enumerate(self.classes_ or [])
        }

    def train_and_evaluate(
        self,
        samples_data: Sequence[Dict[str, Any]],
        ground_truth_labels: Sequence[Optional[str]],
        test_size: float = 0.30,
    ) -> "LearnedFusionResult":
        """Train and evaluate the learned fusion baseline.

        Selection rule (transparent, configurable via MIN_LOO_CV_THRESHOLD in
        learned_fusion.py):
          - N ≥ MIN_LOO_CV_THRESHOLD  →  30% held-out test split
          - MIN_TRAINING_SAMPLES ≤ N < MIN_LOO_CV_THRESHOLD  →  Leave-One-Out CV
          - N < MIN_TRAINING_SAMPLES  →  INSUFFICIENT_DATA

        If labelled samples < MIN_TRAINING_SAMPLES or unique classes < 2:
        Returns LearnedFusionResult with status="INSUFFICIENT_DATA".
        """
        # Filter pairs where ground truth is present
        valid_X: List[List[float]] = []
        valid_y: List[str] = []

        for item, gt in zip(samples_data, ground_truth_labels):
            if gt and str(gt).strip().lower() not in ("n/a", "none", "", "nan"):
                feat = extract_fusion_features(
                    item.get("audio_result", {}),
                    item.get("text_result", {}),
                    item.get("extra", {}),
                )
                valid_X.append(feat)
                valid_y.append(str(gt).strip().lower())

        total_labelled = len(valid_y)

        if total_labelled < MIN_TRAINING_SAMPLES:
            return LearnedFusionResult(
                status="INSUFFICIENT_DATA",
                total_samples=len(samples_data),
                train_samples=0,
                test_samples=0,
                message=(
                    f"Insufficient labelled data for learned fusion model "
                    f"({total_labelled} labelled sample(s) available; "
                    f"minimum required is {MIN_TRAINING_SAMPLES})."
                ),
                feature_names=FEATURE_NAMES,
            )

        unique_classes = set(valid_y)
        if len(unique_classes) < 2:
            return LearnedFusionResult(
                status="INSUFFICIENT_DATA",
                total_samples=len(samples_data),
                train_samples=0,
                test_samples=0,
                message=(
                    f"Insufficient class diversity: all labelled samples belong to "
                    f"'{list(unique_classes)[0]}'. At least 2 distinct classes are required."
                ),
                feature_names=FEATURE_NAMES,
            )

        # Auto-select evaluation strategy
        if total_labelled < MIN_LOO_CV_THRESHOLD:
            return self._loo_evaluate(valid_X, valid_y, len(samples_data))
        else:
            return self._split_evaluate(valid_X, valid_y, len(samples_data), test_size)

    def _loo_evaluate(
        self,
        valid_X: List[List[float]],
        valid_y: List[str],
        total_dataset_samples: int,
    ) -> "LearnedFusionResult":
        """Leave-One-Out Cross-Validation evaluation."""
        loo = LeaveOneOut()
        X_arr = np.array(valid_X, dtype=np.float32)
        y_arr = np.array(valid_y)

        y_true_all: List[str] = []
        y_pred_all: List[str] = []

        try:
            for train_idx, test_idx in loo.split(X_arr):
                X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
                y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

                model = LogisticRegression(
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    solver="lbfgs",
                )
                model.fit(X_tr, y_tr)
                y_pred_all.extend([str(p) for p in model.predict(X_te)])
                y_true_all.extend([str(t) for t in y_te])
        except Exception as e:
            return LearnedFusionResult(
                status="ERROR",
                total_samples=total_dataset_samples,
                train_samples=0,
                test_samples=0,
                message=f"LOO-CV failed: {e}",
                feature_names=FEATURE_NAMES,
            )

        labels = sorted(list(set(valid_y)))
        metrics = compute_classification_metrics(y_true_all, y_pred_all, labels=labels)

        return LearnedFusionResult(
            status="OK",
            total_samples=total_dataset_samples,
            train_samples=len(valid_y) - 1,   # typical LOO train size
            test_samples=len(valid_y),          # n predictions total
            accuracy=metrics.get("accuracy"),
            precision_macro=metrics.get("precision_macro"),
            recall_macro=metrics.get("recall_macro"),
            f1_macro=metrics.get("f1_macro"),
            confusion_matrix=metrics.get("confusion_matrix"),
            message=f"Leave-One-Out CV ({len(valid_y)} folds). Used because N={len(valid_y)} < {MIN_LOO_CV_THRESHOLD}.",
            feature_names=FEATURE_NAMES,
        )

    def _split_evaluate(
        self,
        valid_X: List[List[float]],
        valid_y: List[str],
        total_dataset_samples: int,
        test_size: float,
    ) -> "LearnedFusionResult":
        """Stratified train/test split evaluation."""
        unique_classes = set(valid_y)
        class_counts = {c: valid_y.count(c) for c in unique_classes}
        can_stratify = all(cnt >= 2 for cnt in class_counts.values())

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                valid_X,
                valid_y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=valid_y if can_stratify else None,
            )
        except Exception as e:
            return LearnedFusionResult(
                status="ERROR",
                total_samples=total_dataset_samples,
                train_samples=0,
                test_samples=0,
                message=f"Split failed: {e}",
                feature_names=FEATURE_NAMES,
            )

        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)

        labels = sorted(list(unique_classes))
        metrics = compute_classification_metrics(y_test, y_pred, labels=labels)

        return LearnedFusionResult(
            status="OK",
            total_samples=total_dataset_samples,
            train_samples=len(X_train),
            test_samples=len(X_test),
            accuracy=metrics.get("accuracy"),
            precision_macro=metrics.get("precision_macro"),
            recall_macro=metrics.get("recall_macro"),
            f1_macro=metrics.get("f1_macro"),
            confusion_matrix=metrics.get("confusion_matrix"),
            message="Trained and evaluated on held-out test split.",
            feature_names=FEATURE_NAMES,
        )

