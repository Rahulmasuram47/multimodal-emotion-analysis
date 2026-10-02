"""
Quantitative Evaluation and Ablation Framework for Multimodal Emotion Analysis.
"""

from evaluation.dataset import EvaluationDataset, EvaluationSample, load_evaluation_dataset
from evaluation.metrics import (
    compute_classification_metrics,
    compute_word_error_rate,
    compute_speaker_metrics,
)
from evaluation.harness import EvaluationHarness, EvaluationResult
from evaluation.ablation import AblationEvaluator, AblationResult

__all__ = [
    "EvaluationDataset",
    "EvaluationSample",
    "load_evaluation_dataset",
    "compute_classification_metrics",
    "compute_word_error_rate",
    "compute_speaker_metrics",
    "EvaluationHarness",
    "EvaluationResult",
    "AblationEvaluator",
    "AblationResult",
]
