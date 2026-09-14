"""
Evaluation Harness for Multimodal Emotion Analysis System.

Executes quantitative evaluation across an EvaluationDataset without fabricating data.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from config import AUDIO_EMOTION_CLASSES, TEXT_SENTIMENT_CLASSES
from evaluation.dataset import EvaluationDataset, EvaluationSample, load_evaluation_dataset
from evaluation.metrics import (
    compute_classification_metrics,
    compute_word_error_rate,
    compute_speaker_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SamplePrediction:
    """Predicted outputs for an evaluation sample."""

    filename: str
    pred_voice_emotion: Optional[str] = None
    pred_text_sentiment: Optional[str] = None
    pred_transcript: Optional[str] = None
    pred_speaker_count: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Comprehensive result of evaluation harness execution."""

    total_samples: int
    voice_emotion_metrics: Dict[str, Any]
    text_sentiment_metrics: Dict[str, Any]
    speech_to_text_metrics: Dict[str, Any]
    speaker_count_metrics: Dict[str, Any]
    sample_records: List[Dict[str, Any]]
    has_ground_truth: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "has_ground_truth": self.has_ground_truth,
            "voice_emotion": self.voice_emotion_metrics,
            "text_sentiment": self.text_sentiment_metrics,
            "speech_to_text": self.speech_to_text_metrics,
            "speaker_count": self.speaker_count_metrics,
            "sample_records": self.sample_records,
        }

    def format_summary(self) -> str:
        """Format an explainable plain-text summary report."""
        lines = [
            "=" * 64,
            "MULTIMODAL EMOTION ANALYSIS — EVALUATION REPORT",
            "=" * 64,
            f"Total Evaluated Samples: {self.total_samples}",
            f"Ground Truth Present: {self.has_ground_truth}",
            "",
        ]

        if not self.has_ground_truth:
            lines.append("NOTICE: Empirical metrics are pending user-provided ground-truth data.")
            lines.append("The evaluation harness and metric calculation logic are fully initialized.")
            lines.append("=" * 64)
            return "\n".join(lines)

        # Voice emotion
        lines.append("1. VOICE EMOTION METRICS:")
        if self.voice_emotion_metrics.get("status") == "OK":
            ve = self.voice_emotion_metrics
            lines.append(f"   Evaluated: {ve['total_samples']}")
            lines.append(f"   Accuracy:  {ve['accuracy']}")
            lines.append(f"   Precision: {ve['precision_macro']}")
            lines.append(f"   Recall:    {ve['recall_macro']}")
            lines.append(f"   F1-Score:  {ve['f1_macro']}")
            lines.append("   Confusion Matrix (True v Pred):")
            for true_lbl, preds in ve["confusion_matrix"].items():
                row_str = " | ".join(f"{p}:{cnt}" for p, cnt in preds.items())
                lines.append(f"     {true_lbl:<8}: {row_str}")
        else:
            lines.append("   Status: N/A (no voice emotion ground truth labels present)")

        lines.append("")

        # Text sentiment
        lines.append("2. TEXT SENTIMENT METRICS:")
        if self.text_sentiment_metrics.get("status") == "OK":
            ts = self.text_sentiment_metrics
            lines.append(f"   Evaluated: {ts['total_samples']}")
            lines.append(f"   Accuracy:  {ts['accuracy']}")
            lines.append(f"   Precision: {ts['precision_macro']}")
            lines.append(f"   Recall:    {ts['recall_macro']}")
            lines.append(f"   F1-Score:  {ts['f1_macro']}")
            lines.append("   Confusion Matrix (True v Pred):")
            for true_lbl, preds in ts["confusion_matrix"].items():
                row_str = " | ".join(f"{p}:{cnt}" for p, cnt in preds.items())
                lines.append(f"     {true_lbl:<8}: {row_str}")
        else:
            lines.append("   Status: N/A (no text sentiment ground truth labels present)")

        lines.append("")

        # Speech to text
        lines.append("3. SPEECH-TO-TEXT (WER) METRICS:")
        stt = self.speech_to_text_metrics
        if stt.get("status") == "OK":
            lines.append(f"   Evaluated:         {stt['total_evaluated']}")
            lines.append(f"   Average WER:       {stt['average_wer']}")
            lines.append(f"   Total Ref Words:   {stt['total_reference_words']}")
            lines.append(f"   Total Hyp Words:   {stt['total_hypothesis_words']}")
            lines.append(f"   Substitutions:     {stt['total_substitutions']}")
            lines.append(f"   Deletions:         {stt['total_deletions']}")
            lines.append(f"   Insertions:        {stt['total_insertions']}")
        else:
            lines.append("   Status: N/A (no ground truth transcripts provided)")

        lines.append("")

        # Speaker count
        lines.append("4. SPEAKER COUNT METRICS:")
        spk = self.speaker_count_metrics
        if spk.get("status") == "OK":
            lines.append(f"   Evaluated:            {spk['count']}")
            lines.append(f"   Exact Match Accuracy: {spk['exact_match_accuracy']}")
            lines.append(f"   Mean Absolute Error:  {spk['mae']}")
        else:
            lines.append("   Status: N/A (no speaker count ground truth provided)")

        lines.append("=" * 64)
        return "\n".join(lines)


class EvaluationHarness:
    """Coordinates quantitative evaluation against ground-truth datasets."""

    def __init__(
        self,
        voice_classes: Sequence[str] = AUDIO_EMOTION_CLASSES,
        sentiment_classes: Sequence[str] = TEXT_SENTIMENT_CLASSES,
    ):
        self.voice_classes = list(voice_classes)
        self.sentiment_classes = list(sentiment_classes)

    def evaluate_predictions(
        self,
        dataset: EvaluationDataset,
        predictions: Sequence[Union[SamplePrediction, Dict[str, Any]]],
    ) -> EvaluationResult:
        """
        Evaluate a sequence of pre-computed predictions against the dataset.

        Matches predictions to samples by index or by filename.
        """
        pred_map: Dict[str, SamplePrediction] = {}
        for idx, p in enumerate(predictions):
            if isinstance(p, SamplePrediction):
                key = p.filename or str(idx)
                pred_map[key] = p
            elif isinstance(p, dict):
                key = p.get("filename") or str(idx)
                pred_map[key] = SamplePrediction(
                    filename=key,
                    pred_voice_emotion=p.get("pred_voice_emotion") or p.get("emotion"),
                    pred_text_sentiment=p.get("pred_text_sentiment") or p.get("sentiment"),
                    pred_transcript=p.get("pred_transcript") or p.get("transcript") or p.get("text"),
                    pred_speaker_count=p.get("pred_speaker_count") or p.get("speaker_count"),
                    extra=p,
                )

        ve_true: List[str] = []
        ve_pred: List[str] = []

        ts_true: List[str] = []
        ts_pred: List[str] = []

        wer_results: List[Dict[str, Any]] = []

        spk_true: List[Optional[int]] = []
        spk_pred: List[Optional[int]] = []

        sample_records: List[Dict[str, Any]] = []
        has_any_gt = False

        for idx, sample in enumerate(dataset):
            pred = pred_map.get(sample.filename) or pred_map.get(str(idx))
            if pred is None:
                continue

            record: Dict[str, Any] = {
                "filename": sample.filename,
                "true_voice_emotion": sample.true_voice_emotion,
                "pred_voice_emotion": pred.pred_voice_emotion,
                "true_text_sentiment": sample.true_text_sentiment,
                "pred_text_sentiment": pred.pred_text_sentiment,
                "true_speaker_count": sample.true_speaker_count,
                "pred_speaker_count": pred.pred_speaker_count,
                "true_transcript": sample.true_transcript,
                "pred_transcript": pred.pred_transcript,
                "wer": None,
                "notes": sample.notes,
            }

            # 1. Voice emotion
            if sample.true_voice_emotion is not None:
                has_any_gt = True
                if pred.pred_voice_emotion:
                    ve_true.append(sample.true_voice_emotion)
                    ve_pred.append(pred.pred_voice_emotion)

            # 2. Text sentiment
            if sample.true_text_sentiment is not None:
                has_any_gt = True
                if pred.pred_text_sentiment:
                    ts_true.append(sample.true_text_sentiment)
                    ts_pred.append(pred.pred_text_sentiment)

            # 3. Transcript / WER
            if sample.true_transcript is not None:
                has_any_gt = True
                wer_info = compute_word_error_rate(sample.true_transcript, pred.pred_transcript)
                if wer_info is not None:
                    record["wer"] = wer_info["wer"]
                    wer_results.append(wer_info)

            # 4. Speaker count
            if sample.true_speaker_count is not None:
                has_any_gt = True
                spk_true.append(sample.true_speaker_count)
                spk_pred.append(pred.pred_speaker_count)

            sample_records.append(record)

        # Compute metric aggregates
        ve_metrics = compute_classification_metrics(
            ve_true, ve_pred, labels=self.voice_classes
        )
        ts_metrics = compute_classification_metrics(
            ts_true, ts_pred, labels=self.sentiment_classes
        )
        spk_metrics = compute_speaker_metrics(spk_true, spk_pred)

        # Aggregate WER
        if wer_results:
            total_ref = sum(w["reference_words"] for w in wer_results)
            total_hyp = sum(w["hypothesis_words"] for w in wer_results)
            total_subs = sum(w["substitutions"] for w in wer_results)
            total_dels = sum(w["deletions"] for w in wer_results)
            total_ins = sum(w["insertions"] for w in wer_results)
            macro_wer = round(sum(w["wer"] for w in wer_results) / len(wer_results), 4)
            micro_wer = (
                round(float(total_subs + total_dels + total_ins) / float(total_ref), 4)
                if total_ref > 0
                else 0.0
            )
            stt_metrics = {
                "status": "OK",
                "total_evaluated": len(wer_results),
                "average_wer": macro_wer,
                "overall_wer": micro_wer,
                "total_reference_words": total_ref,
                "total_hypothesis_words": total_hyp,
                "total_substitutions": total_subs,
                "total_deletions": total_dels,
                "total_insertions": total_ins,
            }
        else:
            stt_metrics = {
                "status": "NO_DATA",
                "total_evaluated": 0,
                "average_wer": None,
                "overall_wer": None,
            }

        return EvaluationResult(
            total_samples=len(dataset),
            voice_emotion_metrics=ve_metrics,
            text_sentiment_metrics=ts_metrics,
            speech_to_text_metrics=stt_metrics,
            speaker_count_metrics=spk_metrics,
            sample_records=sample_records,
            has_ground_truth=has_any_gt,
        )
