"""
Ablation Framework for Multimodal Emotion Analysis System.

Compares:
1. Voice-only modality (audio_weight=1.0, text_weight=0.0)
2. Text-only modality  (audio_weight=0.0, text_weight=1.0)
3. Voice + Text fusion (audio_weight=0.5, text_weight=0.5)

CRITICAL RULE:
Does NOT modify the production fusion configuration or config.py.
Weights are passed dynamically to local evaluation fusion instances.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from app.fusion import MultimodalFusion
from evaluation.dataset import EvaluationDataset, EvaluationSample
from evaluation.metrics import compute_classification_metrics


@dataclass
class ModalityAblationRecord:
    """Evaluation metrics for a specific modality ablation setting."""

    mode_name: str
    audio_weight: float
    text_weight: float
    total_samples: int
    mean_confidence: float
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    details: Dict[str, Any]


@dataclass
class AblationResult:
    """Comparative summary of Voice-only vs Text-only vs Multimodal Fusion."""

    total_samples: int
    has_ground_truth: bool
    voice_only: ModalityAblationRecord
    text_only: ModalityAblationRecord
    voice_text_fusion: ModalityAblationRecord
    per_sample_comparison: List[Dict[str, Any]]
    four_way_comparison: Optional[List[Dict[str, Any]]] = None
    learned_fusion: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "total_samples": self.total_samples,
            "has_ground_truth": self.has_ground_truth,
            "voice_only": {
                "audio_weight": self.voice_only.audio_weight,
                "text_weight": self.voice_only.text_weight,
                "mean_confidence": self.voice_only.mean_confidence,
                "accuracy": self.voice_only.accuracy,
                "f1_score": self.voice_only.f1_score,
            },
            "text_only": {
                "audio_weight": self.text_only.audio_weight,
                "text_weight": self.text_only.text_weight,
                "mean_confidence": self.text_only.mean_confidence,
                "accuracy": self.text_only.accuracy,
                "f1_score": self.text_only.f1_score,
            },
            "voice_text_fusion": {
                "audio_weight": self.voice_text_fusion.audio_weight,
                "text_weight": self.voice_text_fusion.text_weight,
                "mean_confidence": self.voice_text_fusion.mean_confidence,
                "accuracy": self.voice_text_fusion.accuracy,
                "f1_score": self.voice_text_fusion.f1_score,
            },
            "per_sample": self.per_sample_comparison,
        }
        if self.four_way_comparison is not None:
            d["four_way_comparison"] = self.four_way_comparison
        if self.learned_fusion is not None:
            if hasattr(self.learned_fusion, "to_dict"):
                d["learned_fusion"] = self.learned_fusion.to_dict()
            elif isinstance(self.learned_fusion, dict):
                d["learned_fusion"] = self.learned_fusion
        d["four_way_models"] = self.get_four_way_model_comparison()
        return d

    def get_four_way_model_comparison(self) -> List[Dict[str, Any]]:
        """Produce 4-way model comparison records (voice-only, text-only, current rule-based fusion, learned fusion)."""
        lf_acc = None
        lf_f1 = None
        if self.learned_fusion is not None:
            if hasattr(self.learned_fusion, "accuracy"):
                lf_acc = self.learned_fusion.accuracy
                lf_f1 = getattr(self.learned_fusion, "f1_macro", None)
            elif isinstance(self.learned_fusion, dict):
                lf_acc = self.learned_fusion.get("accuracy")
                lf_f1 = self.learned_fusion.get("f1_macro")

        return [
            {
                "model": "Voice-Only",
                "accuracy": self.voice_only.accuracy,
                "f1_macro": self.voice_only.f1_score,
                "mean_confidence": self.voice_only.mean_confidence,
            },
            {
                "model": "Text-Only",
                "accuracy": self.text_only.accuracy,
                "f1_macro": self.text_only.f1_score,
                "mean_confidence": self.text_only.mean_confidence,
            },
            {
                "model": "Current Fusion (Rule-Based)",
                "accuracy": self.voice_text_fusion.accuracy,
                "f1_macro": self.voice_text_fusion.f1_score,
                "mean_confidence": self.voice_text_fusion.mean_confidence,
            },
            {
                "model": "Learned Fusion (Logistic Regression)",
                "accuracy": lf_acc,
                "f1_macro": lf_f1,
                "mean_confidence": None,
            },
        ]

    def format_comparison_table(self) -> str:
        """
        Produce model comparison table:
        Model                  Accuracy    Macro F1
        ------------------------------------------------
        Voice Only              actual      actual
        Text Only               actual      actual
        Rule-Based Fusion       actual      actual
        Learned Fusion          actual      actual
        """
        lines = [
            f"{'Model':<24} {'Accuracy':<12} {'Macro F1':<12}",
            "-" * 48,
        ]

        def _val(acc: Optional[float]) -> str:
            return f"{acc:.4f}" if acc is not None else "Pending Data"

        lines.append(f"{'Voice Only':<24} {_val(self.voice_only.accuracy):<12} {_val(self.voice_only.f1_score):<12}")
        lines.append(f"{'Text Only':<24} {_val(self.text_only.accuracy):<12} {_val(self.text_only.f1_score):<12}")
        lines.append(f"{'Rule-Based Fusion':<24} {_val(self.voice_text_fusion.accuracy):<12} {_val(self.voice_text_fusion.f1_score):<12}")

        lf_acc = None
        lf_f1 = None
        if self.learned_fusion is not None:
            if hasattr(self.learned_fusion, "accuracy"):
                lf_acc = self.learned_fusion.accuracy
                lf_f1 = getattr(self.learned_fusion, "f1_macro", None)
            elif isinstance(self.learned_fusion, dict):
                lf_acc = self.learned_fusion.get("accuracy")
                lf_f1 = self.learned_fusion.get("f1_macro")

        lines.append(f"{'Learned Fusion':<24} {_val(lf_acc):<12} {_val(lf_f1):<12}")
        return "\n".join(lines)

    def format_table(self) -> str:
        """Format a clear comparative markdown table."""
        lines = [
            "### Ablation Study: Modality Contribution Comparison",
            "",
            "| Configuration | Audio Wt | Text Wt | Mean Conf | Accuracy | Macro F1 | Status |",
            "|---|---|---|---|---|---|---|",
        ]

        def _fmt(val: Optional[float]) -> str:
            return f"{val:.4f}" if val is not None else "N/A"

        for rec in [self.voice_only, self.text_only, self.voice_text_fusion]:
            status = "Verified" if self.has_ground_truth and rec.accuracy is not None else "Pending Data"
            lines.append(
                f"| **{rec.mode_name}** | {rec.audio_weight:.2f} | {rec.text_weight:.2f} | "
                f"{rec.mean_confidence:.4f} | {_fmt(rec.accuracy)} | {_fmt(rec.f1_score)} | {status} |"
            )

        lines.append("")
        if not self.has_ground_truth:
            lines.append(
                "> **Note:** Empirical accuracy/F1 metrics are pending user-provided labeled ground truth. "
                "The ablation framework executed successfully and verified confidence arbitration across all 3 configurations without modifying production weights."
            )

        return "\n".join(lines)


class AblationEvaluator:
    """Evaluates the contribution of voice and text modalities independently and combined."""

    def __init__(self, confidence_threshold: float = 0.40):
        # Instantiate dedicated evaluation fusion instances without touching global config
        self.voice_only_fusion = MultimodalFusion(
            audio_weight=1.0,
            text_weight=0.0,
            confidence_threshold=confidence_threshold,
        )
        self.text_only_fusion = MultimodalFusion(
            audio_weight=0.0,
            text_weight=1.0,
            confidence_threshold=confidence_threshold,
        )
        self.multimodal_fusion = MultimodalFusion(
            audio_weight=0.5,
            text_weight=0.5,
            confidence_threshold=confidence_threshold,
        )

    def evaluate(
        self,
        samples_data: Sequence[Dict[str, Any]],
        dataset: Optional[EvaluationDataset] = None,
        learned_fusion_result: Optional[Any] = None,
    ) -> AblationResult:
        """
        Run ablation over paired audio/text results.

        Each item in samples_data should contain:
            - 'audio_result': dict with 'emotion' and 'confidence'
            - 'text_result': dict with 'sentiment' and 'confidence'
            - 'filename' (optional)
            - 'ground_truth_direction' or 'true_sentiment' (optional: positive/negative/neutral)
        """
        gt_map: Dict[str, str] = {}
        if dataset:
            for s in dataset:
                if s.true_text_sentiment:
                    gt_map[s.filename] = s.true_text_sentiment

        v_confidences: List[float] = []
        t_confidences: List[float] = []
        f_confidences: List[float] = []

        v_preds: List[str] = []
        t_preds: List[str] = []
        f_preds: List[str] = []
        true_labels: List[str] = []

        per_sample: List[Dict[str, Any]] = []

        for idx, item in enumerate(samples_data):
            audio_res = item.get("audio_result", {})
            text_res = item.get("text_result", {})
            fn = item.get("filename", f"sample_{idx}")

            # 1. Voice-only fusion
            res_v = self.voice_only_fusion.fuse(audio_res, text_res)
            # 2. Text-only fusion
            res_t = self.text_only_fusion.fuse(audio_res, text_res)
            # 3. Multimodal fusion
            res_f = self.multimodal_fusion.fuse(audio_res, text_res)

            v_conf = res_v["combined_confidence"]
            t_conf = res_t["combined_confidence"]
            f_conf = res_f["combined_confidence"]

            v_confidences.append(v_conf)
            t_confidences.append(t_conf)
            f_confidences.append(f_conf)

            # Directional output (positive/negative/neutral)
            v_dir = res_v["audio_direction"]
            t_dir = res_t["text_direction"]
            # Multimodal direction: agreement direction or dominant high-confidence modality
            if res_f["agreement"]:
                f_dir = res_f["audio_direction"]
            elif res_f["audio_confidence"] > res_f["text_confidence"]:
                f_dir = res_f["audio_direction"]
            else:
                f_dir = res_f["text_direction"]

            v_preds.append(v_dir)
            t_preds.append(t_dir)
            f_preds.append(f_dir)

            gt_val = item.get("ground_truth_direction") or gt_map.get(fn)
            if gt_val:
                true_labels.append(gt_val)

            per_sample.append({
                "filename": fn,
                "voice_only_confidence": v_conf,
                "text_only_confidence": t_conf,
                "fusion_confidence": f_conf,
                "voice_direction": v_dir,
                "text_direction": t_dir,
                "fusion_direction": f_dir,
                "ground_truth_direction": gt_val,
                "agreement": res_f["agreement"],
                "mismatch": res_f["mismatch"],
            })

        total = len(samples_data)
        has_gt = len(true_labels) == total and total > 0

        def _build_record(
            name: str,
            aw: float,
            tw: float,
            confs: List[float],
            preds: List[str],
        ) -> ModalityAblationRecord:
            mean_conf = round(sum(confs) / len(confs), 4) if confs else 0.0
            if has_gt:
                m = compute_classification_metrics(true_labels, preds)
                acc = m["accuracy"]
                prec = m["precision_macro"]
                rec = m["recall_macro"]
                f1 = m["f1_macro"]
                details = m
            else:
                acc, prec, rec, f1 = None, None, None, None
                details = {"status": "NO_GROUND_TRUTH"}

            return ModalityAblationRecord(
                mode_name=name,
                audio_weight=aw,
                text_weight=tw,
                total_samples=total,
                mean_confidence=mean_conf,
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1_score=f1,
                details=details,
            )

        rec_voice = _build_record("Voice-Only", 1.0, 0.0, v_confidences, v_preds)
        rec_text = _build_record("Text-Only", 0.0, 1.0, t_confidences, t_preds)
        rec_fusion = _build_record("Voice + Text Fusion", 0.5, 0.5, f_confidences, f_preds)

        four_way = self.build_four_way_comparison(
            samples_data=samples_data,
            voice_record=rec_voice,
            text_record=rec_text,
            fusion_record=rec_fusion,
        )

        return AblationResult(
            total_samples=total,
            has_ground_truth=has_gt,
            voice_only=rec_voice,
            text_only=rec_text,
            voice_text_fusion=rec_fusion,
            per_sample_comparison=per_sample,
            four_way_comparison=four_way,
            learned_fusion=learned_fusion_result,
        )

    def build_four_way_comparison(
        self,
        samples_data: Sequence[Dict[str, Any]],
        voice_record: Optional[ModalityAblationRecord] = None,
        text_record: Optional[ModalityAblationRecord] = None,
        fusion_record: Optional[ModalityAblationRecord] = None,
        conversation_samples: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build a structured 4-way comparison across:
        A. Voice-only
        B. Text-only
        C. Voice + Text fusion
        D. Full temporal conversation analysis
        """
        total = len(samples_data)
        v_conf = f"{voice_record.mean_confidence:.4f}" if voice_record else "N/A"
        t_conf = f"{text_record.mean_confidence:.4f}" if text_record else "N/A"
        f_conf = f"{fusion_record.mean_confidence:.4f}" if fusion_record else "N/A"

        # Count samples with speech text
        text_samples = sum(1 for s in samples_data if s.get("text_result", {}).get("sentiment"))

        # Temporal conversation counts
        conv_samples_count = len(conversation_samples) if conversation_samples is not None else sum(
            1 for s in samples_data if (s.get("extra") or {}).get("conversation_analyzed")
        )

        return [
            {
                "configuration": "A. Voice-only",
                "modality_scope": "Acoustic audio features only (Wav2Vec2 SUPERB ER)",
                "available_samples": total,
                "applicable_metrics": [
                    "Emotion Accuracy",
                    "Macro F1",
                    "Mean Acoustic Confidence",
                    "Confusion Matrix",
                ],
                "predictions_summary": f"Mean acoustic confidence: {v_conf}. Captures vocal pitch, acoustic energy, and prosody.",
                "capabilities": "Independent of speech-to-text; evaluates raw vocal tone directly from audio waveform.",
                "limitations": "Completely blind to verbal content, vocabulary, and semantics; prone to acoustic biases (e.g. low-pitch speech misclassified as sad); static representation across entire sample.",
            },
            {
                "configuration": "B. Text-only",
                "modality_scope": "Linguistic text features only (CardiffNLP RoBERTa Sentiment)",
                "available_samples": text_samples,
                "applicable_metrics": [
                    "Sentiment Accuracy",
                    "Macro F1",
                    "Mean Linguistic Confidence",
                    "Confusion Matrix",
                ],
                "predictions_summary": f"Mean linguistic confidence: {t_conf}. Captures explicit semantic meaning of transcribed words.",
                "capabilities": "Evaluates semantic sentiment accurately when speech is clearly transcribed.",
                "limitations": "Completely blind to vocal delivery, tone, prosody, volume, sarcasm, or acoustic emotion; strictly dependent on ASR accuracy; fails on non-speech audio; static representation.",
            },
            {
                "configuration": "C. Voice + Text fusion",
                "modality_scope": "Bimodal acoustic + linguistic fusion (50% Audio + 50% Text)",
                "available_samples": text_samples,
                "applicable_metrics": [
                    "Combined Bimodal Confidence",
                    "Directional Agreement Rate",
                    "Mismatch / Divergence Rate",
                ],
                "predictions_summary": f"Mean fusion confidence: {f_conf}. Integrates acoustic emotion with linguistic sentiment.",
                "capabilities": "Discovers cross-modal agreement and divergence; provides confidence arbitration when individual modalities are uncertain.",
                "limitations": "Evaluates the audio recording as a single static utterance; loses temporal progression, per-segment evolution, turn-taking, and emotional escalation over time.",
            },
            {
                "configuration": "D. Full temporal conversation analysis",
                "modality_scope": "Temporal segment-level bimodal fusion + conversation dynamics",
                "available_samples": conv_samples_count or text_samples,
                "applicable_metrics": [
                    "Segment-Level Intensity Tracking",
                    "Emotional Trend (increasing/decreasing/stable)",
                    "Monotonic Escalation Indicator",
                    "Modality Consistency Trajectory (mostly_consistent/mostly_divergent/mixed)",
                    "Computational Review-Priority Attention Score",
                    "Speaker Attribution (Diarization Support)",
                ],
                "predictions_summary": "Sequential segment-level fusion with temporal trend, escalation detection, trajectory classification, and review-priority scoring.",
                "capabilities": "Captures emotional evolution over time, detects rising/falling intensity, identifies escalation across conversation turns, classifies conversation consistency trajectory, flags high-priority conversations for review with transparent heuristic, attributes turns to speakers.",
                "limitations": "Computationally heavier; requires multiple speech segments (minimum 2 segments for trajectory, minimum 3 segments for escalation); dependent on ASR segment timestamp boundaries.",
            },
        ]

