"""
Automated Evaluation Runner for Multimodal Emotion Analysis System.

Executes quantitative evaluation, ablation study, temporal analysis,
and diarization validation across the evaluation dataset without fabricating data.

Usage:
    python -m evaluation.run_evaluation
    or:
    python evaluation/run_evaluation.py [--dataset PATH] [--output-dir PATH]
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.emotion_analyzer import VoiceEmotionAnalyzer
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer
from app.fusion import MultimodalFusion
from app.conversation_analysis import ConversationAnalyzer
from evaluation.dataset import EvaluationDataset, EvaluationSample, load_evaluation_dataset
from evaluation.harness import EvaluationHarness, SamplePrediction, EvaluationResult
from evaluation.ablation import AblationEvaluator, AblationResult
from evaluation.diarization_validation import validate_diarization_cases, DiarizationValidationResult
from evaluation.learned_fusion import LearnedFusionBaseline, LearnedFusionResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DATASET_PATH = "data/evaluation_dataset_template.csv"
DEFAULT_RESULTS_DIR = "evaluation/results"


def format_human_readable_report(
    dataset: EvaluationDataset,
    eval_result: EvaluationResult,
    ablation_result: AblationResult,
    learned_fusion_result: LearnedFusionResult,
    missing_annotations: List[str],
    total_evaluated: int,
    total_failed: int,
) -> str:
    """Format human-readable plain text empirical evaluation report."""
    lines = [
        "=" * 64,
        "MULTIMODAL EMOTION ANALYSIS — EMPIRICAL EVALUATION REPORT",
        "=" * 64,
        f"Dataset:                 {getattr(dataset, 'dataset_path', 'data/evaluation_dataset_template.csv')}",
        f"Total Dataset Records:   {len(dataset)}",
        f"Total Samples Evaluated: {total_evaluated}",
        f"Failed Samples:          {total_failed}",
        f"Ground Truth Present:    {'YES' if eval_result.has_ground_truth else 'NO (Pending Data)'}",
        "",
        "ANNOTATION & GROUND TRUTH STATUS:",
    ]
    if missing_annotations:
        lines.append("  Pending / Missing Annotations:")
        for ann in missing_annotations:
            lines.append(f"    - {ann}")
    else:
        lines.append("  All modalities have ground-truth annotations.")
    lines.append("")

    # 1. Voice emotion
    lines.append("1. VOICE EMOTION (ACOUSTIC) METRICS:")
    ve = eval_result.voice_emotion_metrics
    if ve.get("status") == "OK":
        lines.append(f"   Evaluated: {ve['total_samples']}")
        lines.append(f"   Accuracy:  {ve['accuracy']:.4f}")
        lines.append(f"   Macro F1:  {ve['f1_macro']:.4f}")
        lines.append(f"   Precision: {ve['precision_macro']:.4f}")
        lines.append(f"   Recall:    {ve['recall_macro']:.4f}")
        lines.append("   Confusion Matrix (True v Pred):")
        for true_lbl, preds in ve.get("confusion_matrix", {}).items():
            row_str = " | ".join(f"{p}:{cnt}" for p, cnt in preds.items())
            lines.append(f"     {true_lbl:<8}: {row_str}")
    else:
        lines.append("   Status: Pending Data (no ground-truth voice emotion labels present)")
    lines.append("")

    # 2. Text sentiment
    lines.append("2. TEXT SENTIMENT (LINGUISTIC) METRICS:")
    ts = eval_result.text_sentiment_metrics
    if ts.get("status") == "OK":
        lines.append(f"   Evaluated: {ts['total_samples']}")
        lines.append(f"   Accuracy:  {ts['accuracy']:.4f}")
        lines.append(f"   Macro F1:  {ts['f1_macro']:.4f}")
        lines.append(f"   Precision: {ts['precision_macro']:.4f}")
        lines.append(f"   Recall:    {ts['recall_macro']:.4f}")
        lines.append("   Confusion Matrix (True v Pred):")
        for true_lbl, preds in ts.get("confusion_matrix", {}).items():
            row_str = " | ".join(f"{p}:{cnt}" for p, cnt in preds.items())
            lines.append(f"     {true_lbl:<8}: {row_str}")
    else:
        lines.append("   Status: Pending Data (no ground-truth text sentiment labels present)")
    lines.append("")

    # 3. Speech to text (WER)
    lines.append("3. SPEECH-TO-TEXT (WER) METRICS:")
    stt = eval_result.speech_to_text_metrics
    if stt.get("status") == "OK":
        lines.append(f"   Evaluated:         {stt['total_evaluated']}")
        lines.append(f"   Average WER:       {stt['average_wer']:.4f}")
        lines.append(f"   Total Ref Words:   {stt['total_reference_words']}")
        lines.append(f"   Total Hyp Words:   {stt['total_hypothesis_words']}")
    else:
        lines.append("   Status: Pending Data (no reference transcripts provided)")
    lines.append("")

    # 4. Speaker count
    lines.append("4. SPEAKER COUNT METRICS:")
    spk = eval_result.speaker_count_metrics
    if spk.get("status") == "OK":
        lines.append(f"   Evaluated:            {spk['count']}")
        lines.append(f"   Exact Match Accuracy: {spk['exact_match_accuracy']:.4f}")
        lines.append(f"   Mean Absolute Error:  {spk['mae']:.4f}")
    else:
        lines.append("   Status: Pending Data (no speaker count ground truth provided)")
    lines.append("")

    # 5. Model comparison
    lines.append("5. MODEL COMPARISON (ABLATION & LEARNED FUSION BASELINE):")
    lines.append(ablation_result.format_comparison_table())
    lines.append("")
    lines.append("LEARNED FUSION BASELINE DETAILS:")
    lines.append(f"  {learned_fusion_result.format_summary()}")
    lines.append("")

    # 6. Disclaimer
    lines.append("=" * 64)
    lines.append("MANDATORY SCIENTIFIC DISCLAIMER:")
    lines.append(
        "This system performs computational emotion inference from acoustic and linguistic patterns. "
        "It does not diagnose, assess, or make claims about a person's actual psychological or emotional state. "
        "Model confidence reflects classification certainty, not ground-truth correctness."
    )
    lines.append("=" * 64)
    return "\n".join(lines)


def run_pipeline_on_sample(
    sample: EvaluationSample,
    vea: VoiceEmotionAnalyzer,
    stt: SpeechToText,
    sa: SentimentAnalyzer,
    fusion: MultimodalFusion,
    ca: Optional[ConversationAnalyzer] = None,
) -> Dict[str, Any]:
    """
    Run production inference pipeline on a single audio file.
    Does NOT alter production thresholds, weights, or decision rules.
    """
    filepath = sample.filename
    if not os.path.exists(filepath):
        return {
            "status": "FAILED",
            "error": f"File not found: {filepath}",
            "filename": filepath,
        }

    try:
        # 1. Voice emotion (whole-audio acoustic)
        v_res = vea.analyze(filepath)
        pred_voice = v_res.get("emotion")
        v_conf = float(v_res.get("confidence", 0.0))

        # 2. Speech-to-text
        t_res = stt.transcribe(filepath)
        transcript = str(t_res.get("text", "") or "").strip()
        segments = t_res.get("segments", [])

        # 3. Text sentiment
        if transcript:
            s_res = sa.analyze(transcript)
            pred_sentiment = s_res.get("sentiment")
            s_conf = float(s_res.get("confidence", 0.0))
            f_res = fusion.fuse(v_res, s_res)
        else:
            s_res = {"sentiment": None, "confidence": 0.0, "probabilities": {}}
            pred_sentiment = None
            s_conf = 0.0
            f_res = {
                "combined_confidence": v_conf,
                "agreement": False,
                "mismatch": False,
                "audio_direction": "neutral",
                "text_direction": "neutral",
            }

        # 4. Conversation analysis (if usable segments exist)
        conv_res: Optional[Dict[str, Any]] = None
        pred_speaker_count = 1  # Default baseline for single audio clip
        if ca is not None and len(segments) > 0:
            try:
                conv_res = ca.analyze(filepath)
                # Count distinct speakers if diarization identified them
                unique_speakers = {
                    s.get("speaker")
                    for s in conv_res.get("segments", [])
                    if s.get("speaker") and s.get("speaker") != "Unknown"
                }
                if unique_speakers:
                    pred_speaker_count = len(unique_speakers)
            except Exception as e:
                logger.warning(f"Conversation analysis skipped for {filepath}: {e}")

        return {
            "status": "EVALUATED",
            "filename": filepath,
            "pred_voice_emotion": pred_voice,
            "voice_confidence": v_conf,
            "pred_text_sentiment": pred_sentiment,
            "text_confidence": s_conf,
            "pred_transcript": transcript,
            "pred_speaker_count": pred_speaker_count,
            "audio_result": v_res,
            "text_result": s_res,
            "fusion_result": f_res,
            "conversation_result": conv_res,
            "extra": {
                "segment_count": len(segments),
                "conversation_analyzed": conv_res is not None,
            },
        }
    except Exception as e:
        logger.exception(f"Inference failed for {filepath}: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "filename": filepath,
        }


def execute_evaluation(
    dataset_path: str = DEFAULT_DATASET_PATH,
    results_dir: str = DEFAULT_RESULTS_DIR,
) -> Dict[str, Any]:
    """
    Complete automated evaluation workflow.
    """
    logger.info(f"Loading evaluation dataset from: {dataset_path}")
    dataset = load_evaluation_dataset(dataset_path)
    logger.info(f"Loaded {len(dataset)} evaluation records.")

    # Initialize production models
    logger.info("Initializing production pipeline models for evaluation...")
    vea = VoiceEmotionAnalyzer()
    stt = SpeechToText()
    sa = SentimentAnalyzer()
    fusion = MultimodalFusion()
    ca = ConversationAnalyzer(
        emotion_analyzer=vea,
        speech_to_text=stt,
        sentiment_analyzer=sa,
        fusion=fusion,
    )

    sample_predictions: List[SamplePrediction] = []
    ablation_sample_data: List[Dict[str, Any]] = []
    evaluated_samples: List[Dict[str, Any]] = []
    failed_samples: List[Dict[str, Any]] = []
    skipped_samples: List[Dict[str, Any]] = []

    temporal_rising_tone_record: Optional[Dict[str, Any]] = None

    for idx, sample in enumerate(dataset):
        logger.info(f"[{idx+1}/{len(dataset)}] Evaluating sample: {sample.filename}")
        res = run_pipeline_on_sample(sample, vea, stt, sa, fusion, ca)

        if res["status"] == "FAILED":
            failed_samples.append(res)
            continue

        evaluated_samples.append(res)

        pred = SamplePrediction(
            filename=sample.filename,
            pred_voice_emotion=res.get("pred_voice_emotion"),
            pred_text_sentiment=res.get("pred_text_sentiment"),
            pred_transcript=res.get("pred_transcript"),
            pred_speaker_count=res.get("pred_speaker_count"),
            extra=res,
        )
        sample_predictions.append(pred)

        # For ablation study
        ablation_sample_data.append({
            "filename": sample.filename,
            "audio_result": res.get("audio_result", {}),
            "text_result": res.get("text_result", {}),
            "ground_truth_direction": sample.true_text_sentiment,
            "extra": res.get("extra", {}),
        })

        # Check rising-tone sample for temporal experiment
        if "rising_tone.wav" in sample.filename:
            temporal_rising_tone_record = res.get("conversation_result")

    # Run Quantitative Metrics Evaluation Harness
    logger.info("Computing metrics via EvaluationHarness...")
    harness = EvaluationHarness()
    eval_result: EvaluationResult = harness.evaluate_predictions(dataset, sample_predictions)

    # Run Learned Fusion Baseline
    logger.info("Evaluating Learned Fusion Baseline (Logistic Regression)...")
    learned_fusion_baseline = LearnedFusionBaseline(random_state=42)
    true_directions = [item.get("ground_truth_direction") for item in ablation_sample_data]
    learned_fusion_result: LearnedFusionResult = learned_fusion_baseline.train_and_evaluate(
        ablation_sample_data, true_directions
    )

    # Run 4-way Ablation Experiment
    logger.info("Running 4-way Ablation Experiment...")
    ablation_evaluator = AblationEvaluator()
    ablation_result: AblationResult = ablation_evaluator.evaluate(
        ablation_sample_data, dataset=dataset, learned_fusion_result=learned_fusion_result
    )

    # Run Diarization Validation
    logger.info("Running Diarization Validation Procedure...")
    diarization_res: DiarizationValidationResult = validate_diarization_cases()

    # Process Temporal Rising-Tone Experiment Results
    rising_tone_experiment: Dict[str, Any] = {}
    if temporal_rising_tone_record:
        segments = temporal_rising_tone_record.get("segments", [])
        trend = temporal_rising_tone_record.get("emotional_trend", {})
        escalation = temporal_rising_tone_record.get("escalation", {})
        ct = temporal_rising_tone_record.get("consistency_trajectory", {})
        att = temporal_rising_tone_record.get("attention_score", {})

        rising_tone_experiment = {
            "status": "COMPLETED",
            "filename": "sample_audio/rising_tone.wav",
            "segment_count": len(segments),
            "segment_intensities": [s.get("intensity") for s in segments],
            "segment_details": [
                {
                    "segment_index": s.get("segment_index"),
                    "start_seconds": s.get("start_seconds"),
                    "end_seconds": s.get("end_seconds"),
                    "voice_emotion": (s.get("voice_emotion") or {}).get("emotion"),
                    "text_sentiment": (s.get("text_sentiment") or {}).get("sentiment"),
                    "intensity": s.get("intensity"),
                }
                for s in segments
            ],
            "emotional_trend": trend,
            "escalation": escalation,
            "consistency_trajectory": ct,
            "attention_score": att,
            "empirical_finding": (
                f"Rising-tone recording yielded {len(segments)} segments. "
                f"Intensities: {[s.get('intensity') for s in segments]}. "
                f"Overall trend was '{trend.get('trend')}' (change={trend.get('change'):+.4f}). "
                f"Escalation detected={escalation.get('escalation_detected')}: {escalation.get('reason')}"
            ),
        }
    else:
        rising_tone_experiment = {
            "status": "NOT_EVALUATED",
            "reason": "Escalation was not evaluated/detected because the minimum usable-segment requirement was not satisfied.",
        }

    # Summary dictionary
    summary_data: Dict[str, Any] = {
        "metadata": {
            "dataset_path": dataset_path,
            "total_dataset_records": len(dataset),
            "evaluated_samples_count": len(evaluated_samples),
            "failed_samples_count": len(failed_samples),
            "skipped_samples_count": len(skipped_samples),
            "has_ground_truth": eval_result.has_ground_truth,
        },
        "metrics": eval_result.to_dict(),
        "ablation": ablation_result.to_dict(),
        "learned_fusion": learned_fusion_result.to_dict(),
        "diarization_validation": diarization_res.to_dict(),
        "temporal_rising_tone_experiment": rising_tone_experiment,
        "disclaimer": (
            "This system performs computational emotion inference from acoustic and linguistic patterns. "
            "It does not diagnose, assess, or make claims about a person's actual psychological or emotional state. "
            "Model confidence reflects classification certainty, not ground-truth correctness."
        ),
    }

    # Missing annotations breakdown
    missing_annotations = []
    if not dataset.has_voice_emotion_labels:
        missing_annotations.append("Voice Emotion ground truth (true_voice_emotion)")
    if not dataset.has_text_sentiment_labels:
        missing_annotations.append("Text Sentiment ground truth (true_text_sentiment)")
    if not dataset.has_transcript_labels:
        missing_annotations.append("Reference Transcripts (true_transcript for WER)")
    if not dataset.has_speaker_count_labels:
        missing_annotations.append("Speaker Count ground truth (true_speaker_count for MAE)")

    # Latest evaluation dictionary matching Task 1 format
    latest_eval_data: Dict[str, Any] = {
        "metadata": {
            "dataset_path": dataset_path,
            "total_dataset_records": len(dataset),
            "evaluated_samples_count": len(evaluated_samples),
            "failed_samples_count": len(failed_samples),
            "skipped_samples_count": len(skipped_samples),
            "has_ground_truth": eval_result.has_ground_truth,
            "missing_annotations": missing_annotations,
        },
        "voice_emotion": {
            "status": eval_result.voice_emotion_metrics.get("status"),
            "evaluated_samples": eval_result.voice_emotion_metrics.get("total_samples", 0),
            "accuracy": eval_result.voice_emotion_metrics.get("accuracy"),
            "f1_macro": eval_result.voice_emotion_metrics.get("f1_macro"),
            "precision_macro": eval_result.voice_emotion_metrics.get("precision_macro"),
            "recall_macro": eval_result.voice_emotion_metrics.get("recall_macro"),
            "confusion_matrix": eval_result.voice_emotion_metrics.get("confusion_matrix"),
        },
        "text_sentiment": {
            "status": eval_result.text_sentiment_metrics.get("status"),
            "evaluated_samples": eval_result.text_sentiment_metrics.get("total_samples", 0),
            "accuracy": eval_result.text_sentiment_metrics.get("accuracy"),
            "f1_macro": eval_result.text_sentiment_metrics.get("f1_macro"),
            "precision_macro": eval_result.text_sentiment_metrics.get("precision_macro"),
            "recall_macro": eval_result.text_sentiment_metrics.get("recall_macro"),
            "confusion_matrix": eval_result.text_sentiment_metrics.get("confusion_matrix"),
        },
        "rule_based_fusion": {
            "status": "OK" if ablation_result.voice_text_fusion.accuracy is not None else "PENDING_DATA",
            "evaluated_samples": ablation_result.voice_text_fusion.total_samples,
            "accuracy": ablation_result.voice_text_fusion.accuracy,
            "f1_macro": ablation_result.voice_text_fusion.f1_score,
            "mean_confidence": ablation_result.voice_text_fusion.mean_confidence,
        },
        "learned_fusion": learned_fusion_result.to_dict(),
        "speech_to_text": eval_result.speech_to_text_metrics,
        "speaker_count": eval_result.speaker_count_metrics,
        "ablation_comparison_table": ablation_result.format_comparison_table(),
        "disclaimer": summary_data["disclaimer"],
    }

    # Export machine-readable results
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)
    logger.info(f"Saved evaluation summary JSON to: {json_path}")

    # Export latest_evaluation.json
    latest_json_path = os.path.join(results_dir, "latest_evaluation.json")
    with open(latest_json_path, "w", encoding="utf-8") as f:
        json.dump(latest_eval_data, f, indent=2, default=str)
    logger.info(f"Saved latest evaluation JSON to: {latest_json_path}")

    # Export latest_evaluation.txt
    txt_report = format_human_readable_report(
        dataset=dataset,
        eval_result=eval_result,
        ablation_result=ablation_result,
        learned_fusion_result=learned_fusion_result,
        missing_annotations=missing_annotations,
        total_evaluated=len(evaluated_samples),
        total_failed=len(failed_samples),
    )
    latest_txt_path = os.path.join(results_dir, "latest_evaluation.txt")
    with open(latest_txt_path, "w", encoding="utf-8") as f:
        f.write(txt_report)
    logger.info(f"Saved latest evaluation report TXT to: {latest_txt_path}")

    # Export evaluation predictions CSV
    csv_preds_path = os.path.join(results_dir, "evaluation_predictions.csv")
    with open(csv_preds_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "filename",
            "true_voice_emotion",
            "pred_voice_emotion",
            "voice_match",
            "true_text_sentiment",
            "pred_text_sentiment",
            "sentiment_match",
            "true_speaker_count",
            "pred_speaker_count",
            "speaker_count_match",
            "true_transcript",
            "pred_transcript",
            "wer",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in eval_result.sample_records:
            t_v = rec.get("true_voice_emotion")
            p_v = rec.get("pred_voice_emotion")
            v_match = (t_v == p_v) if t_v and p_v else "N/A"

            t_s = rec.get("true_text_sentiment")
            p_s = rec.get("pred_text_sentiment")
            s_match = (t_s == p_s) if t_s and p_s else "N/A"

            t_spk = rec.get("true_speaker_count")
            p_spk = rec.get("pred_speaker_count")
            spk_match = (t_spk == p_spk) if t_spk is not None and p_spk is not None else "N/A"

            writer.writerow({
                "filename": rec.get("filename"),
                "true_voice_emotion": t_v or "N/A",
                "pred_voice_emotion": p_v or "N/A",
                "voice_match": v_match,
                "true_text_sentiment": t_s or "N/A",
                "pred_text_sentiment": p_s or "N/A",
                "sentiment_match": s_match,
                "true_speaker_count": t_spk if t_spk is not None else "N/A",
                "pred_speaker_count": p_spk if p_spk is not None else "N/A",
                "speaker_count_match": spk_match,
                "true_transcript": rec.get("true_transcript") or "N/A",
                "pred_transcript": rec.get("pred_transcript") or "N/A",
                "wer": rec.get("wer") if rec.get("wer") is not None else "N/A",
                "notes": rec.get("notes") or "",
            })
    logger.info(f"Saved evaluation predictions CSV to: {csv_preds_path}")

    # Export ablation results CSV
    csv_ablation_path = os.path.join(results_dir, "ablation_results.csv")
    with open(csv_ablation_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "configuration",
            "modality_scope",
            "available_samples",
            "applicable_metrics",
            "predictions_summary",
            "capabilities",
            "limitations",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if ablation_result.four_way_comparison:
            for item in ablation_result.four_way_comparison:
                writer.writerow({
                    "configuration": item.get("configuration"),
                    "modality_scope": item.get("modality_scope"),
                    "available_samples": item.get("available_samples"),
                    "applicable_metrics": "; ".join(item.get("applicable_metrics", [])),
                    "predictions_summary": item.get("predictions_summary"),
                    "capabilities": item.get("capabilities"),
                    "limitations": item.get("limitations"),
                })
    logger.info(f"Saved ablation results CSV to: {csv_ablation_path}")

    # Export human-readable Markdown summary
    summary_md_path = os.path.join(results_dir, "experiment_summary.md")
    with open(summary_md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 — Evaluation, Validation & Experimental Evidence\n\n")
        f.write(f"**Total Samples Evaluated**: {len(evaluated_samples)}\n")
        f.write(f"**Failed Samples**: {len(failed_samples)}\n\n")

        f.write("## 1. Quantitative Evaluation Metrics\n\n")
        f.write(f"```\n{eval_result.format_summary()}\n```\n\n")

        f.write("## 2. 4-Way Modality Ablation Study\n\n")
        f.write(f"{ablation_result.format_table()}\n\n")
        f.write("### Configuration Comparison (A, B, C, D)\n\n")
        for item in ablation_result.four_way_comparison or []:
            f.write(f"#### {item['configuration']}\n")
            f.write(f"- **Modality Scope**: {item['modality_scope']}\n")
            f.write(f"- **Available Samples**: {item['available_samples']}\n")
            f.write(f"- **Predictions Summary**: {item['predictions_summary']}\n")
            f.write(f"- **Capabilities**: {item['capabilities']}\n")
            f.write(f"- **Limitations**: {item['limitations']}\n\n")

        f.write("## 3. Temporal / Rising-Tone Experiment\n\n")
        f.write(f"- **Target File**: `{rising_tone_experiment.get('filename')}`\n")
        f.write(f"- **Finding**: {rising_tone_experiment.get('empirical_finding')}\n\n")

        f.write("## 4. Speaker Diarization Validation\n\n")
        f.write(f"- **Status**: {diarization_res.honest_assessment}\n")
        f.write(f"- **Total Cases**: {diarization_res.total_cases}, **Matched Cases**: {diarization_res.matched_cases}\n")
        f.write("### Observed Limitations:\n")
        for lim in diarization_res.observed_limitations:
            f.write(f"- {lim}\n")
        f.write("\n")

        f.write("## 5. Mandatory Disclaimer\n\n")
        f.write(f"> {summary_data['disclaimer']}\n")

    logger.info(f"Saved experiment summary Markdown to: {summary_md_path}")

    # Also print the summary to stdout
    print("\n" + eval_result.format_summary())
    print("\n" + ablation_result.format_table())
    print(f"\nDiarization Assessment: {diarization_res.honest_assessment}")

    return summary_data


def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Automated Evaluation")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Path to evaluation dataset CSV")
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR, help="Path to output results directory")
    args = parser.parse_args()

    execute_evaluation(dataset_path=args.dataset, results_dir=args.output_dir)


if __name__ == "__main__":
    main()
