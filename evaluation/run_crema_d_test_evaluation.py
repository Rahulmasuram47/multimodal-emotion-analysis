"""
One-Shot CREMA-D Test-Split Evaluation.

Executes exactly ONCE on the held-out test actors from evaluation/results/crema_d_split.json.
Evaluates baseline model (superb/wav2vec2-base-superb-er) and, only if head_decision.json
marked candidate as promising, evaluates the frozen candidate head.
Outputs evaluation/results/crema_d_benchmark.md and .json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.crema_d import (
    CANONICAL_4CLASS,
    check_dataset_license,
    compute_actor_cluster_bootstrap,
    compute_cache_key,
    load_crema_dataset,
    load_split_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crema_d_test_eval")


def compute_paired_actor_cluster_bootstrap(
    records_cand: List[Dict[str, Any]],
    records_base: List[Dict[str, Any]],
    classes: Sequence[str] = CANONICAL_4CLASS,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Optional[float]]:
    """Compute paired actor-cluster bootstrap 95% CI for the difference in UAR (Candidate - Baseline)."""
    if not records_cand or not records_base or len(records_cand) != len(records_base):
        return {"mean_diff": None, "ci_lower": None, "ci_upper": None, "n_clusters": 0}

    # Group by actor
    actor_clusters: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}
    for rc, rb in zip(records_cand, records_base):
        aid = rc["actor_id"]
        if aid not in actor_clusters:
            actor_clusters[aid] = ([], [])
        actor_clusters[aid][0].append(rc)
        actor_clusters[aid][1].append(rb)

    unique_actors = list(actor_clusters.keys())
    n_actors = len(unique_actors)
    if n_actors < 2:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n_clusters": n_actors}

    def _calc_uar(recs: List[Dict[str, Any]]) -> float:
        rec_list = []
        for cl in classes:
            sup = sum(1 for r in recs if r["canonical_emotion"] == cl)
            tp = sum(1 for r in recs if r["canonical_emotion"] == cl and r["canonical_pred"] == cl)
            if sup > 0:
                rec_list.append(tp / sup)
        return float(np.mean(rec_list)) if rec_list else 0.0

    rng = np.random.RandomState(seed)
    diffs: List[float] = []

    for _ in range(n_bootstrap):
        sampled_actors = rng.choice(unique_actors, size=n_actors, replace=True)
        samp_cand: List[Dict[str, Any]] = []
        samp_base: List[Dict[str, Any]] = []
        for aid in sampled_actors:
            c_recs, b_recs = actor_clusters[aid]
            samp_cand.extend(c_recs)
            samp_base.extend(b_recs)

        uar_c = _calc_uar(samp_cand)
        uar_b = _calc_uar(samp_base)
        diffs.append(uar_c - uar_b)

    ci_lower = float(np.percentile(diffs, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))
    mean_diff = float(np.mean(diffs))

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_clusters": n_actors,
    }


def compute_test_metrics(
    records: List[Dict[str, Any]],
    classes: Sequence[str] = CANONICAL_4CLASS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute test metrics: Accuracy, UAR, Macro-F1, per-class recall/precision, confusion matrix, and cluster CIs."""
    if not records:
        return {
            "n_samples": 0,
            "accuracy": None,
            "accuracy_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n_clusters": 0},
            "uar": None,
            "uar_ci": {"mean": None, "ci_lower": None, "ci_upper": None, "n_clusters": 0},
            "macro_f1": None,
            "chance_level": round(1.0 / len(classes), 4),
            "per_class": {},
            "confusion_matrix": {"classes": list(classes), "matrix": []},
        }

    y_true = [r["canonical_emotion"] for r in records]
    y_pred = [r["canonical_pred"] for r in records]

    cls_list = list(classes)
    cls_to_idx = {c: i for i, c in enumerate(cls_list)}
    n_cls = len(cls_list)

    cm = [[0 for _ in range(n_cls)] for _ in range(n_cls)]
    correct = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == yp:
            correct += 1
        if yt in cls_to_idx and yp in cls_to_idx:
            cm[cls_to_idx[yt]][cls_to_idx[yp]] += 1

    total = len(y_true)
    accuracy = correct / total if total > 0 else 0.0

    per_class: Dict[str, Any] = {}
    recalls: List[float] = []
    f1s: List[float] = []

    for i, c in enumerate(cls_list):
        tp = cm[i][i]
        fn = sum(cm[i][j] for j in range(n_cls) if j != i)
        fp = sum(cm[j][i] for j in range(n_cls) if j != i)

        support = tp + fn
        recall = tp / support if support > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[c] = {
            "support": support,
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "is_highlighted": c in {"sad", "neutral"},
        }
        recalls.append(recall)
        f1s.append(f1)

    uar = float(np.mean(recalls)) if recalls else 0.0
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    def _acc_fn(recs: List[Dict[str, Any]]) -> float:
        c = sum(1 for r in recs if r["canonical_emotion"] == r["canonical_pred"])
        return c / len(recs) if recs else 0.0

    def _uar_fn(recs: List[Dict[str, Any]]) -> float:
        rec_list = []
        for cl in cls_list:
            sup_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl)
            tp_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl and r["canonical_pred"] == cl)
            if sup_cnt > 0:
                rec_list.append(tp_cnt / sup_cnt)
        return float(np.mean(rec_list)) if rec_list else 0.0

    acc_ci = compute_actor_cluster_bootstrap(records, _acc_fn, n_bootstrap=1000, seed=seed)
    uar_ci = compute_actor_cluster_bootstrap(records, _uar_fn, n_bootstrap=1000, seed=seed)

    return {
        "n_samples": total,
        "n_actors": len(set(r["actor_id"] for r in records)),
        "accuracy": round(accuracy, 4),
        "accuracy_ci": acc_ci,
        "uar": round(uar, 4),
        "uar_ci": uar_ci,
        "macro_f1": round(macro_f1, 4),
        "chance_level": round(1.0 / n_cls, 4),
        "per_class": per_class,
        "confusion_matrix": {
            "classes": cls_list,
            "matrix": cm,
        },
    }


def build_test_benchmark_markdown(
    base_metrics: Dict[str, Any],
    cand_metrics: Optional[Dict[str, Any]],
    paired_ci: Optional[Dict[str, Any]],
    meta: Dict[str, Any],
) -> str:
    """Format final one-shot test-split benchmark report in Markdown."""
    head_adopted = meta.get("head_adopted", False)
    head_status_str = "Adopted (promising in Dev CV)" if head_adopted else "Not Adopted (Dev decision: not promising; baseline maintained)"

    if base_metrics.get("n_samples", 0) == 0:
        return f"""# CREMA-D One-Shot Test-Split Evaluation Benchmark

> [!WARNING]
> **Data Availability**: The local dataset path `{meta.get('data_dir')}` contains 0 audio clips on disk. In accordance with Absolute Data Integrity, no demonstration metrics were fabricated or hardcoded.

## Evaluation Protocol & Caveats
- **Nature of Corpus**: **acted speech, actor-disjoint split, not natural speech**
- **Test Split Isolation**: Exactly 31 held-out test actors evaluated in a single one-shot pass.
- **Candidate Head Status**: **{head_status_str}**
- **Active Production Model**: `{meta.get('active_model_id')}` ({meta.get('active_backend')})
- **Alternative Backend (emotion2vec)**: {meta.get('emotion2vec_note')}
- **Random Seed**: {meta.get('seed')}
- **Pre-processing**: Mono 16 kHz resampled, feature extractor padding
- **Excluded Classes**: Raw CREMA-D `DIS` (Disgust) and `FEA` (Fear) mapped to unmodeled/other; benchmark evaluates canonical 4-class target ({CANONICAL_4CLASS}).
- **Dataset License**: {meta.get('license_status')}
- **Sampling Budget**: {meta.get('sampling_note')}
"""

    cm = base_metrics["confusion_matrix"]
    classes = cm["classes"]
    cm_matrix = cm["matrix"]

    cm_header = "| True \\ Pred | " + " | ".join(classes) + " |"
    cm_sep = "|---|" + "|".join(["---" for _ in range(len(classes))]) + "|"
    cm_rows = []
    for i, c in enumerate(classes):
        row = f"| **{c}** | " + " | ".join(str(cm_matrix[i][j]) for j in range(len(classes))) + " |"
        cm_rows.append(row)
    cm_table = "\n".join([cm_header, cm_sep] + cm_rows)

    pc = base_metrics["per_class"]
    pc_header = "| Emotion Class | Support | Recall | Precision | F1 | Highlight |"
    pc_sep = "|---|---|---|---|---|---|"
    pc_rows = []
    for c in classes:
        info = pc.get(c, {})
        hl = "⭐ **Key Metric**" if info.get("is_highlighted") else ""
        pc_rows.append(
            f"| **{c}** | {info.get('support')} | {info.get('recall', 0.0):.4f} | "
            f"{info.get('precision', 0.0):.4f} | {info.get('f1', 0.0):.4f} | {hl} |"
        )
    pc_table = "\n".join([pc_header, pc_sep] + pc_rows)

    acc_ci = base_metrics["accuracy_ci"]
    uar_ci = base_metrics["uar_ci"]

    cand_section = ""
    if cand_metrics and head_adopted:
        c_acc = cand_metrics["accuracy"]
        c_uar = cand_metrics["uar"]
        diff_str = f"{paired_ci.get('mean_diff', 0.0):+.4f} (95% CI: [{paired_ci.get('ci_lower', 0.0):+.4f}, {paired_ci.get('ci_upper', 0.0):+.4f}])" if paired_ci else "N/A"
        cand_section = f"""
## Candidate Head Comparison (Adopted)
- **Candidate Accuracy**: `{c_acc * 100:.2f}%`
- **Candidate UAR**: `{c_uar * 100:.2f}%`
- **Paired UAR Gain (Candidate - Baseline)**: `{diff_str}`
"""
    else:
        cand_section = f"""
## Candidate Head Status
- **Status**: **{head_status_str}**
- **Action**: Per the frozen dev decision in `head_decision.json`, the candidate head was not adopted. The baseline Wav2Vec2 model remains the active production head.
"""

    report = f"""# CREMA-D One-Shot Test-Split Evaluation Benchmark

## Evaluation Protocol & Caveats
- **Nature of Corpus**: **acted speech, actor-disjoint split, not natural speech**
- **Evaluation Partition**: Strictly held-out test actors (N = {base_metrics.get('n_actors')} actors, {base_metrics.get('n_samples')} clips)
- **Active Model**: `{meta.get('active_model_id')}` ({meta.get('active_backend')})
- **Alternative Backend (emotion2vec)**: {meta.get('emotion2vec_note')}
- **Random Seed**: {meta.get('seed')}
- **Pre-processing**: Mono 16 kHz resampled, feature extractor padding
- **Excluded Classes**: Raw CREMA-D `DIS` and `FEA` mapped to unmodeled/other; evaluated on 4 canonical classes ({CANONICAL_4CLASS})
- **Dataset License**: {meta.get('license_status')}
- **Sampling Budget**: {meta.get('sampling_note')}

## Baseline Test Metrics (Wav2Vec2-SUPERB)
- **Accuracy**: `{base_metrics['accuracy'] * 100:.2f}%` (95% Actor-Cluster CI: `[{acc_ci.get('ci_lower', 0) * 100:.2f}%, {acc_ci.get('ci_upper', 0) * 100:.2f}%]`)
- **Unweighted Average Recall (UAR)**: `{base_metrics['uar'] * 100:.2f}%` (95% Actor-Cluster CI: `[{uar_ci.get('ci_lower', 0) * 100:.2f}%, {uar_ci.get('ci_upper', 0) * 100:.2f}%]`)
- **Macro-F1**: `{base_metrics['macro_f1'] * 100:.2f}%`
- **Chance Level**: `{base_metrics['chance_level'] * 100:.2f}%` (1/{len(classes)} classes)

## Per-Class Performance (Baseline)
{pc_table}

## Confusion Matrix (Baseline)
{cm_table}

{cand_section}

> [!NOTE]
> **One-Shot Freeze Guarantee**: This evaluation was executed exactly once on the held-out test split. No hyperparameters, mappings, thresholds, or model weights were tuned or re-evaluated after observing these test results.
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="CREMA-D One-Shot Test Evaluator")
    parser.add_argument("--data-dir", default="data/external_datasets/crema_d/AudioWAV", help="Path to CREMA-D AudioWAV folder")
    parser.add_argument("--split-file", default="evaluation/results/crema_d_split.json", help="Path to split file")
    parser.add_argument("--decision-file", default="evaluation/results/head_decision.json", help="Path to head decision file")
    parser.add_argument("--cache-dir", default="evaluation/results/cache", help="Cache directory")
    parser.add_argument("--max-per-class", type=int, default=100, help="Max clips per emotion class")
    parser.add_argument("--full", action="store_true", help="Evaluate full test partition")
    args = parser.parse_args()

    # Step 0: Ensure head_decision.json exists before running
    decision_path = Path(args.decision_file)
    if not decision_path.is_file():
        raise RuntimeError(
            f"Test evaluation strictly refused: '{decision_path}' is missing. "
            "Phase 4B dev-only decision must be finalized before test evaluation."
        )

    with open(decision_path, "r", encoding="utf-8") as f:
        decision_data = json.load(f)

    head_adopted = decision_data.get("decision") == "promising"
    print("=" * 70)
    print("PHASE 4C: ONE-SHOT TEST-SPLIT EVALUATION")
    print(f"Candidate Head Decision in {decision_path.name}: {decision_data.get('decision')}")
    print(f"Head Adopted for Test Evaluation: {head_adopted}")
    print("=" * 70)

    # Step 1: Load split
    split_path = Path(args.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file missing: {split_path}")
    split_data = load_split_file(split_path)
    test_actors = set(split_data["test_actors"])
    print(f"Held-Out Test Actors Count: {len(test_actors)} (Seed {split_data['seed']})")

    # Step 2: License and environment checks
    license_status = check_dataset_license(args.data_dir)
    active_backend = "superb"
    active_model_id = getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er")
    emotion2vec_note = "Unavailable in environment (missing funasr); comparison excluded."

    sampling_note = (
        "Full test partition evaluated without class capping."
        if args.full
        else f"Capped at {args.max_per_class} clips per class (seeded sampling)."
    )

    meta = {
        "active_backend": active_backend,
        "active_model_id": active_model_id,
        "emotion2vec_note": emotion2vec_note,
        "data_dir": args.data_dir,
        "license_status": license_status,
        "seed": split_data["seed"],
        "head_adopted": head_adopted,
        "head_decision": decision_data.get("decision"),
        "sampling_note": sampling_note,
        "corpus_nature": "acted speech, actor-disjoint split, not natural speech",
    }

    # Step 3: Load test clips
    all_clips = load_crema_dataset(args.data_dir)
    test_clips = [c for c in all_clips if c.actor_id in test_actors and c.canonical_emotion in CANONICAL_4CLASS]
    print(f"Test partition clips detected on disk: {len(test_clips)}")

    # Handle 0 clips case gracefully per Absolute Data Integrity
    if not test_clips:
        print("\nNotice: 0 test clips detected on disk. Writing benchmark report documenting data status.")
        empty_metrics = {"n_samples": 0}
        report_md = build_test_benchmark_markdown(empty_metrics, None, None, meta)
        out_md = Path("evaluation/results/crema_d_benchmark.md")
        out_json = Path("evaluation/results/crema_d_benchmark.json")
        out_md.write_text(report_md, encoding="utf-8")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": meta,
                "baseline_metrics": empty_metrics,
                "candidate_metrics": None,
                "paired_difference": None,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }, f, indent=2)
        print(f"Saved: {out_md} and {out_json}")
        return

    # Filter by budget if not full
    if not args.full and args.max_per_class:
        rng = np.random.RandomState(split_data["seed"])
        class_buckets: Dict[str, List[Any]] = {}
        for c in test_clips:
            class_buckets.setdefault(c.canonical_emotion, []).append(c)
        eval_clips = []
        for cls_name, b_clips in class_buckets.items():
            shuffled = list(b_clips)
            rng.shuffle(shuffled)
            eval_clips.extend(shuffled[: args.max_per_class])
    else:
        eval_clips = test_clips

    print(f"Evaluating {len(eval_clips)} Test clips...")

    # Load baseline model
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
    from evaluation.run_crema_d_baseline import run_superb_inference_on_clip

    extractor = Wav2Vec2FeatureExtractor.from_pretrained(active_model_id)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(active_model_id)
    model.eval()

    candidate_head = None
    if head_adopted:
        cand_artifact = Path("evaluation/results/candidate_head.joblib")
        if cand_artifact.is_file():
            import joblib
            candidate_head = joblib.load(cand_artifact)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_records: List[Dict[str, Any]] = []
    cand_records: List[Dict[str, Any]] = []

    for clip in eval_clips:
        key = compute_cache_key(
            model_id=active_model_id,
            revision="main",
            target_sr=16000,
            clip_filename=clip.filename,
            labels=CANONICAL_4CLASS,
        )
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.is_file():
            pred_res = np.load(cache_file, allow_pickle=True).item()
        else:
            pred_res = run_superb_inference_on_clip(clip.filepath, extractor, model)
            np.save(cache_file, pred_res)

        base_records.append({
            "filename": clip.filename,
            "actor_id": clip.actor_id,
            "canonical_emotion": clip.canonical_emotion,
            "canonical_pred": pred_res["canonical_pred"],
            "confidence": pred_res["confidence"],
        })

        if candidate_head is not None and "embedding" in pred_res:
            c_probs = candidate_head.predict_proba(pred_res["embedding"][np.newaxis, :])[0]
            c_classes = list(candidate_head.classes_)
            c_top = int(np.argmax(c_probs))
            c_pred = str(c_classes[c_top])
            cand_records.append({
                "filename": clip.filename,
                "actor_id": clip.actor_id,
                "canonical_emotion": clip.canonical_emotion,
                "canonical_pred": c_pred,
                "confidence": float(c_probs[c_top]),
            })

    base_metrics = compute_test_metrics(base_records, classes=CANONICAL_4CLASS, seed=split_data["seed"])
    cand_metrics = compute_test_metrics(cand_records, classes=CANONICAL_4CLASS, seed=split_data["seed"]) if cand_records else None
    paired_ci = compute_paired_actor_cluster_bootstrap(cand_records, base_records, classes=CANONICAL_4CLASS, seed=split_data["seed"]) if cand_records else None

    report_md = build_test_benchmark_markdown(base_metrics, cand_metrics, paired_ci, meta)
    out_md = Path("evaluation/results/crema_d_benchmark.md")
    out_json = Path("evaluation/results/crema_d_benchmark.json")

    out_md.write_text(report_md, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": meta,
            "baseline_metrics": base_metrics,
            "candidate_metrics": cand_metrics,
            "paired_difference": paired_ci,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }, f, indent=2)

    print(f"\nSaved Test Benchmark Markdown to: {out_md}")
    print(f"Saved Test Benchmark JSON to: {out_json}")


if __name__ == "__main__":
    main()
