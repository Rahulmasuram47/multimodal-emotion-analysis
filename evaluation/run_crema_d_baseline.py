"""
CREMA-D Dev Baseline Evaluation Runner.

Evaluates the active voice emotion model on the actor-independent DEV split of CREMA-D.
Enforces that test-split metrics cannot be computed or displayed until Phase 4B head decision.
Caches logits and embeddings into .npy cache files with stable hash keys.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.crema_d import (
    CANONICAL_4CLASS,
    CREMA_RAW_EMOTIONS,
    CREMA_TO_SUPERB_MAP,
    check_dataset_license,
    compute_actor_cluster_bootstrap,
    compute_cache_key,
    enforce_test_evaluation_lock,
    generate_actor_split,
    load_crema_dataset,
    load_split_file,
    save_split_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("crema_d_runner")


def inspect_voice_backends() -> Dict[str, Any]:
    """Inspect active and alternative voice backends in the current environment."""
    configured_backend = getattr(config, "VOICE_BACKEND", "superb")
    superb_id = getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er")
    emotion2vec_id = getattr(config, "EMOTION2VEC_MODEL_NAME", "emotion2vec/emotion2vec_plus_base")

    superb_status: Dict[str, Any] = {"available": False, "model_id": superb_id, "exposes_logits": False, "exposes_embeddings": False}
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        superb_status["available"] = True
        superb_status["exposes_logits"] = True
        superb_status["exposes_embeddings"] = True
    except Exception as e:
        superb_status["error"] = str(e)

    e2v_status: Dict[str, Any] = {"available": False, "model_id": emotion2vec_id, "exposes_logits": False, "exposes_embeddings": False}
    try:
        import funasr
        e2v_status["available"] = True
        e2v_status["exposes_logits"] = False  # produces classification scores via generate()
        e2v_status["exposes_embeddings"] = True
    except Exception as e:
        e2v_status["error"] = f"Missing dependency or load failure: {e}"

    # Determine genuinely active backend used by app
    if configured_backend == "emotion2vec" and not e2v_status["available"]:
        active_backend = "superb"
        active_reason = "Configured for emotion2vec, but funasr is missing; VoiceEmotionAnalyzer fell back to SUPERB."
    elif configured_backend == "emotion2vec" and e2v_status["available"]:
        active_backend = "emotion2vec"
        active_reason = "Configured for emotion2vec and funasr is loadable."
    else:
        active_backend = "superb"
        active_reason = "Configured for superb."

    active_model_id = superb_id if active_backend == "superb" else emotion2vec_id

    return {
        "configured_backend": configured_backend,
        "active_backend": active_backend,
        "active_reason": active_reason,
        "active_model_id": active_model_id,
        "superb": superb_status,
        "emotion2vec": e2v_status,
    }


def run_superb_inference_on_clip(
    clip_path: str,
    extractor: Any,
    model: Any,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """Execute forward pass of SUPERB Wav2Vec2 model, extracting logits and embeddings."""
    import soundfile as sf
    import torch
    import librosa

    audio, sr = sf.read(clip_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    else:
        audio = audio.astype(np.float32)

    inputs = extractor(audio, sampling_rate=target_sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits[0].cpu().numpy()
        # Pool last hidden state as clip embedding
        last_hidden = outputs.hidden_states[-1][0].cpu().numpy()
        embedding = np.mean(last_hidden, axis=0)

    probs = np.exp(logits - np.max(logits))
    probs = probs / np.sum(probs)
    pred_idx = int(np.argmax(probs))
    raw_label = model.config.id2label[pred_idx]
    canonical_pred = CREMA_TO_SUPERB_MAP.get(raw_label.upper(), raw_label.lower())
    if canonical_pred == "neu":
        canonical_pred = "neutral"
    elif canonical_pred == "hap":
        canonical_pred = "happy"
    elif canonical_pred == "ang":
        canonical_pred = "angry"

    return {
        "logits": logits,
        "embedding": embedding,
        "probabilities": probs,
        "pred_idx": pred_idx,
        "raw_label": raw_label,
        "canonical_pred": canonical_pred,
        "confidence": float(probs[pred_idx]),
    }


def run_emotion2vec_inference_on_clip(
    clip_path: str,
    backend: Any,
    target_sr: int = 16000,
) -> Dict[str, Any]:
    """Execute emotion2vec inference on clip, extracting class scores, prediction, and embedding."""
    import soundfile as sf
    import librosa

    audio, sr = sf.read(clip_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=target_sr)
    else:
        audio = audio.astype(np.float32)

    # 1. Run app's own emotion2vec backend predict to get class scores and prediction
    pred_res = backend.predict(audio, sr=target_sr)

    # 2. Extract clip embedding from FunASR model
    raw_gen = backend._model.generate(
        audio.astype(np.float32),
        granularity="utterance",
        extract_embedding=True,
    )
    res_dict = raw_gen[0] if raw_gen else {}
    embedding = res_dict.get("feats")
    if embedding is not None and hasattr(embedding, "cpu"):
        embedding = embedding.cpu().numpy()
    elif embedding is not None:
        embedding = np.asarray(embedding)

    raw_label = pred_res.get("raw_label", "unk")
    canonical_pred = pred_res.get("emotion", "uncertain")

    return {
        "embedding": embedding,
        "scores": pred_res.get("scores", {}),
        "raw_label": raw_label,
        "canonical_pred": canonical_pred,
        "confidence": float(pred_res.get("confidence", 0.0)),
        "raw_confidence": float(pred_res.get("raw_confidence", 0.0)),
    }


def compute_dev_metrics(
    records: List[Dict[str, Any]],
    classes: Sequence[str] = CANONICAL_4CLASS,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute dev-only classification metrics: Accuracy, UAR, Macro-F1, per-class recall, and confusion matrix."""
    if not records:
        return {
            "n_samples": 0,
            "accuracy": None,
            "uar": None,
            "macro_f1": None,
            "per_class": {},
            "confusion_matrix": {},
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

    # Actor-cluster bootstrap for accuracy and UAR
    def _acc_fn(recs: List[Dict[str, Any]]) -> float:
        c = sum(1 for r in recs if r["canonical_emotion"] == r["canonical_pred"])
        return c / len(recs) if recs else 0.0

    def _uar_fn(recs: List[Dict[str, Any]]) -> float:
        rec_list = []
        for cl in cls_list:
            tp_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl and r["canonical_pred"] == cl)
            sup_cnt = sum(1 for r in recs if r["canonical_emotion"] == cl)
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


def build_markdown_report(metrics: Dict[str, Any], meta: Dict[str, Any]) -> str:
    """Format baseline results as structured Markdown report."""
    if metrics.get("n_samples", 0) == 0:
        return (
            "# CREMA-D Dev Baseline Evaluation Report\n\n"
            "> [!WARNING]\n"
            "> No audio clips were found in the dataset directory. Metrics could not be computed.\n\n"
            f"- **Backend Active**: {meta.get('active_backend')} ({meta.get('active_model_id')})\n"
            f"- **Dataset Path**: {meta.get('data_dir')}\n"
            f"- **License Notice**: {meta.get('license_status')}\n"
        )

    cm = metrics["confusion_matrix"]
    classes = cm["classes"]
    cm_matrix = cm["matrix"]

    cm_header = "| True \\ Pred | " + " | ".join(classes) + " |"
    cm_sep = "|---|" + "|".join(["---" for _ in range(len(classes))]) + "|"
    cm_rows = []
    for i, c in enumerate(classes):
        row = f"| **{c}** | " + " | ".join(str(cm_matrix[i][j]) for j in range(len(classes))) + " |"
        cm_rows.append(row)
    cm_table = "\n".join([cm_header, cm_sep] + cm_rows)

    pc = metrics["per_class"]
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

    acc_ci = metrics["accuracy_ci"]
    uar_ci = metrics["uar_ci"]

    report = f"""# CREMA-D Dev-Only Baseline Evaluation Report

## Experiment Metadata
- **Active Voice Model**: `{meta.get('active_model_id')}` ({meta.get('active_backend')})
- **Active Backend Selection**: {meta.get('active_reason')}
- **Alternative Backend (emotion2vec)**: {meta.get('emotion2vec_note')}
- **Evaluation Split**: DEV actors only (N = {metrics.get('n_actors')} actors, {metrics.get('n_samples')} clips)
- **Random Seed**: {meta.get('seed')}
- **Pre-processing**: Mono 16 kHz resampled; feature extractor padding
- **Dataset License**: {meta.get('license_status')}

## Aggregate Metrics
- **Accuracy**: `{metrics['accuracy'] * 100:.2f}%` (95% Actor-Cluster CI: `[{acc_ci.get('ci_lower', 0) * 100:.2f}%, {acc_ci.get('ci_upper', 0) * 100:.2f}%]`)
- **Unweighted Average Recall (UAR)**: `{metrics['uar'] * 100:.2f}%` (95% Actor-Cluster CI: `[{uar_ci.get('ci_lower', 0) * 100:.2f}%, {uar_ci.get('ci_upper', 0) * 100:.2f}%]`)
- **Macro-F1**: `{metrics['macro_f1'] * 100:.2f}%`
- **Chance Level**: `{metrics['chance_level'] * 100:.2f}%` (1/{len(classes)} classes)

## Per-Class Performance
{pc_table}

## Confusion Matrix
{cm_table}

> [!IMPORTANT]
> **Test Split Isolation**: Test actor metrics were strictly excluded from this report. Test evaluation is locked until `evaluation/results/head_decision.json` is generated.
"""
    return report


def main():
    parser = argparse.ArgumentParser(description="CREMA-D Dev Baseline Runner")
    parser.add_argument("--data-dir", default="data/external_datasets/crema_d/AudioWAV", help="Path to CREMA-D AudioWAV folder")
    parser.add_argument("--split-file", default="evaluation/results/crema_d_split.json", help="Path to split file")
    parser.add_argument("--cache-dir", default="evaluation/results/cache", help="Cache directory for logits and embeddings")
    parser.add_argument("--max-per-class", type=int, default=100, help="Max clips per emotion class to evaluate")
    parser.add_argument("--full", action="store_true", help="Run full dev evaluation without class limit")
    parser.add_argument("--eval-test", action="store_true", help="Attempt test split evaluation (locked)")
    args = parser.parse_args()

    # Step 0: Check test-metrics lock if requested
    if args.eval_test:
        enforce_test_evaluation_lock()

    # Step 1: Inspect backends
    backend_info = inspect_voice_backends()
    print("=" * 70)
    print("STEP 0: VOICE BACKEND INSPECTION")
    print(f"Configured VOICE_BACKEND: {backend_info['configured_backend']}")
    print(f"Genuinely Active Backend: {backend_info['active_backend']} ({backend_info['active_model_id']})")
    print(f"Reason: {backend_info['active_reason']}")
    if not backend_info["emotion2vec"]["available"]:
        print(f"Notice: emotion2vec is unavailable ({backend_info['emotion2vec'].get('error')}). SUPERB is used as active baseline.")
    print("=" * 70)

    # Step 2: License check
    license_status = check_dataset_license(args.data_dir)
    print("\nDATASET LICENSE STATUS:")
    print(license_status)
    print("=" * 70)

    # Step 3: Split management
    split_path = Path(args.split_file)
    if not split_path.is_file():
        logger.info("Generating new split file at: %s", split_path)
        split_data = generate_actor_split(seed=42, n_dev=60)
        save_split_file(split_data, split_path)
    else:
        split_data = load_split_file(split_path)
    print(f"\nActor Split Loaded: {split_data['n_dev_actors']} Dev Actors, {split_data['n_test_actors']} Test Actors (Seed {split_data['seed']})")

    # Step 4: Dataset loading
    all_clips = load_crema_dataset(args.data_dir)
    print(f"Total CREMA-D audio clips detected: {len(all_clips)}")

    # Filter to Dev clips
    dev_actors = set(split_data["dev_actors"])
    dev_clips = [c for c in all_clips if c.actor_id in dev_actors]
    print(f"Dev partition audio clips detected: {len(dev_clips)}")

    if not dev_clips:
        print("\nNo audio clips found in data directory. Creating zero-clip baseline metadata files.")
        meta = {
            "active_backend": backend_info["active_backend"],
            "active_model_id": backend_info["active_model_id"],
            "active_reason": backend_info["active_reason"],
            "emotion2vec_note": backend_info["emotion2vec"].get("error", "Not installed"),
            "data_dir": args.data_dir,
            "license_status": license_status,
            "seed": split_data["seed"],
        }
        empty_metrics = {"n_samples": 0}
        report_md = build_markdown_report(empty_metrics, meta)
        out_md = Path("evaluation/results/crema_d_dev_baseline.md")
        out_json = Path("evaluation/results/crema_d_dev_baseline.json")
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(report_md, encoding="utf-8")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"metadata": meta, "metrics": empty_metrics}, f, indent=2)
        print(f"Saved: {out_md} and {out_json}")
        return

    # Select subset by class budget
    if not args.full and args.max_per_class:
        class_buckets: Dict[str, List[Any]] = {}
        for c in dev_clips:
            if c.canonical_emotion in CANONICAL_4CLASS:
                class_buckets.setdefault(c.canonical_emotion, []).append(c)
        selected_clips = []
        for cls_name, b_clips in class_buckets.items():
            selected_clips.extend(b_clips[: args.max_per_class])
        eval_clips = selected_clips
    else:
        eval_clips = [c for c in dev_clips if c.canonical_emotion in CANONICAL_4CLASS]

    print(f"Evaluating {len(eval_clips)} Dev clips (Budget: {'FULL' if args.full else args.max_per_class} per class)")

    # Model inference with caching
    if backend_info["active_backend"] == "emotion2vec":
        from app.voice_backends.emotion2vec import Emotion2VecBackend
        backend = Emotion2VecBackend(model_id=backend_info["active_model_id"])
        backend._load()
        extractor, model = None, None
    else:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(backend_info["active_model_id"])
        model = Wav2Vec2ForSequenceClassification.from_pretrained(backend_info["active_model_id"])
        model.eval()
        backend = None

    # Check that prints model's raw label list and mapping used to the 4 app classes plus fear/disgust
    print("\n" + "=" * 80)
    print("MODEL RAW LABELS AND LABEL MAPPING CHECK")
    print(f"Active Backend: {backend_info['active_backend']} ({backend_info['active_model_id']})")
    if backend_info["active_backend"] == "emotion2vec":
        from app.voice_backends.emotion2vec import _DIRECT, _OTHER_SET, _map_label, _parse_english
        raw_labels = [
            "生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy",
            "中立/neutral", "其他/other", "难过/sad", "吃惊/surprised", "<unk>"
        ]
        print(f"Model raw labels ({len(raw_labels)}): {raw_labels}")
        print("Mapping to App Classes:")
        for r in raw_labels:
            eng = _parse_english(r)
            mapped = _map_label(r)
            if eng in {"disgusted", "fearful"}:
                handling = "mapped to 'other' -> 'uncertain' (fear/disgust handled as non-4class uncertain)"
            elif eng in _DIRECT:
                handling = f"mapped directly to 4-class '{mapped}'"
            else:
                handling = f"mapped to '{mapped}' (non-4class)"
            print(f"  {r:<20} -> English: {eng:<12} -> {handling}")
    else:
        print(f"Active backend is {backend_info['active_backend']}; emotion2vec mapping check skipped.")
    print("=" * 80 + "\n")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    for clip in eval_clips:
        key = compute_cache_key(
            model_id=backend_info["active_model_id"],
            revision="main",
            target_sr=16000,
            clip_filename=clip.filename,
            labels=CANONICAL_4CLASS,
        )
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.is_file():
            cached = np.load(cache_file, allow_pickle=True).item()
            pred_res = cached
        else:
            if backend_info["active_backend"] == "emotion2vec":
                pred_res = run_emotion2vec_inference_on_clip(clip.filepath, backend)
            else:
                pred_res = run_superb_inference_on_clip(clip.filepath, extractor, model)
            np.save(cache_file, pred_res)

        records.append({
            "filename": clip.filename,
            "actor_id": clip.actor_id,
            "sentence": clip.sentence,
            "raw_emotion": clip.raw_emotion,
            "canonical_emotion": clip.canonical_emotion,
            "canonical_pred": pred_res["canonical_pred"],
            "confidence": pred_res["confidence"],
            "raw_label": pred_res.get("raw_label", ""),
        })

    elapsed = time.perf_counter() - t_start
    print(f"\nInference completed in {elapsed:.2f}s ({elapsed / len(eval_clips):.3f}s/clip)")

    # Per-clip predicted label next to true label comparison table
    print("\n" + "=" * 80)
    print("PER-CLIP PREDICTIONS VS TRUE LABELS")
    print(f"{'Filename':<25} {'True Label':<12} {'Predicted':<12} {'Raw Model Label':<20} {'Confidence':<10}")
    print("-" * 80)
    for r in records:
        print(f"{r['filename']:<25} {r['canonical_emotion']:<12} {r['canonical_pred']:<12} {r.get('raw_label', ''):<20} {r.get('confidence', 0.0):<10.2%}")
    print("-" * 80)
    pred_classes = set(r["canonical_pred"] for r in records)
    all_one_class = len(pred_classes) == 1
    print(f"Prediction classes observed: {sorted(pred_classes)} (Total: {len(pred_classes)} distinct)")
    print(f"Are all predictions one class? {'YES (COLLAPSE)' if all_one_class else 'NO (DIVERSE)'}")
    print("=" * 80 + "\n")

    # Compute metrics
    metrics = compute_dev_metrics(records, classes=CANONICAL_4CLASS, seed=split_data["seed"])
    meta = {
        "active_backend": backend_info["active_backend"],
        "active_model_id": backend_info["active_model_id"],
        "active_reason": backend_info["active_reason"],
        "emotion2vec_note": backend_info["emotion2vec"].get("error", "Not installed"),
        "data_dir": args.data_dir,
        "license_status": license_status,
        "seed": split_data["seed"],
        "elapsed_seconds": round(elapsed, 2),
    }

    report_md = build_markdown_report(metrics, meta)
    out_md = Path("evaluation/results/crema_d_dev_baseline.md")
    out_json = Path("evaluation/results/crema_d_dev_baseline.json")
    out_md.write_text(report_md, encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "metrics": metrics}, f, indent=2)

    print(f"\nSaved Dev Baseline Report to: {out_md}")
    print(f"Saved Dev Baseline JSON to: {out_json}")


if __name__ == "__main__":
    main()
