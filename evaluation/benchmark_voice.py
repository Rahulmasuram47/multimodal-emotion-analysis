"""Voice backend benchmark.

Usage:
    python evaluation/benchmark_voice.py \\
        --manifest data/benchmark/manifest.csv \\
        --backends superb,emotion2vec

Manifest CSV columns: filename, true_voice_emotion
Output: evaluation/results/voice_benchmark.csv + .md
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# Make repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import (  # type: ignore
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from app.audio_processor import load_and_preprocess_audio
import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["neutral", "happy", "sad", "angry"]   # "other" = wrong
RESULTS_DIR = Path(__file__).parent / "results"


# ------------------------------------------------------------------ #
# Backend factory                                                      #
# ------------------------------------------------------------------ #

def _make_backend(name: str):
    """Return a VoiceBackend instance for *name* without loading via VEA."""
    if name == "superb":
        from app.emotion_analyzer import VoiceEmotionAnalyzer
        from app.voice_backends.superb import SuperbBackend
        vea = VoiceEmotionAnalyzer(backend="superb")
        return SuperbBackend(analyzer=vea)
    elif name == "emotion2vec":
        from app.voice_backends.emotion2vec import Emotion2VecBackend
        return Emotion2VecBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")


# ------------------------------------------------------------------ #
# Benchmark logic                                                      #
# ------------------------------------------------------------------ #

def _load_manifest(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_backend(backend_name: str, rows: List[Dict[str, str]]) -> List[Dict]:
    backend = _make_backend(backend_name)
    results = []
    for row in rows:
        filepath = row["filename"]
        true_label = (row.get("true_voice_emotion") or "").strip().lower()
        if not os.path.isfile(filepath):
            logger.warning("Missing file, skipping: %s", filepath)
            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": "MISSING",
                "confidence": None,
                "seconds": None,
                "skipped": True,
            })
            continue
        try:
            audio, sr, _ = load_and_preprocess_audio(filepath)
            t0 = time.perf_counter()
            pred = backend.predict(audio, sr)
            elapsed = time.perf_counter() - t0
            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": pred["emotion"],
                "confidence": pred["confidence"],
                "seconds": elapsed,
                "skipped": False,
            })
        except Exception as exc:
            logger.warning("Inference failed for %s: %s", filepath, exc)
            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": "ERROR",
                "confidence": None,
                "seconds": None,
                "skipped": True,
            })
    return results


def _wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate 95% Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    # 95% confidence z-score ~ 1.95996
    z = 1.95996
    p = k / n
    denominator = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denominator
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
    return max(0.0, float(centre - spread)), min(1.0, float(centre + spread))


def _compute_metrics(rows: List[Dict]) -> Dict:
    valid = [r for r in rows if not r["skipped"] and r["true"] in CANONICAL_CLASSES]
    if not valid:
        return {
            "n": 0,
            "accuracy": None,
            "wilson_ci": None,
            "macro_f1": None,
            "conf_matrix": None,
            "class_share": {},
            "mean_conf_correct": None,
            "mean_conf_wrong": None,
            "mean_seconds": None,
        }

    y_true = [r["true"] for r in valid]
    y_pred = [r["pred"] for r in valid]

    # treat "other" predictions as wrong (wrong label not in CANONICAL_CLASSES)
    acc = accuracy_score(y_true, y_pred)
    k_correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    wilson_ci = _wilson_score_interval(k_correct, len(valid), confidence=0.95)

    f1 = f1_score(y_true, y_pred, labels=CANONICAL_CLASSES,
                  average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CANONICAL_CLASSES)

    correct = [r for r in valid if r["true"] == r["pred"]]
    wrong = [r for r in valid if r["true"] != r["pred"]]
    mean_conf_ok = float(np.mean([r["confidence"] for r in correct if r["confidence"] is not None])) if correct else None
    mean_conf_ng = float(np.mean([r["confidence"] for r in wrong if r["confidence"] is not None])) if wrong else None
    mean_sec = float(np.mean([r["seconds"] for r in valid if r["seconds"] is not None]))

    # Predicted-class share
    pred_counts: Dict[str, int] = {}
    for p in y_pred:
        pred_counts[p] = pred_counts.get(p, 0) + 1
    class_share = {lbl: count / len(valid) for lbl, count in sorted(pred_counts.items())}

    return {
        "n": len(valid),
        "accuracy": acc,
        "wilson_ci": wilson_ci,
        "macro_f1": f1,
        "conf_matrix": cm,
        "class_share": class_share,
        "mean_conf_correct": mean_conf_ok,
        "mean_conf_wrong": mean_conf_ng,
        "mean_seconds": mean_sec,
    }


# ------------------------------------------------------------------ #
# Formatting                                                           #
# ------------------------------------------------------------------ #

def _print_per_clip_table(backend_name: str, rows: List[Dict]):
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"{'='*60}")
    print(f"{'File':<40} {'True':<10} {'Pred':<10} {'Conf':>6}")
    print("-" * 70)
    for r in rows:
        fname = Path(r["filename"]).name
        conf_str = f"{r['confidence']:.3f}" if r["confidence"] is not None else "  N/A"
        print(f"{fname:<40} {r['true']:<10} {r['pred']:<10} {conf_str:>6}")


def _format_conf_matrix(cm, labels: List[str]) -> str:
    header = "      " + "  ".join(f"{l[:4]:>6}" for l in labels)
    lines = [header]
    for i, row in enumerate(cm):
        line = f"{labels[i][:6]:<6}" + "  ".join(f"{v:>6}" for v in row)
        lines.append(line)
    return "\n".join(lines)


def _print_summary(backend_name: str, m: Dict):
    print(f"\n--- {backend_name} summary (N={m['n']}) ---")
    if m["n"] == 0:
        print("  No annotated samples found.")
        return
    ci = m.get("wilson_ci")
    ci_str = f" [95% Wilson CI: {ci[0]:.1%} - {ci[1]:.1%}]" if ci else ""
    print(f"  Accuracy : {m['accuracy']:.1%}{ci_str}")
    print(f"  Macro-F1 : {m['macro_f1']:.3f} (4 classes; other=wrong)")
    print(f"  Conf (correct) : {m['mean_conf_correct']:.3f}" if m["mean_conf_correct"] is not None else "  Conf (correct) : N/A")
    print(f"  Conf (wrong)   : {m['mean_conf_wrong']:.3f}" if m["mean_conf_wrong"] is not None else "  Conf (wrong)   : N/A")
    print(f"  Mean s/clip    : {m['mean_seconds']:.3f}")
    if m.get("class_share"):
        shares_str = ", ".join(f"{k}: {v:.1%}" for k, v in m["class_share"].items())
        print(f"  Predicted share: {shares_str}")
    if m["conf_matrix"] is not None:
        print("\n  Confusion matrix (rows=true, cols=pred):")
        for line in _format_conf_matrix(m["conf_matrix"], CANONICAL_CLASSES).splitlines():
            print("  " + line)


# ------------------------------------------------------------------ #
# CSV / Markdown output                                                #
# ------------------------------------------------------------------ #

def _write_csv(all_rows: List[Dict], metrics: Dict[str, Dict], mode: str = "clip"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = "voice_benchmark_segments.csv" if mode == "segments" else "voice_benchmark.csv"
    csv_path = RESULTS_DIR / fname
    fieldnames = ["backend", "filename", "true", "pred", "confidence", "seconds", "skipped"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in fieldnames})
    return csv_path


def _write_md(all_rows: List[Dict], metrics: Dict[str, Dict], mode: str = "clip"):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fname = "voice_benchmark_segments.md" if mode == "segments" else "voice_benchmark.md"
    md_path = RESULTS_DIR / fname
    lines = [f"# Voice Backend Benchmark (Mode: {mode})\n"]
    backends = list(metrics.keys())

    for backend_name in backends:
        m = metrics[backend_name]
        rows = [r for r in all_rows if r["backend"] == backend_name]
        lines.append(f"## {backend_name}\n")
        lines.append(f"| File | True | Pred | Conf |")
        lines.append(f"|------|------|------|------|")
        for r in rows:
            fname_str = Path(r["filename"]).name
            conf_str = f"{r['confidence']:.3f}" if r["confidence"] is not None else "N/A"
            lines.append(f"| {fname_str} | {r['true']} | {r['pred']} | {conf_str} |")
        lines.append("")
        if m["n"] > 0:
            ci = m.get("wilson_ci")
            ci_str = f" (95% Wilson CI: {ci[0]:.1%} - {ci[1]:.1%})" if ci else ""
            lines.append(f"**N:** {m['n']}  ")
            lines.append(f"**Accuracy:** {m['accuracy']:.1%}{ci_str}  ")
            lines.append(f"**Macro-F1:** {m['macro_f1']:.3f} (4 classes; other=wrong)  ")
            lines.append(f"**Mean conf correct:** {m['mean_conf_correct']:.3f}  " if m["mean_conf_correct"] is not None else "**Mean conf correct:** N/A  ")
            lines.append(f"**Mean conf wrong:** {m['mean_conf_wrong']:.3f}  " if m["mean_conf_wrong"] is not None else "**Mean conf wrong:** N/A  ")
            lines.append(f"**Mean s/clip:** {m['mean_seconds']:.3f}  ")
            if m.get("class_share"):
                shares_str = ", ".join(f"{k}: {v:.1%}" for k, v in m["class_share"].items())
                lines.append(f"**Predicted-class share:** {shares_str}  ")
            if m["conf_matrix"] is not None:
                lines.append("\n**Confusion matrix** (rows=true, cols=pred):\n")
                lines.append("```")
                lines.append(_format_conf_matrix(m["conf_matrix"], CANONICAL_CLASSES))
                lines.append("```")
        else:
            lines.append("_No annotated samples._")
        lines.append("")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return md_path


def _run_backend_segments(backend_name: str, rows: List[Dict[str, str]]) -> List[Dict]:
    """Run backend using existing ConversationAnalyzer and derive_segment_summary aggregation."""
    from app.emotion_analyzer import VoiceEmotionAnalyzer
    from app.speech_to_text import SpeechToText
    from app.sentiment_analyzer import SentimentAnalyzer
    from app.fusion import MultimodalFusion
    from app.conversation_analysis import ConversationAnalyzer
    from ui.utils import derive_segment_summary

    vea = VoiceEmotionAnalyzer(backend=backend_name)
    stt = SpeechToText()
    sa = SentimentAnalyzer()
    fusion = MultimodalFusion()
    ca = ConversationAnalyzer(
        emotion_analyzer=vea,
        speech_to_text=stt,
        sentiment_analyzer=sa,
        fusion=fusion,
    )

    results = []
    for row in rows:
        filepath = row["filename"]
        true_label = (row.get("true_voice_emotion") or "").strip().lower()
        if not os.path.isfile(filepath):
            logger.warning("Missing file, skipping: %s", filepath)
            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": "MISSING",
                "confidence": None,
                "seconds": None,
                "skipped": True,
            })
            continue
        try:
            t0 = time.perf_counter()
            conv_res = ca.analyze(filepath)
            elapsed = time.perf_counter() - t0
            segs = conv_res.get("segments", [])
            if segs:
                dom_voice, _, _, _ = derive_segment_summary(segs)
                pred_label = dom_voice
                # Average segment voice confidence for dominant label
                matched_confs = [
                    float((s.get("voice_emotion") or {}).get("confidence", 0.0))
                    for s in segs
                    if (s.get("voice_emotion") or {}).get("emotion") == dom_voice
                ]
                pred_conf = float(np.mean(matched_confs)) if matched_confs else 0.0
            else:
                pred_label = "other"
                pred_conf = 0.0

            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": pred_label,
                "confidence": pred_conf,
                "seconds": elapsed,
                "skipped": False,
            })
        except Exception as exc:
            logger.warning("Inference failed for %s: %s", filepath, exc)
            results.append({
                "backend": backend_name,
                "filename": filepath,
                "true": true_label,
                "pred": "ERROR",
                "confidence": None,
                "seconds": None,
                "skipped": True,
            })
    return results


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="Benchmark voice emotion backends")
    parser.add_argument("--manifest", default="data/benchmark/manifest.csv",
                        help="CSV with columns: filename, true_voice_emotion")
    parser.add_argument("--backends", default="superb,emotion2vec",
                        help="Comma-separated list of backends to compare")
    parser.add_argument("--mode", default="clip", choices=["clip", "segments"],
                        help="Benchmark mode: 'clip' (whole audio) or 'segments' (ConversationAnalyzer segments + aggregation)")
    args = parser.parse_args()

    rows_manifest = _load_manifest(args.manifest)
    backend_names = [b.strip() for b in args.backends.split(",") if b.strip()]

    all_rows: List[Dict] = []
    metrics: Dict[str, Dict] = {}

    for bname in backend_names:
        print(f"\nRunning backend: {bname} (mode: {args.mode}) ...")
        if args.mode == "segments":
            brows = _run_backend_segments(bname, rows_manifest)
        else:
            brows = _run_backend(bname, rows_manifest)
        all_rows.extend(brows)
        _print_per_clip_table(bname, brows)
        m = _compute_metrics(brows)
        metrics[bname] = m
        _print_summary(bname, m)

    csv_path = _write_csv(all_rows, metrics, mode=args.mode)
    md_path = _write_md(all_rows, metrics, mode=args.mode)
    print(f"\nResults written to:\n  {csv_path}\n  {md_path}")


if __name__ == "__main__":
    main()
