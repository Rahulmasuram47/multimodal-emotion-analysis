"""Compare app voice step vs baseline emotion2vec on sample audio and CREMA-D demo clips.

Titles and outputs:
evaluation/results/app_vs_baseline.md
evaluation/results/app_vs_baseline.json
Title: "small, self-recorded and acted clips; not an accuracy benchmark"
"""
import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure root directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from app.audio_processor import load_and_preprocess_audio
from app.conversation_analysis import ConversationAnalyzer
from evaluation.run_crema_d_baseline import run_emotion2vec_inference_on_clip


def parse_true_label(filename: str) -> str:
    """Parse true emotion label from filename (CREMA-D pattern or toy names like Sad1, Happy2)."""
    stem = Path(filename).stem
    # CREMA-D pattern: e.g. 1001_DFA_NEU_XX or ActorID_Sentence_Emotion_Level
    m_crema = re.match(r"^\d+_[A-Za-z0-9]+_([A-Za-z]{3})_[A-Za-z0-9]+", stem)
    if m_crema:
        code = m_crema.group(1).upper()
        mapping = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fear",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad",
        }
        return mapping.get(code, code.lower())

    # Toy names: e.g. Sad1, Happy2, angry_sample, neutral_sample
    lower_stem = stem.lower()
    if "angry" in lower_stem:
        return "angry"
    if "happy" in lower_stem:
        return "happy"
    if "sad" in lower_stem:
        return "sad"
    if "neutral" in lower_stem:
        return "neutral"

    # Fallback to leading letters
    m_alpha = re.match(r"^([a-zA-Z]+)", stem)
    if m_alpha:
        return m_alpha.group(1).lower()
    return "unknown"


def run_app_voice_whole_clip(
    analyzer: ConversationAnalyzer,
    audio_path: str,
) -> Dict[str, Any]:
    """Run app's voice step on the whole clip."""
    audio, sr, duration = load_and_preprocess_audio(audio_path)
    # Use analyzer.emotion_analyzer._analyze_audio_array or analyze
    res = analyzer.emotion_analyzer.analyze(audio_path)
    backend_info = analyzer.emotion_analyzer.backend_info() if hasattr(analyzer.emotion_analyzer, "backend_info") else {}
    backend_used = backend_info.get("backend_used", getattr(analyzer.emotion_analyzer, "backend_used", "unknown"))

    raw_conf = float(res.get("raw_confidence", res.get("confidence", 0.0)))
    final_conf = float(res.get("confidence", 0.0))
    conf_informative = getattr(analyzer.emotion_analyzer, "confidence_informative", True)
    if not conf_informative:
        cap_val = getattr(config, "VOICE_CONF_CAP", 0.65)
        if final_conf > cap_val:
            final_conf = cap_val

    return {
        "file": os.path.basename(audio_path),
        "backend_used": backend_used,
        "top_label": res.get("emotion", "unknown"),
        "final_confidence": round(final_conf, 4),
        "raw_confidence": round(raw_conf, 4),
        "mode": "whole-clip",
    }


def run_app_voice_per_segment(
    analyzer: ConversationAnalyzer,
    audio_path: str,
) -> List[Dict[str, Any]]:
    """Run app's voice step via the standard ConversationAnalyzer pipeline per segment."""
    res = analyzer.analyze(audio_path)
    backend_info = res.get("voice_backend", {})
    backend_used = backend_info.get("backend_used", getattr(analyzer.emotion_analyzer, "backend_used", "unknown"))
    segments = res.get("segments", []) or []

    rows = []
    if not segments:
        # Empty/no speech fallback
        rows.append({
            "file": os.path.basename(audio_path),
            "backend_used": backend_used,
            "top_label": "no_speech",
            "final_confidence": 0.0,
            "raw_confidence": 0.0,
            "mode": "per-segment (none)",
        })
        return rows

    for idx, seg in enumerate(segments):
        ve = seg.get("voice_emotion", {}) or {}
        raw_conf = float(ve.get("raw_confidence", ve.get("confidence", 0.0)))
        final_conf = float(ve.get("confidence", 0.0))
        top_lbl = ve.get("emotion", "unknown")
        rows.append({
            "file": os.path.basename(audio_path),
            "backend_used": backend_used,
            "top_label": top_lbl,
            "final_confidence": round(final_conf, 4),
            "raw_confidence": round(raw_conf, 4),
            "mode": f"segment-{idx+1}",
        })
    return rows


def run_comparison(
    audio_dirs: List[str],
    max_files: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute app vs baseline comparison across target directories."""
    # Initialize ConversationAnalyzer with configured backend
    backend_name = getattr(config, "VOICE_BACKEND", "emotion2vec")
    analyzer = ConversationAnalyzer(
        whisper_model_size="base",
        whisper_language="auto",
        voice_backend=backend_name,
    )

    # Initialize baseline backend object directly from app's backend to avoid code duplication
    from app.voice_backends.emotion2vec import Emotion2VecBackend
    baseline_backend = Emotion2VecBackend()
    baseline_backend._load()

    all_records: List[Dict[str, Any]] = []

    for d in audio_dirs:
        dir_path = Path(d)
        if not dir_path.exists():
            continue

        wav_files = sorted(dir_path.glob("*.wav"))
        if max_files and len(wav_files) > max_files:
            wav_files = wav_files[:max_files]

        for wav_path in wav_files:
            # Skip 0-byte or corrupted files
            if wav_path.stat().st_size == 0 or "corrupted" in wav_path.name:
                continue

            str_path = str(wav_path)
            true_lbl = parse_true_label(wav_path.name)

            # 1. Baseline emotion2vec inference
            try:
                base_res = run_emotion2vec_inference_on_clip(str_path, backend=baseline_backend)
                base_label = base_res.get("canonical_pred", "unknown")
                base_conf = float(base_res.get("confidence", 0.0))
            except Exception as e:
                base_label = f"error: {e}"
                base_conf = 0.0

            # 2. App whole-clip voice step
            try:
                app_whole = run_app_voice_whole_clip(analyzer, str_path)
            except Exception as e:
                app_whole = {
                    "file": wav_path.name,
                    "backend_used": backend_name,
                    "top_label": f"error: {e}",
                    "final_confidence": 0.0,
                    "raw_confidence": 0.0,
                    "mode": "whole-clip",
                }

            # Comparison flags for whole-clip
            diff_label = (app_whole["top_label"] != base_label)
            diff_conf = abs(app_whole["final_confidence"] - base_conf) > 0.001
            cause_reasons = []
            if diff_label:
                cause_reasons.append(f"Label difference: app='{app_whole['top_label']}' vs baseline='{base_label}'")
            if diff_conf:
                cause_reasons.append(
                    f"Confidence difference: app={app_whole['final_confidence']:.4f} vs baseline={base_conf:.4f} "
                    f"(app raw={app_whole['raw_confidence']:.4f}, cap={getattr(config, 'VOICE_CONF_CAP', 0.65)})"
                )

            whole_record = {
                "file": wav_path.name,
                "directory": d,
                "true_label": true_lbl,
                "backend_used": app_whole["backend_used"],
                "app_top_label": app_whole["top_label"],
                "app_final_confidence": app_whole["final_confidence"],
                "app_raw_confidence": app_whole["raw_confidence"],
                "mode": app_whole["mode"],
                "baseline_label": base_label,
                "baseline_confidence": round(base_conf, 4),
                "flag_diff_label": diff_label,
                "flag_diff_confidence": diff_conf,
                "cause": "; ".join(cause_reasons) if cause_reasons else "Identical",
            }
            all_records.append(whole_record)

            # 3. App per-segment voice step
            try:
                app_segs = run_app_voice_per_segment(analyzer, str_path)
            except Exception as e:
                app_segs = [{
                    "file": wav_path.name,
                    "backend_used": backend_name,
                    "top_label": f"error: {e}",
                    "final_confidence": 0.0,
                    "raw_confidence": 0.0,
                    "mode": "per-segment",
                }]

            for seg_item in app_segs:
                s_diff_label = (seg_item["top_label"] != base_label)
                s_diff_conf = abs(seg_item["final_confidence"] - base_conf) > 0.001
                s_causes = []
                if s_diff_label:
                    s_causes.append(f"Segment boundary/padding effect or VAD vs whole clip ({seg_item['top_label']} vs {base_label})")
                if s_diff_conf:
                    s_causes.append(
                        f"Segment confidence calibration/cap (app={seg_item['final_confidence']:.4f} vs base={base_conf:.4f})"
                    )

                seg_record = {
                    "file": wav_path.name,
                    "directory": d,
                    "true_label": true_lbl,
                    "backend_used": seg_item["backend_used"],
                    "app_top_label": seg_item["top_label"],
                    "app_final_confidence": seg_item["final_confidence"],
                    "app_raw_confidence": seg_item["raw_confidence"],
                    "mode": seg_item["mode"],
                    "baseline_label": base_label,
                    "baseline_confidence": round(base_conf, 4),
                    "flag_diff_label": s_diff_label,
                    "flag_diff_confidence": s_diff_conf,
                    "cause": "; ".join(s_causes) if s_causes else "Identical",
                }
                all_records.append(seg_record)

    summary_meta = {
        "title": "small, self-recorded and acted clips; not an accuracy benchmark",
        "total_records": len(all_records),
        "total_files": len(set(r["file"] for r in all_records)),
        "directories_evaluated": audio_dirs,
    }

    return all_records, summary_meta


def generate_markdown_report(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    """Generate Markdown report table."""
    lines = [
        f"# Evaluation: {meta['title']}",
        "",
        "> **Notice**: This comparison analyzes differences between the app's online voice inference "
        "(whole-clip and Whisper-segmented) and the offline baseline emotion2vec pipeline on local "
        "sample and acted demo clips. This is NOT an accuracy benchmark.",
        "",
        f"- **Total files**: {meta['total_files']}",
        f"- **Total comparison entries**: {meta['total_records']}",
        f"- **Directories**: {', '.join(meta['directories_evaluated'])}",
        "",
        "| File | True Label | Mode | Backend | App Label | App Final Conf | App Raw Conf | Baseline Label | Baseline Conf | Label Diff | Conf Diff | Cause |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for r in records:
        f_lbl = "⚠️ YES" if r["flag_diff_label"] else "No"
        f_cnf = "⚠️ YES" if r["flag_diff_confidence"] else "No"
        lines.append(
            f"| `{r['file']}` | **{r['true_label']}** | {r['mode']} | {r['backend_used']} | "
            f"`{r['app_top_label']}` | {r['app_final_confidence']:.4f} | {r['app_raw_confidence']:.4f} | "
            f"`{r['baseline_label']}` | {r['baseline_confidence']:.4f} | {f_lbl} | {f_cnf} | {r['cause']} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare App vs Baseline Emotion Predictions")
    parser.add_argument("--dirs", nargs="+", default=["sample_audio", r"C:\Project\crema_demo"])
    parser.add_argument("--max-files", type=int, default=40)
    args = parser.parse_args()

    print(f"Starting comparison on directories: {args.dirs} (max_files={args.max_files})")
    records, meta = run_comparison(audio_dirs=args.dirs, max_files=args.max_files)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "app_vs_baseline.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": records}, f, indent=2)

    md_path = out_dir / "app_vs_baseline.md"
    md_content = generate_markdown_report(records, meta)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Comparison completed. Outputs written to {md_path} and {json_path}.")


if __name__ == "__main__":
    main()
