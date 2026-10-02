"""
Prepare a Balanced Subset from RAVDESS or CREMA-D with Contamination Warnings.

Usage:
    python -m evaluation.scripts.prepare_public_subset --source-dir /path/to/dataset --dataset-type ravdess --output-dir data/public_subset --target-per-class 5
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
from typing import Dict, List, Optional, Tuple

from evaluation.dataset import EXTENDED_COLUMNS


# RAVDESS emotion codes:
# 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
RAVDESS_MAP = {
    "01": "neutral",
    "02": "neutral",  # calm mapped to neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
}

# CREMA-D emotion codes:
# ANG = angry, DIS = disgust, FEA = fear, HAP = happy, NEU = neutral, SAD = sad
CREMAD_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

TARGET_CLASSES = ["angry", "happy", "neutral", "sad"]

CONTAMINATION_WARNINGS = """
================================================================================
⚠️  TRAINING-SET CONTAMINATION & DOMAIN-SHIFT WARNINGS (MANDATORY DISCLOSURE)  ⚠️
================================================================================
1. Wav2Vec2 (`superb/wav2vec2-base-superb-er`):
   - Fine-tuned specifically on IEMOCAP 5-fold cross-validation.
   - Self-supervised foundation backbone was pre-trained on LibriSpeech (960 hours).
   - Evaluating on RAVDESS or CREMA-D provides out-of-domain acted speech evaluation,
     which avoids IEMOCAP test-set leakage, but acoustic distributions differ sharply
     from real conversational audio.

2. CardiffNLP RoBERTa (`twitter-roberta-base-sentiment-latest`):
   - Pre-trained on ~124M tweets and fine-tuned on TweetEval sentiment benchmark.
   - Text transcripts from RAVDESS ("Kids are talking by the door") or CREMA-D
     are neutral statements spoken with varying emotion; text models will predict
     'neutral' almost exclusively regardless of the vocal emotion, producing
     deliberate synthetic modality divergence.

3. Faster-Whisper (`base` / `small` INT8):
   - Pre-trained on 680,000 hours of multilingual web-scraped audio.
   - High likelihood of having encountered public benchmark speech datasets during
     unsupervised or semi-supervised pre-training. WER on public benchmarks may be
     optimistically biased compared to novel conversational audio.
================================================================================
"""


def parse_ravdess_emotion(filename: str) -> Optional[str]:
    """Parse emotion from standard RAVDESS filename (e.g., 03-01-05-01-02-01-12.wav)."""
    parts = Path(filename).stem.split("-")
    if len(parts) >= 3:
        code = parts[2]
        return RAVDESS_MAP.get(code)
    return None


def parse_cremad_emotion(filename: str) -> Optional[str]:
    """Parse emotion from standard CREMA-D filename (e.g., 1001_DFA_ANG_XX.wav)."""
    parts = Path(filename).stem.split("_")
    if len(parts) >= 3:
        code = parts[2].upper()
        return CREMAD_MAP.get(code)
    return None


def prepare_subset(
    source_dir: str | Path,
    dataset_type: str,
    output_dir: str | Path,
    target_per_class: int = 5,
) -> Tuple[List[Dict[str, str]], Path]:
    src = Path(source_dir)
    out = Path(output_dir)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    print(CONTAMINATION_WARNINGS)

    if not src.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src}")

    # Gather matching clips by class
    parser_func = parse_ravdess_emotion if dataset_type.lower() == "ravdess" else parse_cremad_emotion
    files_by_class: Dict[str, List[Path]] = {cls: [] for cls in TARGET_CLASSES}

    for f in sorted(src.rglob("*.wav")):
        emo = parser_func(f.name)
        if emo and emo in files_by_class:
            if len(files_by_class[emo]) < target_per_class:
                files_by_class[emo].append(f)

    # Check counts
    total_selected = sum(len(flist) for flist in files_by_class.values())
    print(f"[+] Found clips for subset across {dataset_type.upper()}:")
    for cls in TARGET_CLASSES:
        print(f"    - {cls:<8}: {len(files_by_class[cls])}/{target_per_class}")

    if total_selected == 0:
        print(f"[!] Warning: No matching {dataset_type} audio files found in {source_dir}.")
        return [], out / "starter_subset.csv"

    # Copy files and build starter CSV records
    records: List[Dict[str, str]] = []
    for cls in TARGET_CLASSES:
        for f in files_by_class[cls]:
            dest_file = audio_out / f.name
            shutil.copy2(f, dest_file)
            rel_audio_path = f"data/public_subset/audio/{f.name}"
            records.append({
                "file": rel_audio_path,
                "source": f"public_{dataset_type.lower()}",
                "true_voice_emotion": cls,
                "true_text_sentiment": "",  # pending annotation / validation
                "true_transcript": "",
                "true_speaker_count": "1",
                "true_escalation": "no",
                "true_incongruent": "",
                "notes": f"Selected from {dataset_type.upper()} ({f.name}); subject to cross-dataset evaluation",
                "annotator": "dataset_ground_truth",
            })

    # Write starter CSV
    csv_path = out / "starter_subset.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXTENDED_COLUMNS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"\n[+] Successfully created subset with {len(records)} clips.")
    print(f"[+] Audio files copied to: {audio_out}")
    print(f"[+] Starter CSV written to: {csv_path}")
    return records, csv_path


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced subset from RAVDESS or CREMA-D.")
    parser.add_argument("--source-dir", required=True, help="Directory containing original WAV files")
    parser.add_argument("--dataset-type", choices=["ravdess", "cremad"], required=True, help="Dataset format")
    parser.add_argument("--output-dir", default="data/public_subset", help="Target output folder")
    parser.add_argument("--target-per-class", type=int, default=5, help="Number of clips per emotion class")
    args = parser.parse_args()

    prepare_subset(
        source_dir=args.source_dir,
        dataset_type=args.dataset_type,
        output_dir=args.output_dir,
        target_per_class=args.target_per_class,
    )


if __name__ == "__main__":
    main()
