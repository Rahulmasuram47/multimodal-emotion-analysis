"""
Natural-Speech Label Sheet Generator.

Generates independent, randomized rating sheets for each annotator.
Guarantees complete blinding:
- Speaker intent is NEVER included in rater sheets.
- Model predictions and acoustic confidences are NEVER included in rater sheets.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ALLOWED_EMOTIONS = ["neutral", "happy", "sad", "angry", "other"]
RATER_CSV_COLUMNS = ["clip_id", "filename", "perceived_emotion", "confidence", "notes"]


def generate_rater_sheets(
    clips: List[Dict[str, Any]],
    n_raters: int = 2,
    output_dir: str | Path = "evaluation/natural_set/sheets",
    base_seed: int = 42,
) -> List[Path]:
    """
    Generate randomized, blinded annotation sheets for n_raters.
    Each sheet contains clips in a different random order.
    Strictly strips 'intent', 'model_prediction', or other non-perceptual fields.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # If no clips provided, produce a single template CSV
    if not clips:
        template_file = out_path / "label_sheet_template.csv"
        with open(template_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RATER_CSV_COLUMNS)
            writer.writeheader()
            writer.writerow({
                "clip_id": "spk01_001",
                "filename": "spk01_001.wav",
                "perceived_emotion": "",
                "confidence": "",
                "notes": "Allowed emotions: neutral, happy, sad, angry, other",
            })
        generated_files.append(template_file)
        return generated_files

    for rater_idx in range(1, n_raters + 1):
        rater_file = out_path / f"rater_{rater_idx}.csv"
        rng = random.Random(base_seed + rater_idx * 100)

        # Copy and shuffle clips for this rater
        shuffled = list(clips)
        rng.shuffle(shuffled)

        with open(rater_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RATER_CSV_COLUMNS)
            writer.writeheader()
            for row in shuffled:
                # Blinded representation: strictly exclude intent & model output
                writer.writerow({
                    "clip_id": row.get("clip_id", ""),
                    "filename": row.get("filename", ""),
                    "perceived_emotion": "",
                    "confidence": "",
                    "notes": "",
                })
        generated_files.append(rater_file)

    return generated_files


def main():
    parser = argparse.ArgumentParser(description="Generate blinded rater annotation sheets for natural speech dataset.")
    parser.add_argument("--clips-csv", default=None, help="Optional master manifest containing clip_id, filename")
    parser.add_argument("--n-raters", type=int, default=2, help="Number of independent raters")
    parser.add_argument("--output-dir", default="evaluation/natural_set/sheets", help="Output directory for CSV sheets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for rater shuffling")
    args = parser.parse_args()

    clips = []
    if args.clips_csv and Path(args.clips_csv).is_file():
        with open(args.clips_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            clips = list(reader)

    files = generate_rater_sheets(
        clips=clips,
        n_raters=args.n_raters,
        output_dir=args.output_dir,
        base_seed=args.seed,
    )
    print(f"Generated {len(files)} rater sheet(s) in {args.output_dir}:")
    for fp in files:
        print(f"  - {fp.name}")


if __name__ == "__main__":
    main()
