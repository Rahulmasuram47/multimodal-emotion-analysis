"""
Merge Multi-Annotator CSVs, Flag Disagreements, Compute Cohen's Kappa, and Build Consensus.

Usage:
    python scripts/merge_annotations.py [--annotations-dir data/annotations] [--output data/evaluation_dataset.csv]
"""

from __future__ import annotations

import argparse
import csv
import io
import itertools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evaluation.metrics import compute_cohens_kappa
from evaluation.dataset import EXTENDED_COLUMNS


CATEGORICAL_FIELDS = [
    "true_voice_emotion",
    "true_text_sentiment",
    "true_speaker_count",
    "true_escalation",
    "true_incongruent",
]


def load_annotator_csvs(annotations_dir: str | Path) -> Dict[str, Dict[str, Dict[str, str]]]:
    """Load all annotator_*.csv files.

    Returns:
        Dict mapping annotator_name -> {file_path -> {field: value}}
    """
    p = Path(annotations_dir)
    if not p.exists():
        return {}

    annotator_data: Dict[str, Dict[str, Dict[str, str]]] = {}
    for csv_file in sorted(p.glob("annotator_*.csv")):
        annotator_id = csv_file.stem.replace("annotator_", "")
        records: Dict[str, Dict[str, str]] = {}
        with open(csv_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                norm_file = Path(row.get("file", "")).as_posix()
                if norm_file:
                    records[norm_file] = dict(row)
        annotator_data[annotator_id] = records

    return annotator_data


def analyze_agreement(
    annotator_data: Dict[str, Dict[str, Dict[str, str]]]
) -> Dict[str, Any]:
    """Compute pairwise Cohen's kappa and identify disagreements across files.

    Returns:
        Dict with:
        - annotators: list of annotator names
        - pairwise_kappa: Dict[(a1, a2), Dict[field, kappa_result]]
        - field_kappas: Dict[field, float] (average kappa across pairs)
        - disagreements: List[Dict[str, Any]] (files with conflicting labels)
        - consensus_records: Dict[file_path, Dict[str, Any]] (files where all annotators agree)
    """
    annotators = sorted(list(annotator_data.keys()))
    all_files = sorted(list({f for records in annotator_data.values() for f in records.keys()}))

    # Pairwise Cohen's Kappa per categorical field
    pairwise_kappa: Dict[Tuple[str, str], Dict[str, Any]] = {}
    field_kappas: Dict[str, List[float]] = {field: [] for field in CATEGORICAL_FIELDS}

    for a1, a2 in itertools.combinations(annotators, 2):
        pair_res: Dict[str, Any] = {}
        shared_files = sorted(list(set(annotator_data[a1].keys()) & set(annotator_data[a2].keys())))
        for field in CATEGORICAL_FIELDS:
            r1 = [annotator_data[a1][f].get(field, "").strip() for f in shared_files]
            r2 = [annotator_data[a2][f].get(field, "").strip() for f in shared_files]
            res = compute_cohens_kappa(r1, r2)
            pair_res[field] = res
            if res.get("kappa") is not None:
                field_kappas[field].append(float(res["kappa"]))
        pairwise_kappa[(a1, a2)] = pair_res

    avg_field_kappas: Dict[str, Optional[float]] = {}
    for field, kappas in field_kappas.items():
        avg_field_kappas[field] = round(sum(kappas) / len(kappas), 4) if kappas else None

    # Detect conflicts / disagreements per file
    disagreements: List[Dict[str, Any]] = []
    consensus_records: Dict[str, Dict[str, str]] = {}

    for f in all_files:
        per_annotator_rows = {
            a: annotator_data[a][f]
            for a in annotators
            if f in annotator_data[a]
        }
        has_conflict = False
        conflict_details: Dict[str, Dict[str, str]] = {}
        merged_row: Dict[str, str] = {"file": f, "source": "", "notes": "", "annotator": "+".join(per_annotator_rows.keys())}

        for field in CATEGORICAL_FIELDS + ["true_transcript"]:
            values = {
                a: row.get(field, "").strip()
                for a, row in per_annotator_rows.items()
                if row.get(field, "").strip()
            }
            distinct_values = set(values.values())
            if len(distinct_values) > 1:
                has_conflict = True
                conflict_details[field] = values
            elif len(distinct_values) == 1:
                merged_row[field] = next(iter(distinct_values))
            else:
                merged_row[field] = ""

        # Set source & notes from first annotator if present
        for row in per_annotator_rows.values():
            if row.get("source"):
                merged_row["source"] = row["source"]
            if row.get("notes"):
                merged_row["notes"] = row["notes"]

        if has_conflict:
            disagreements.append({
                "file": f,
                "annotators": list(per_annotator_rows.keys()),
                "conflicts": conflict_details,
                "partial_row": merged_row,
            })
        else:
            consensus_records[f] = merged_row

    return {
        "annotators": annotators,
        "total_files": len(all_files),
        "pairwise_kappa": pairwise_kappa,
        "avg_field_kappas": avg_field_kappas,
        "disagreements": disagreements,
        "consensus_records": consensus_records,
    }


def write_consensus_dataset(
    consensus_records: Dict[str, Dict[str, str]],
    output_path: str | Path,
) -> int:
    """Write verified consensus rows to output CSV with EXTENDED_COLUMNS."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTENDED_COLUMNS)
        writer.writeheader()
        for fpath, row in sorted(consensus_records.items()):
            clean_row = {col: row.get(col, "") for col in EXTENDED_COLUMNS}
            writer.writerow(clean_row)

    return len(consensus_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multi-annotator datasets and flag disagreements.")
    parser.add_argument("--annotations-dir", default="data/annotations", help="Directory containing annotator_*.csv files")
    parser.add_argument("--output", default="data/evaluation_dataset.csv", help="Path for consensus output CSV")
    args = parser.parse_args()

    data = load_annotator_csvs(args.annotations_dir)
    if not data:
        print(f"[!] No annotator_*.csv files found in '{args.annotations_dir}'.")
        return

    analysis = analyze_agreement(data)
    print(f"[+] Loaded {len(analysis['annotators'])} annotators: {', '.join(analysis['annotators'])}")
    print(f"[+] Total distinct files annotated: {analysis['total_files']}")
    print("\n--- Cohen's Kappa per Field ---")
    for field, k in analysis["avg_field_kappas"].items():
        k_str = f"{k:.4f}" if k is not None else "N/A (insufficient overlap)"
        print(f"  {field:<22}: {k_str}")

    print(f"\n[+] Full consensus files: {len(analysis['consensus_records'])}")
    print(f"[!] Files with disagreements: {len(analysis['disagreements'])}")

    if analysis["disagreements"]:
        print("\n--- Disagreements to Resolve ---")
        for d in analysis["disagreements"]:
            print(f"  File: {d['file']}")
            for fld, vals in d["conflicts"].items():
                print(f"    - {fld}: {vals}")

    saved_count = write_consensus_dataset(analysis["consensus_records"], args.output)
    print(f"\n[+] Wrote {saved_count} consensus records to {args.output}")


if __name__ == "__main__":
    main()
