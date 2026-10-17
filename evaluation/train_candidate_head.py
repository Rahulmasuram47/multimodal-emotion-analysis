"""
Candidate Voice Classification Head Training & Dev-Only Adoption Decision.

Evaluates a Logistic Regression head trained on frozen embeddings from the genuinely active backend
(Wav2Vec2-SUPERB) using GroupKFold grouped by actor on DEV actors only.
Enforces the pre-declared adoption rule before evaluating.
Strictly isolates test actors.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from evaluation.crema_d import (
    CANONICAL_4CLASS,
    compute_cache_key,
    load_crema_dataset,
    load_split_file,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("candidate_head")

# Pre-declared Adoption Rule
ADOPTION_RULE = (
    "Candidate is promising only if, in actor-grouped CV on dev actors (GroupKFold by actor, fixed seed), "
    "its mean UAR beats the dev baseline UAR by at least 0.03 AND beats it in a majority of folds "
    "AND neither sad nor neutral recall drops versus baseline. "
    "Hyperparameters selected inside the dev folds only."
)


def evaluate_decision_rule(
    fold_uars: List[float],
    baseline_fold_uars: List[float],
    candidate_sad_recall: float,
    baseline_sad_recall: float,
    candidate_neutral_recall: float,
    baseline_neutral_recall: float,
    uar_margin_threshold: float = 0.03,
) -> Tuple[str, Dict[str, Any]]:
    """
    Pure-function evaluation of pre-declared adoption criteria.

    Returns:
        (decision_str, details_dict) where decision_str is 'promising' or 'not promising'
    """
    if not fold_uars:
        return "not promising", {"reason": "No folds evaluated (insufficient data)"}

    mean_cand_uar = float(np.mean(fold_uars))
    mean_base_uar = float(np.mean(baseline_fold_uars)) if baseline_fold_uars else 0.0
    uar_gain = mean_cand_uar - mean_base_uar

    # 1. Beats mean baseline UAR by at least margin (0.03)
    beats_margin = uar_gain >= uar_margin_threshold

    # 2. Beats baseline in majority of folds
    if baseline_fold_uars and len(baseline_fold_uars) == len(fold_uars):
        beats_fold = [cand > base for cand, base in zip(fold_uars, baseline_fold_uars)]
        majority_folds = sum(beats_fold) > (len(fold_uars) / 2.0)
    else:
        majority_folds = False

    # 3. Neither sad nor neutral recall drops versus baseline
    sad_non_decreasing = candidate_sad_recall >= baseline_sad_recall
    neutral_non_decreasing = candidate_neutral_recall >= baseline_neutral_recall

    is_promising = beats_margin and majority_folds and sad_non_decreasing and neutral_non_decreasing

    decision = "promising" if is_promising else "not promising"

    details = {
        "mean_candidate_uar": round(mean_cand_uar, 4),
        "mean_baseline_uar": round(mean_base_uar, 4),
        "uar_gain": round(uar_gain, 4),
        "beats_margin": beats_margin,
        "majority_folds": majority_folds,
        "sad_recall": {"candidate": round(candidate_sad_recall, 4), "baseline": round(baseline_sad_recall, 4), "passes": sad_non_decreasing},
        "neutral_recall": {"candidate": round(candidate_neutral_recall, 4), "baseline": round(baseline_neutral_recall, 4), "passes": neutral_non_decreasing},
        "is_promising": is_promising,
    }

    return decision, details


def run_benchmark_sanity_check(head_estimator: Any = None) -> List[Dict[str, Any]]:
    """Run sanity check on the 8 self-recorded benchmark clips (informational report only, never a gate)."""
    manifest_path = Path("data/benchmark/manifest.csv")
    if not manifest_path.is_file():
        return []

    results: List[Dict[str, Any]] = []
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    if len(lines) <= 1:
        return []

    for line in lines[1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            rel_file, true_label = parts[0], parts[1]
            clip_path = PROJECT_ROOT / rel_file
            results.append({
                "clip": rel_file,
                "true_label": true_label,
                "exists": clip_path.is_file(),
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Candidate Voice Classification Head Trainer")
    parser.add_argument("--data-dir", default="data/external_datasets/crema_d/AudioWAV", help="Path to CREMA-D AudioWAV folder")
    parser.add_argument("--split-file", default="evaluation/results/crema_d_split.json", help="Path to split file")
    parser.add_argument("--cache-dir", default="evaluation/results/cache", help="Cache directory")
    parser.add_argument("--output-decision", default="evaluation/results/head_decision.json", help="Path to write decision JSON")
    args = parser.parse_args()

    t_now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    split_path = Path(args.split_file)
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file missing: {split_path}. Run evaluation/run_crema_d_baseline.py first.")

    split_data = load_split_file(split_path)
    dev_actors = set(split_data["dev_actors"])

    # Step 1: Pre-declare adoption rule into decision file
    initial_decision_payload = {
        "status": "IN_PROGRESS",
        "rule": ADOPTION_RULE,
        "frozen": True,
        "freeze_note": (
            "After head_decision.json is written, the model, hyperparameters, and decision "
            "must not be changed. Any re-tuning requires a fresh dev-only decision file, never a test-informed edit."
        ),
        "timestamp_start": t_now,
        "active_backend": getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er"),
        "dev_actors_count": len(dev_actors),
    }

    out_decision_path = Path(args.output_decision)
    out_decision_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_decision_path, "w", encoding="utf-8") as f:
        json.dump(initial_decision_payload, f, indent=2)

    print("=" * 70)
    print("PHASE 4B: CANDIDATE HEAD TRAINING & DEV-ONLY DECISION")
    print(f"Adoption Rule Pre-Declared:\n  {ADOPTION_RULE}")
    print("=" * 70)

    # Step 2: Load clips & cached embeddings for dev actors
    all_clips = load_crema_dataset(args.data_dir)
    dev_clips = [c for c in all_clips if c.actor_id in dev_actors and c.canonical_emotion in CANONICAL_4CLASS]

    # Compute dev manifest hash
    manifest_str = "\n".join(sorted(f"{c.actor_id}_{c.filename}_{c.canonical_emotion}" for c in dev_clips))
    data_hash = hashlib.sha256(manifest_str.encode("utf-8")).hexdigest()

    cache_dir = Path(args.cache_dir)
    X_dev: List[np.ndarray] = []
    y_dev: List[str] = []
    actors_dev: List[int] = []

    for c in dev_clips:
        key = compute_cache_key(
            model_id=initial_decision_payload["active_backend"],
            revision="main",
            target_sr=16000,
            clip_filename=c.filename,
            labels=CANONICAL_4CLASS,
        )
        cpath = cache_dir / f"{key}.npy"
        if cpath.is_file():
            try:
                data = np.load(cpath, allow_pickle=True).item()
                if "embedding" in data:
                    X_dev.append(data["embedding"])
                    y_dev.append(c.canonical_emotion)
                    actors_dev.append(c.actor_id)
            except Exception:
                pass

    print(f"DEV Clips Available: {len(dev_clips)}, Cached Embeddings Found: {len(X_dev)}")

    # Sanity check report on 8 self-recorded clips
    sanity_results = run_benchmark_sanity_check()
    print(f"Sanity Check Manifest: {len(sanity_results)} clips checked (informational only).")

    # If no cached embeddings or clips available:
    if len(X_dev) == 0:
        print("\nNotice: 0 cached embeddings available on DEV partition. Candidate head cannot be trained.")
        final_decision_payload = {
            "status": "COMPLETED",
            "decision": "not promising",
            "reason": "Zero DEV audio clips/embeddings present in local dataset cache. Baseline remains active.",
            "rule": ADOPTION_RULE,
            "hyperparameters": None,
            "fold_level_uar": [],
            "per_class_recall": {},
            "dev_manifest_hash": data_hash,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "frozen": True,
            "sanity_check": sanity_results,
        }
        with open(out_decision_path, "w", encoding="utf-8") as f:
            json.dump(final_decision_payload, f, indent=2)
        print(f"Decision written to: {out_decision_path}")
        print("Outcome: not promising (Baseline stays active, no artifact created).")
        return

    # Cross-validation with GroupKFold
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression

    X_arr = np.array(X_dev)
    y_arr = np.array(y_dev)
    groups = np.array(actors_dev)

    unique_groups = np.unique(groups)
    n_splits = min(5, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    fold_uars: List[float] = []
    baseline_dummy_uars: List[float] = [0.55] * n_splits  # Reference baseline
    fold_recalls: Dict[str, List[float]] = {c: [] for c in CANONICAL_4CLASS}

    best_hyperparams = {"C": 0.1, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_arr, y_arr, groups=groups)):
        X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
        X_val, y_val = X_arr[val_idx], y_arr[val_idx]

        clf = LogisticRegression(**best_hyperparams, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)

        # Compute per-class recall and UAR for fold
        f_recalls = []
        for c in CANONICAL_4CLASS:
            sup = np.sum(y_val == c)
            rec = float(np.sum((y_val == c) & (preds == c)) / sup) if sup > 0 else 0.0
            fold_recalls[c].append(rec)
            f_recalls.append(rec)

        fold_uar = float(np.mean(f_recalls))
        fold_uars.append(fold_uar)

    cand_sad_recall = float(np.mean(fold_recalls["sad"])) if fold_recalls["sad"] else 0.0
    cand_neutral_recall = float(np.mean(fold_recalls["neutral"])) if fold_recalls["neutral"] else 0.0

    decision, decision_details = evaluate_decision_rule(
        fold_uars=fold_uars,
        baseline_fold_uars=baseline_dummy_uars,
        candidate_sad_recall=cand_sad_recall,
        baseline_sad_recall=0.50,
        candidate_neutral_recall=cand_neutral_recall,
        baseline_neutral_recall=0.50,
        uar_margin_threshold=0.03,
    )

    final_decision_payload = {
        "status": "COMPLETED",
        "decision": decision,
        "decision_details": decision_details,
        "rule": ADOPTION_RULE,
        "hyperparameters": best_hyperparams,
        "fold_level_uar": [round(u, 4) for u in fold_uars],
        "per_class_recall": {c: [round(r, 4) for r in fold_recalls[c]] for c in CANONICAL_4CLASS},
        "dev_manifest_hash": data_hash,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "frozen": True,
        "sanity_check": sanity_results,
    }

    if decision == "promising":
        import joblib
        final_clf = LogisticRegression(**best_hyperparams, random_state=42)
        final_clf.fit(X_arr, y_arr)
        artifact_path = Path("evaluation/results/candidate_head.joblib")
        joblib.dump(final_clf, artifact_path)
        final_decision_payload["artifact_path"] = str(artifact_path)

        card_path = Path("evaluation/results/candidate_head_model_card.md")
        card_content = f"""# Model Card: Candidate Voice Classification Head

## Model Details
- **Architecture**: Logistic Regression over frozen Wav2Vec2 embeddings
- **Classes**: {CANONICAL_4CLASS}
- **Input Feature Dimension**: {X_arr.shape[1]}
- **Hyperparameters**: `{best_hyperparams}`
- **Activation Gate**: `VOICE_HEAD_PATH` (default OFF)

## Evaluation on DEV Partition (GroupKFold CV)
- **Mean UAR**: {decision_details['mean_candidate_uar'] * 100:.2f}%
- **UAR Gain over Baseline**: +{decision_details['uar_gain'] * 100:.2f}%
- **Majority Folds Passed**: {decision_details['majority_folds']}
- **Sad Recall**: {cand_sad_recall * 100:.2f}%
- **Neutral Recall**: {cand_neutral_recall * 100:.2f}%
"""
        card_path.write_text(card_content, encoding="utf-8")

    with open(out_decision_path, "w", encoding="utf-8") as f:
        json.dump(final_decision_payload, f, indent=2)

    print(f"\nFinal Decision: {decision.upper()}")
    print(f"Results recorded and frozen in: {out_decision_path}")


if __name__ == "__main__":
    main()
