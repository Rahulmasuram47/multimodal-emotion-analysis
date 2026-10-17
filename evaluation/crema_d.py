"""
CREMA-D dataset loader, actor-level splitter, and evaluation utilities.

Dataset specifications:
- AudioWAV contains recordings of 91 actors (IDs 1001-1091).
- Filename convention: <ActorID>_<Sentence>_<Emotion>_<Level>.wav
- Emotions: ANG (Anger), DIS (Disgust), FEA (Fear), HAP (Happiness), NEU (Neutral), SAD (Sadness)
- Intensity Levels: LO (Low), MD (Medium), HI (High), XX (Unspecified / Neutral)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Valid CREMA-D metadata vocabulary
CREMA_RAW_EMOTIONS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
CREMA_LEVELS = ["LO", "MD", "HI", "XX"]

# Explicit mapping from CREMA-D 6-class raw emotions to active voice model canonical labels
# Active model: superb/wav2vec2-base-superb-er (classes: neu, hap, ang, sad)
CREMA_TO_SUPERB_MAP: Dict[str, str] = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
    "DIS": "other",
    "FEA": "other",
}

# 4-class collapsed subset directly evaluated by SUPERB ER
CANONICAL_4CLASS = ["neutral", "happy", "angry", "sad"]

# Regex for CREMA-D audio filenames: e.g. 1001_DFA_ANG_XX.wav
CREMA_FILENAME_PATTERN = re.compile(
    r"^(?P<actor_id>\d{4})_(?P<sentence>[A-Za-z0-9]+)_(?P<emotion>[A-Z]{3})_(?P<level>[A-Z]{2,3})\.wav$",
    re.IGNORECASE,
)


@dataclass
class CremaClip:
    filename: str
    filepath: str
    actor_id: int
    sentence: str
    raw_emotion: str
    level: str
    canonical_emotion: str
    is_4class_subset: bool


def parse_crema_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Parse a CREMA-D audio filename into its constituent metadata components."""
    base = os.path.basename(filename)
    match = CREMA_FILENAME_PATTERN.match(base)
    if not match:
        return None

    actor_str, sentence, emotion, level = (
        match.group("actor_id"),
        match.group("sentence").upper(),
        match.group("emotion").upper(),
        match.group("level").upper(),
    )

    actor_id = int(actor_str)
    if not (1001 <= actor_id <= 1091):
        return None

    if emotion not in CREMA_RAW_EMOTIONS:
        return None

    if level not in CREMA_LEVELS:
        return None

    canonical = CREMA_TO_SUPERB_MAP.get(emotion, "other")
    is_4class = canonical in CANONICAL_4CLASS

    return {
        "filename": base,
        "actor_id": actor_id,
        "sentence": sentence,
        "raw_emotion": emotion,
        "level": level,
        "canonical_emotion": canonical,
        "is_4class_subset": is_4class,
    }


def load_crema_dataset(data_dir: str | Path) -> List[CremaClip]:
    """Scan data directory and load all valid CREMA-D audio clip metadata."""
    dir_path = Path(data_dir)
    clips: List[CremaClip] = []
    if not dir_path.is_dir():
        logger.warning("CREMA-D audio directory not found: %s", dir_path)
        return clips

    for fname in sorted(os.listdir(dir_path)):
        if not fname.lower().endswith(".wav"):
            continue
        parsed = parse_crema_filename(fname)
        if parsed:
            clips.append(
                CremaClip(
                    filename=parsed["filename"],
                    filepath=str(dir_path / parsed["filename"]),
                    actor_id=parsed["actor_id"],
                    sentence=parsed["sentence"],
                    raw_emotion=parsed["raw_emotion"],
                    level=parsed["level"],
                    canonical_emotion=parsed["canonical_emotion"],
                    is_4class_subset=parsed["is_4class_subset"],
                )
            )

    return clips


def generate_actor_split(
    actors: Optional[Sequence[int]] = None,
    seed: int = 42,
    n_dev: int = 60,
) -> Dict[str, Any]:
    """Generate deterministic speaker-independent split by ACTOR (seed 42, ~60 dev actors)."""
    if actors is None:
        actor_list = list(range(1001, 1092))
    else:
        actor_list = sorted(list(set(actors)))

    rng = np.random.RandomState(seed)
    shuffled = list(actor_list)
    rng.shuffle(shuffled)

    dev_actors = sorted(shuffled[:n_dev])
    test_actors = sorted(shuffled[n_dev:])

    # Strict assertion for zero speaker overlap
    overlap = set(dev_actors).intersection(set(test_actors))
    if overlap:
        raise ValueError(f"Actor split has illegal overlap between dev and test: {overlap}")

    return {
        "seed": seed,
        "n_total_actors": len(actor_list),
        "n_dev_actors": len(dev_actors),
        "n_test_actors": len(test_actors),
        "dev_actors": dev_actors,
        "test_actors": test_actors,
    }


def save_split_file(split_data: Dict[str, Any], output_path: str | Path) -> None:
    """Save the train/dev/test split JSON to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split_data, f, indent=2)


def load_split_file(split_path: str | Path) -> Dict[str, Any]:
    """Load split JSON from disk, or raise error if missing."""
    p = Path(split_path)
    if not p.is_file():
        raise FileNotFoundError(f"Split file not found at: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_cache_key(
    model_id: str,
    revision: str,
    target_sr: int,
    clip_filename: str,
    labels: Sequence[str],
) -> str:
    """Generate a deterministic, stable SHA-256 cache key for model inference on a clip."""
    sorted_labels = ",".join(sorted(labels))
    key_str = f"{model_id}|{revision}|{target_sr}|{clip_filename}|{sorted_labels}"
    return hashlib.sha256(key_str.encode("utf-8")).hexdigest()


def check_dataset_license(dataset_dir: str | Path) -> str:
    """Check for local dataset license file or return explicit notice."""
    p = Path(dataset_dir)
    search_dirs = [p, p.parent] if p.is_dir() else [p.parent]
    license_files = ["LICENSE", "LICENSE.txt", "license.txt", "README.txt", "README.md"]
    for d in search_dirs:
        for lf in license_files:
            target = d / lf
            if target.is_file():
                try:
                    content = target.read_text(encoding="utf-8")
                    if "license" in content.lower() or "odbl" in content.lower():
                        return f"Found license file at {target}:\n\n{content[:1000]}"
                except Exception:
                    pass
    return (
        "Dataset license file is missing from data/external_datasets/crema_d/.\n"
        "(Note: CREMA-D is publicly distributed under the Open Database License (ODbL) v1.0 / "
        "Open Data Commons Open Database License by the CREMA-D consortium)."
    )


def enforce_test_evaluation_lock(decision_file: str | Path = "evaluation/results/head_decision.json") -> None:
    """
    Enforce that test split evaluation is strictly refused until Phase 4B creates head_decision.json.
    Raises RuntimeError if file is missing.
    """
    p = Path(decision_file)
    if not p.is_file():
        raise RuntimeError(
            f"Test-split evaluation is strictly locked and refused: '{p}' is missing. "
            "Evaluation on test actors can only proceed after the classification head decision is documented in Phase 4B."
        )


def compute_actor_cluster_bootstrap(
    records: List[Dict[str, Any]],
    metric_fn: Callable[[List[Dict[str, Any]]], float],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Optional[float]]:
    """
    Compute cluster bootstrap confidence intervals resampling at the ACTOR level.

    Instead of i.i.d. clip sampling, we resample clusters of clips grouped by actor_id
    to account for intra-speaker acoustic correlation.
    """
    if not records:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n_clusters": 0}

    # Group records by actor_id
    actor_clusters: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        aid = r["actor_id"]
        actor_clusters.setdefault(aid, []).append(r)

    unique_actors = list(actor_clusters.keys())
    n_actors = len(unique_actors)
    if n_actors < 2:
        val = metric_fn(records)
        return {"mean": float(val), "ci_lower": float(val), "ci_upper": float(val), "n_clusters": n_actors}

    rng = np.random.RandomState(seed)
    scores: List[float] = []

    for _ in range(n_bootstrap):
        sampled_actors = rng.choice(unique_actors, size=n_actors, replace=True)
        sampled_records: List[Dict[str, Any]] = []
        for aid in sampled_actors:
            sampled_records.extend(actor_clusters[aid])
        try:
            score = metric_fn(sampled_records)
            scores.append(score)
        except Exception:
            pass

    if not scores:
        return {"mean": None, "ci_lower": None, "ci_upper": None, "n_clusters": n_actors}

    ci_lower = float(np.percentile(scores, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(scores, 100 * (1.0 - alpha / 2.0)))
    mean_val = float(np.mean(scores))

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "n_clusters": n_actors,
    }
