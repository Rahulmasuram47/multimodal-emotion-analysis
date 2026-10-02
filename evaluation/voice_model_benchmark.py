"""
Phase 3 — Voice Emotion Model Benchmark.

Compares candidate Wav2Vec2-family speech-emotion checkpoints against the
current production model, using whatever ground-truth voice-emotion labels
exist in data/evaluation_dataset.csv (via evaluation/dataset.py).

Run:
    python -m evaluation.voice_model_benchmark

This script NEVER changes config.py or production behavior on its own —
it only reports numbers. Changing config.AUDIO_EMOTION_MODEL_NAME is a
manual, evidence-based decision made after reading this report.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

import config
from app.audio_processor import load_and_preprocess_audio
from evaluation.dataset import load_evaluation_dataset, EvaluationSample

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

TARGET_CLASSES = config.AUDIO_EMOTION_CLASSES  # ['neutral', 'happy', 'angry', 'sad']

# ─────────────────────────────────────────────────────────────────────────
# Candidate models. Each entry:
#   model_id     : HuggingFace repo id
#   label_map    : maps the MODEL's own output labels -> one of TARGET_CLASSES.
#                  A label mapped to None is dropped from consideration
#                  (its probability mass is redistributed proportionally).
#   notes        : what to manually verify on the model's HF page before
#                  trusting this benchmark (license, size, training data).
#
# VERIFY BEFORE TRUSTING RESULTS: open each model's HuggingFace page and
# confirm license and training-data overlap with your own eval clips.
# ─────────────────────────────────────────────────────────────────────────
CANDIDATE_MODELS: List[Dict[str, Any]] = [
    {
        "model_id": config.AUDIO_EMOTION_MODEL_NAME,  # current production baseline
        "label_map": {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
                      "neutral": "neutral", "happy": "happy", "angry": "angry"},
        "notes": "Current production model (superb/wav2vec2-base-superb-er). Baseline.",
    },
    {
        "model_id": "superb/wav2vec2-large-superb-er",
        "label_map": {"neu": "neutral", "hap": "happy", "ang": "angry", "sad": "sad",
                      "neutral": "neutral", "happy": "happy", "angry": "angry"},
        "notes": "Same SUPERB 4-class label set as baseline, larger model. "
                 "VERIFY: license (apache-2.0 expected), model size (~1.2GB vs ~360MB baseline), "
                 "CPU latency may be notably higher — check before adopting.",
    },
    {
        "model_id": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        "label_map": {
            "angry": "angry", "happy": "happy", "neutral": "neutral", "sad": "sad",
            "calm": None, "disgust": None, "fearful": None, "surprised": None,
        },
        "notes": "8-class RAVDESS-trained model, remapped to our 4 classes "
                 "(calm/disgust/fearful/surprised dropped, mass redistributed). "
                 "VERIFY: license and model size on HF page before adopting.",
    },
]


@dataclass
class BenchmarkResult:
    model_id: str
    n_samples: int = 0
    n_correct: int = 0
    total_latency_seconds: float = 0.0
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    per_class_correct: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    load_error: Optional[str] = None

    @property
    def accuracy(self) -> Optional[float]:
        if self.n_samples == 0:
            return None
        return self.n_correct / self.n_samples

    @property
    def avg_latency_seconds(self) -> Optional[float]:
        if self.n_samples == 0:
            return None
        return self.total_latency_seconds / self.n_samples

    def macro_f1(self) -> Optional[float]:
        if self.n_samples == 0:
            return None
        f1_scores = []
        for cls in TARGET_CLASSES:
            tp = self.confusion.get(cls, {}).get(cls, 0)
            fp = sum(self.confusion.get(other, {}).get(cls, 0) for other in TARGET_CLASSES if other != cls)
            fn = sum(self.confusion.get(cls, {}).get(other, 0) for other in TARGET_CLASSES if other != cls)
            if tp + fp == 0 or tp + fn == 0:
                if self.per_class_support.get(cls, 0) == 0:
                    continue  # class not present in this eval set at all
                f1_scores.append(0.0)
                continue
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            if precision + recall == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * precision * recall / (precision + recall))
        return sum(f1_scores) / len(f1_scores) if f1_scores else None


def _remap_scores(raw_scores: Dict[str, float], label_map: Dict[str, Optional[str]]) -> Dict[str, float]:
    """Remap a model's raw label->probability dict onto TARGET_CLASSES.

    Labels mapped to None are dropped; the remaining probabilities are
    renormalized so they still sum to 1.0 (approximately), preserving
    relative confidence rather than just taking argmax pre-remap.
    """
    remapped: Dict[str, float] = {cls: 0.0 for cls in TARGET_CLASSES}
    for raw_label, prob in raw_scores.items():
        target = label_map.get(raw_label.lower())
        if target is not None and target in remapped:
            remapped[target] += prob
    total = sum(remapped.values())
    if total > 0:
        remapped = {k: v / total for k, v in remapped.items()}
    return remapped


def _resolve_audio_path(filename: str) -> Optional[Path]:
    """Find the actual audio file for a dataset row's filename.

    Checks sample_audio/ and recordings/ (config.RECORDINGS_DIR), in that
    order, since annotated clips may live in either.
    """
    candidates = [
        Path(config.SAMPLE_AUDIO_DIR) / filename,
        Path(config.RECORDINGS_DIR) / filename,
        Path(filename),  # already a full/relative path
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def benchmark_model(model_spec: Dict[str, Any], samples: List[EvaluationSample]) -> BenchmarkResult:
    model_id = model_spec["model_id"]
    label_map = model_spec["label_map"]
    result = BenchmarkResult(model_id=model_id)

    print(f"\n{'=' * 70}\nLoading: {model_id}\n{model_spec['notes']}\n{'=' * 70}")

    try:
        load_start = time.time()
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        model.eval()
        load_time = time.time() - load_start
        print(f"Loaded in {load_time:.1f}s. Model labels: {list(model.config.id2label.values())}")
    except Exception as exc:
        result.load_error = f"{type(exc).__name__}: {exc}"
        print(f"FAILED TO LOAD: {result.load_error}")
        return result

    for sample in samples:
        audio_path = _resolve_audio_path(sample.filename)
        if audio_path is None:
            result.errors.append(f"{sample.filename}: audio file not found")
            continue

        true_label = sample.true_voice_emotion
        if true_label not in TARGET_CLASSES:
            continue  # unrecognized/blank label, already filtered upstream but double-check

        try:
            audio, sr, _duration = load_and_preprocess_audio(str(audio_path))
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)

            t0 = time.time()
            inputs = extractor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
            elapsed = time.time() - t0

            raw_scores = {model.config.id2label[i]: float(probs[i]) for i in range(len(probs))}
            remapped = _remap_scores(raw_scores, label_map)
            predicted = max(remapped, key=remapped.get)

            result.n_samples += 1
            result.total_latency_seconds += elapsed
            result.per_class_support[true_label] = result.per_class_support.get(true_label, 0) + 1
            result.confusion.setdefault(true_label, {})
            result.confusion[true_label][predicted] = result.confusion[true_label].get(predicted, 0) + 1

            if predicted == true_label:
                result.n_correct += 1
                result.per_class_correct[true_label] = result.per_class_correct.get(true_label, 0) + 1

        except Exception as exc:
            result.errors.append(f"{sample.filename}: {type(exc).__name__}: {exc}")

    return result


def print_report(results: List[BenchmarkResult]) -> None:
    print(f"\n{'=' * 70}\nBENCHMARK SUMMARY\n{'=' * 70}")
    print(f"{'Model':<55} {'N':>4} {'Acc':>7} {'MacroF1':>8} {'Latency/clip':>14}")
    print("-" * 92)
    for r in results:
        if r.load_error:
            print(f"{r.model_id:<55} LOAD FAILED: {r.load_error}")
            continue
        acc_str = f"{r.accuracy:.1%}" if r.accuracy is not None else "N/A"
        f1 = r.macro_f1()
        f1_str = f"{f1:.3f}" if f1 is not None else "N/A"
        lat_str = f"{r.avg_latency_seconds:.2f}s" if r.avg_latency_seconds is not None else "N/A"
        print(f"{r.model_id:<55} {r.n_samples:>4} {acc_str:>7} {f1_str:>8} {lat_str:>14}")

    print(f"\n{'=' * 70}\nPER-CLASS BREAKDOWN\n{'=' * 70}")
    for r in results:
        if r.load_error or r.n_samples == 0:
            continue
        print(f"\n{r.model_id}:")
        for cls in TARGET_CLASSES:
            support = r.per_class_support.get(cls, 0)
            correct = r.per_class_correct.get(cls, 0)
            if support == 0:
                print(f"  {cls:<10} — no samples in eval set")
                continue
            print(f"  {cls:<10} {correct}/{support} correct ({correct/support:.1%})")
        if r.errors:
            print(f"  Errors ({len(r.errors)}): {r.errors[:3]}{'...' if len(r.errors) > 3 else ''}")

    if results and not any(r.load_error for r in results):
        min_n = min(r.n_samples for r in results if r.n_samples > 0)
        if min_n < config.EVAL_MIN_N:
            print(f"\n⚠️  WARNING: smallest evaluated N = {min_n}, below EVAL_MIN_N = {config.EVAL_MIN_N}. "
                  f"These numbers are indicative only — do not treat them as a reliable comparison "
                  f"until more clips are annotated. Do NOT report these as final accuracy in your project report.")


def main() -> None:
    dataset = load_evaluation_dataset()
    samples = dataset.filter_with_voice_emotion()

    print(f"Loaded {len(dataset)} total dataset rows, {len(samples)} with a voice-emotion label.")
    if not samples:
        print("No voice-emotion-labeled samples found. Nothing to benchmark. "
              "Use the Annotation Helper tab to label some clips first.")
        return

    results = [benchmark_model(spec, samples) for spec in CANDIDATE_MODELS]
    print_report(results)

    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "voice_model_benchmark.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        import contextlib, io as _io
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_report(results)
        f.write(buf.getvalue())
    print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()