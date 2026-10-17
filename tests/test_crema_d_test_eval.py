"""
Unit Tests for Phase 4C: One-Shot Test-Split Evaluation.
Mock models and synthetic data only; no network, dataset, or HF token required.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run_crema_d_test_evaluation import (
    build_test_benchmark_markdown,
    compute_paired_actor_cluster_bootstrap,
    compute_test_metrics,
)


class TestCremaDTestEval(unittest.TestCase):
    def test_01_paired_actor_cluster_bootstrap(self):
        """Verify paired actor-cluster bootstrap CI calculation on synthetic paired data."""
        cand_recs = []
        base_recs = []
        # 10 actors, 4 clips each
        for aid in range(1001, 1011):
            for i in range(4):
                true_label = "happy" if (aid + i) % 2 == 0 else "sad"
                # Candidate gets 80% correct, baseline gets 50% correct
                cand_pred = true_label if i != 0 else "neutral"
                base_pred = true_label if i > 1 else "neutral"
                cand_recs.append({"actor_id": aid, "canonical_emotion": true_label, "canonical_pred": cand_pred})
                base_recs.append({"actor_id": aid, "canonical_emotion": true_label, "canonical_pred": base_pred})

        res = compute_paired_actor_cluster_bootstrap(cand_recs, base_recs, n_bootstrap=100, seed=42)
        self.assertEqual(res["n_clusters"], 10)
        self.assertIsNotNone(res["mean_diff"])
        self.assertIsNotNone(res["ci_lower"])
        self.assertIsNotNone(res["ci_upper"])
        self.assertLessEqual(res["ci_lower"], res["mean_diff"])
        self.assertGreaterEqual(res["ci_upper"], res["mean_diff"])

    def test_02_compute_test_metrics(self):
        """Verify standard multiclass metrics and highlight markers for sad and neutral."""
        recs = [
            {"actor_id": 1001, "canonical_emotion": "neutral", "canonical_pred": "neutral"},
            {"actor_id": 1001, "canonical_emotion": "sad", "canonical_pred": "sad"},
            {"actor_id": 1002, "canonical_emotion": "angry", "canonical_pred": "angry"},
            {"actor_id": 1002, "canonical_emotion": "happy", "canonical_pred": "neutral"},
        ]
        metrics = compute_test_metrics(recs, seed=42)
        self.assertEqual(metrics["n_samples"], 4)
        self.assertEqual(metrics["n_actors"], 2)
        self.assertAlmostEqual(metrics["accuracy"], 0.75, places=2)
        self.assertTrue(metrics["per_class"]["sad"]["is_highlighted"])
        self.assertTrue(metrics["per_class"]["neutral"]["is_highlighted"])
        self.assertFalse(metrics["per_class"]["angry"]["is_highlighted"])

    def test_03_benchmark_markdown_required_phrases(self):
        """Verify that generated benchmark markdown includes required caveats and metadata."""
        meta = {
            "active_backend": "superb",
            "active_model_id": "superb/wav2vec2-base-superb-er",
            "emotion2vec_note": "Unavailable (missing funasr)",
            "data_dir": "data/external_datasets/crema_d/AudioWAV",
            "license_status": "ODbL v1.0",
            "seed": 42,
            "head_adopted": False,
            "sampling_note": "Capped at 100 clips per class (seeded sampling).",
            "corpus_nature": "acted speech, actor-disjoint split, not natural speech",
        }
        empty_metrics = {"n_samples": 0}
        report = build_test_benchmark_markdown(empty_metrics, None, None, meta)
        self.assertIn("acted speech, actor-disjoint split, not natural speech", report)
        self.assertIn("superb/wav2vec2-base-superb-er", report)
        self.assertIn("Not Adopted", report)


if __name__ == "__main__":
    unittest.main()
