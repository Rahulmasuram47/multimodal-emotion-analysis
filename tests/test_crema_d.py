"""
Unit Tests for CREMA-D loader, splitting, cluster bootstrap, caching, and evaluation lock.
Mock models in unit tests; tests need no real dataset, network, or HF token.
"""
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.crema_d import (
    CANONICAL_4CLASS,
    CREMA_RAW_EMOTIONS,
    CREMA_TO_SUPERB_MAP,
    check_dataset_license,
    compute_actor_cluster_bootstrap,
    compute_cache_key,
    enforce_test_evaluation_lock,
    generate_actor_split,
    parse_crema_filename,
)


class TestCremaDSuite(unittest.TestCase):
    def test_01_filename_parsing(self):
        """Verify filename parsing against CREMA-D naming conventions."""
        # Standard valid filenames
        meta1 = parse_crema_filename("1001_DFA_ANG_XX.wav")
        self.assertIsNotNone(meta1)
        self.assertEqual(meta1["actor_id"], 1001)
        self.assertEqual(meta1["sentence"], "DFA")
        self.assertEqual(meta1["raw_emotion"], "ANG")
        self.assertEqual(meta1["level"], "XX")
        self.assertEqual(meta1["canonical_emotion"], "angry")
        self.assertTrue(meta1["is_4class_subset"])

        meta2 = parse_crema_filename("1091_IWW_SAD_HI.wav")
        self.assertIsNotNone(meta2)
        self.assertEqual(meta2["actor_id"], 1091)
        self.assertEqual(meta2["raw_emotion"], "SAD")
        self.assertEqual(meta2["canonical_emotion"], "sad")

        meta3 = parse_crema_filename("1042_TIE_FEA_MD.wav")
        self.assertIsNotNone(meta3)
        self.assertEqual(meta3["raw_emotion"], "FEA")
        self.assertEqual(meta3["canonical_emotion"], "other")
        self.assertFalse(meta3["is_4class_subset"])

        # Invalid filenames
        self.assertIsNone(parse_crema_filename("random_file.wav"))
        self.assertIsNone(parse_crema_filename("0999_DFA_ANG_XX.wav"))  # actor < 1001
        self.assertIsNone(parse_crema_filename("1092_DFA_ANG_XX.wav"))  # actor > 1091
        self.assertIsNone(parse_crema_filename("1001_DFA_XYZ_XX.wav"))  # invalid emotion
        self.assertIsNone(parse_crema_filename("1001_DFA_ANG_ZZ.wav"))  # invalid level
        self.assertIsNone(parse_crema_filename("1001_DFA_ANG_XX.mp3"))  # not wav

    def test_02_label_mapping(self):
        """Verify explicit mapping from 6-class raw emotions to active backend canonical labels."""
        self.assertEqual(len(CREMA_RAW_EMOTIONS), 6)
        for raw in ["ANG", "HAP", "NEU", "SAD", "DIS", "FEA"]:
            self.assertIn(raw, CREMA_TO_SUPERB_MAP)

        self.assertEqual(CREMA_TO_SUPERB_MAP["ANG"], "angry")
        self.assertEqual(CREMA_TO_SUPERB_MAP["HAP"], "happy")
        self.assertEqual(CREMA_TO_SUPERB_MAP["NEU"], "neutral")
        self.assertEqual(CREMA_TO_SUPERB_MAP["SAD"], "sad")
        self.assertEqual(CREMA_TO_SUPERB_MAP["DIS"], "other")
        self.assertEqual(CREMA_TO_SUPERB_MAP["FEA"], "other")

        for cl in CANONICAL_4CLASS:
            self.assertIn(cl, ["neutral", "happy", "angry", "sad"])

    def test_03_zero_actor_overlap_split(self):
        """Verify actor split partitions 91 actors into ~60 dev and 31 test with 0 overlap."""
        split = generate_actor_split(seed=42, n_dev=60)
        self.assertEqual(split["n_total_actors"], 91)
        self.assertEqual(split["n_dev_actors"], 60)
        self.assertEqual(split["n_test_actors"], 31)

        dev_set = set(split["dev_actors"])
        test_set = set(split["test_actors"])
        self.assertEqual(len(dev_set), 60)
        self.assertEqual(len(test_set), 31)
        self.assertEqual(len(dev_set.intersection(test_set)), 0, "Dev and test actors must have zero overlap")
        self.assertEqual(dev_set.union(test_set), set(range(1001, 1092)))

    def test_04_toy_cluster_bootstrap(self):
        """Verify actor-cluster bootstrap CI calculation on toy synthetic data."""
        # Create synthetic records clustered across 10 actors
        records = []
        for aid in range(1001, 1011):
            for i in range(5):
                records.append({
                    "actor_id": aid,
                    "canonical_emotion": "happy" if (aid + i) % 2 == 0 else "sad",
                    "canonical_pred": "happy" if (aid + i) % 2 == 0 else "neutral",
                })

        def dummy_accuracy(recs):
            c = sum(1 for r in recs if r["canonical_emotion"] == r["canonical_pred"])
            return c / len(recs) if recs else 0.0

        ci = compute_actor_cluster_bootstrap(records, dummy_accuracy, n_bootstrap=100, seed=42)
        self.assertEqual(ci["n_clusters"], 10)
        self.assertIsNotNone(ci["mean"])
        self.assertIsNotNone(ci["ci_lower"])
        self.assertIsNotNone(ci["ci_upper"])
        self.assertLessEqual(ci["ci_lower"], ci["mean"])
        self.assertGreaterEqual(ci["ci_upper"], ci["mean"])

    def test_05_stable_cache_key(self):
        """Verify deterministic generation of SHA-256 cache keys."""
        key1 = compute_cache_key(
            model_id="superb/wav2vec2-base-superb-er",
            revision="main",
            target_sr=16000,
            clip_filename="1001_DFA_ANG_XX.wav",
            labels=["angry", "happy", "neutral", "sad"],
        )
        key2 = compute_cache_key(
            model_id="superb/wav2vec2-base-superb-er",
            revision="main",
            target_sr=16000,
            clip_filename="1001_DFA_ANG_XX.wav",
            labels=["sad", "angry", "neutral", "happy"],  # Different label order
        )
        self.assertEqual(key1, key2, "Cache key must be deterministic and invariant to label argument ordering")
        self.assertEqual(len(key1), 64, "Key should be valid SHA-256 hex string")

    def test_06_test_metrics_refusal_when_decision_missing(self):
        """Verify that attempting to evaluate test split without head_decision.json raises RuntimeError."""
        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                enforce_test_evaluation_lock("non_existent_head_decision.json")
            self.assertIn("strictly locked and refused", str(ctx.exception))

    def test_07_inspect_voice_backends(self):
        """Verify inspect_voice_backends returns valid schema and config matching."""
        from evaluation.run_crema_d_baseline import inspect_voice_backends
        info = inspect_voice_backends()
        self.assertIn("configured_backend", info)
        self.assertIn("active_backend", info)
        self.assertIn("active_model_id", info)
        self.assertIn(info["active_backend"], ["superb", "emotion2vec"])

    def test_08_run_emotion2vec_inference_mocked(self):
        """Verify run_emotion2vec_inference_on_clip extracts scores and embeddings with mocked backend."""
        from evaluation.run_crema_d_baseline import run_emotion2vec_inference_on_clip
        from unittest.mock import MagicMock
        import numpy as np

        mock_backend = MagicMock()
        mock_backend.predict.return_value = {
            "emotion": "happy",
            "confidence": 0.65,
            "raw_confidence": 0.99,
            "scores": {"angry": 0.0, "happy": 1.0, "neutral": 0.0, "sad": 0.0},
            "raw_label": "开心/happy",
        }
        mock_backend._model.generate.return_value = [
            {"feats": np.ones((768,), dtype=np.float32), "labels": ["开心/happy"], "scores": [0.99]}
        ]

        dummy_audio = np.zeros(16000, dtype=np.float32)
        with patch("soundfile.read", return_value=(dummy_audio, 16000)):
            res = run_emotion2vec_inference_on_clip("dummy.wav", mock_backend)

        self.assertEqual(res["canonical_pred"], "happy")
        self.assertEqual(res["raw_label"], "开心/happy")
        self.assertAlmostEqual(res["confidence"], 0.65)
        self.assertIsNotNone(res["embedding"])
        self.assertEqual(res["embedding"].shape, (768,))


if __name__ == "__main__":
    unittest.main()
