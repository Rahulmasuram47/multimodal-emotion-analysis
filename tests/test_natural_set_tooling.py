"""
Unit Tests for Phase 6: Natural-Speech Evaluation Tooling.

Tests:
1. Known Cohen's kappa values (perfect agreement = 1.0, chance agreement ~ 0.0, negative kappa for inverted agreement).
2. Known Fleiss' kappa calculation.
3. Consensus resolution and tie handling (ties resolved as "disputed", unanimous detection).
4. Blinding verification in make_label_sheet.py (intent and model outputs strictly excluded).
"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.natural_set.make_label_sheet import generate_rater_sheets
from evaluation.natural_set.score import (
    compute_cohens_kappa,
    compute_fleiss_kappa,
    resolve_consensus,
)


class TestNaturalSetTooling(unittest.TestCase):
    def test_01_known_cohens_kappa_values(self):
        """Verify Cohen's kappa matches known theoretical values on deterministic test cases."""
        # Case A: Perfect agreement -> Kappa = 1.0
        r1_perf = ["happy", "sad", "neutral", "angry", "happy"]
        r2_perf = ["happy", "sad", "neutral", "angry", "happy"]
        res_perf = compute_cohens_kappa(r1_perf, r2_perf)
        self.assertEqual(res_perf["kappa"], 1.0)
        self.assertEqual(res_perf["raw_agreement"], 1.0)

        # Case B: Chance agreement / Independence -> Kappa ~ 0.0
        # Rater 1: alternating [A, B, A, B, ...]; Rater 2: [A, A, B, B, ...]
        r1_chance = ["happy"] * 50 + ["sad"] * 50
        r2_chance = (["happy"] * 25 + ["sad"] * 25) * 2
        res_chance = compute_cohens_kappa(r1_chance, r2_chance)
        self.assertAlmostEqual(res_chance["kappa"], 0.0, places=2)

        # Case C: Complete inverted disagreement -> Kappa < 0.0
        r1_inv = ["happy", "happy", "sad", "sad"]
        r2_inv = ["sad", "sad", "happy", "happy"]
        res_inv = compute_cohens_kappa(r1_inv, r2_inv)
        self.assertLess(res_inv["kappa"], 0.0)

    def test_02_fleiss_kappa_calculation(self):
        """Verify Fleiss' kappa on a known agreement matrix."""
        # Perfect agreement across 3 raters on 4 subjects
        matrix_perfect = np.array([
            [3, 0, 0],
            [0, 3, 0],
            [0, 0, 3],
            [3, 0, 0],
        ])
        res = compute_fleiss_kappa(matrix_perfect)
        self.assertEqual(res["kappa"], 1.0)
        self.assertEqual(res["p_observed"], 1.0)

    def test_03_consensus_and_tie_handling(self):
        """Verify consensus resolution rules: unanimous, majority, and ties -> 'disputed'."""
        # Unanimous agreement
        label_u, is_u = resolve_consensus(["happy", "happy", "happy"])
        self.assertEqual(label_u, "happy")
        self.assertTrue(is_u)

        # Majority agreement (2 vs 1)
        label_m, is_m = resolve_consensus(["happy", "happy", "sad"])
        self.assertEqual(label_m, "happy")
        self.assertFalse(is_m)

        # Exact tie (1 vs 1) -> must resolve to "disputed"
        label_tie2, is_tie2 = resolve_consensus(["happy", "sad"])
        self.assertEqual(label_tie2, "disputed")
        self.assertFalse(is_tie2)

        # Exact tie (2 vs 2 in 4 raters) -> "disputed"
        label_tie4, is_tie4 = resolve_consensus(["happy", "happy", "sad", "sad"])
        self.assertEqual(label_tie4, "disputed")
        self.assertFalse(is_tie4)

    def test_04_rater_sheet_blinding_and_shuffling(self):
        """Verify that rater sheets omit intent, omit model output, and are independently shuffled."""
        raw_clips = [
            {"clip_id": "spk01_001", "filename": "spk01_001.wav", "intent": "happy", "model_pred": "sad"},
            {"clip_id": "spk01_002", "filename": "spk01_002.wav", "intent": "angry", "model_pred": "angry"},
            {"clip_id": "spk02_001", "filename": "spk02_001.wav", "intent": "neutral", "model_pred": "neutral"},
            {"clip_id": "spk02_002", "filename": "spk02_002.wav", "intent": "sad", "model_pred": "happy"},
            {"clip_id": "spk03_001", "filename": "spk03_001.wav", "intent": "happy", "model_pred": "happy"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            sheets = generate_rater_sheets(raw_clips, n_raters=2, output_dir=tmpdir, base_seed=42)
            self.assertEqual(len(sheets), 2)

            # Inspect rater 1 sheet
            with open(sheets[0], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                self.assertNotIn("intent", headers, "Speaker intent must be strictly blinded")
                self.assertNotIn("model_pred", headers, "Model output must be strictly blinded")
                r1_rows = list(reader)

            # Inspect rater 2 sheet
            with open(sheets[1], "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                r2_rows = list(reader)

            # Check that row ordering differs (randomly shuffled per rater)
            r1_ids = [r["clip_id"] for r in r1_rows]
            r2_ids = [r["clip_id"] for r in r2_rows]
            self.assertEqual(set(r1_ids), set(r2_ids))
            self.assertNotEqual(r1_ids, r2_ids, "Rater sheets should be independently shuffled")


if __name__ == "__main__":
    unittest.main()
