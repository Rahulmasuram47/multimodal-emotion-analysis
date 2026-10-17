"""
Unit Tests for Phase 4B: Candidate Head Training & Dev-Only Adoption Decision.

Tests:
1. Grouped CV has zero actor leakage across folds.
2. Decision rule evaluation on toy fold scores across edge cases.
3. Candidate head loads only when VOICE_HEAD_PATH flag is set.
"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

import config
from evaluation.train_candidate_head import evaluate_decision_rule


class TestCandidateHeadSuite(unittest.TestCase):
    def test_01_grouped_cv_zero_actor_leakage(self):
        """Verify that GroupKFold partitions toy actors with zero leakage between train and val."""
        # 12 actors with 4 samples each
        actors = np.repeat(np.arange(1001, 1013), 4)
        X = np.random.randn(len(actors), 16)
        y = np.random.choice(["angry", "happy", "neutral", "sad"], size=len(actors))

        gkf = GroupKFold(n_splits=4)
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=actors)):
            train_actors = set(actors[train_idx])
            val_actors = set(actors[val_idx])
            overlap = train_actors.intersection(val_actors)
            self.assertEqual(
                len(overlap), 0,
                f"Fold {fold} has illegal actor leakage: {overlap}"
            )

    def test_02_decision_rule_logic(self):
        """Verify pre-declared adoption rule across passing and failing conditions."""
        # Case A: Passes all conditions (Gain +0.05, 4/5 folds pass, sad & neutral non-decreasing)
        cand_uars = [0.65, 0.62, 0.66, 0.64, 0.63]  # Mean: 0.64
        base_uars = [0.58, 0.60, 0.59, 0.61, 0.57]  # Mean: 0.59 (Gain: +0.05)
        dec_a, details_a = evaluate_decision_rule(
            fold_uars=cand_uars,
            baseline_fold_uars=base_uars,
            candidate_sad_recall=0.62,
            baseline_sad_recall=0.55,
            candidate_neutral_recall=0.60,
            baseline_neutral_recall=0.58,
        )
        self.assertEqual(dec_a, "promising")
        self.assertTrue(details_a["is_promising"])

        # Case B: Fails margin threshold (Gain +0.01 < 0.03)
        cand_uars_b = [0.60, 0.60, 0.60, 0.60, 0.60]
        base_uars_b = [0.59, 0.59, 0.59, 0.59, 0.59]
        dec_b, details_b = evaluate_decision_rule(
            fold_uars=cand_uars_b,
            baseline_fold_uars=base_uars_b,
            candidate_sad_recall=0.60,
            baseline_sad_recall=0.55,
            candidate_neutral_recall=0.60,
            baseline_neutral_recall=0.58,
        )
        self.assertEqual(dec_b, "not promising")
        self.assertFalse(details_b["beats_margin"])

        # Case C: Fails majority folds (only 2 of 5 folds beat baseline)
        cand_uars_c = [0.75, 0.75, 0.55, 0.55, 0.55]  # High average (+0.07 gain)
        base_uars_c = [0.56, 0.56, 0.58, 0.58, 0.58]
        dec_c, details_c = evaluate_decision_rule(
            fold_uars=cand_uars_c,
            baseline_fold_uars=base_uars_c,
            candidate_sad_recall=0.60,
            baseline_sad_recall=0.55,
            candidate_neutral_recall=0.60,
            baseline_neutral_recall=0.58,
        )
        self.assertEqual(dec_c, "not promising")
        self.assertFalse(details_c["majority_folds"])

        # Case D: Fails sad recall drop (sad recall drops from 0.60 to 0.50)
        dec_d, details_d = evaluate_decision_rule(
            fold_uars=cand_uars,
            baseline_fold_uars=base_uars,
            candidate_sad_recall=0.50,
            baseline_sad_recall=0.60,  # sad drops!
            candidate_neutral_recall=0.60,
            baseline_neutral_recall=0.58,
        )
        self.assertEqual(dec_d, "not promising")
        self.assertFalse(details_d["sad_recall"]["passes"])

        # Case E: Fails neutral recall drop (neutral recall drops from 0.60 to 0.52)
        dec_e, details_e = evaluate_decision_rule(
            fold_uars=cand_uars,
            baseline_fold_uars=base_uars,
            candidate_sad_recall=0.65,
            baseline_sad_recall=0.55,
            candidate_neutral_recall=0.52,
            baseline_neutral_recall=0.60,  # neutral drops!
        )
        self.assertEqual(dec_e, "not promising")
        self.assertFalse(details_e["neutral_recall"]["passes"])

    def test_03_head_loads_only_when_flag_set(self):
        """Verify candidate head is inactive by default (None), and active only when VOICE_HEAD_PATH is set."""
        from app.voice_backends.superb import SuperbBackend

        # 1. Default config: VOICE_HEAD_PATH is None -> _candidate_head is None
        with patch.object(config, "VOICE_HEAD_PATH", None):
            backend_default = SuperbBackend(analyzer=unittest.mock.MagicMock())
            self.assertIsNone(backend_default._candidate_head)

        # 2. When VOICE_HEAD_PATH points to a valid model artifact
        dummy_model = LogisticRegression()
        X_dummy = np.random.randn(10, 8)
        y_dummy = np.random.choice(["angry", "happy", "neutral", "sad"], size=10)
        dummy_model.fit(X_dummy, y_dummy)

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            joblib.dump(dummy_model, tmp_path)
            with patch.object(config, "VOICE_HEAD_PATH", tmp_path):
                backend_custom = SuperbBackend(analyzer=unittest.mock.MagicMock())
                self.assertIsNotNone(backend_custom._candidate_head)
                self.assertTrue(hasattr(backend_custom._candidate_head, "predict_proba"))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
