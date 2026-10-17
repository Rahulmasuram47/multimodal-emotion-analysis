"""
Unit Tests for Phase 5A: Calibration and Latency Metrics.

Tests:
1. Temperature recovery (recovers true scalar temperature T on synthetic logits).
2. ECE on calibrated data (evaluates ECE ~ 0 on well-calibrated synthetic predictions).
3. RTF arithmetic (computes processing_time / audio_duration with zero-division handling).
"""
import unittest
import numpy as np

from evaluation.calibration import compute_ece, compute_nll, fit_temperature
from scripts.benchmark_latency import compute_rtf


class TestCalibrationAndLatency(unittest.TestCase):
    def test_01_temperature_recovery(self):
        """Verify that minimizing NLL recovers true scaling temperature T on synthetic logits."""
        np.random.seed(42)
        n_samples = 1500
        n_classes = 4
        true_T = 1.8

        # Generate base logits
        raw_logits = np.random.randn(n_samples, n_classes) * 2.5

        # Probabilities under true_T
        scaled_logits = raw_logits / true_T
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        true_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Sample discrete labels from true probabilities
        labels = np.array([
            np.random.choice(n_classes, p=true_probs[i])
            for i in range(n_samples)
        ])

        recovered_T = fit_temperature(raw_logits, labels)
        self.assertAlmostEqual(recovered_T, true_T, delta=0.25,
                               msg=f"Expected recovered T ~ {true_T}, got {recovered_T}")

    def test_02_ece_on_calibrated_data(self):
        """Verify that Expected Calibration Error (ECE) is ~0 on well-calibrated probabilities."""
        np.random.seed(42)
        n_samples = 5000
        n_classes = 3

        # Generate random probabilities
        logits = np.random.randn(n_samples, n_classes)
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)

        # Sample labels from probs so data is perfectly calibrated by construction
        labels = np.array([
            np.random.choice(n_classes, p=probs[i])
            for i in range(n_samples)
        ])

        ece, bins = compute_ece(probs, labels, n_bins=10)
        self.assertLess(ece, 0.05, f"Well-calibrated data should have ECE < 0.05, got {ece}")
        self.assertEqual(len(bins), 10)

    def test_03_rtf_arithmetic(self):
        """Verify Real-Time Factor (RTF) computation across normal and edge cases."""
        # Standard sub-realtime: 2.5s proc for 5.0s audio -> RTF = 0.5
        self.assertEqual(compute_rtf(2.5, 5.0), 0.5)

        # Slower than realtime: 15.0s proc for 10.0s audio -> RTF = 1.5
        self.assertEqual(compute_rtf(15.0, 10.0), 1.5)

        # Faster than realtime: 0.1s proc for 10.0s audio -> RTF = 0.01
        self.assertEqual(compute_rtf(0.1, 10.0), 0.01)

        # Zero duration handling (must avoid ZeroDivisionError)
        self.assertEqual(compute_rtf(1.0, 0.0), 0.0)
        self.assertEqual(compute_rtf(0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
