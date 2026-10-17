"""
Unit Tests for Phase 5B: Synthetic Noise Robustness.

Tests:
1. White noise SNR mixing accuracy (within 0.5 dB of target SNR for 20 dB, 10 dB, 0 dB).
2. 8 kHz telephony resampling round-trip preserves signal length and dimensionality.
3. Robustness reporting includes mandatory label 'synthetic noise, not real environments'.
"""
import unittest
import numpy as np

from evaluation.robustness import (
    add_white_noise_snr,
    apply_8khz_telephony_roundtrip,
    compute_empirical_snr_db,
    generate_robustness_markdown,
)


class TestRobustness(unittest.TestCase):
    def test_01_snr_mixing_within_half_db(self):
        """Verify white noise injection achieves requested SNR within 0.5 dB tolerance."""
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        # 440 Hz + 880 Hz harmonic test signal
        clean_signal = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 880 * t)

        target_snrs = [20.0, 10.0, 0.0]

        for target_snr in target_snrs:
            noisy = add_white_noise_snr(clean_signal, target_snr_db=target_snr, seed=42)
            measured_snr = compute_empirical_snr_db(clean_signal, noisy)

            delta = abs(measured_snr - target_snr)
            self.assertLess(
                delta, 0.5,
                f"Target SNR {target_snr} dB, but measured {measured_snr:.2f} dB (delta {delta:.2f} dB > 0.5 dB)"
            )

    def test_02_telephony_roundtrip_preserves_length(self):
        """Verify 8 kHz down-and-up resampling round-trip matches original duration and shape."""
        sr = 16000
        n_samples = 48000  # 3 seconds
        audio = np.random.randn(n_samples).astype(np.float32) * 0.1

        roundtrip = apply_8khz_telephony_roundtrip(audio, sr=sr)
        self.assertEqual(len(roundtrip), len(audio))
        self.assertEqual(roundtrip.ndim, 1)
        self.assertEqual(roundtrip.dtype, np.float32)

    def test_03_markdown_required_scope_label(self):
        """Verify markdown report explicitly states 'synthetic noise, not real environments'."""
        meta = {
            "model_id": "superb/wav2vec2-base-superb-er",
            "max_clips_per_class": 20,
            "seed": 42,
            "notice": "synthetic noise, not real environments",
        }
        report = generate_robustness_markdown({}, meta)
        self.assertIn("synthetic noise, not real environments", report)


if __name__ == "__main__":
    unittest.main()
