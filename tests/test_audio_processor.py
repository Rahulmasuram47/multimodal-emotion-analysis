"""
Test Suite for Milestone 2: Audio Processing Module.
Verifies valid file processing, unsupported format rejection, corrupted audio handling,
empty file handling, and acoustic feature extraction.
"""
import os
import sys
import unittest
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.audio_processor import (
    validate_audio_file,
    load_and_preprocess_audio,
    extract_acoustic_features,
    AudioValidationError,
    AudioProcessingError,
)
from config import TARGET_SAMPLE_RATE, N_MFCC


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.valid_file = os.path.join("sample_audio", "sample_16k.wav")
        self.unsupported_file = os.path.join("sample_audio", "test_unsupported.xyz")
        self.corrupted_file = os.path.join("sample_audio", "test_corrupted.wav")
        self.empty_file = os.path.join("sample_audio", "test_empty.wav")

    def test_01_valid_audio_loading(self):
        """Test that a valid 16kHz WAV file is correctly validated, loaded, and normalized."""
        meta = validate_audio_file(self.valid_file)
        self.assertIn("file_path", meta)
        self.assertEqual(meta["extension"], ".wav")
        self.assertGreater(meta["file_size_bytes"], 0)

        audio, sr, duration = load_and_preprocess_audio(self.valid_file)
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.ndim, 1, "Audio should be single-channel mono")
        self.assertEqual(sr, TARGET_SAMPLE_RATE)
        self.assertAlmostEqual(duration, 3.0, places=1)
        self.assertLessEqual(np.max(np.abs(audio)), 1.0001, "Audio peak should be normalized <= 1.0")

    def test_02_unsupported_format_rejection(self):
        """Test that unsupported file extensions are caught gracefully with AudioValidationError."""
        with self.assertRaises(AudioValidationError) as ctx:
            validate_audio_file(self.unsupported_file)
        self.assertIn("Unsupported audio format", str(ctx.exception))

        with self.assertRaises(AudioValidationError):
            load_and_preprocess_audio(self.unsupported_file)

    def test_03_corrupted_file_handling(self):
        """Test that corrupted audio files are caught gracefully without unhandled crashes."""
        with self.assertRaises(AudioValidationError) as ctx:
            validate_audio_file(self.corrupted_file)
        self.assertTrue(
            "corrupted" in str(ctx.exception).lower() or "unreadable" in str(ctx.exception).lower(),
            f"Expected corruption error message, got: {ctx.exception}"
        )

        with self.assertRaises(AudioValidationError):
            load_and_preprocess_audio(self.corrupted_file)

    def test_04_empty_file_handling(self):
        """Test that empty (0-byte) files are rejected gracefully."""
        with self.assertRaises(AudioValidationError) as ctx:
            validate_audio_file(self.empty_file)
        self.assertIn("empty", str(ctx.exception).lower())

        with self.assertRaises(AudioValidationError):
            load_and_preprocess_audio(self.empty_file)

    def test_05_feature_extraction(self):
        """Test extraction of acoustic features (MFCC, RMS energy, Pitch, ZCR, Spectral Centroid)."""
        audio, sr, duration = load_and_preprocess_audio(self.valid_file)
        features = extract_acoustic_features(audio, sr=sr, n_mfcc=N_MFCC)

        # Check MFCC features
        self.assertEqual(features["n_mfcc"], N_MFCC)
        self.assertEqual(features["mfcc_shape"][0], N_MFCC)
        self.assertGreater(features["mfcc_shape"][1], 0)
        self.assertEqual(len(features["mfcc_mean"]), N_MFCC)
        self.assertEqual(len(features["mfcc_std"]), N_MFCC)

        # Check RMS Energy
        self.assertGreater(features["rms_mean"], 0)
        self.assertGreater(features["rms_max"], 0)

        # Check Pitch / F0
        self.assertIn("pitch_mean_hz", features)
        self.assertIn("voiced_fraction", features)
        self.assertGreaterEqual(features["voiced_fraction"], 0.0)
        self.assertLessEqual(features["voiced_fraction"], 1.0)

        # Check Zero-Crossing Rate & Spectral Centroid
        self.assertIn("zcr_mean", features)
        self.assertIn("spectral_centroid_mean", features)
        self.assertGreater(features["spectral_centroid_mean"], 0)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioProcessor)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)