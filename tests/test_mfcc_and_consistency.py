"""
Tests for MFCC timbral feature extraction and Modality Consistency Trajectory analysis.
"""

import pytest
import numpy as np

from app.audio_processor import extract_acoustic_features as extract_features
from app.consistency_analysis import (
    compute_consistency_trajectory,
    compute_attention_score,
    classify_segment_relationship,
)


class TestMFCCExtraction:
    """Verify MFCC features are extracted according to specification."""

    def test_extract_features_includes_13_mfccs(self):
        sr = 16000
        # 1-second synthetic 440Hz sine wave
        t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
        y = 0.5 * np.sin(2 * np.pi * 440 * t)

        feats = extract_features(y, sample_rate=sr, n_mfcc=13)
        assert feats["n_mfcc"] == 13
        assert len(feats["mfcc_mean"]) == 13
        assert len(feats["mfcc_std"]) == 13
        assert feats["mfcc_raw"].shape[0] == 13


class TestConsistencyTrajectory:
    """Verify segment relationship classification and consistency trajectory dynamics."""

    def test_classify_relationship_agreement(self):
        seg = {
            "fusion": {
                "agreement": True,
                "mismatch": False,
                "audio_direction": "positive",
                "text_direction": "positive",
            },
        }
        rel = classify_segment_relationship(seg)
        assert rel == "agreement"

    def test_classify_relationship_divergence(self):
        seg = {
            "fusion": {
                "agreement": False,
                "mismatch": True,
                "audio_direction": "negative",
                "text_direction": "positive",
            },
        }
        rel = classify_segment_relationship(seg)
        assert rel == "divergence"

    def test_classify_relationship_low_confidence(self):
        seg = {
            "fusion": {
                "agreement": False,
                "mismatch": False,
                "audio_direction": "negative",
                "text_direction": "positive",
            },
        }
        rel = classify_segment_relationship(seg)
        assert rel == "low_confidence"

    def test_trajectory_metrics_across_mixed_conversation(self):
        segments = [
            # 2 agreeing segments
            {"fusion": {"agreement": True, "mismatch": False, "audio_direction": "positive", "text_direction": "positive"}, "intensity": 0.2},
            {"fusion": {"agreement": True, "mismatch": False, "audio_direction": "positive", "text_direction": "positive"}, "intensity": 0.3},
            # 1 divergent segment
            {"fusion": {"agreement": False, "mismatch": True, "audio_direction": "negative", "text_direction": "positive"}, "intensity": 0.5},
            # 1 low-confidence segment
            {"fusion": {"agreement": False, "mismatch": False, "audio_direction": "negative", "text_direction": "positive"}, "intensity": 0.4},
        ]
        ct = compute_consistency_trajectory(segments)
        assert ct.total_segments == 4
        assert ct.agreement_count == 2
        assert ct.divergence_count == 1
        assert ct.low_confidence_count == 1
        assert pytest.approx(ct.agreement_rate, 0.01) == 0.50
        assert pytest.approx(ct.mismatch_rate, 0.01) == 0.25

        att = compute_attention_score(segments, trajectory=ct)
        assert 0.0 <= att.attention_score <= 1.0


