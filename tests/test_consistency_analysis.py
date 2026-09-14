"""
Tests for app/consistency_analysis.py.

Covers:
- Segment-level relationship classification (agreement, divergence, low_confidence)
- Trajectory computation:  mostly_consistent, mostly_divergent, mixed, insufficient_data
- Attention score:
    * deterministic output for fixed inputs
    * score is always in [0.0, 1.0]
    * mismatch_contribution is exactly zero when all segments agree
    * comparable high-mismatch cases score higher when intensity and escalation held similar
    * individual component contributions are non-negative
    * escalation_contribution is correctly zero / non-zero
    * intensity_contribution is correctly non-zero even when mismatch_rate is zero

NOTE: We do NOT test that all-agreement gives zero TOTAL attention score,
because the intensity component may still be non-zero.
We DO verify that the MISMATCH CONTRIBUTION is zero for all-agreement cases.
"""

import pytest

from app.consistency_analysis import (
    AttentionScore,
    ConsistencyTrajectory,
    MOSTLY_CONSISTENT_THRESHOLD,
    MOSTLY_DIVERGENT_THRESHOLD,
    MIN_TRAJECTORY_SEGMENTS,
    SEGMENT_RELATION_AGREEMENT,
    SEGMENT_RELATION_DIVERGENCE,
    SEGMENT_RELATION_LOW_CONFIDENCE,
    classify_segment_relationship,
    compute_attention_score,
    compute_consistency_trajectory,
    DEFAULT_W_MISMATCH,
    DEFAULT_W_INTENSITY,
    DEFAULT_W_ESCALATION,
)


# ---------------------------------------------------------
# Segment factory helpers
# ---------------------------------------------------------

def make_segment(agreement: bool, mismatch: bool, intensity: float = 0.50,
                 audio_dir: str = "negative", text_dir: str = "negative",
                 segment_index: int = 1):
    """Create a minimal segment dict matching ConversationAnalyzer output."""
    return {
        "segment_index": segment_index,
        "start_seconds": 0.0,
        "end_seconds": 5.0,
        "intensity": intensity,
        "fusion": {
            "agreement": agreement,
            "mismatch": mismatch,
            "audio_direction": audio_dir,
            "text_direction": text_dir,
            "combined_confidence": 0.70,
        },
    }


def agreement_seg(intensity: float = 0.60, index: int = 1):
    return make_segment(True, False, intensity, "negative", "negative", index)


def divergence_seg(intensity: float = 0.60, index: int = 1):
    return make_segment(False, True, intensity, "negative", "positive", index)


def low_conf_seg(intensity: float = 0.50, index: int = 1):
    return make_segment(False, False, intensity, "negative", "positive", index)


# ---------------------------------------------------------
# classify_segment_relationship
# ---------------------------------------------------------

class TestClassifySegmentRelationship:

    def test_agreement_classification(self):
        seg = agreement_seg()
        assert classify_segment_relationship(seg) == SEGMENT_RELATION_AGREEMENT

    def test_divergence_classification(self):
        seg = divergence_seg()
        assert classify_segment_relationship(seg) == SEGMENT_RELATION_DIVERGENCE

    def test_low_confidence_classification(self):
        seg = low_conf_seg()
        assert classify_segment_relationship(seg) == SEGMENT_RELATION_LOW_CONFIDENCE

    def test_agreement_takes_priority_over_mismatch_flag(self):
        """If agreement=True, result must be 'agreement' regardless of mismatch."""
        seg = make_segment(True, True)  # edge-case: both flags True
        assert classify_segment_relationship(seg) == SEGMENT_RELATION_AGREEMENT

    def test_missing_fusion_key_treated_as_low_confidence(self):
        seg = {"segment_index": 1, "intensity": 0.5}
        result = classify_segment_relationship(seg)
        assert result == SEGMENT_RELATION_LOW_CONFIDENCE


# ---------------------------------------------------------
# compute_consistency_trajectory
# ---------------------------------------------------------

class TestConsistencyTrajectory:

    def test_all_agreement_gives_mostly_consistent(self):
        segs = [agreement_seg(index=i) for i in range(1, 5)]
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "mostly_consistent"
        assert t.agreement_rate == 1.0
        assert t.mismatch_rate == 0.0

    def test_all_divergence_gives_mostly_divergent(self):
        segs = [divergence_seg(index=i) for i in range(1, 5)]
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "mostly_divergent"
        assert t.mismatch_rate == 1.0
        assert t.agreement_rate == 0.0

    def test_mixed_trajectory(self):
        # 2 agreement, 2 divergence, 2 low-confidence → rate = 2/6 = 0.333 agreement
        segs = (
            [agreement_seg(index=i) for i in range(1, 3)]
            + [divergence_seg(index=i) for i in range(3, 5)]
            + [low_conf_seg(index=i) for i in range(5, 7)]
        )
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "mixed"
        assert t.agreement_count == 2
        assert t.divergence_count == 2
        assert t.low_confidence_count == 2

    def test_insufficient_data_for_single_segment(self):
        segs = [agreement_seg()]
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "insufficient_data"
        assert t.total_segments == 1

    def test_insufficient_data_for_zero_segments(self):
        t = compute_consistency_trajectory([])
        assert t.trajectory == "insufficient_data"
        assert t.total_segments == 0

    def test_agreement_rate_boundary_mostly_consistent(self):
        # 7 agreements, 1 divergence → rate = 0.875 >= 0.70
        segs = (
            [agreement_seg(index=i) for i in range(1, 8)]
            + [divergence_seg(index=8)]
        )
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "mostly_consistent"

    def test_agreement_rate_boundary_mostly_divergent(self):
        # 1 agreement, 9 divergences → rate = 0.10 < 0.30
        segs = (
            [agreement_seg(index=1)]
            + [divergence_seg(index=i) for i in range(2, 11)]
        )
        t = compute_consistency_trajectory(segs)
        assert t.trajectory == "mostly_divergent"

    def test_per_segment_relationships_are_populated(self):
        segs = [agreement_seg(index=1), divergence_seg(index=2)]
        t = compute_consistency_trajectory(segs)
        assert len(t.per_segment_relationships) == 2
        assert t.per_segment_relationships[0]["relationship"] == SEGMENT_RELATION_AGREEMENT
        assert t.per_segment_relationships[1]["relationship"] == SEGMENT_RELATION_DIVERGENCE

    def test_trajectory_to_dict_has_required_keys(self):
        segs = [agreement_seg(index=i) for i in range(1, 3)]
        t = compute_consistency_trajectory(segs)
        d = t.to_dict()
        for key in ("trajectory", "agreement_rate", "mismatch_rate",
                    "agreement_count", "divergence_count", "total_segments"):
            assert key in d


# ---------------------------------------------------------
# compute_attention_score
# ---------------------------------------------------------

class TestAttentionScore:

    def test_deterministic_output(self):
        """Same inputs must always produce the same score."""
        segs = [agreement_seg(0.70, i) for i in range(1, 4)]
        escalation = {"escalation_detected": False}

        s1 = compute_attention_score(segs, escalation)
        s2 = compute_attention_score(segs, escalation)
        assert s1.attention_score == s2.attention_score

    def test_score_in_valid_range(self):
        """Score must always be in [0.0, 1.0]."""
        segs = [divergence_seg(1.0, i) for i in range(1, 10)]
        escalation = {"escalation_detected": True}
        s = compute_attention_score(segs, escalation)
        assert 0.0 <= s.attention_score <= 1.0

    def test_mismatch_contribution_zero_for_all_agreement(self):
        """
        When every segment is agreement, the mismatch_contribution must be 0.0.
        The total attention score may still be non-zero because intensity contributes.
        """
        segs = [agreement_seg(0.60, i) for i in range(1, 5)]
        escalation = {"escalation_detected": False}
        s = compute_attention_score(segs, escalation)
        assert s.mismatch_contribution == 0.0
        # Intensity contribution can be non-zero
        assert s.intensity_contribution > 0.0

    def test_high_mismatch_scores_higher_than_low_mismatch(self):
        """
        High-mismatch conversation should receive higher attention than a
        low-mismatch conversation when average_intensity and escalation are held equal.
        """
        intensity = 0.50
        escalation = {"escalation_detected": False}

        low_mismatch = [agreement_seg(intensity, i) for i in range(1, 5)]
        high_mismatch = [divergence_seg(intensity, i) for i in range(1, 5)]

        s_low = compute_attention_score(low_mismatch, escalation)
        s_high = compute_attention_score(high_mismatch, escalation)

        assert s_high.attention_score > s_low.attention_score

    def test_escalation_contribution_is_zero_without_escalation(self):
        segs = [divergence_seg(0.50, i) for i in range(1, 4)]
        escalation = {"escalation_detected": False}
        s = compute_attention_score(segs, escalation)
        assert s.escalation_contribution == 0.0

    def test_escalation_contribution_is_nonzero_with_escalation(self):
        segs = [agreement_seg(0.50, i) for i in range(1, 4)]
        escalation = {"escalation_detected": True}
        s = compute_attention_score(segs, escalation)
        assert s.escalation_contribution > 0.0
        assert s.escalation_detected is True

    def test_intensity_contribution_nonzero_when_mismatch_is_zero(self):
        """Verify intensity still contributes even when all segments agree."""
        segs = [agreement_seg(0.80, i) for i in range(1, 4)]
        escalation = {"escalation_detected": False}
        s = compute_attention_score(segs, escalation)
        assert s.mismatch_contribution == 0.0
        assert s.intensity_contribution > 0.0

    def test_components_are_non_negative(self):
        segs = [low_conf_seg(0.30, i) for i in range(1, 4)]
        escalation = {"escalation_detected": False}
        s = compute_attention_score(segs, escalation)
        assert s.mismatch_contribution >= 0.0
        assert s.intensity_contribution >= 0.0
        assert s.escalation_contribution >= 0.0

    def test_custom_weights_are_used(self):
        """Custom weights should change the resulting score."""
        segs = [divergence_seg(0.60, i) for i in range(1, 4)]
        escalation = {"escalation_detected": True}

        default = compute_attention_score(segs, escalation)
        custom = compute_attention_score(
            segs, escalation,
            w_mismatch=0.10, w_intensity=0.10, w_escalation=0.80
        )
        # Escalation-heavy weighting should differ from the default
        assert custom.attention_score != default.attention_score
        assert custom.w_mismatch == 0.10
        assert custom.w_escalation == 0.80

    def test_no_segments_gives_zero_score(self):
        s = compute_attention_score([], None)
        assert s.attention_score == 0.0
        assert s.mismatch_contribution == 0.0
        assert s.intensity_contribution == 0.0
        assert s.escalation_contribution == 0.0

    def test_formula_is_documented_in_result(self):
        segs = [agreement_seg()]
        s = compute_attention_score(segs, None)
        assert "mismatch_rate" in s.formula
        assert "average_intensity" in s.formula
        assert "escalation_indicator" in s.formula

    def test_attention_score_to_dict_contains_notice(self):
        segs = [agreement_seg(0.50, i) for i in range(1, 3)]
        s = compute_attention_score(segs, {"escalation_detected": False})
        d = s.to_dict()
        assert "notice" in d
        assert "heuristic" in d["notice"].lower()
        assert "formula" in d
        assert "weights" in d

    def test_precomputed_trajectory_is_used(self):
        """When trajectory is pre-computed, mismatch_rate should match it."""
        segs = [divergence_seg(0.50, i) for i in range(1, 5)]
        traj = compute_consistency_trajectory(segs)
        s = compute_attention_score(segs, None, trajectory=traj)
        assert s.mismatch_rate == traj.mismatch_rate

    def test_score_formula_addition_holds(self):
        """Verify the three components sum to the reported attention_score within float tolerance."""
        segs = [divergence_seg(0.60, i) for i in range(1, 4)]
        escalation = {"escalation_detected": True}
        s = compute_attention_score(segs, escalation)
        expected = round(
            s.mismatch_contribution + s.intensity_contribution + s.escalation_contribution, 4
        )
        assert abs(s.attention_score - expected) < 1e-6
