"""Tests for compute_attention_score and build_key_finding single-turn guard."""
import json
import pathlib
import pytest

from app.consistency_analysis import compute_attention_score

GOLDEN = json.loads(
    (pathlib.Path(__file__).parent / "golden_attention.json").read_text()
)


def _seg(mismatch=False, intensity=0.5):
    return {
        "intensity": intensity,
        "fusion": {"mismatch": mismatch, "agreement": not mismatch},
        "text_unavailable": False,
    }


# ---------------------------------------------------------------------------
# Pinned outputs for 2/3/4-segment fixtures — must be unchanged after the fix
# ---------------------------------------------------------------------------


def test_pinned_2seg_nomismatch():
    segs = [_seg(False, 0.4), _seg(False, 0.6)]
    r = compute_attention_score(segs)
    g = GOLDEN["2seg_nomismatch"]
    assert r.attention_score == g["attention_score"]
    assert r.mismatch_contribution == g["mismatch_contribution"]
    assert r.intensity_contribution == g["intensity_contribution"]
    assert r.mismatch_rate == g["mismatch_rate"]


def test_pinned_2seg_mismatch1():
    segs = [_seg(True, 0.7), _seg(False, 0.5)]
    r = compute_attention_score(segs)
    g = GOLDEN["2seg_mismatch1"]
    assert r.attention_score == g["attention_score"]
    assert r.mismatch_contribution == g["mismatch_contribution"]
    assert r.mismatch_rate == g["mismatch_rate"]


def test_pinned_3seg_mismatch2():
    segs = [_seg(True, 0.8), _seg(True, 0.6), _seg(False, 0.4)]
    r = compute_attention_score(segs)
    g = GOLDEN["3seg_mismatch2"]
    assert r.attention_score == g["attention_score"]
    assert r.mismatch_contribution == g["mismatch_contribution"]
    assert r.mismatch_rate == g["mismatch_rate"]


def test_pinned_4seg_mismatch1():
    segs = [_seg(True, 0.9), _seg(False, 0.3), _seg(False, 0.5), _seg(False, 0.7)]
    r = compute_attention_score(segs)
    g = GOLDEN["4seg_mismatch1"]
    assert r.attention_score == g["attention_score"]
    assert r.mismatch_contribution == g["mismatch_contribution"]
    assert r.mismatch_rate == g["mismatch_rate"]


# ---------------------------------------------------------------------------
# Single-segment behaviour (new requirements)
# ---------------------------------------------------------------------------


def test_1seg_mismatch_contribution_positive():
    """A single-turn mismatch should give mismatch_contribution > 0."""
    segs = [_seg(True, 0.5)]
    r = compute_attention_score(segs)
    assert r.mismatch_contribution > 0, f"mismatch_contribution={r.mismatch_contribution}"
    assert r.mismatch_rate == 1.0


def test_1seg_agreement_low_mismatch():
    """A single-turn agreement should give mismatch_contribution == 0."""
    segs = [_seg(False, 0.3)]
    r = compute_attention_score(segs)
    assert r.mismatch_contribution == 0.0
    assert r.mismatch_rate == 0.0


# ---------------------------------------------------------------------------
# build_key_finding single-turn guard
# ---------------------------------------------------------------------------


def test_build_key_finding_single_turn():
    from ui.components.verdict_card import build_key_finding

    seg = _seg(True, 0.7)
    result = build_key_finding("anger", "neutral", [seg], "stable", False)
    assert result == "Single turn: too few turns to assess a trend."


def test_build_key_finding_two_turns_not_single():
    from ui.components.verdict_card import build_key_finding

    segs = [_seg(False, 0.4), _seg(False, 0.6)]
    result = build_key_finding("happiness", "positive", segs, "stable", False)
    assert "Single turn" not in result
