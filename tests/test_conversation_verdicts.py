"""
Unit tests for conversation-level verdicts and escalation detection.

Synthetic / mocked inputs clearly marked test-only.
Verifies Prompt B Task 4 & Prompt G:
- Clean monotonic rise -> 'escalating'
- Rise-with-one-dip within tolerance -> 'escalating' (acceptance criterion)
- Strict mode vs tolerant mode on rise-with-dip
- Flat / small variation -> 'stable'
- Monotonic fall -> 'de-escalating'
- Back-and-forth swings -> 'fluctuating'
- <2 or <3 segments edge cases -> 'insufficient_data'
"""

import pytest

import config
from app.conversation_analysis import ConversationAnalyzer


def _make_segments(intensities):
    """Helper to build minimal synthetic segment dicts (test-only input)."""
    return [{"intensity": float(val), "transcript": f"Segment {i+1}"} for i, val in enumerate(intensities)]


def test_verdict_monotonic_rise():
    segments = _make_segments([0.25, 0.40, 0.55, 0.70])
    res = ConversationAnalyzer.compute_unified_verdict(segments)
    assert res["verdict"] == "escalating"
    assert res["slope"] > 0
    assert res["net_change"] >= 0.20


def test_verdict_rise_with_one_dip_acceptance_criterion():
    """Key acceptance test: A conversation with rising intensity that has
    one minor dip (e.g. 0.05 drop <= 0.08 tolerance) is recognized as 'escalating'
    by the tolerant model.
    """
    # 0.30 -> 0.50 (+0.20), 0.50 -> 0.45 (-0.05 dip <= 0.08), 0.45 -> 0.65 (+0.20), 0.65 -> 0.80 (+0.15)
    segments = _make_segments([0.30, 0.50, 0.45, 0.65, 0.80])
    res = ConversationAnalyzer.compute_unified_verdict(segments)
    assert res["verdict"] == "escalating"

    # Tolerant advanced escalation detection should detect escalation
    esc_tolerant = ConversationAnalyzer.detect_escalation_advanced(segments, mode="tolerant")
    assert esc_tolerant["escalation_detected"] is True

    # Strict escalation detection should REJECT due to the dip
    esc_strict = ConversationAnalyzer.detect_escalation(segments)
    assert esc_strict["escalation_detected"] is False


def test_verdict_flat_stable():
    segments = _make_segments([0.50, 0.52, 0.49, 0.51])
    res = ConversationAnalyzer.compute_unified_verdict(segments)
    assert res["verdict"] == "stable"


def test_verdict_fall_deescalating():
    segments = _make_segments([0.80, 0.65, 0.50, 0.35])
    res = ConversationAnalyzer.compute_unified_verdict(segments)
    assert res["verdict"] == "de-escalating"
    assert res["slope"] < 0
    assert res["net_change"] <= -0.20


def test_verdict_fluctuating():
    segments = _make_segments([0.20, 0.75, 0.25, 0.80])
    res = ConversationAnalyzer.compute_unified_verdict(segments)
    assert res["verdict"] == "fluctuating"


def test_verdict_insufficient_data():
    # 1 segment
    res1 = ConversationAnalyzer.compute_unified_verdict(_make_segments([0.50]))
    assert res1["verdict"] == "insufficient_data"

    # Empty
    res0 = ConversationAnalyzer.compute_unified_verdict([])
    assert res0["verdict"] == "insufficient_data"


def test_escalation_min_segments():
    # Only 2 segments should not trigger advanced escalation when min_segments=3
    segments = _make_segments([0.30, 0.60])
    esc = ConversationAnalyzer.detect_escalation_advanced(segments, mode="tolerant")
    assert esc["escalation_detected"] is False
