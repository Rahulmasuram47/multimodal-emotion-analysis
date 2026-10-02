"""
Unit tests for UI contracts, visual components, WCAG AA contrast, and chart data transformations.
All tests use synthetic / mocked data and execute without launching a Streamlit server.
"""

import pytest
import numpy as np

from scripts.check_contrast import run_contrast_audit, contrast_ratio
from ui.charts.dynamics import (
    render_agreement_strip,
    render_valence_arousal_map,
    render_intensity_trajectory_chart,
    render_speaker_comparison_chart,
)
from ui.components.verdict_card import build_key_finding, render_verdict_hero
from ui.components.modality_cards import render_modality_cards


def test_wcag_aa_contrast_audit_passes():
    """Verify all tokens in dark and light themes pass WCAG AA contrast."""
    assert run_contrast_audit() is True


def test_contrast_ratio_calculation():
    """Verify mathematically accurate contrast ratio calculation."""
    # Pure black against pure white is 21:1
    cr = contrast_ratio("#000000", "#FFFFFF")
    assert round(cr, 1) == 21.0

    # Same color is 1:1
    assert round(contrast_ratio("#101318", "#101318"), 1) == 1.0


def test_key_finding_synthesis():
    """Verify build_key_finding creates honest factual summaries."""
    finding = build_key_finding(
        dominant_voice="angry",
        dominant_text="positive",
        segments=[
            {"fusion": {"audio_direction": "negative"}},
            {"fusion": {"audio_direction": "negative"}},
        ],
        trend_label="increasing",
        escalation_detected=False,
    )
    assert "predominantly angry" in finding.lower()
    assert "predominantly positive" in finding.lower()
    assert "increased" in finding.lower()


def test_dynamics_charts_handle_empty_and_valid_segments():
    """Verify visual chart renderers accept data without raising exceptions."""
    # Empty segments shouldn't crash
    render_agreement_strip([])
    render_valence_arousal_map([])
    render_intensity_trajectory_chart([])
    render_speaker_comparison_chart([])

    # Synthetic multi-turn segments
    segments = [
        {
            "segment_index": 0,
            "start_seconds": 0.0,
            "end_seconds": 2.5,
            "transcript": "Good morning team",
            "intensity": 0.25,
            "text_sentiment": {"sentiment": "positive", "confidence": 0.88},
            "voice_emotion": {"emotion": "neutral", "confidence": 0.75},
            "fusion": {
                "agreement": False,
                "mismatch": False,
                "audio_direction": "neutral",
                "text_direction": "positive",
                "audio_arousal": 0.30,
                "audio_reliability": 1.0,
                "text_reliability": 1.0,
                "combined_reliability": 1.0,
                "abstention": False,
                "pattern_label": "weak or low-confidence signal",
                "counterfactuals": ["Test counterfactual"],
            },
            "speaker": "Speaker A",
        },
        {
            "segment_index": 1,
            "start_seconds": 2.5,
            "end_seconds": 5.0,
            "transcript": "Let's review the report",
            "intensity": 0.40,
            "text_sentiment": {"sentiment": "neutral", "confidence": 0.80},
            "voice_emotion": {"emotion": "angry", "confidence": 0.82},
            "fusion": {
                "agreement": False,
                "mismatch": True,
                "audio_direction": "negative",
                "text_direction": "neutral",
                "audio_arousal": 0.70,
                "audio_reliability": 1.0,
                "text_reliability": 0.90,
                "combined_reliability": 0.95,
                "abstention": False,
                "pattern_label": "incongruence: directional mismatch",
                "counterfactuals": [],
            },
            "speaker": "Speaker B",
        },
    ]

    # These plotly chart functions should build traces cleanly
    render_agreement_strip(segments)
    render_valence_arousal_map(segments)
    render_intensity_trajectory_chart(segments)

    speaker_analytics = [
        {"speaker": "Speaker A", "talk_time_share": 0.5, "mean_intensity": 0.25},
        {"speaker": "Speaker B", "talk_time_share": 0.5, "mean_intensity": 0.40},
    ]
    render_speaker_comparison_chart(speaker_analytics)
