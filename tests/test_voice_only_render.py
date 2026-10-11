"""
Tests verifying that voice-only (text unavailable) results render cleanly across all UI components.
Mocks streamlit calls to assert none raises exceptions for both single-segment, multi-segment,
and normal text-available fixtures.
"""

from unittest.mock import MagicMock, patch
import pytest

from ui.charts.dynamics import (
    render_agreement_strip,
    render_intensity_trajectory_chart,
    render_valence_arousal_map,
)
from ui.components.export_bar import render_export_bar
from ui.components.modality_cards import render_modality_cards
from ui.components.segment_timeline import render_segment_timeline
from ui.components.verdict_card import build_key_finding, render_verdict_card, render_verdict_hero
from ui.pages.why_result_view import render_why_result_page


def _make_voice_only_segment(idx=1, start=0.0, end=2.0, emotion="angry", conf=0.85):
    return {
        "segment_index": idx,
        "start_seconds": start,
        "end_seconds": end,
        "transcript": "blabla repetitif repetitif repetitif repetitif",
        "speaker": "Speaker A",
        "voice_emotion": {
            "emotion": emotion,
            "confidence": conf,
            "scores": {emotion: conf, "neutral": 1.0 - conf},
            "voice_reliability": 0.9,
        },
        "text_sentiment": None,
        "text_unavailable": True,
        "asr_flags": ["non_english"],
        "voice_reliability": 0.9,
        "text_reliability": 0.0,
        "fusion": {
            "pattern_label": "voice-only (text unavailable)",
            "pattern_explanation": "Operating in voice-only analysis mode.",
            "rule_ids_fired": ["voice_only_text_unavailable"],
            "agreement": False,
            "mismatch": False,
            "abstention": False,
            "combined_confidence": conf,
            "audio_direction": "negative",
            "text_direction": "unavailable",
            "audio_reliability": 0.9,
            "text_reliability": 0.0,
            "combined_reliability": 0.9,
            "effective_weights": {"audio_weight": 1.0, "text_weight": 0.0},
            "audio_arousal": 0.8,
            "counterfactuals": ["Abstention buffer: voice reliability is above cut-off."],
        },
        "intensity": 0.70,
        "intensity_components": {
            "formula_version": "v2",
            "prosodic_arousal": 0.65,
            "prosody_components": {},
            "voice_arousal": 0.75,
            "text_negativity": None,
            "weights": {"prosody": 0.5, "voice_arousal": 0.5, "text_negativity": 0.0},
        },
    }


def _make_normal_segment(idx=1, start=0.0, end=2.0, emotion="happy", sent="positive"):
    return {
        "segment_index": idx,
        "start_seconds": start,
        "end_seconds": end,
        "transcript": "Hello this is a good meeting",
        "speaker": "Speaker A",
        "voice_emotion": {
            "emotion": emotion,
            "confidence": 0.8,
            "scores": {emotion: 0.8, "neutral": 0.2},
            "voice_reliability": 1.0,
        },
        "text_sentiment": {
            "sentiment": sent,
            "confidence": 0.85,
            "scores": {sent: 0.85, "neutral": 0.15},
            "text_reliability": 1.0,
        },
        "text_unavailable": False,
        "asr_flags": [],
        "voice_reliability": 1.0,
        "text_reliability": 1.0,
        "fusion": {
            "pattern_label": "positive pattern",
            "pattern_explanation": "Voice and text both point positive.",
            "rule_ids_fired": ["agreement_positive"],
            "agreement": True,
            "mismatch": False,
            "abstention": False,
            "combined_confidence": 0.825,
            "audio_direction": "positive",
            "text_direction": "positive",
            "audio_reliability": 1.0,
            "text_reliability": 1.0,
            "combined_reliability": 1.0,
            "effective_weights": {"audio_weight": 0.5, "text_weight": 0.5},
            "audio_arousal": 0.75,
            "counterfactuals": [],
        },
        "intensity": 0.75,
        "intensity_components": {
            "formula_version": "v2",
            "prosodic_arousal": 0.7,
            "prosody_components": {},
            "voice_arousal": 0.75,
            "text_negativity": 0.05,
            "weights": {"prosody": 0.35, "voice_arousal": 0.35, "text_negativity": 0.30},
        },
    }


@pytest.fixture
def voice_only_1seg():
    seg = _make_voice_only_segment(1, 0.0, 2.5, "angry", 0.85)
    return {
        "full_transcript": "blabla repetitif repetitif repetitif repetitif",
        "language": "fr",
        "language_probability": 0.92,
        "asr_flags": ["non_english"],
        "segments": [seg],
        "verdict": {
            "verdict": "insufficient_data",
            "reason": "At least 3 turns are needed to assess a trend.",
            "slope": 0.0,
            "net_change": 0.0,
            "counterfactuals": [],
        },
        "escalation": {"escalation_detected": False, "reason": "No escalation"},
        "attention_score": {"attention_score": 0.3, "priority_band": "LOW"},
        "recommendation": {"status": "NORMAL", "recommended_action": "Routine check"},
    }


@pytest.fixture
def voice_only_3seg():
    segs = [
        _make_voice_only_segment(1, 0.0, 2.0, "neutral", 0.6),
        _make_voice_only_segment(2, 2.5, 4.5, "angry", 0.75),
        _make_voice_only_segment(3, 5.0, 7.0, "angry", 0.90),
    ]
    return {
        "full_transcript": "Segment 1 text. Segment 2 text. Segment 3 text.",
        "language": "es",
        "language_probability": 0.88,
        "asr_flags": ["non_english"],
        "segments": segs,
        "verdict": {
            "verdict": "escalating",
            "reason": "Intensity rose steadily across turns.",
            "slope": 0.10,
            "net_change": 0.20,
            "counterfactuals": ["Verdict flips if slope drops."],
        },
        "escalation": {"escalation_detected": True, "reason": "Consistent rise"},
        "attention_score": {"attention_score": 0.65, "priority_band": "HIGH"},
        "recommendation": {"status": "REVIEW RECOMMENDED", "recommended_action": "Review clip"},
    }


@pytest.fixture
def normal_text_avail_3seg():
    segs = [
        _make_normal_segment(1, 0.0, 2.0, "happy", "positive"),
        _make_normal_segment(2, 2.5, 4.5, "happy", "positive"),
        _make_normal_segment(3, 5.0, 7.0, "happy", "positive"),
    ]
    return {
        "full_transcript": "Hello this is a good meeting. Thank you very much.",
        "language": "en",
        "language_probability": 0.96,
        "asr_flags": [],
        "segments": segs,
        "verdict": {
            "verdict": "stable",
            "reason": "Intensity stayed roughly constant.",
            "slope": 0.01,
            "net_change": 0.02,
            "counterfactuals": [],
        },
        "escalation": {"escalation_detected": False, "reason": "Stable intensity"},
        "attention_score": {"attention_score": 0.2, "priority_band": "LOW"},
        "recommendation": {"status": "NORMAL", "recommended_action": "Routine"},
    }


def _run_all_renderers(fixture):
    segments = fixture["segments"]
    verdict = fixture["verdict"]
    escalation = fixture["escalation"]
    attention_score = fixture["attention_score"]
    recommendation = fixture["recommendation"]

    # 1. Verdict & Key finding
    finding = build_key_finding(
        dominant_voice="angry",
        dominant_text="N/A",
        segments=segments,
        trend_label="increasing",
        escalation_detected=escalation.get("escalation_detected", False),
    )
    render_verdict_card(finding)
    render_verdict_hero(
        verdict=verdict,
        finding=finding,
        attention_score=attention_score,
        recommendation=recommendation,
    )

    # 2. Modality cards
    render_modality_cards(
        dominant_voice="angry",
        dominant_text="N/A",
        segments=segments,
        agree_count=0,
        mismatch_count=0,
        voice_reliability=0.9,
        text_reliability=0.0,
        combined_reliability=0.9,
        abstention=False,
    )

    # 3. Charts & timeline
    render_agreement_strip(segments)
    render_valence_arousal_map(segments)
    render_intensity_trajectory_chart(segments, verdict=verdict, escalation=escalation)
    render_segment_timeline(segments)

    # 4. Why this result tab
    render_why_result_page(result=fixture, segments=segments)

    # 5. Export bar
    render_export_bar(fixture)


def _mock_columns(spec):
    n = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
    return [MagicMock() for _ in range(n)]


def test_voice_only_render_1_segment(voice_only_1seg):
    """Calling all renderers with a 1-segment voice-only fixture does not raise."""
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.expander", return_value=MagicMock()), \
         patch("streamlit.caption"), \
         patch("streamlit.dataframe"), \
         patch("streamlit.download_button"), \
         patch("streamlit.plotly_chart"), \
         patch("streamlit.container", return_value=MagicMock()):
        _run_all_renderers(voice_only_1seg)


def test_voice_only_render_3_segments(voice_only_3seg):
    """Calling all renderers with a 3-segment voice-only fixture does not raise."""
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.expander", return_value=MagicMock()), \
         patch("streamlit.caption"), \
         patch("streamlit.dataframe"), \
         patch("streamlit.download_button"), \
         patch("streamlit.plotly_chart"), \
         patch("streamlit.container", return_value=MagicMock()):
        _run_all_renderers(voice_only_3seg)


def test_normal_text_available_render(normal_text_avail_3seg):
    """Calling all renderers with normal text-available fixture does not raise."""
    with patch("streamlit.markdown"), \
         patch("streamlit.columns", side_effect=_mock_columns), \
         patch("streamlit.expander", return_value=MagicMock()), \
         patch("streamlit.caption"), \
         patch("streamlit.dataframe"), \
         patch("streamlit.download_button"), \
         patch("streamlit.plotly_chart"), \
         patch("streamlit.container", return_value=MagicMock()):
        _run_all_renderers(normal_text_avail_3seg)
