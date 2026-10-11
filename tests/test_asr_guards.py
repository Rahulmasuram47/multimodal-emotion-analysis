"""
Tests for ASR guards, voice-only pipeline path, and golden fusion outputs.
All models (Whisper, RoBERTa, etc.) are mocked.
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from app.conversation_analysis import ConversationAnalyzer
from app.fusion import MultimodalFusion
from app.speech_to_text import SpeechToText
from ui.charts.dynamics import render_agreement_strip
from ui.components.export_bar import build_json_export_payload


# ---------------------------------------------------------------------------
# Golden test
# ---------------------------------------------------------------------------

def test_golden_fusion_outputs():
    """Verify current MultimodalFusion matches golden outputs for fixed cases."""
    golden_path = os.path.join(os.path.dirname(__file__), "golden_fusion.json")
    assert os.path.exists(golden_path), "golden_fusion.json must exist"

    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    fusion = MultimodalFusion()
    cases = {
        "agree_neg_high": (
            {"emotion": "angry", "confidence": 0.85, "voice_reliability": 1.0},
            {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0},
        ),
        "agree_pos": (
            {"emotion": "happy", "confidence": 0.80, "voice_reliability": 1.0},
            {"sentiment": "positive", "confidence": 0.75, "text_reliability": 1.0},
        ),
        "mismatch": (
            {"emotion": "happy", "confidence": 0.80, "voice_reliability": 1.0},
            {"sentiment": "negative", "confidence": 0.80, "text_reliability": 1.0},
        ),
        "weak": (
            {"emotion": "neutral", "confidence": 0.55, "voice_reliability": 0.8},
            {"sentiment": "positive", "confidence": 0.40, "text_reliability": 0.7},
        ),
        "abstention": (
            {"emotion": "sad", "confidence": 0.30, "voice_reliability": 0.3},
            {"sentiment": "neutral", "confidence": 0.25, "text_reliability": 0.3},
        ),
    }

    keep = [
        "pattern_label", "rule_ids_fired", "agreement", "mismatch", "abstention",
        "audio_direction", "text_direction", "combined_reliability",
    ]
    for key, (voice_in, text_in) in cases.items():
        actual = fusion.fuse(voice_in, text_in)
        for field in keep:
            assert actual[field] == golden[key][field], (
                f"Mismatch in {key} for field {field}: expected {golden[key][field]}, got {actual[field]}"
            )


# ---------------------------------------------------------------------------
# Whisper transcribe backward compatibility & keys
# ---------------------------------------------------------------------------

def test_whisper_transcribe_backward_compatible_keys():
    """SpeechToText.transcribe() returns language and language_probability."""
    mock_seg = SimpleNamespace(
        start=0.0,
        end=2.0,
        text="Hello world test",
        avg_logprob=-0.2,
        no_speech_prob=0.01,
        words=[],
    )
    mock_info = SimpleNamespace(
        language="en",
        language_probability=0.98,
    )

    with patch("app.speech_to_text.WhisperModel") as MockModel:
        instance = MockModel.return_value
        instance.transcribe.return_value = ([mock_seg], mock_info)

        stt = SpeechToText()
        res = stt.transcribe("dummy.wav")

        assert "language" in res
        assert "language_probability" in res
        assert res["language"] == "en"
        assert res["language_probability"] == pytest.approx(0.98)
        assert res["text"] == "Hello world test"
        assert len(res["segments"]) == 1


# ---------------------------------------------------------------------------
# Voice-only fusion path
# ---------------------------------------------------------------------------

def test_fusion_voice_only_path():
    """MultimodalFusion with text_result=None or text_unavailable produces voice_only rule."""
    fusion = MultimodalFusion()
    voice = {"emotion": "angry", "confidence": 0.82, "voice_reliability": 0.9}

    # None text
    res = fusion.fuse(voice, None)
    assert res["pattern_label"] == "voice-only (text unavailable)"
    assert res["rule_ids_fired"] == ["voice_only_text_unavailable"]
    assert res["effective_weights"]["audio_weight"] == 1.00
    assert res["effective_weights"]["text_weight"] == 0.00
    assert res["agreement"] is False
    assert res["mismatch"] is False
    assert res["audio_direction"] == "negative"
    assert res["combined_confidence"] == pytest.approx(0.82)
    assert res["combined_reliability"] == pytest.approx(0.9)

    # text_unavailable flag
    res2 = fusion.fuse(voice, {"sentiment": None, "text_unavailable": True})
    assert res2["pattern_label"] == "voice-only (text unavailable)"
    assert res2["rule_ids_fired"] == ["voice_only_text_unavailable"]


def test_fusion_voice_only_abstention():
    """Voice-only path respects low reliability abstention threshold."""
    fusion = MultimodalFusion(abstention_threshold=0.35)
    voice = {"emotion": "sad", "confidence": 0.5, "voice_reliability": 0.20}
    res = fusion.fuse(voice, None)
    assert res["abstention"] is True
    assert "abstention_low_reliability" in res["rule_ids_fired"]


# ---------------------------------------------------------------------------
# Intensity v2 renormalization
# ---------------------------------------------------------------------------

def test_intensity_v2_renormalization_on_unavailable_text():
    """Intensity v2 drops text_negativity and renormalizes weights when text is unavailable."""
    ca = ConversationAnalyzer.__new__(ConversationAnalyzer)
    audio_seg = np.zeros(16000, dtype=np.float32)

    with patch("app.conversation_analysis.compute_prosodic_arousal") as mock_prosody, \
         patch("app.conversation_analysis.compute_voice_arousal") as mock_va:
        mock_prosody.return_value = SimpleNamespace(arousal=0.6, components={})
        mock_va.return_value = 0.8

        # When text_scores is empty or None: prosody (0.35) and voice (0.35) sum to 0.70 -> 50% each
        intensity, comp = ca._calculate_intensity_v2(
            voice_scores={"angry": 0.8},
            text_scores={},
            audio_segment=audio_seg,
            sr=16000,
            word_count=5,
            duration_seconds=1.0,
        )

        assert comp["text_negativity"] is None
        assert comp["weights"]["text_negativity"] == 0.0
        assert comp["weights"]["prosody"] == pytest.approx(0.5, abs=0.01)
        assert comp["weights"]["voice_arousal"] == pytest.approx(0.5, abs=0.01)
        # Expected: 0.5 * 0.6 + 0.5 * 0.8 = 0.70
        assert intensity == pytest.approx(0.70, abs=0.01)


# ---------------------------------------------------------------------------
# ASR Guards: non_english, forced language bypass, repetition, low_language_confidence
# ---------------------------------------------------------------------------

def test_asr_guards_non_english_and_forced_bypass():
    """Non-English transcript sets text_unavailable; forced language bypasses it."""
    vea = MagicMock()
    vea.analyze_segment.return_value = {
        "emotion": "happy", "confidence": 0.8, "scores": {"happy": 0.8, "neutral": 0.2},
    }
    sa = MagicMock()
    sa.analyze.return_value = {"sentiment": "positive", "confidence": 0.9, "scores": {"positive": 0.9}}

    stt = MagicMock()
    stt.language = "auto"
    stt.transcribe.return_value = {
        "text": "Bonjour tout le monde comment allez vous aujourd hui",
        "language": "fr",
        "language_probability": 0.95,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Bonjour tout le monde comment allez vous aujourd hui"}
        ],
    }

    ca = ConversationAnalyzer(
        emotion_analyzer=vea,
        speech_to_text=stt,
        sentiment_analyzer=sa,
        fusion=MultimodalFusion(),
    )

    dummy_audio = np.zeros(32000, dtype=np.float32)
    with patch("app.conversation_analysis.load_and_preprocess_audio", return_value=(dummy_audio, 16000, 2.0)):
        res = ca.analyze("french_test.wav")

        assert "non_english" in res["asr_flags"]
        seg = res["segments"][0]
        assert seg["text_unavailable"] is True
        assert seg["text_sentiment"] is None
        assert seg["fusion"]["pattern_label"] == "voice-only (text unavailable)"
        sa.analyze.assert_not_called()

    # Now test forced language bypass: user forced 'fr' or 'en'
    stt.language = "fr"
    sa.reset_mock()
    with patch("app.conversation_analysis.load_and_preprocess_audio", return_value=(dummy_audio, 16000, 2.0)):
        res_forced = ca.analyze("french_forced.wav")
        assert "non_english" not in res_forced["asr_flags"]
        seg_forced = res_forced["segments"][0]
        assert seg_forced["text_unavailable"] is False
        sa.analyze.assert_called_once()


def test_asr_guard_repetition():
    """Repetition ratio >= REPETITION_RATIO_MAX with >= 8 tokens flags repetition and skips sentiment."""
    vea = MagicMock()
    vea.analyze_segment.return_value = {
        "emotion": "neutral", "confidence": 0.7, "scores": {"neutral": 0.7},
    }
    sa = MagicMock()
    stt = MagicMock()
    stt.language = "en"
    # Repetitive hallucination text: 10 tokens, only 2 unique ("the", "dog") -> rep_ratio = 1 - 2/10 = 0.80
    rep_text = "the dog the dog the dog the dog the dog"
    stt.transcribe.return_value = {
        "text": rep_text,
        "language": "en",
        "language_probability": 0.95,
        "segments": [
            {"start": 0.0, "end": 3.0, "text": rep_text}
        ],
    }

    ca = ConversationAnalyzer(
        emotion_analyzer=vea,
        speech_to_text=stt,
        sentiment_analyzer=sa,
        fusion=MultimodalFusion(),
    )

    dummy_audio = np.zeros(48000, dtype=np.float32)
    with patch("app.conversation_analysis.load_and_preprocess_audio", return_value=(dummy_audio, 16000, 3.0)):
        res = ca.analyze("rep_test.wav")
        seg = res["segments"][0]
        assert "repetition" in seg["asr_flags"]
        assert seg["text_unavailable"] is True
        assert seg["text_sentiment"] is None
        sa.analyze.assert_not_called()


def test_asr_guard_low_language_confidence_warning_only():
    """Low language confidence flags a warning but does NOT skip sentiment."""
    vea = MagicMock()
    vea.analyze_segment.return_value = {
        "emotion": "neutral", "confidence": 0.7, "scores": {"neutral": 0.7},
    }
    sa = MagicMock()
    sa.analyze.return_value = {"sentiment": "neutral", "confidence": 0.8, "scores": {"neutral": 0.8}}
    stt = MagicMock()
    stt.language = "auto"
    stt.transcribe.return_value = {
        "text": "Hello this is a short audio test sample",
        "language": "en",
        "language_probability": 0.40,  # below LOW_LANG_CONF (0.50)
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello this is a short audio test sample"}
        ],
    }

    ca = ConversationAnalyzer(
        emotion_analyzer=vea,
        speech_to_text=stt,
        sentiment_analyzer=sa,
        fusion=MultimodalFusion(),
    )

    dummy_audio = np.zeros(32000, dtype=np.float32)
    with patch("app.conversation_analysis.load_and_preprocess_audio", return_value=(dummy_audio, 16000, 2.0)):
        res = ca.analyze("low_conf_test.wav")
        assert "low_language_confidence" in res["asr_flags"]
        assert "non_english" not in res["asr_flags"]
        seg = res["segments"][0]
        assert seg["text_unavailable"] is False
        assert seg["text_sentiment"] is not None
        sa.analyze.assert_called_once()


# ---------------------------------------------------------------------------
# Ribbon and Exports None-safety
# ---------------------------------------------------------------------------

def test_exports_and_ribbon_none_safe():
    """Verify build_json_export_payload handles ASR fields and text_unavailable safely."""
    conv_result = {
        "language": "es",
        "language_probability": 0.85,
        "asr_flags": ["non_english"],
        "segments": [
            {
                "segment_index": 1,
                "start_seconds": 0.0,
                "end_seconds": 1.5,
                "transcript": "Hola amigo",
                "voice_emotion": {"emotion": "happy", "confidence": 0.8},
                "text_sentiment": None,
                "text_unavailable": True,
                "asr_flags": ["non_english"],
                "fusion": {
                    "pattern_label": "voice-only (text unavailable)",
                    "agreement": False,
                    "mismatch": False,
                    "audio_direction": "positive",
                    "text_direction": "unavailable",
                },
                "intensity": 0.65,
            }
        ],
    }

    payload = build_json_export_payload(conv_result)
    assert payload["language"] == "es"
    assert payload["language_probability"] == 0.85
    assert payload["asr_flags"] == ["non_english"]
    seg0 = payload["segments"][0]
    assert seg0["language"] == "es"
    assert seg0["asr_flags"] == ["non_english"]
    assert seg0["text_sentiment"] is None
