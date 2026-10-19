"""Phase 12-fix regression tests.

1. Pre-analysis UI helper reads analyzer display_name from voice_backend_live.
2. Legacy result (no voice_backend key) returns 'not recorded'.
3. backend_info() is safe on a pre-Phase-12 cached analyzer that lacks new attrs.
"""
from unittest.mock import patch, MagicMock
import pytest


# ---------------------------------------------------------------------------
# Test 1: pre-analysis helper returns analyzer display_name via voice_backend_live
# ---------------------------------------------------------------------------
def test_pre_analysis_helper_returns_live_display_name():
    """get_voice_backend_info() with no result returns display_name from voice_backend_live."""
    from ui.utils import get_voice_backend_info

    live = {
        "configured_backend": "emotion2vec",
        "backend_used": "emotion2vec",
        "model_id": "emotion2vec/emotion2vec_plus_base",
        "display_name": "emotion2vec+ base",
        "fallback_active": False,
        "fallback_reason": None,
        "confidence_informative": False,
    }
    mock_ss = MagicMock()
    mock_ss.get.side_effect = (
        lambda k, d=None: live if k == "voice_backend_live" else (None if k == "analysis_result" else d)
    )
    import streamlit as _st
    with patch.object(_st, "session_state", mock_ss):
        info = get_voice_backend_info()
    assert info["display_name"] == "emotion2vec+ base"
    assert info["backend_used"] == "emotion2vec"
    assert info["configured_backend"] == "emotion2vec"


# ---------------------------------------------------------------------------
# Test 2: legacy result without voice_backend key → "not recorded"
# ---------------------------------------------------------------------------
def test_legacy_result_without_voice_backend_returns_not_recorded():
    """Explicit result dict with no voice_backend key returns 'not recorded' strings."""
    from ui.utils import get_voice_backend_info

    legacy_result = {"segments": [], "language": "en"}  # no voice_backend key
    info = get_voice_backend_info(legacy_result)
    assert info["backend_used"] == "not recorded"
    assert info["model_id"] == "not recorded"
    assert info["configured_backend"] == "not recorded"
    assert info["display_name"] == "not recorded"


# ---------------------------------------------------------------------------
# Test 3: backend_info() safe on pre-Phase-12 cached analyzer (missing new attrs)
# ---------------------------------------------------------------------------
def test_backend_info_safe_on_pre_phase12_cached_analyzer():
    """backend_info() must not raise AttributeError on old cached analyzer instances.

    Simulates a @st.cache_resource-cached VoiceEmotionAnalyzer created before
    Phase 12 that lacks backend_used, configured_backend, fallback_active,
    fallback_reason attributes.
    """
    from app.emotion_analyzer import VoiceEmotionAnalyzer

    with (
        patch("app.voice_backends.emotion2vec.Emotion2VecBackend._load", side_effect=ImportError("no funasr")),
        patch("app.voice_backends.superb.SuperbBackend"),
    ):
        analyzer = VoiceEmotionAnalyzer(backend="superb")

    # Simulate pre-Phase-12 cached instance: delete the new attributes
    for attr in ("backend_used", "configured_backend", "fallback_active", "fallback_reason"):
        try:
            delattr(analyzer, attr)
        except AttributeError:
            pass

    # Must not raise
    info = analyzer.backend_info()

    # Should fall back to safe defaults
    assert "backend_used" in info
    assert "configured_backend" in info
    assert info["fallback_active"] is False
    assert info["fallback_reason"] is None
