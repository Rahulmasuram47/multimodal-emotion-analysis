"""Phase 12: Tests for voice backend provenance and display transparency."""
from unittest.mock import patch, MagicMock
import pytest

from app.emotion_analyzer import VoiceEmotionAnalyzer
from ui.utils import get_voice_backend_info
from ui.components.export_bar import build_json_export_payload
from ui.components.transparency_panels import (
    render_model_transparency_section,
    render_fusion_explainer,
)


def test_emotion2vec_load_fallback_on_importerror():
    """Mocked ImportError in emotion2vec load gives fallback_active True and backend_used 'superb'."""
    with patch("app.voice_backends.emotion2vec.Emotion2VecBackend._load", side_effect=ImportError("No module named 'funasr'")), \
         patch("app.voice_backends.superb.SuperbBackend"):
        analyzer = VoiceEmotionAnalyzer(backend="emotion2vec")
        info = analyzer.backend_info()

        assert analyzer.fallback_active is True
        assert analyzer.backend_used == "superb"
        assert info["configured_backend"] == "emotion2vec"
        assert info["backend_used"] == "superb"
        assert info["fallback_active"] is True
        assert "funasr" in str(info["fallback_reason"])


def test_ui_helper_returns_model_id_and_not_recorded():
    """UI helper returns stored model_id for recorded result and 'not recorded' when missing."""
    res_recorded = {
        "voice_backend": {
            "configured_backend": "emotion2vec",
            "backend_used": "emotion2vec",
            "model_id": "emotion2vec/emotion2vec_plus_base",
            "display_name": "emotion2vec+ base",
            "fallback_active": False,
            "fallback_reason": None,
            "confidence_informative": False,
        }
    }
    info_rec = get_voice_backend_info(res_recorded)
    assert info_rec["model_id"] == "emotion2vec/emotion2vec_plus_base"
    assert info_rec["backend_used"] == "emotion2vec"

    # Empty result or result lacking voice_backend
    info_missing = get_voice_backend_info({})
    assert info_missing["model_id"] == "not recorded"
    assert info_missing["backend_used"] == "not recorded"
    assert info_missing["configured_backend"] == "not recorded"


def test_transparency_table_no_unknown_when_backend_set():
    """No 'unknown' appears in transparency table when backend metadata is present."""
    res = {
        "voice_backend": {
            "configured_backend": "emotion2vec",
            "backend_used": "emotion2vec",
            "model_id": "emotion2vec/emotion2vec_plus_base",
            "display_name": "emotion2vec+ base",
            "confidence_informative": False,
        }
    }
    with patch("streamlit.markdown") as mock_markdown, patch("streamlit.expander"):
        render_model_transparency_section(res)
        assert mock_markdown.called
        md_text = mock_markdown.call_args[0][0]
        assert "Source ID" in md_text
        assert "unknown" not in md_text.lower()
        assert "`emotion2vec/emotion2vec_plus_base`" in md_text


def test_export_contains_voice_backend_used():
    """Export payload contains voice_backend_used, model_id, fallback_reason, voice_confidence_informative."""
    res = {
        "segments": [],
        "voice_backend": {
            "configured_backend": "emotion2vec",
            "backend_used": "superb",
            "model_id": "superb/wav2vec2-base-superb-er",
            "fallback_active": True,
            "fallback_reason": "funasr missing",
            "confidence_informative": True,
        },
    }
    payload = build_json_export_payload(res)
    assert payload["voice_backend_used"] == "superb"
    assert payload["model_id"] == "superb/wav2vec2-base-superb-er"
    assert payload["fallback_reason"] == "funasr missing"
    assert payload["voice_confidence_informative"] is True


def test_explainer_not_calibrated_for_non_informative_backend():
    """'(not calibrated)' appears in the explainer for a non-informative backend."""
    res = {
        "voice_backend": {
            "configured_backend": "emotion2vec",
            "backend_used": "emotion2vec",
            "model_id": "emotion2vec/emotion2vec_plus_base",
            "confidence_informative": False,
        }
    }
    with patch("streamlit.markdown") as mock_markdown, patch("streamlit.expander"), patch("streamlit.latex"):
        render_fusion_explainer(res)
        assert mock_markdown.called
        explainer_text = "\n".join(str(call[0][0]) for call in mock_markdown.call_args_list)
        assert "(not calibrated)" in explainer_text


def test_helper_returns_live_info_when_no_result():
    """Returns voice_backend_live info when no analysis_result in session state."""
    live = {
        "configured_backend": "emotion2vec",
        "backend_used": "superb",
        "model_id": "superb/wav2vec2-base-superb-er",
        "display_name": "Wav2Vec2-SUPERB ER",
        "fallback_active": True,
        "fallback_reason": "funasr missing",
        "confidence_informative": True,
    }
    mock_ss = MagicMock()
    mock_ss.get.side_effect = lambda k, d=None: live if k == "voice_backend_live" else (None if k == "analysis_result" else d)
    # st is imported inside get_voice_backend_info() — patch the real streamlit module's session_state
    import streamlit as _st
    with patch.object(_st, "session_state", mock_ss):
        info = get_voice_backend_info()
    assert info["backend_used"] == "superb"
    assert info["fallback_active"] is True
    assert info["fallback_reason"] == "funasr missing"


def test_stored_result_wins_over_live_info():
    """analysis_result['voice_backend'] wins over voice_backend_live (Case 1 short-circuits)."""
    result_with_backend = {
        "voice_backend": {
            "configured_backend": "superb",
            "backend_used": "superb",
            "model_id": "superb/wav2vec2-base-superb-er",
            "display_name": "Wav2Vec2-SUPERB ER",
            "fallback_active": False,
            "fallback_reason": None,
            "confidence_informative": True,
        }
    }
    # Case 1: explicit result arg is passed — no session state needed
    info = get_voice_backend_info(result_with_backend)
    assert info["backend_used"] == "superb"
    assert info["fallback_active"] is False
    assert info["model_id"] == "superb/wav2vec2-base-superb-er"
