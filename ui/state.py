"""
Session state management for the Multimodal Emotion Analysis dashboard.
"""

from typing import Any, Dict
import streamlit as st

from app.microphone import RESET_SESSION_KEYS, reset_analysis_session_state

__all__ = [
    "init_session_state",
    "get_active_uploader_keys",
    "reset_session",
    "reset_analysis_session_state",
    "RESET_SESSION_KEYS",
]


def init_session_state() -> None:
    """Initialize default session state keys if not already set."""
    defaults = {
        "analysis_result": None,
        "active_file_id": None,
        "audio_bytes": None,
        "audio_format": None,
        "seek_time": None,
        "uploader_version": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_uploader_keys() -> tuple[str, str]:
    """Return versioned widget keys for file uploader and microphone recorder."""
    version = st.session_state.get("uploader_version", 0)
    uploader_key = f"audio_file_uploader_{version}" if version > 0 else "audio_file_uploader"
    mic_key = f"audio_mic_recorder_{version}" if version > 0 else "audio_mic_recorder"
    return uploader_key, mic_key


def reset_session() -> None:
    """Reset all analysis state and increment uploader version."""
    reset_analysis_session_state(st.session_state)
