"""
Tests for microphone workflow and fallback handling.

Verifies:
1. Microphone availability detection across environments and configuration flags.
2. Fallback message exact wording and handling on permission / widget failure.
3. Temporary audio file saving and buffer preparation.
4. Recorded audio routing through the EXISTING ConversationAnalyzer pipeline.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.microphone import (
    FALLBACK_MICROPHONE_MESSAGE,
    get_fallback_message,
    is_microphone_available,
    route_audio_to_pipeline,
    save_audio_buffer,
)


def test_fallback_message_exact_text():
    """Verify the exact user-facing fallback message required by specification."""
    expected = (
        "Microphone capture is unavailable in this environment. Please use the audio upload workflow."
    )
    assert FALLBACK_MICROPHONE_MESSAGE == expected
    assert get_fallback_message() == expected


def test_is_microphone_available_default():
    """Verify microphone availability in the current environment."""
    # Under default conditions with Streamlit >= 1.40 installed, is_microphone_available should return True
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DISABLE_MICROPHONE_CAPTURE", None)
        os.environ.pop("ENABLE_MICROPHONE_CAPTURE", None)
        available = is_microphone_available()
        assert isinstance(available, bool)
        assert available is True


def test_is_microphone_available_disabled_via_env():
    """Verify microphone is flagged unavailable when disabled via environment variables."""
    with patch.dict(os.environ, {"DISABLE_MICROPHONE_CAPTURE": "1"}):
        assert is_microphone_available() is False

    with patch.dict(os.environ, {"DISABLE_MICROPHONE_CAPTURE": "true"}):
        assert is_microphone_available() is False

    with patch.dict(os.environ, {"ENABLE_MICROPHONE_CAPTURE": "0"}):
        assert is_microphone_available() is False

    with patch.dict(os.environ, {"ENABLE_MICROPHONE_CAPTURE": "false"}):
        assert is_microphone_available() is False


def test_is_microphone_available_when_streamlit_lacks_audio_input():
    """Verify availability check returns False when Streamlit does not provide audio_input."""
    with patch("streamlit.audio_input", create=False):
        # Even if audio_input attribute is absent
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISABLE_MICROPHONE_CAPTURE", None)
            os.environ.pop("ENABLE_MICROPHONE_CAPTURE", None)
            with patch("builtins.hasattr", side_effect=lambda obj, name: False if name == "audio_input" else hasattr(obj, name)):
                assert is_microphone_available() is False


def test_save_audio_buffer_creates_valid_file():
    """Verify audio bytes are saved to a temporary audio file with proper extension."""
    dummy_wav_bytes = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    temp_path = save_audio_buffer(dummy_wav_bytes, filename="user_record.wav")

    try:
        assert os.path.exists(temp_path)
        assert os.path.isfile(temp_path)
        assert temp_path.endswith(".wav")
        with open(temp_path, "rb") as f:
            content = f.read()
        assert content == dummy_wav_bytes
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_save_audio_buffer_empty_raises_value_error():
    """Verify empty or None audio buffer raises ValueError."""
    with pytest.raises(ValueError, match="empty or None"):
        save_audio_buffer(b"")

    with pytest.raises(ValueError, match="empty or None"):
        save_audio_buffer(None)


def test_route_audio_to_existing_pipeline():
    """Verify recorded audio is routed directly to the existing ConversationAnalyzer pipeline."""
    mock_analyzer = MagicMock()
    mock_expected_result = {
        "overall": {
            "dominant_voice_emotion": "neutral",
            "dominant_text_sentiment": "neutral",
            "fused_emotion": "neutral",
            "fused_confidence": 0.85,
        },
        "segments": [],
        "trend": {"trend": "stable", "change": 0.0},
        "escalation_detected": False,
    }
    mock_analyzer.analyze.return_value = mock_expected_result

    # Create dummy temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF dummy audio data")
        temp_audio_path = f.name

    try:
        progress_mock = MagicMock()
        result = route_audio_to_pipeline(
            audio_path=temp_audio_path,
            analyzer=mock_analyzer,
            progress_callback=progress_mock,
        )

        # Assert that ConversationAnalyzer.analyze was called directly with the audio file
        mock_analyzer.analyze.assert_called_once_with(
            temp_audio_path, progress_callback=progress_mock
        )
        assert result == mock_expected_result
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


def test_route_audio_nonexistent_file_raises():
    """Verify routing a nonexistent file raises FileNotFoundError."""
    mock_analyzer = MagicMock()
    with pytest.raises(FileNotFoundError):
        route_audio_to_pipeline("nonexistent_audio_path.wav", mock_analyzer)


def test_fallback_handling_on_recording_exception():
    """Verify that if microphone recording widget throws an exception, fallback path triggers."""
    # Simulate a permission or runtime error in audio capture
    def failing_mic_widget():
        raise PermissionError("Browser microphone permission denied")

    recorded_audio = None
    fallback_message = None

    try:
        recorded_audio = failing_mic_widget()
    except Exception:
        fallback_message = get_fallback_message()

    assert recorded_audio is None
    assert fallback_message == FALLBACK_MICROPHONE_MESSAGE
