"""
Microphone Recording Workflow & Fallback Utilities.

Provides helper routines for browser microphone capture, audio saving/conversion,
fallback detection, and routing recorded audio into the existing multimodal
analysis pipeline.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Mandatory user-facing fallback message when microphone capture is unavailable
FALLBACK_MICROPHONE_MESSAGE = (
    "Microphone capture is unavailable in this environment. Please use the audio upload workflow."
)


def is_microphone_available() -> bool:
    """
    Check if microphone audio capture is supported in the current environment.

    Verifies:
    1. Streamlit library provides the `audio_input` widget.
    2. Microphone capture is not explicitly disabled via environment configuration
       (e.g., DISABLE_MICROPHONE_CAPTURE=1 or ENABLE_MICROPHONE_CAPTURE=0).

    Returns:
        bool: True if microphone recording widget is available and enabled.
    """
    # Check environment override flags (useful for headless / server restrictions / tests)
    if os.environ.get("DISABLE_MICROPHONE_CAPTURE", "").lower() in ("1", "true", "yes"):
        return False
    if os.environ.get("ENABLE_MICROPHONE_CAPTURE", "").lower() in ("0", "false", "no"):
        return False

    try:
        import streamlit as st
        return hasattr(st, "audio_input") and callable(getattr(st, "audio_input", None))
    except ImportError:
        return False


def get_fallback_message() -> str:
    """Return the exact user-facing fallback message."""
    return FALLBACK_MICROPHONE_MESSAGE


def save_audio_buffer(
    audio_bytes: bytes,
    filename: Optional[str] = None,
    suffix: Optional[str] = None,
) -> str:
    """
    Save raw audio bytes to a temporary audio file suitable for pipeline ingestion.

    Args:
        audio_bytes: The raw byte content of the recorded or uploaded audio.
        filename: Optional filename used to infer file extension.
        suffix: Optional explicit suffix (e.g. '.wav').

    Returns:
        str: The absolute path to the saved temporary audio file.

    Raises:
        ValueError: If audio_bytes is empty or None.
    """
    if not audio_bytes:
        raise ValueError("Audio buffer is empty or None.")

    if not suffix:
        if filename:
            suffix = Path(filename).suffix or ".wav"
        else:
            suffix = ".wav"

    if not suffix.startswith("."):
        suffix = f".{suffix}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(audio_bytes)
        return temp_audio.name


def route_audio_to_pipeline(
    audio_path: str,
    analyzer: Any,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> Dict[str, Any]:
    """
    Pass the audio file through the existing ConversationAnalyzer pipeline.

    This ensures recorded microphone audio reuses the exact same
    AudioProcessor -> Wav2Vec2 -> Whisper -> RoBERTa -> Fusion -> Conversation Analysis
    pipeline without code duplication.

    Args:
        audio_path: Path to the audio file.
        analyzer: An instance of ConversationAnalyzer.
        progress_callback: Optional progress reporter callback.

    Returns:
        Dict[str, Any]: Analysis result dictionary from the pipeline.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at path: {audio_path}")

    return analyzer.analyze(audio_path, progress_callback=progress_callback)
