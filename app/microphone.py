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
from typing import Any, Callable, Dict, Optional, Tuple

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


SYSTEM_NOTICES_AND_LIMITATIONS = {
    "system_notice": (
        "This application is a decision-support heuristic designed to extract computational "
        "acoustic and linguistic features. Outputs are technical indicators, not clinical, "
        "psychological, psychiatric, or deception assessments."
    ),
    "heuristic_nature": "Outputs are algorithmic heuristics intended strictly for advisory decision-support.",
    "sarcasm_and_deception": (
        "The system measures computational modality disagreement (incongruence); "
        "it does NOT detect sarcasm, lies, truth, deception, concealment, or a speaker's inner psychological state."
    ),
    "human_in_the_loop": (
        "All automated outputs, recommendations, and metrics are strictly advisory. "
        "Final decisions, interventions, and assessments must always be made by qualified human professionals exercising domain judgment."
    ),
    "operational_limitations": [
        "Absence of Ground Truth: Performance metrics reflect runtime heuristics until empirical evaluation is conducted on labelled corpora.",
        "Pretrained Cross-Corpus Shift: Models were pretrained on external datasets; acoustic and domain shifts may affect accuracy.",
        "Acoustic Discretization: Categorizes segments into discrete labels (neutral, happy, angry, sad); complex affective states may not be fully represented.",
        "Sarcasm & Deception Non-Detection: Modality incongruence does not indicate dishonesty or deception.",
        "Audio Overlap & Acoustics: Overlapping speech, background noise, microphone differences, and reverberation alter acoustic contours.",
        "Linguistic & Dialect Diversity: Speech recognition and sentiment vary with accents, dialects, and speaking styles.",
        "Escalation Heuristic: Escalation indicates continuous acoustic/linguistic intensity rise, not psychological crisis.",
        "Attention Score Heuristic: Attention score is a triage prioritization metric, not a risk or danger index.",
        "Mandatory Human-in-the-Loop: Automated inferences must be supervised and validated by qualified personnel.",
    ],
}


def build_json_export_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Create additive export payload containing full analysis result plus structured notices and limitations."""
    payload = dict(result) if result else {}
    payload["notices_and_limitations"] = SYSTEM_NOTICES_AND_LIMITATIONS
    return payload


def categorize_pipeline_error(exc: Exception) -> Tuple[str, str]:
    """
    Categorize a pipeline exception into a user-facing title and actionable guidance.

    Returns:
        Tuple[str, str]: (error_title, user_action_message)
    """
    msg = str(exc).lower()

    # 1. Empty or silent audio
    if (
        "empty" in msg
        or "silent" in msg
        or "0 duration" in msg
        or "0 bytes" in msg
        or "no audio frames" in msg
        or "no speech or conversation" in msg
        or "no usable segments" in msg
        or "no valid conversation segments" in msg
        or "could not be decoded" in msg
    ):
        return (
            "Empty or Silent Audio Detected",
            "The audio recording appears to be silent or contains no detectable spoken conversation. Please verify that your microphone is working properly, speak clearly into the microphone, or upload an audio file containing audible conversation.",
        )

    # 2. Extremely short audio
    if (
        "shorter than minimum" in msg
        or "too short" in msg
        or "minimum required" in msg
        or "minimum analysis threshold" in msg
        or ("duration" in msg and "short" in msg)
    ):
        return (
            "Audio Too Short for Analysis",
            "The audio recording is too short for reliable emotion extraction and conversation dynamics. Please provide a recording of at least 1.0 to 2.0 seconds with spoken dialogue.",
        )

    # 3. Corrupted or unsupported format
    if (
        "unsupported" in msg
        or "corrupt" in msg
        or "failed to load audio" in msg
        or "soundfile" in msg
        or "audiovalidationerror" in msg
        or ("format" in msg and ("invalid" in msg or "not supported" in msg))
    ):
        return (
            "Unsupported or Corrupted Audio Format",
            "The provided audio file could not be decoded or is corrupted. Please try a different audio file in a supported format (WAV, FLAC, MP3, OGG, or M4A).",
        )

    # 4. Model inference failure
    if (
        "cuda" in msg
        or "torch" in msg
        or "whisper" in msg
        or "wav2vec" in msg
        or "roberta" in msg
        or "out of memory" in msg
        or "inference" in msg
        or isinstance(exc, (RuntimeError, MemoryError))
    ):
        return (
            "Model Inference Failure",
            "A neural model inference error occurred during acoustic or linguistic processing. Please verify available system resources (CPU/RAM) and retry your analysis.",
        )

    # 5. General fallback
    return (
        "Multimodal Analysis Pipeline Failed",
        "An unexpected error occurred during multimodal analysis. Please check your audio input and try again.",
    )



def extract_escalation_intensity_change(escalation_data: Dict[str, Any]) -> float:
    """Safely extract intensity change, correctly reading 'intensity_change' key produced by backend."""
    if not isinstance(escalation_data, dict):
        return 0.0
    return float(escalation_data.get("intensity_change", escalation_data.get("intensity_increase", 0.0)) or 0.0)


# Session state keys required for a clean reset to the initial landing screen
RESET_SESSION_KEYS = (
    "analysis_result",
    "active_file_id",
    "audio_bytes",
    "audio_format",
    "seek_time",
    "_last_rec_bytes",
    "_last_up_bytes",
    "_last_active_source",
    "audio_file_uploader",
    "audio_mic_recorder",
)


def reset_analysis_session_state(session_state: Any) -> None:
    """
    Clear all session state keys associated with analysis results and audio inputs,
    allowing the application to cleanly return to the initial upload/record landing screen.
    """
    if session_state is None:
        return

    # Increment uploader version to mount fresh widget instances
    current_ver = session_state.get("uploader_version", 0)
    try:
        session_state["uploader_version"] = current_ver + 1
    except Exception:
        pass

    # Pop explicit keys
    for key in RESET_SESSION_KEYS:
        if hasattr(session_state, "pop"):
            session_state.pop(key, None)
        elif key in session_state:
            del session_state[key]

    # Pop any versioned widget keys
    try:
        keys_to_clear = [
            k for k in list(session_state.keys())
            if str(k).startswith("audio_file_uploader") or str(k).startswith("audio_mic_recorder")
        ]
        for k in keys_to_clear:
            if hasattr(session_state, "pop"):
                session_state.pop(k, None)
            elif k in session_state:
                del session_state[k]
    except Exception:
        pass

