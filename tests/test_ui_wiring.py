"""
Tests for UI Wiring, Escalation Bugfix, Panel Reachability, Error Categorization, and JSON Export.

Verifies:
1. Escalation card reads the backend 'intensity_change' key instead of defaulting to 0.00.
2. Both previously-orphaned panels are defined, reachable, and invoked in the main flow.
3. Specific user-facing error messages for corrupted audio, silent/empty audio, short audio, and inference failures.
4. JSON export output includes structured notices/limitations field additively.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.audio_processor import AudioProcessingError, AudioValidationError
from app.microphone import (
    RESET_SESSION_KEYS,
    SYSTEM_NOTICES_AND_LIMITATIONS,
    build_json_export_payload,
    categorize_pipeline_error,
    extract_escalation_intensity_change,
    reset_analysis_session_state,
)


def test_escalation_intensity_reads_backend_key():
    """Verify that the escalation intensity extractor correctly reads 'intensity_change' from backend."""
    # Backend conversation_analysis.py produces 'intensity_change'
    backend_escalation = {
        "escalation_detected": True,
        "reason": "Emotional intensity increased continuously.",
        "intensity_change": 0.3542,
    }

    val = extract_escalation_intensity_change(backend_escalation)
    assert val == 0.3542
    assert f"+{val:.2f}" == "+0.35"
    assert f"+{val:.2f}" != "+0.00"


def test_escalation_intensity_fallback_handling():
    """Verify fallback handling when intensity_change or intensity_increase is present or absent."""
    # Legacy key fallback
    legacy_data = {"intensity_increase": 0.28}
    assert extract_escalation_intensity_change(legacy_data) == 0.28

    # Empty / non-dict
    assert extract_escalation_intensity_change({}) == 0.0
    assert extract_escalation_intensity_change(None) == 0.0


def test_both_panels_called_in_streamlit_main_flow():
    """Verify via AST that render_model_transparency_section and render_professional_limitations_section are invoked."""
    app_path = Path("streamlit_app.py")
    assert app_path.exists()

    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    called_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_functions.add(node.func.id)

    assert "render_model_transparency_section" in called_functions, (
        "render_model_transparency_section is defined but not invoked in streamlit_app.py"
    )
    assert "render_professional_limitations_section" in called_functions, (
        "render_professional_limitations_section is defined but not invoked in streamlit_app.py"
    )


def test_error_message_corrupted_and_unsupported_audio():
    """Verify specific error message and action guidance for corrupted / unsupported audio format."""
    err = AudioValidationError("Unsupported audio format: '.xyz'. Supported: wav, flac, mp3, ogg, m4a")
    title, guidance = categorize_pipeline_error(err)

    assert "Unsupported" in title or "Corrupted" in title
    assert "supported format" in guidance.lower() or "different audio file" in guidance.lower()

    err2 = AudioProcessingError("Failed to load audio: corrupted header chunk (soundfile fallback failed)")
    title2, guidance2 = categorize_pipeline_error(err2)
    assert "Unsupported" in title2 or "Corrupted" in title2


def test_error_message_empty_and_silent_audio():
    """Verify specific error message and action guidance for empty or silent audio."""
    err = ValueError("The audio file appears to be empty or could not be decoded.")
    title, guidance = categorize_pipeline_error(err)

    assert "Empty" in title or "Silent" in title
    assert "microphone" in guidance.lower() or "audible" in guidance.lower()

    err2 = AudioValidationError("Audio file contains no audio frames or has 0 duration")
    title2, guidance2 = categorize_pipeline_error(err2)
    assert "Empty" in title2 or "Silent" in title2


def test_error_message_short_audio():
    """Verify specific error message and action guidance for extremely short audio."""
    err = ValueError("The audio duration (0.35s) is shorter than minimum analysis threshold (1.0s).")
    title, guidance = categorize_pipeline_error(err)

    assert "Short" in title
    assert "seconds" in guidance.lower() or "too brief" in guidance.lower()


def test_error_message_model_inference_failure():
    """Verify specific error message and action guidance for model inference failure."""
    err = RuntimeError("CUDA/Torch out of memory during Wav2Vec2 forward pass inference")
    title, guidance = categorize_pipeline_error(err)

    assert "Inference" in title
    assert "resources" in guidance.lower() or "memory" in guidance.lower()


def test_json_export_notices_and_limitations_field():
    """Verify JSON export payload additively includes structured notices/limitations field."""
    sample_result = {
        "overall": {"dominant_voice_emotion": "neutral", "fused_emotion": "neutral"},
        "segments": [{"segment_index": 0, "speaker": "Speaker 0"}],
        "recommendation": {"primary_recommendation": "Maintain neutral tone"},
        "processing_time": {"total_processing_seconds": 1.25},
    }

    export_payload = build_json_export_payload(sample_result)

    # Existing fields must be completely preserved
    assert export_payload["overall"] == sample_result["overall"]
    assert export_payload["segments"] == sample_result["segments"]
    assert export_payload["recommendation"] == sample_result["recommendation"]
    assert export_payload["processing_time"] == sample_result["processing_time"]

    # Structured notices and limitations must be present
    assert "notices_and_limitations" in export_payload
    notices = export_payload["notices_and_limitations"]

    assert "system_notice" in notices
    assert "heuristic" in notices["system_notice"].lower()
    assert "sarcasm_and_deception" in notices
    assert "human_in_the_loop" in notices
    assert "operational_limitations" in notices
    assert isinstance(notices["operational_limitations"], list)
    assert len(notices["operational_limitations"]) >= 5


def test_reset_analysis_session_state_clears_required_keys():
    """Verify that reset_analysis_session_state purges all required keys and increments uploader version."""
    expected_required_keys = {
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
    }

    # Verify that RESET_SESSION_KEYS constant contains all required keys
    assert expected_required_keys.issubset(set(RESET_SESSION_KEYS)), (
        f"Missing keys in RESET_SESSION_KEYS: {expected_required_keys - set(RESET_SESSION_KEYS)}"
    )

    # Populate mock session state dictionary
    mock_session = {
        "analysis_result": {"overall": "happy"},
        "active_file_id": "test_audio_12345",
        "audio_bytes": b"fake_wav_data",
        "audio_format": "audio/wav",
        "seek_time": 12.5,
        "_last_rec_bytes": b"fake_mic_data",
        "_last_up_bytes": b"fake_file_data",
        "_last_active_source": "mic",
        "audio_file_uploader": "some_file_obj",
        "audio_mic_recorder": "some_rec_obj",
        "audio_file_uploader_1": "versioned_uploader_obj",
        "audio_mic_recorder_1": "versioned_mic_obj",
        "uploader_version": 1,
        "persistent_unrelated_key": "should_remain",
    }

    reset_analysis_session_state(mock_session)

    # All analysis and input keys must be purged
    for key in expected_required_keys:
        assert key not in mock_session, f"Key {key} was not cleared from session state"

    # Versioned keys must also be purged
    assert "audio_file_uploader_1" not in mock_session
    assert "audio_mic_recorder_1" not in mock_session

    # Uploader version must have been incremented to guarantee fresh widget instantiation
    assert mock_session.get("uploader_version") == 2

    # Unrelated state must be preserved
    assert mock_session.get("persistent_unrelated_key") == "should_remain"


def test_wavesurfer_html_dark_theme_styling():
    """Verify that WaveSurfer template uses dark mode colors and prevents white iframe canvas."""
    app_source = Path("streamlit_app.py").read_text(encoding="utf-8")

    # Dark background color for HTML/body
    assert "background-color: #181D24" in app_source or "background: #181D24" in app_source
    # Dark color scheme meta tag to prevent browser engine white iframe canvas
    assert 'meta name="color-scheme" content="dark"' in app_source
    # Waveform background set to dark palette token
    assert "background: #101318" in app_source or "backgroundColor: '#101318'" in app_source
    assert "color-scheme: dark" in app_source


def test_css_audio_input_and_audio_styling():
    """Verify that custom CSS targets stAudioInput and stAudio widgets with dark theme rules."""
    app_source = Path("streamlit_app.py").read_text(encoding="utf-8")

    # stAudioInput container styling
    assert 'data-testid="stAudioInput"' in app_source
    assert ".stAudioInput" in app_source

    # stAudio / native player container styling
    assert 'data-testid="stAudio"' in app_source or ".stAudio" in app_source
    assert "color-scheme: dark" in app_source

    # WaveSurfer canvas container inside stAudioInput
    assert 'data-testid="stAudioInputWaveSurfer"' in app_source

    # Custom component iframe styling
    assert 'data-testid="stCustomComponentV1"' in app_source
