"""
Tests for Phase 8: Web Hygiene, Deployment Hardening, and Upload Safety.
Verifies upload hygiene, safe error handling without tracebacks, privacy/limitations
claims, and web static asset structure.
"""

import os
import tempfile
from pathlib import Path
import pytest

from app.audio_processor import AudioValidationError, validate_audio_file, load_and_preprocess_audio
from app.microphone import categorize_pipeline_error
import config
from ui.pages.privacy_view import render_privacy_page
from ui.pages.limitations_view import render_limitations_page


def test_upload_whitelist_and_size_limits():
    """Verify supported extensions and max file size boundaries in config."""
    assert ".wav" in config.SUPPORTED_AUDIO_EXTENSIONS
    assert ".mp3" in config.SUPPORTED_AUDIO_EXTENSIONS
    assert ".m4a" in config.SUPPORTED_AUDIO_EXTENSIONS
    assert ".flac" in config.SUPPORTED_AUDIO_EXTENSIONS
    assert config.MAX_AUDIO_FILE_SIZE_MB == 50.0
    assert config.MIN_AUDIO_DURATION_SECONDS <= 1.0


def test_empty_file_validation_error():
    """Verify 0-byte audio file triggers AudioValidationError and safe friendly message."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    try:
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(temp_path)
        assert "empty" in str(exc_info.value).lower()

        title, guidance = categorize_pipeline_error(exc_info.value)
        assert "Empty" in title or "Silent" in title
        assert "traceback" not in guidance.lower()
        assert "file:" not in guidance.lower()
    finally:
        os.unlink(temp_path)


def test_oversized_file_validation_error(monkeypatch):
    """Verify file exceeding size limit triggers AudioValidationError and friendly guidance."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"RIFF" + b"\x00" * 200)
        temp_path = f.name
    try:
        # Temporarily set MAX_AUDIO_FILE_SIZE_MB to a tiny limit (0.0001 MB = ~100 bytes)
        monkeypatch.setattr("app.audio_processor.MAX_AUDIO_FILE_SIZE_MB", 0.0001)
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(temp_path)
        assert "exceeds maximum allowed limit" in str(exc_info.value).lower()

        title, guidance = categorize_pipeline_error(exc_info.value)
        assert "Size Exceeded" in title
        assert "traceback" not in guidance.lower()
    finally:
        os.unlink(temp_path)


def test_corrupted_file_validation_error():
    """Verify non-audio corrupt file triggers AudioValidationError and friendly guidance."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(b"NOT_A_VALID_AUDIO_HEADER_CORRUPTED_BYTES_1234567890")
        temp_path = f.name
    try:
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(temp_path)
        assert "corrupted" in str(exc_info.value).lower() or "unreadable" in str(exc_info.value).lower()

        title, guidance = categorize_pipeline_error(exc_info.value)
        assert "Unsupported" in title or "Corrupted" in title
        assert "traceback" not in guidance.lower()
    finally:
        os.unlink(temp_path)


def test_unsupported_extension_error():
    """Verify rejected extension (.exe, .txt, etc.) triggers AudioValidationError."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Hello world")
        temp_path = f.name
    try:
        with pytest.raises(AudioValidationError) as exc_info:
            validate_audio_file(temp_path)
        assert "unsupported audio format" in str(exc_info.value).lower()

        title, guidance = categorize_pipeline_error(exc_info.value)
        assert "Unsupported" in title or "Corrupted" in title
        assert "traceback" not in guidance.lower()
    finally:
        os.unlink(temp_path)


def test_privacy_and_limitations_views_content():
    """Verify claims in Privacy and Limitations views comply with constraints."""
    import inspect
    privacy_src = inspect.getsource(render_privacy_page)
    limitations_src = inspect.getsource(render_limitations_page)

    # Disclaimers present
    assert "NOT" in privacy_src
    assert "clinical" in privacy_src.lower()
    assert "hiring" in privacy_src.lower()
    assert "surveillance" in privacy_src.lower()
    assert "deception" in privacy_src.lower()
    assert "educational" in privacy_src.lower() or "academic" in privacy_src.lower()

    # Limitations boundaries
    assert "heuristics" in limitations_src.lower()
    assert "human-in-the-loop" in limitations_src.lower()
    assert "deception" in limitations_src.lower()
    assert "sarcasm" in limitations_src.lower()


def test_static_assets_and_config():
    """Verify static robots.txt, sitemap.xml, and config.toml settings exist."""
    static_robots = Path("static/robots.txt")
    static_sitemap = Path("static/sitemap.xml")
    config_toml = Path(".streamlit/config.toml")

    assert static_robots.exists()
    robots_txt = static_robots.read_text(encoding="utf-8")
    assert "User-agent" in robots_txt
    assert "Disallow: /data/" in robots_txt

    assert static_sitemap.exists()
    sitemap_xml = static_sitemap.read_text(encoding="utf-8")
    assert "<urlset" in sitemap_xml

    assert config_toml.exists()
    toml_txt = config_toml.read_text(encoding="utf-8")
    assert "enableStaticServing = true" in toml_txt
    assert "maxUploadSize = 50" in toml_txt
