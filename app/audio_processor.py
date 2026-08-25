"""
Audio Processing Module for Multimodal Emotion Analysis.
Handles validation, loading, resampling, normalization, and acoustic feature extraction.
"""
import os
import logging
from typing import Dict, Any, Tuple
import numpy as np
import soundfile as sf
import librosa

from config import (
    TARGET_SAMPLE_RATE,
    SUPPORTED_AUDIO_EXTENSIONS,
    MAX_AUDIO_DURATION_SECONDS,
    MIN_AUDIO_DURATION_SECONDS,
    N_MFCC,
    N_FFT,
    HOP_LENGTH,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class AudioValidationError(Exception):
    """Raised when an audio file fails validation (unsupported format, missing, empty, or corrupted)."""
    pass


class AudioProcessingError(Exception):
    """Raised when an error occurs during audio loading, resampling, or feature extraction."""
    pass


def validate_audio_file(file_path: str) -> Dict[str, Any]:
    """
    Validate that an audio file exists, is non-empty, has a supported format,
    and is not corrupted.

    Args:
        file_path: Path to the audio file.

    Returns:
        Dict containing basic metadata (file_path, file_size_bytes, extension).

    Raises:
        AudioValidationError: If the file is missing, unsupported, empty, or unreadable.
    """
    if not file_path:
        raise AudioValidationError("No audio file path provided.")

    if not os.path.exists(file_path):
        raise AudioValidationError(f"Audio file not found: '{file_path}'")

    if not os.path.isfile(file_path):
        raise AudioValidationError(f"Path is not a file: '{file_path}'")

    # Extension check
    _, ext = os.path.splitext(file_path)
    ext_lower = ext.lower()
    if ext_lower not in SUPPORTED_AUDIO_EXTENSIONS:
        supported_str = ", ".join(SUPPORTED_AUDIO_EXTENSIONS)
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. Supported formats: {supported_str}"
        )

    # Empty file check
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise AudioValidationError(f"Audio file is empty (0 bytes): '{file_path}'")

    # Corruption check by reading header/metadata
    try:
        info = sf.info(file_path)
        if info.frames == 0 or info.duration == 0:
            raise AudioValidationError(f"Audio file contains no audio frames or has 0 duration: '{file_path}'")
    except Exception as e:
        # If soundfile cannot read header, try librosa as fallback cleanly
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dur = librosa.get_duration(path=file_path)
                if dur <= 0:
                    raise AudioValidationError(f"Audio file has zero duration: '{file_path}'")
            except Exception as inner_e:
                raise AudioValidationError(
                    f"Audio file is corrupted or unreadable: {e}"
                ) from inner_e

    logger.info(f"Audio validation passed: '{file_path}' (size: {file_size} bytes)")
    return {
        "file_path": file_path,
        "file_size_bytes": file_size,
        "extension": ext_lower,
    }


def load_and_preprocess_audio(
    file_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    max_duration: float = MAX_AUDIO_DURATION_SECONDS,
    min_duration: float = MIN_AUDIO_DURATION_SECONDS,
) -> Tuple[np.ndarray, int, float]:
    """
    Load, convert to mono, resample to target_sr, normalize amplitude, and validate duration.

    Args:
        file_path: Path to the audio file.
        target_sr: Target sampling rate in Hz (default: 16000).
        max_duration: Maximum allowed audio duration in seconds.
        min_duration: Minimum required audio duration in seconds.

    Returns:
        Tuple of (audio_signal as 1D np.ndarray, sample_rate as int, duration_seconds as float).

    Raises:
        AudioValidationError: If validation fails or duration is out of allowed bounds.
        AudioProcessingError: If loading or preprocessing fails.
    """
    validate_audio_file(file_path)

    logger.info(f"Loading audio from: '{file_path}' with target_sr={target_sr} Hz")
    try:
        y, sr = librosa.load(file_path, sr=target_sr, mono=True)
    except Exception as e:
        # Fallback to soundfile reading
        try:
            data, orig_sr = sf.read(file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Convert stereo to mono
            if orig_sr != target_sr:
                y = librosa.resample(data.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
                sr = target_sr
            else:
                y = data.astype(np.float32)
                sr = orig_sr
        except Exception as sf_err:
            raise AudioProcessingError(f"Failed to load audio: {e} (soundfile fallback: {sf_err})") from e

    if len(y) == 0:
        raise AudioValidationError("Loaded audio data is empty.")

    duration = float(len(y) / sr)

    if duration < min_duration:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) is shorter than minimum required ({min_duration:.2f}s)."
        )

    if duration > max_duration:
        raise AudioValidationError(
            f"Audio duration ({duration:.2f}s) exceeds maximum allowed limit ({max_duration:.2f}s)."
        )

    # Peak normalization: scale to [-1.0, 1.0] safely
    max_amp = float(np.max(np.abs(y)))
    if max_amp > 1e-6:
        y = y / max_amp
    else:
        logger.warning("Audio signal has near-zero amplitude (near silence).")

    logger.info(f"Audio preprocessed successfully: {len(y)} samples, {duration:.3f}s at {sr} Hz")
    return y.astype(np.float32), sr, duration


def extract_acoustic_features(
    y: np.ndarray,
    sr: int = TARGET_SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> Dict[str, Any]:
    """
    Extract key acoustic features from the audio signal:
    - MFCCs (matrix and summary statistics)
    - RMS Energy / Intensity (mean, max, std)
    - Pitch / Fundamental Frequency F0 (mean, min, max, voiced fraction)
    - Zero-Crossing Rate (mean, std)
    - Spectral Centroid (mean)

    Args:
        y: 1D audio signal array.
        sr: Sampling rate in Hz.
        n_mfcc: Number of MFCC coefficients to extract.

    Returns:
        Dict containing raw feature arrays and summarized scalar statistics.
    """
    if y is None or len(y) == 0:
        raise AudioProcessingError("Cannot extract features from empty or None audio array.")

    try:
        duration = float(len(y) / sr)

        # 1. MFCC Extraction
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
        mfcc_mean = np.mean(mfcc, axis=1).tolist()
        mfcc_std = np.std(mfcc, axis=1).tolist()

        # 2. RMS Energy / Intensity
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)
        rms_mean = float(np.mean(rms))
        rms_max = float(np.max(rms))
        rms_std = float(np.std(rms))

        # 3. Pitch / F0 (Fundamental frequency) using Yin algorithm
        # Speech range: 50 Hz to 500 Hz
        try:
            f0 = librosa.yin(y=y, fmin=50, fmax=500, sr=sr, hop_length=HOP_LENGTH)
            valid_f0 = f0[np.isfinite(f0) & (f0 > 0)]
            if len(valid_f0) > 0:
                pitch_mean_hz = float(np.mean(valid_f0))
                pitch_min_hz = float(np.min(valid_f0))
                pitch_max_hz = float(np.max(valid_f0))
                voiced_fraction = float(len(valid_f0) / len(f0))
            else:
                pitch_mean_hz = 0.0
                pitch_min_hz = 0.0
                pitch_max_hz = 0.0
                voiced_fraction = 0.0
        except Exception as pitch_err:
            logger.warning(f"Pitch extraction warning: {pitch_err}")
            pitch_mean_hz = 0.0
            pitch_min_hz = 0.0
            pitch_max_hz = 0.0
            voiced_fraction = 0.0

        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH, frame_length=N_FFT)
        zcr_mean = float(np.mean(zcr))
        zcr_std = float(np.std(zcr))

        # 5. Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, hop_length=HOP_LENGTH, n_fft=N_FFT
        )
        spectral_centroid_mean = float(np.mean(spectral_centroid))

        features = {
            "duration_seconds": duration,
            "sample_rate": sr,
            "total_samples": len(y),
            # MFCC Summary
            "n_mfcc": n_mfcc,
            "mfcc_shape": list(mfcc.shape),
            "mfcc_mean": mfcc_mean,
            "mfcc_std": mfcc_std,
            "mfcc_raw": mfcc,
            # Energy Summary
            "rms_mean": rms_mean,
            "rms_max": rms_max,
            "rms_std": rms_std,
            "rms_raw": rms,
            # Pitch Summary
            "pitch_mean_hz": pitch_mean_hz,
            "pitch_min_hz": pitch_min_hz,
            "pitch_max_hz": pitch_max_hz,
            "voiced_fraction": voiced_fraction,
            # Spectral / ZCR
            "zcr_mean": zcr_mean,
            "zcr_std": zcr_std,
            "spectral_centroid_mean": spectral_centroid_mean,
        }

        logger.info(
            f"Extracted features: MFCC shape={mfcc.shape}, RMS mean={rms_mean:.4f}, "
            f"Pitch mean={pitch_mean_hz:.1f}Hz, ZCR mean={zcr_mean:.4f}"
        )
        return features

    except Exception as e:
        raise AudioProcessingError(f"Failed to extract acoustic features: {e}") from e