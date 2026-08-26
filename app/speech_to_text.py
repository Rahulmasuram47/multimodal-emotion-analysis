"""
Speech-to-Text Module for Multimodal Emotion Analysis.

Uses faster-whisper to convert speech audio into text.
"""

import logging
from typing import Any, Dict, List

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribe speech using the faster-whisper model."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type

        logger.info(
            "Loading Whisper model: %s "
            "(device=%s, compute_type=%s)",
            model_size,
            device,
            compute_type,
        )

        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )

        logger.info("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary containing the transcript, language,
            language probability, and timestamped segments.
        """

        logger.info("Transcribing audio: %s", audio_path)

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
        )

        segment_results: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []

        for segment in segments:
            text = segment.text.strip()

            segment_results.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                }
            )

            if text:
                transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()

        result = {
            "text": transcript,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": segment_results,
        }

        logger.info(
            "Transcription completed: %d characters",
            len(transcript),
        )

        return result