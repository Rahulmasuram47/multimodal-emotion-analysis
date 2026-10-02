"""
Speech-to-Text Module for Multimodal Emotion Analysis.

Uses faster-whisper to convert speech audio into text
with timestamped speech segments.
"""

import logging
from typing import Any, Dict, List, Optional

from faster_whisper import WhisperModel
import numpy as np

import config

logger = logging.getLogger(__name__)


class SpeechToText:
    """Transcribe speech using the faster-whisper model."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ):
        # Falls back to config.STT_MODEL_SIZE when not explicitly passed,
        # so existing callers (ConversationAnalyzer, tests that pass
        # model_size="base" explicitly) keep working exactly as before —
        # config.py is now the single source of truth for the production
        # default, matching how VoiceEmotionAnalyzer was updated in Phase 3.
        self.model_size = model_size or config.STT_MODEL_SIZE
        self.device = device
        self.compute_type = compute_type

        # Falls back to config.STT_LANGUAGE. 'auto' preserves the exact
        # original behavior (Whisper auto-detects language per file).
        self.language = language or config.STT_LANGUAGE

        logger.info(
            "Loading Whisper model: %s "
            "(device=%s, compute_type=%s, language=%s)",
            self.model_size,
            device,
            compute_type,
            self.language,
        )

        self.model = WhisperModel(
            self.model_size,
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
            Dictionary containing:
            - full transcript
            - detected language
            - language probability
            - timestamped speech segments, each with avg_logprob and
              no_speech_prob (faster-whisper's native per-segment
              confidence signals) plus a derived is_low_confidence flag
        """

        logger.info("Transcribing audio: %s", audio_path)

        # language=None tells faster-whisper to auto-detect, matching the
        # original behavior exactly. Any other value forces that language
        # and skips detection uncertainty for that file.
        whisper_language = None if self.language == "auto" else self.language

        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
            },
            language=whisper_language,
        )

        segment_results: List[Dict[str, Any]] = []
        transcript_parts: List[str] = []

        for segment in segments:
            text = segment.text.strip()

            # Ignore empty Whisper segments.
            if not text:
                continue

            avg_logprob = float(getattr(segment, "avg_logprob", 0.0))
            no_speech_prob = float(getattr(segment, "no_speech_prob", 0.0))

            # Continuous text reliability calculation from avg_logprob and no_speech_prob
            # exp(avg_logprob) ranges in (0, 1]; scaled by (1 - no_speech_prob)
            raw_asr_conf = float(np.exp(np.clip(avg_logprob, -3.0, 0.0))) * (1.0 - np.clip(no_speech_prob, 0.0, 1.0))
            text_reliability = float(np.clip(round(raw_asr_conf, 3), 0.20, 1.0))

            is_low_confidence = (
                avg_logprob <= config.STT_LOW_CONFIDENCE_AVG_LOGPROB
                or no_speech_prob >= config.STT_HIGH_NO_SPEECH_PROB_THRESHOLD
                or text_reliability < 0.65
            )

            # Extract word-level timestamps if available
            words_list: List[Dict[str, Any]] = []
            for w in getattr(segment, "words", []) or []:
                words_list.append({
                    "word": getattr(w, "word", ""),
                    "start": float(getattr(w, "start", 0.0)),
                    "end": float(getattr(w, "end", 0.0)),
                    "probability": float(getattr(w, "probability", 1.0)),
                })

            segment_results.append(
                {
                    "start": float(segment.start),
                    "end": float(segment.end),
                    "text": text,
                    "avg_logprob": avg_logprob,
                    "no_speech_prob": no_speech_prob,
                    "text_reliability": text_reliability,
                    "is_low_confidence": is_low_confidence,
                    "words": words_list,
                }
            )

            transcript_parts.append(text)

        transcript = " ".join(transcript_parts).strip()
        no_speech = (len(segment_results) == 0)

        result = {
            "text": transcript,
            "language": getattr(info, "language", "unknown") if info else "unknown",
            "language_probability": float(getattr(info, "language_probability", 0.0)) if info else 0.0,
            "segments": segment_results,
            "no_speech_detected": no_speech,
        }

        logger.info(
            "Transcription completed: %d characters, %d segments (no_speech=%s)",
            len(transcript),
            len(segment_results),
            no_speech,
        )

        return result